#!/usr/bin/env python3

# Tkinter GUI front end for adversarial_watermark.py
# Designed to run without importing torch in frozen exe
# Build with: py -3.12 -m PyInstaller build_exe.spec

from __future__ import annotations

import json
import multiprocessing
import os
import platform
import queue
import shutil
import ssl
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zipfile

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

if os.name != "nt":
    raise SystemExit("Cloak is a Windows application. It does not run on this platform.")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

APP_NAME = "Cloak"
APP_TAGLINE = "your image to look like something harmful to CLIP models."
FIRST_RUN_PACKAGES = ["open_clip_torch", "torch", "torchvision", "pillow"]
FROZEN = bool(getattr(sys, "frozen", False))

# ---- Python version policy -----------------------------------------------------------

PREFERRED_PY = (3, 12)
SUPPORTED_PY = ((3, 12), (3, 11), (3, 10))
MIN_PY, MAX_PY = (3, 10), (3, 13)      # [MIN, MAX)

# ---- Default models -----------------------------------------------------------------

MODEL_PRESETS = {
    "ViT-B-32": "laion2b_s34b_b79k",
    "ViT-B-16": "laion2b_s34b_b88k",
    "ViT-L-14": "laion2b_s32b_b82k",
    "ViT-H-14": "laion2b_s32b_b79k",
    "ViT-g-14": "laion2b_s34b_b88k",
    "RN50": "openai",
}
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = MODEL_PRESETS[DEFAULT_MODEL]
DEFAULT_PROMPT = "watermark"
FIXED_CONTRAST = "image"       # Used to calculate meaningful P(target) percentage

# ---- Paths -------------------------------------------------------------------------

def resource_path(name: str) -> str:
    base = getattr(sys, "_MEIPASS", None)
    if base is None:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, name)

def app_data_dir() -> str:
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    path = os.path.join(root, "Cloak")
    os.makedirs(path, exist_ok=True)
    return path

DATA_DIR = app_data_dir()
PY_DIR = os.path.join(DATA_DIR, "python")
MARKER = os.path.join(DATA_DIR, "runtime.json")        # Private embedded Python
LOG_FILE = os.path.join(DATA_DIR, "install.log")       # Records the interpreter

def save_runtime(python_cmd: list[str], version: tuple[int, int] | None = None) -> None:
    with open(MARKER, "w", encoding="utf-8") as fh:
        json.dump({
            "python": python_cmd,
            "python_version": list(version) if version else None,
            "host_version": list(sys.version_info[:3]),
            "preferred": list(PREFERRED_PY),
        }, fh)

def load_runtime() -> list[str] | None:
    try:
        with open(MARKER, "r", encoding="utf-8") as fh:
            cmd = json.load(fh)["python"]
        return cmd if cmd and os.path.exists(cmd[0]) else None
    except Exception:
        return None

# ---- Environment and process helpers ---------------------------------------------------------------------------

def _child_env() -> dict:
    env = os.environ.copy()
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "_PYI_ARCHIVE_FILE"):
        env.pop(var, None)
    return env

def python_version(cmd: list[str]) -> tuple[int, int] | None:
    """(major, minor) of `cmd` if it is a supported Python WITH pip, else None."""
    try:
        probe = subprocess.run(
            cmd + ["-c", "import sys,importlib.util;"
                          "print(sys.version_info[0], sys.version_info[1],"
                          " 1 if importlib.util.find_spec('pip') else 0)"],
            capture_output=True, text=True, timeout=90,
            env=_child_env(), creationflags=CREATE_NO_WINDOW,
        )
        if probe.returncode != 0:
            return None
        major, minor, has_pip = (int(x) for x in probe.stdout.split())
        if not has_pip:
            return None
        if not (MIN_PY <= (major, minor) < MAX_PY):
            return None
        return (major, minor)
    except Exception:
        return None

def usable_python(cmd: list[str]) -> bool:
    return python_version(cmd) is not None

def worker_can_import(python_cmd: list[str]) -> bool:
    """True if torch / open_clip / PIL import in the given interpreter."""
    try:
        probe = subprocess.run(
            python_cmd + ["-c", "import torch, torchvision, open_clip; import PIL; print('OK')"],
            capture_output=True, text=True, timeout=120,
            env=_child_env(), creationflags=CREATE_NO_WINDOW,
        )
        return probe.returncode == 0 and probe.stdout.strip().endswith("OK")
    except Exception:
        return False

# ---- Private embedded Python -------------------------------------------------------

EMBED_VERSION = "3.12.7"
EMBED_URLS = {
    "arm64": f"https://www.python.org/ftp/python/{EMBED_VERSION}/python-{EMBED_VERSION}-embed-arm64.zip",
    "amd64": f"https://www.python.org/ftp/python/{EMBED_VERSION}/python-{EMBED_VERSION}-embed-amd64.zip",
    "win32": f"https://www.python.org/ftp/python/{EMBED_VERSION}/python-{EMBED_VERSION}-embed-win32.zip",
}
EMBED_SERIES = f"{PREFERRED_PY[0]}.{PREFERRED_PY[1]}"         # EG "3.12"
GET_PIP_URL = f"https://bootstrap.pypa.io/pip/{EMBED_SERIES}/get-pip.py" # Tied to embedded interpreter
GET_PIP_FALLBACK_URL = "https://bootstrap.pypa.io/get-pip.py"

def _embed_url() -> str:
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return EMBED_URLS["arm64"]
    if machine in ("amd64", "x86_64"):
        return EMBED_URLS["amd64"]
    return EMBED_URLS["win32"]

def _download(url: str, dest: str, emit) -> None:
    emit(f"Downloading {url}")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = step = 0
        with open(dest, "wb") as out:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total and done * 10 // total > step:
                    step = done * 10 // total
                    emit(f"  {done // 1048576} / {total // 1048576} MB")
    emit(f"Saved {os.path.basename(dest)}")

def install_embedded_python(emit) -> list[str]:
    exe = os.path.join(PY_DIR, "python.exe")
    if os.path.isfile(exe) and usable_python([exe]):
        emit("Reusing the Python Cloak installed earlier.")
        return [exe]

    emit("")
    emit(f"No suitable Python found. Installing a private Python {EMBED_VERSION} for Cloak "
         f"(~10 MB). Nothing already on this PC is changed.")

    shutil.rmtree(PY_DIR, ignore_errors=True)
    os.makedirs(PY_DIR, exist_ok=True)
    archive = os.path.join(DATA_DIR, "python-embed.zip")

    _download(_embed_url(), archive, emit)
    emit("Unpacking...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(PY_DIR)
    try:
        os.remove(archive)
    except OSError:
        pass

    for name in os.listdir(PY_DIR): # re-enables site imports
        if name.endswith("._pth"):
            path = os.path.join(PY_DIR, name)
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            rewritten = [ln[1:].strip() if ln.strip().startswith("#import site") else ln
                         for ln in lines]
            if "import site" not in rewritten:
                rewritten.append("import site")
            if "Lib\\site-packages" not in rewritten:
                rewritten.insert(1, "Lib\\site-packages")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(rewritten) + "\n")
            emit(f"Enabled site-packages in {name}")

    os.makedirs(os.path.join(PY_DIR, "Lib", "site-packages"), exist_ok=True)

    get_pip = os.path.join(PY_DIR, "get-pip.py")
    try:
        _download(GET_PIP_URL, get_pip, emit)
    except Exception as exc:  # noqa: BLE001
        emit(f"Pinned get-pip unavailable ({exc}); trying the generic one.")
        _download(GET_PIP_FALLBACK_URL, get_pip, emit)
    emit("Installing pip...")
    proc = subprocess.run([exe, get_pip, "--no-warn-script-location"],
                          capture_output=True, text=True, timeout=900,
                          env=_child_env(), creationflags=CREATE_NO_WINDOW)
    for line in (proc.stdout or "").splitlines()[-6:]:
        emit(line)
    if proc.returncode != 0 or not usable_python([exe]):
        emit((proc.stderr or "").strip()[-800:])
        raise RuntimeError("Could not set up the private Python. See " + LOG_FILE)

    emit(f"Private Python ready at {PY_DIR}")
    emit("")
    return [exe]

def find_python(emit=lambda _m: None) -> list[str]:
    if not FROZEN:
        version = python_version([sys.executable])
        if version:
            emit(f"Using the current interpreter (Python {version[0]}.{version[1]}).")
            return [sys.executable]

    candidates: list[list[str]] = []
    private = os.path.join(PY_DIR, "python.exe")
    if os.path.isfile(private):
        candidates.append([private])
    py = shutil.which("py")
    if py:
        for major, minor in SUPPORTED_PY:
            candidates.append([py, f"-{major}.{minor}"])
    for major, minor in SUPPORTED_PY:
        found = shutil.which(f"python{major}.{minor}")
        if found:
            candidates.append([found])
    generic = shutil.which("python")
    if generic and os.path.abspath(generic) != os.path.abspath(sys.executable):
        candidates.append([generic])

    best: tuple[int, list[str], tuple[int, int]] | None = None
    for cmd in candidates:
        version = python_version(cmd)
        if version is None:
            continue
        rank = 1000 if version == PREFERRED_PY else version[1]
        emit(f"Found Python {version[0]}.{version[1]}: {' '.join(cmd)}")
        if best is None or rank > best[0]:
            best = (rank, cmd, version)

    if best is not None:
        _, cmd, version = best
        if version != PREFERRED_PY:
            emit(f"Note: using Python {version[0]}.{version[1]}. "
                 f"{PREFERRED_PY[0]}.{PREFERRED_PY[1]} is the tested version, but this one is "
                 f"supported. The worker runs in its own process, so its version is independent "
                 f"of the app's.")
        emit(f"Chosen: {' '.join(cmd)}  (Python {version[0]}.{version[1]})")
        return cmd

    return install_embedded_python(emit)

def resolve_python(python_cmd: list[str]) -> list[str]:
    try:
        out = subprocess.run(
            python_cmd + ["-c", "import sys;print(sys.executable)"],
            capture_output=True, text=True, timeout=60,
            env=_child_env(), creationflags=CREATE_NO_WINDOW,
        )
        real = out.stdout.strip()
        if out.returncode == 0 and real and os.path.isfile(real):
            return [real]
    except Exception:
        pass
    return python_cmd

# ---- Installation -----------------------------------------------------------------------

def host_banner() -> str:
    kind = "frozen exe" if FROZEN else "source"
    v = sys.version_info
    return (f"Cloak host: {kind}, Python {v.major}.{v.minor}.{v.micro} "
            f"({platform.machine()}). The host never imports torch. The worker Python does.")

def run_install(emit) -> list[str]:
    # Find Python, pip-install everything, verify torch imports, return resolved python command
    emit(host_banner())
    python = find_python(emit)

    install = python + ["-m", "pip", "install", "--no-input", "--disable-pip-version-check"]
    commands = [install + FIRST_RUN_PACKAGES]
    reqs = resource_path("requirements.txt")
    if os.path.isfile(reqs):
        commands.append(install + ["-r", reqs])

    with open(LOG_FILE, "w", encoding="utf-8", errors="replace") as logf:
        for cmd in commands:
            emit("$ " + " ".join(os.path.basename(c) if i == 0 else c
                                 for i, c in enumerate(cmd)))
            logf.write("$ " + " ".join(cmd) + "\n")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env=_child_env(), creationflags=CREATE_NO_WINDOW,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                logf.write(line + "\n")
                if line:
                    emit(line)
            code = proc.wait()
            if code != 0:
                raise RuntimeError(f"pip exited with code {code}. Full output: {LOG_FILE}")

    emit("Checking that torch and open_clip import in that Python...")
    resolved = resolve_python(python)
    if not worker_can_import(resolved):
        raise RuntimeError(
            "pip reported success but torch / open_clip won't import in the target "
            f"Python.\nThat usually means a broken or mismatched wheel. See {LOG_FILE}."
        )
    version = python_version(resolved)
    emit(f"Import check passed on Python "
         f"{version[0]}.{version[1]}." if version else "Import check passed.")
    return resolved

# ---- Theme --------------------------------------------------------------------------

ACCENT = "#5ee0c0"
ACCENT_DIM = "#3aa88f"
BAD = "#ef6f6f"
INK = "#12161c"
LINE = "#2f3947"
MUTED = "#8b98a9"
PANEL = "#1a2029"
PANEL_2 = "#222a35"
TEXT = "#e6ebf2"
WARN = "#f2b45f"

MONO = ("Consolas", 12)
UI = ("Segoe UI", 12)
UI_BOLD = (UI[0], 12, "bold")
UI_TITLE = (UI[0], 24, "bold")
UI_SMALL = (UI[0], 11)
UI_SCORE = (MONO[0], 34, "bold")

def style_app(root: tk.Tk) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    root.configure(bg=INK)
    style.configure(".", background=INK, foreground=TEXT, font=UI)
    style.configure("TFrame", background=INK)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("TLabel", background=INK, foreground=TEXT)
    style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
    style.configure("Muted.TLabel", background=INK, foreground=MUTED, font=UI_SMALL)
    style.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED, font=UI_SMALL)
    style.configure("Title.TLabel", background=INK, foreground=TEXT, font=UI_TITLE)
    style.configure("Head.TLabel", background=INK, foreground=ACCENT, font=UI_BOLD)
    style.configure("TLabelframe", background=INK, foreground=MUTED, bordercolor=LINE)
    style.configure("TLabelframe.Label", background=INK, foreground=MUTED, font=UI_SMALL)
    style.configure("TEntry", fieldbackground=PANEL_2, foreground=TEXT,
                    bordercolor=LINE, insertcolor=TEXT, padding=7)
    style.configure("TSpinbox", fieldbackground=PANEL_2, foreground=TEXT,
                    bordercolor=LINE, arrowcolor=TEXT, insertcolor=TEXT, padding=6)
    style.map("TSpinbox",
              fieldbackground=[("readonly", PANEL_2), ("disabled", PANEL)],
              foreground=[("disabled", MUTED)])
    style.configure("TCombobox", fieldbackground=PANEL_2, background=PANEL_2,
                    foreground=TEXT, arrowcolor=TEXT, bordercolor=LINE, padding=6)
    style.map("TCombobox",
              fieldbackground=[("readonly", PANEL_2), ("disabled", PANEL),
                               ("focus", PANEL_2), ("active", PANEL_2)],
              background=[("readonly", PANEL_2), ("active", PANEL_2)],
              foreground=[("readonly", TEXT), ("disabled", MUTED)],
              selectbackground=[("readonly", PANEL_2)],   # kill the blue text highlight
              selectforeground=[("readonly", TEXT)])
    root.option_add("*TCombobox*Listbox.background", PANEL_2)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT_DIM)
    root.option_add("*TCombobox*Listbox.font", UI)
    style.configure("TButton", background=PANEL_2, foreground=TEXT,
                    bordercolor=LINE, focuscolor=LINE, padding=(14, 9))
    style.map("TButton", background=[("active", LINE), ("disabled", PANEL)],
              foreground=[("disabled", MUTED)])
    style.configure("Accent.TButton", background=ACCENT, foreground=INK,
                    font=UI_BOLD, padding=(14, 11), bordercolor=ACCENT)
    style.map("Accent.TButton", background=[("active", ACCENT_DIM), ("disabled", PANEL_2)],
              foreground=[("disabled", MUTED)])
    style.configure("Link.TButton", background=PANEL, foreground=ACCENT,
                    bordercolor=PANEL, focuscolor=PANEL, font=UI_SMALL, padding=(6, 3))
    style.map("Link.TButton", background=[("active", PANEL_2)],
              foreground=[("disabled", MUTED), ("active", ACCENT)])
    style.configure("TProgressbar", background=ACCENT, troughcolor=PANEL_2,
                    bordercolor=PANEL_2, lightcolor=ACCENT, darkcolor=ACCENT)

def make_log(parent, height=8) -> tk.Text:
    wrap = tk.Frame(parent, bg=LINE)
    text = tk.Text(wrap, height=height, bg=PANEL, fg=MUTED, font=MONO, bd=0,
                   padx=10, pady=8, wrap="word", insertbackground=TEXT,
                   highlightthickness=0, state="disabled")
    bar = ttk.Scrollbar(wrap, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=bar.set)
    text.pack(side="left", fill="both", expand=True, padx=1, pady=1)
    bar.pack(side="right", fill="y")
    wrap.pack(fill="both", expand=True)
    return text

def log_write(widget: tk.Text, line: str) -> None:
    widget.configure(state="normal")
    widget.insert("end", line + "\n")
    widget.see("end")
    widget.configure(state="disabled")

# ---- Setup screen ------------------------------------------------------------------

class SetupScreen(ttk.Frame):
    def __init__(self, master, on_ready):
        super().__init__(master, padding=28)
        self.on_ready = on_ready             # Called by resolved python cmd
        self.events: queue.Queue = queue.Queue()

        ttk.Label(self, text="Setting up " + APP_NAME, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            self,
            text="First launch only. Cloak is fetching PyTorch and OpenCLIP.",
            style="Muted.TLabel", wraplength=700, justify="left",
        ).pack(anchor="w", pady=(6, 18))

        self.status = ttk.Label(self, text="Starting...", style="Head.TLabel")
        self.status.pack(anchor="w")
        self.bar = ttk.Progressbar(self, mode="indeterminate", length=700)
        self.bar.pack(fill="x", pady=(10, 16))
        self.bar.start(12)
        self.log = make_log(self, height=15)
        self.retry = ttk.Button(self, text="Try again", command=self.start, style="Accent.TButton")
        self.start()

    def emit(self, line: str) -> None:
        self.events.put(("log", line))

    def start(self) -> None:
        self.retry.pack_forget()
        self.bar.start(12)
        self.status.configure(text="Preparing Python and dependencies...", foreground=ACCENT)
        threading.Thread(target=self._work, daemon=True).start()
        self.after(60, self._pump)

    def _work(self) -> None:
        try:
            existing = load_runtime()
            if existing and worker_can_import(existing):
                self.events.put(("log", "Dependencies already present."))
                self.events.put(("done", existing))
                return
            python = run_install(self.emit)
            save_runtime(python, python_version(python))
            self.events.put(("done", python))
        except Exception as exc:  # noqa: BLE001
            self.events.put(("log", ""))
            self.events.put(("log", traceback.format_exc(limit=1).strip()))
            self.events.put(("error", str(exc)))

    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    log_write(self.log, payload)
                elif kind == "done":
                    self.bar.stop()
                    self.status.configure(text="Ready.", foreground=ACCENT)
                    self.after(400, lambda p=payload: self.on_ready(p))
                    return
                elif kind == "error":
                    self.bar.stop()
                    self.status.configure(text="Setup failed", foreground=BAD)
                    log_write(self.log, payload)
                    self.retry.pack(anchor="w", pady=(14, 0))
                    return
        except queue.Empty:
            pass
        self.after(60, self._pump)

# ---- Main screen -------------------------------------------------------------------

class ProtectorScreen(ttk.Frame):
    PREVIEW = 300

    def __init__(self, master, python_cmd: list[str]):
        super().__init__(master, padding=20)
        self.python_cmd = python_cmd
        self.events: queue.Queue = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.worker_thread: threading.Thread | None = None
        self.cancelling = False

        self.source_path: str | None = None
        self.result_path: str | None = None
        self.result_meta: dict | None = None
        self._photo_before = None
        self._photo_after = None
        self._img_w = self._img_h = 0

        self.v_prompt = tk.StringVar(value=DEFAULT_PROMPT)
        self.v_model = tk.StringVar(value=DEFAULT_MODEL)
        self.v_eps = tk.DoubleVar(value=3)
        self.v_alpha = tk.DoubleVar(value=1.0)
        self.v_steps = tk.IntVar(value=250)
        self.v_seed = tk.IntVar(value=0)
        self.v_device = tk.StringVar(value="cpu")

        self._build()
        version = python_version(python_cmd)
        tag = f"Python {version[0]}.{version[1]}" if version else "Python"
        self.log_line(f"Worker: {tag} at {python_cmd[0]}")
        if version and version != PREFERRED_PY:
            self.log_line(f"(Tested on {PREFERRED_PY[0]}.{PREFERRED_PY[1]}; "
                          f"{version[0]}.{version[1]} is supported.)")
        self.after(80, self._pump)

    def _load_preview(self, path: str):   # Loads image without importing PIL into GUI
        ext = os.path.splitext(path)[1].lower()
        if ext in (".png", ".gif"):
            try:
                photo = tk.PhotoImage(file=path)
                return photo, photo.width(), photo.height()
            except Exception:
                pass
        tmp = os.path.join(DATA_DIR, "preview_src.png")    # Asks worker.py to transcode to temp PNG
        code = (
            "import sys;from PIL import Image;"
            "im=Image.open(sys.argv[1]).convert('RGB');"
            "im.thumbnail((1600,1600));im.save(sys.argv[2])"
        )
        proc = subprocess.run(self.python_cmd + ["-c", code, path, tmp],
                              capture_output=True, text=True, timeout=120,
                              env=_child_env(), creationflags=CREATE_NO_WINDOW)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "Could not read image.")
        photo = tk.PhotoImage(file=tmp)
        return photo, photo.width(), photo.height()

    def _fit(self, photo: tk.PhotoImage) -> tk.PhotoImage:
        w, h = photo.width(), photo.height()
        factor = max(1, -(-max(w, h) // (self.PREVIEW - 8)))  # Ceil division on the subsample
        if factor > 1:
            photo = photo.subsample(factor, factor)
        return photo

    # Round panels
    def _panel_rect(self, canvas, x1, y1, x2, y2, **kw):
        """Square-cornered filled rectangle for the hand-drawn panels."""
        return canvas.create_rectangle(x1, y1, x2, y2, **kw)

    # Layout
    def _build(self) -> None:
        header = ttk.Frame(self); header.pack(fill="x")
        ttk.Label(header, text=APP_NAME, style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="   " + APP_TAGLINE, style="Muted.TLabel").pack(side="left", pady=(10, 0))

        body = ttk.Frame(self); body.pack(fill="both", expand=True, pady=(14, 0))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, minsize=360)
        body.rowconfigure(0, weight=1)
        self._build_left(body)
        self._build_controls(body)

    # Left column: image panels, cosine similarity scores, progress bar, log
    def _build_left(self, parent) -> None:
        left = ttk.Frame(parent); left.grid(row=0, column=0, sticky="nsew", padx=(0, 18))

        strip = ttk.Frame(left); strip.pack(fill="x")
        self.canvas_before, self.btn_choose = self._image_panel(
            strip, "Original", "Upload Image", self.choose_image,
            empty="Upload an image...")
        self.canvas_after, self.btn_save = self._image_panel(
            strip, "Protected", "Download Image", self.save,
            empty="")
        self.btn_save.configure(state="disabled")

        self._build_score(left)
        self._build_footer(left)

    def _image_panel(self, parent, caption, btn_text, command, empty):
        """Centered caption, preview canvas, and a real ttk.Button underneath it."""
        holder = ttk.Frame(parent); holder.pack(side="left", expand=True, fill="both", padx=(0, 12))

        # caption centered over the frame
        ttk.Label(holder, text=caption, style="Head.TLabel", anchor="center").pack(
            fill="x", pady=(0, 6))

        size = self.PREVIEW
        canvas = tk.Canvas(holder, width=size, height=size, bg=INK,
                           highlightthickness=0, bd=0)
        canvas.pack()
        canvas._empty_text = empty
        self._draw_panel_bg(canvas, empty)

        btn = ttk.Button(holder, text=btn_text, command=command)
        btn.pack(fill="x", pady=(8, 0))
        return canvas, btn

    def _draw_panel_bg(self, canvas, placeholder=None):
        size = self.PREVIEW
        canvas.delete("all")
        self._panel_rect(canvas, 1, 1, size - 1, size - 1, fill=PANEL, outline=LINE,
                         tags="panelbg")
        if placeholder:
            canvas.create_text(size / 2, size / 2, text=placeholder, fill=MUTED, font=UI_SMALL)

    def _build_score(self, parent) -> None:
        size_h = 150
        card = tk.Canvas(parent, height=size_h, bg=INK, highlightthickness=0, bd=0)
        card.pack(fill="x", pady=(16, 0))
        self._score_card = card
        card.bind("<Configure>", lambda e: self._draw_score_bg(e.width, size_h))

        # widgets floated on top of the rounded canvas
        self.score_before = tk.Label(card, text="0.0000", bg=PANEL, fg=MUTED, font=UI_SCORE)
        self.score_arrow = tk.Label(card, text="\u2192", bg=PANEL, fg=MUTED, font=(UI[0], 26))
        self.score_after = tk.Label(card, text="0.0000", bg=PANEL, fg=MUTED, font=UI_SCORE)
        self.lbl_before = tk.Label(card, text="Before", bg=PANEL, fg=MUTED, font=UI_SMALL)
        self.lbl_after = tk.Label(card, text="After", bg=PANEL, fg=MUTED, font=UI_SMALL)
        self.card_title = tk.Label(card, text="Cosine Similarity Score to Prompt",
                                   bg=PANEL, fg=TEXT, font=UI_BOLD)
        self.detail = tk.Label(card, text="Upload an image to begin.", bg=PANEL, fg=MUTED,
                               font=UI_SMALL, justify="left", anchor="w")

        self.card_title.place(x=22, y=14)
        self.score_before.place(x=22, y=44)
        self.lbl_before.place(x=24, y=104)
        self.score_arrow.place(x=185, y=52)
        self.score_after.place(x=240, y=44)
        self.lbl_after.place(x=242, y=104)
        self.detail.place(x=430, y=54)

    def _draw_score_bg(self, w, h) -> None:
        c = self._score_card
        c.delete("bg")
        self._panel_rect(c, 1, 1, w - 1, h - 1, fill=PANEL, outline=LINE, tags="bg")
        c.tag_lower("bg")
        self.detail.configure(wraplength=max(120, w - 450))

    def _build_footer(self, parent) -> None:
        footer = ttk.Frame(parent); footer.pack(fill="both", expand=True, pady=(16, 0))
        line = ttk.Frame(footer); line.pack(fill="x")
        self.status = ttk.Label(line, text="Ready.", style="Head.TLabel"); self.status.pack(side="left")
        self.bar = ttk.Progressbar(line, mode="determinate", length=260); self.bar.pack(side="right")
        logbox = ttk.Frame(footer); logbox.pack(fill="both", expand=True, pady=(8, 0))
        self.log = make_log(logbox, height=7)

    # Right column: settings, description, credit
    def _build_controls(self, parent) -> None:
        right = ttk.Frame(parent); right.grid(row=0, column=1, sticky="nsew")

        settings = ttk.Labelframe(right, text="SETTINGS", padding=14); settings.pack(fill="x")
        settings.columnconfigure(1, weight=1)

        r = 0
        self._setting(settings, r, "Prompt",
                      ttk.Entry(settings, textvariable=self.v_prompt, font=UI)); r += 1
        self._setting(settings, r, "Model",
                      ttk.Combobox(settings, textvariable=self.v_model, state="readonly",
                                   values=list(MODEL_PRESETS.keys()), font=UI)); r += 1
        self._setting(settings, r, "Device",
                      ttk.Combobox(settings, textvariable=self.v_device, state="readonly",
                                   values=["cpu", "cuda"], width=8, font=UI)); r += 1
        self._setting(settings, r, "Budget",
                      self._num_entry(settings, self.v_eps, 1, 64, 1, integer=True)); r += 1
        self._setting(settings, r, "Steps",
                      self._num_entry(settings, self.v_steps, 10, 5000, 10, integer=True)); r += 1
        self._setting(settings, r, "Step Size",
                      self._num_entry(settings, self.v_alpha, 0.1, 16, 0.1)); r += 1
        self._setting(settings, r, "Seed",
                      self._num_entry(settings, self.v_seed, 0, 999999, 1, integer=True)); r += 1
        ttk.Label(settings, text="Higher budget is stronger but visible. More steps is stronger but slower. Smaller step size is stronger but requires more steps. Different seeds produce different noise. CPU is slower but CUDA requires Nvidia chips.\n\nCreated by Timothy.",
                  style="Muted.TLabel", wraplength=310, justify="left").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(12, 0))

        actions = ttk.Frame(right); actions.pack(fill="x", pady=(16, 0))
        self.btn_apply = ttk.Button(actions, text="Apply Protection", style="Accent.TButton",
                                    command=self.apply, state="disabled")
        self.btn_apply.pack(fill="x")
        self.btn_stop = ttk.Button(actions, text="Cancel", command=self.stop, state="disabled")
        self.btn_stop.pack(fill="x", pady=(8, 0))

    # A plain entry with scroll enabled
    def _num_entry(self, parent, var, lo, hi, step, integer=False):
        entry = ttk.Entry(parent, textvariable=var, font=UI, justify="left")

        def nudge(direction):
            try:
                value = float(var.get())
            except (tk.TclError, ValueError):
                value = lo
            value = min(hi, max(lo, value + direction * step))
            if integer:
                var.set(int(round(value)))
            else:
                var.set(round(value, 4))

        # Windows sends <MouseWheel> with delta +/-120 and scroll up = increase
        entry.bind("<MouseWheel>", lambda e: (nudge(1 if e.delta > 0 else -1), "break")[1])
        return entry

    def _setting(self, parent, row, label, widget) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
        widget.grid(row=row, column=1, sticky="ew", pady=5)

    # Helpers
    def log_line(self, text: str) -> None:
        log_write(self.log, text)

    def _place(self, canvas: tk.Canvas, photo: tk.PhotoImage) -> None:
        self._draw_panel_bg(canvas)
        canvas.create_image(self.PREVIEW / 2, self.PREVIEW / 2, image=photo)

    def clear_after(self) -> None:
        self.result_path = None
        self.result_meta = None
        self._photo_after = None
        self._draw_panel_bg(self.canvas_after, "Apply protection...")
        self.score_after.configure(text="0.0000", fg=MUTED)
        self.btn_save.configure(state="disabled")

    # Actions
    def choose_image(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.gif"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        try:
            photo, w, h = self._load_preview(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(APP_NAME, f"That file wouldn't open as an image.\n\n{exc}")
            return
        self.source_path = path
        self._img_w, self._img_h = w, h
        self._photo_before = self._fit(photo)
        self._place(self.canvas_before, self._photo_before)
        self.clear_after()
        self.score_before.configure(text="0.0000", fg=MUTED)
        self.detail.configure(text=f"Scores will appear after protection is applied.")
        self.btn_apply.configure(state="normal")
        self.log_line(f"Loaded {os.path.basename(path)} ({w}x{h})")

    def read_settings(self) -> dict | None:
        try:
            eps = float(self.v_eps.get()) / 255.0
            alpha = float(self.v_alpha.get()) / 255.0
            steps = int(self.v_steps.get())
            seed = int(self.v_seed.get())
        except (tk.TclError, ValueError):
            messagebox.showerror(APP_NAME, "Budget, steps, step size and seed must be numbers.")
            return None
        prompt = self.v_prompt.get().strip()
        if not prompt:
            messagebox.showerror(APP_NAME, "Type a prompt first, such as 'watermark' or 'blood'.")
            return None
        if steps < 1:
            messagebox.showerror(APP_NAME, "Steps must be at least 1.")
            return None
        return {
            "input": self.source_path,
            "output": os.path.join(DATA_DIR, "protected_result.png"),
            "prompt": prompt,
            "contrast_prompt": FIXED_CONTRAST,   # Drives P(target) readout
            "eps": eps,
            "alpha": alpha,
            "steps": steps,
            "seed": seed,
            "model": self.v_model.get().strip(),
            "pretrained": MODEL_PRESETS.get(self.v_model.get(), DEFAULT_PRETRAINED),
            "device": self.v_device.get(),
        }

    def apply(self) -> None:
        if not self.source_path or (self.proc and self.proc.poll() is None):
            return
        cfg = self.read_settings()
        if cfg is None:
            return
        self.clear_after()
        self.cancelling = False
        self.btn_apply.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_choose.configure(state="disabled")
        self.bar.configure(mode="determinate", maximum=cfg["steps"], value=0)
        self.status.configure(text="Loading model...", foreground=ACCENT)
        self.worker_thread = threading.Thread(target=self._run_worker, args=(cfg,), daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        self.cancelling = True
        self.status.configure(text="Stopping...", foreground=WARN)
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _run_worker(self, cfg: dict) -> None:
        put = self.events.put
        started = time.time()
        cfg_path = os.path.join(DATA_DIR, "job.json")
        try:
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            worker = resource_path("worker.py")
            put(("log", f"eps={round(cfg['eps'] * 255)}/255  alpha={cfg['alpha'] * 255:.1f}/255  "
                        f"steps={cfg['steps']}  seed={cfg['seed']}  prompt='{cfg['prompt']}'"))
            self.proc = subprocess.Popen(
                self.python_cmd + ["-u", worker, cfg_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env=_child_env(), creationflags=CREATE_NO_WINDOW,
            )
            assert self.proc.stdout is not None
            for raw in self.proc.stdout:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    put(("log", raw))
                    continue
                kind = event.get("t")
                if kind == "log":
                    put(("log", event["m"]))
                elif kind == "progress":
                    put(("progress", (event["step"], event["total"], event["sim"])))
                elif kind == "result":
                    put(("result", (event, time.time() - started)))
                elif kind == "error":
                    put(("log", event.get("trace", "").strip()))
                    put(("error", event.get("m", "worker failed")))
            code = self.proc.wait()
            if self.cancelling:
                put(("cancelled", None))
            elif code != 0:
                put(("error", f"The worker exited with code {code}. See the log above."))
        except Exception as exc:  # noqa: BLE001
            put(("log", traceback.format_exc(limit=2).strip()))
            put(("error", str(exc)))

    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.log_line(payload)
                elif kind == "progress":
                    step, total, sim = payload
                    self.bar.configure(value=step)
                    self.status.configure(text=f"Progress: {step}/{total}    Cosine Similarity Score: {sim:.4f}",
                                          foreground=ACCENT)
                elif kind == "result":
                    self._finish(*payload)
                elif kind == "cancelled":
                    self._idle("Stopped.", WARN)
                    self.log_line("Stopped before finishing. Nothing was saved.")
                elif kind == "error":
                    self._idle("Failed.", BAD)
                    messagebox.showerror(APP_NAME, payload)
        except queue.Empty:
            pass
        self.after(80, self._pump)

    def _idle(self, message: str, colour: str) -> None:
        self.status.configure(text=message, foreground=colour)
        self.btn_apply.configure(state="normal" if self.source_path else "disabled")
        self.btn_stop.configure(state="disabled")
        self.btn_choose.configure(state="normal")
        self.bar.configure(value=0)

    def _finish(self, event: dict, elapsed: float) -> None:
        self.result_path = event["output"]
        self.result_meta = event
        try:
            photo = tk.PhotoImage(file=self.result_path)
            self._photo_after = self._fit(photo)
            self._place(self.canvas_after, self._photo_after)
        except Exception:
            self._draw_panel_bg(self.canvas_after, "saved (no preview)")

        before, after = event["before"], event["after"]
        gain = after["cos_target"] - before["cos_target"]
        self.score_before.configure(text=f"{before['cos_target']:.4f}", fg=MUTED)
        self.score_after.configure(text=f"{after['cos_target']:.4f}",
                                   fg=ACCENT if gain > 0 else BAD)
        self.score_arrow.configure(fg=ACCENT if gain > 0 else MUTED)
        self.detail.configure(
            text=(f"Change  {gain:+.4f}\n"
                  f"L\u221e  {round(event['linf'] * 255)}/255\n"
                  f"{event['steps_run']} steps in {elapsed:.0f}s"))
        self.btn_save.configure(state="normal")
        self._idle("Done. Generally, scores above 0.4 are strong.", ACCENT)
        self.log_line(f"cos sim {before['cos_target']:.4f} -> {after['cos_target']:.4f} "
                      f"({gain:+.4f}) in {elapsed:.1f}s")

    def save(self) -> None:
        if not self.result_path or not os.path.isfile(self.result_path):
            return
        stem = os.path.splitext(os.path.basename(self.source_path or "image"))[0]
        path = filedialog.asksaveasfilename(
            title="Save protected image", defaultextension=".png",
            initialfile=f"{stem}_protected.png",
            filetypes=[("PNG (recommended)", "*.png"),
                       ("WebP", "*.webp"),
                       ("TIFF", "*.tiff"),
                       ("JPEG (not recommended)", "*.jpg")])
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".png", ""):
                shutil.copyfile(self.result_path, path)
            else:
                if ext in (".jpg", ".jpeg") and not messagebox.askyesno(
                    APP_NAME,
                    "JPEG compression can destroy the perturbation and undo the protection.\n\n"
                    "Save as JPEG anyway?"):
                    return
                code = ("import sys;from PIL import Image;"
                        "im=Image.open(sys.argv[1]);"
                        "im.save(sys.argv[2], quality=100, subsampling=0) if sys.argv[2].lower().endswith(('.jpg','.jpeg')) "
                        "else (im.save(sys.argv[2], lossless=True) if sys.argv[2].lower().endswith('.webp') "
                        "else im.save(sys.argv[2]))")
                proc = subprocess.run(self.python_cmd + ["-c", code, self.result_path, path],
                                      capture_output=True, text=True, timeout=120,
                                      env=_child_env(), creationflags=CREATE_NO_WINDOW)
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.strip() or "conversion failed")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(APP_NAME, f"Couldn't save the file.\n\n{exc}")
            return
        self.log_line(f"Saved {path}")
        self.status.configure(text=f"Saved to {os.path.basename(path)}", foreground=ACCENT)

# ---- Root ------------------------------------------------------------------------------

class CloakApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1140x840")
        self.minsize(1040, 760)
        style_app(self)
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        runtime = load_runtime()
        if runtime and worker_can_import(runtime):
            self.show_main(runtime)
        else:
            SetupScreen(self.container, on_ready=self.show_main).pack(fill="both", expand=True)

    def show_main(self, python_cmd: list[str]) -> None:
        for child in self.container.winfo_children():
            child.destroy()
        try:
            ProtectorScreen(self.container, python_cmd).pack(fill="both", expand=True)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            messagebox.showerror(APP_NAME, f"Cloak couldn't start:\n\n{exc}")
            self.destroy()

def enable_dpi_awareness() -> None:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)   # Windows 8.1+
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()     # Windows 7
        except Exception:
            pass

def main() -> None:
    multiprocessing.freeze_support()
    enable_dpi_awareness()
    CloakApp().mainloop()

if __name__ == "__main__":
    main()
