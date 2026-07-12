# -*- mode: python ; coding: utf-8 -*-

# Spec for compiling Cloak into an exe file using PyInstaller.
# Installs torch, torchvision and open_clip into %LOCALAPPDATA%\\Cloak\\deps on first launch.
# Build with: py -3.12 -m PyInstaller build_exe.spec

block_cipher = None

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("requirements.txt", "."),          # Read on first run by pip bootstrap
        ("worker.py", "."),                 # Run by installed Python instead of exe
        ("adversarial_watermark.py", "."),  # Imported by worker.py
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "torchvision", "open_clip", "PIL", "numpy", "transformers"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="Cloak",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # Set True while debugging to see tracebacks
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="cloak.ico",
)
