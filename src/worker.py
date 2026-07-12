#!/usr/bin/env python3

# Runs the adversarial attack in a subprocess using installed Python.
# Invoke with:  python -u worker.py <config.json>
# The config carries every setting including the input and output image paths.

"""
Contract (all one-line JSON on stdout):
    {"t": "log",      "m": str}
    {"t": "progress", "step": int, "total": int, "sim": float}
    {"t": "result",   "output": path, "before": {...}, "after": {...}, "linf": float,
                      "steps_run": int}
    {"t": "error",    "m": str, "trace": str}
"""

import json
import os
import sys
import traceback

def emit(**event) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()

def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    import adversarial_watermark as engine
    from PIL import Image

    bundle = engine.load_model(
        cfg["model"], cfg["pretrained"], cfg["device"], log=lambda m: emit(t="log", m=m)
    )
    emit(t="log", m=f"Device: {bundle.device}")

    image = Image.open(cfg["input"]).convert("RGB")

    result = engine.protect_image(
        bundle, image,
        prompt=cfg["prompt"],
        contrast_prompt=cfg["contrast_prompt"],
        eps=cfg["eps"],
        alpha=cfg["alpha"],
        steps=cfg["steps"],
        seed=cfg["seed"],
        progress=lambda step, total, sim: emit(t="progress", step=step, total=total, sim=sim),
    )

    result.image.save(cfg["output"])
    emit(
        t="result",
        output=cfg["output"],
        before=result.before,
        after=result.after,
        linf=result.linf,
        steps_run=result.steps_run,
    )
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        emit(t="error", m=str(exc), trace=traceback.format_exc())
        sys.exit(1)
