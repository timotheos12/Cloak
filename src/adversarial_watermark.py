#!/usr/bin/env python3

# Adds subtle noise to images to trick OpenCLIP into thinking there is a 'watermark'.
# Optimizes L-infinity perturbation using projected gradient ascent to steer CLIP embedding.
# CLI command: python adversarial_watermark.py -i photo.jpg -o photo_protected.png

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from torchvision.transforms import Normalize, Resize, CenterCrop
from torchvision.transforms.functional import to_tensor, to_pil_image

# ---- Model presets ------------------------------------------------------------------

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
DEFAULT_CONTRAST_PROMPT = "image"

class Cancelled(Exception):
    """Raised inside the optimization loop when the caller asks it to stop."""

# ---- Device and preprocessing helpers -----------------------------------------------

def available_devices() -> list[str]:
    """Windows offers CUDA or CPU."""
    devices = ["auto"]
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices

def pick_device(requested: str) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_preprocess_params(preprocess):
    """Pull the exact Normalize mean/std and the resize/crop sizes out of the OpenCLIP
    val transform, so our differentiable pipeline matches inference as closely as possible."""
    mean = std = None
    resize_size = crop_size = None
    for t in preprocess.transforms:
        if isinstance(t, Normalize):
            mean = torch.tensor(t.mean).view(1, 3, 1, 1)
            std = torch.tensor(t.std).view(1, 3, 1, 1)
        elif isinstance(t, Resize):
            s = t.size
            resize_size = s if isinstance(s, int) else min(s)
        elif isinstance(t, CenterCrop):
            s = t.size
            crop_size = s if isinstance(s, int) else min(s)
    if crop_size is None:
        crop_size = resize_size
    if resize_size is None:
        resize_size = crop_size
    if mean is None:  # OpenAI and LAION default normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return mean, std, resize_size, crop_size

def differentiable_preprocess(img01, resize_size, crop_size, mean, std):
    """Differentiable equivalent of: Resize(shortest side -> resize_size, bicubic) +
    CenterCrop(crop_size) + Normalize. Input `img01` is (1, 3, H, W) in [0, 1]."""
    _, _, h, w = img01.shape
    scale = resize_size / min(h, w)
    new_h = max(crop_size, round(h * scale))
    new_w = max(crop_size, round(w * scale))
    resized = F.interpolate(
        img01, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True
    )
    top = (new_h - crop_size) // 2
    left = (new_w - crop_size) // 2
    cropped = resized[:, :, top:top + crop_size, left:left + crop_size]
    return (cropped - mean) / std

# ---- Cached model bundle ------------------------------------------------------------

@dataclass
class ModelBundle:
    name: str
    pretrained: str
    device: torch.device
    model: object
    tokenizer: object
    preprocess: object
    mean: object
    std: object
    resize_size: int
    crop_size: int
    logit_scale: float
    _text_cache: dict = field(default_factory=dict)

    def text_embedding(self, prompt: str):
        if prompt not in self._text_cache:
            with torch.no_grad():
                tokens = self.tokenizer([prompt]).to(self.device)
                emb = F.normalize(self.model.encode_text(tokens).float(), dim=-1)
            self._text_cache[prompt] = emb
        return self._text_cache[prompt]

_BUNDLE_CACHE: dict[tuple, ModelBundle] = {}
_LOAD_LOCK = threading.Lock()

def load_model(
    model_name: str = DEFAULT_MODEL,
    pretrained: str = DEFAULT_PRETRAINED,
    device: str = "auto",
    log: Optional[Callable[[str], None]] = None,
) -> ModelBundle:
    """Load (and cache) an OpenCLIP model. First call for a given model downloads weights."""
    log = log or (lambda _msg: None)
    dev = pick_device(device)
    key = (model_name, pretrained, str(dev))

    with _LOAD_LOCK:
        if key in _BUNDLE_CACHE:
            return _BUNDLE_CACHE[key]

        log(f"Loading OpenCLIP {model_name} / {pretrained} on {dev} "
            f"(the first load downloads weights and needs internet)...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(dev).float().eval()
        for param in model.parameters():
            param.requires_grad_(False)  # Only need perturbation gradients

        mean, std, resize_size, crop_size = extract_preprocess_params(preprocess)
        mean, std = mean.to(dev), std.to(dev)
        with torch.no_grad():
            logit_scale = model.logit_scale.exp().item()

        bundle = ModelBundle(
            name=model_name,
            pretrained=pretrained,
            device=dev,
            model=model,
            tokenizer=tokenizer,
            preprocess=preprocess,
            mean=mean,
            std=std,
            resize_size=resize_size,
            crop_size=crop_size,
            logit_scale=logit_scale,
        )
        _BUNDLE_CACHE[key] = bundle
        log(f"Model ready. Input {crop_size}px (shortest side resized to {resize_size}px).")
        return bundle

# ---- Cosine similarity scoring --------------------------------------------------------

def evaluate(bundle: ModelBundle, pil_img: Image.Image, prompt: str, contrast_prompt: str) -> dict:
    """Honest numbers using the model's *real* preprocessing on the 8-bit image."""
    target_emb = bundle.text_embedding(prompt)
    contrast_emb = bundle.text_embedding(contrast_prompt)
    with torch.no_grad():
        x = bundle.preprocess(pil_img.convert("RGB")).unsqueeze(0).to(bundle.device)
        emb = F.normalize(bundle.model.encode_image(x).float(), dim=-1)
        sim_t = (emb @ target_emb.T).item()
        sim_c = (emb @ contrast_emb.T).item()
        logits = torch.tensor([sim_t, sim_c]) * bundle.logit_scale
        prob = torch.softmax(logits, dim=0)[0].item()
    return {"cos_target": sim_t, "cos_contrast": sim_c, "prob_target": prob}

# ---- Adversarial attack --------------------------------------------------------------

@dataclass
class ProtectResult:
    image: Image.Image
    before: dict
    after: dict
    linf: float
    steps_run: int
    prompt: str
    contrast_prompt: str

def protect_image(
    bundle: ModelBundle,
    pil_img: Image.Image,
    prompt: str = DEFAULT_PROMPT,
    contrast_prompt: str = DEFAULT_CONTRAST_PROMPT,
    eps: float = 8 / 255,
    alpha: float = 1 / 255,
    steps: int = 250,
    seed: int = 0,
    progress: Optional[Callable[[int, int, float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> ProtectResult:
    """Projected gradient ascent on the pixels, maximizing cos(image_emb, text_emb).

    `progress(step, total, cos_sim)` is called after every step; raise-free.
    `cancel_event`, if set mid-run, aborts with `Cancelled`.
    """
    torch.manual_seed(seed)
    device = bundle.device
    pil_img = pil_img.convert("RGB")

    target_emb = bundle.text_embedding(prompt)
    before = evaluate(bundle, pil_img, prompt, contrast_prompt)

    img01 = to_tensor(pil_img).unsqueeze(0).to(device)      # (1, 3, H, W) in [0, 1]
    delta = torch.zeros_like(img01, requires_grad=True)

    steps_run = 0
    for step in range(steps):
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled("stopped by user")

        adv01 = torch.clamp(img01 + delta, 0.0, 1.0)
        feats = differentiable_preprocess(
            adv01, bundle.resize_size, bundle.crop_size, bundle.mean, bundle.std
        )
        emb = F.normalize(bundle.model.encode_image(feats).float(), dim=-1)
        sim = (emb @ target_emb.T).squeeze()   # Normalize cosine similarity score
        loss = sim                             # Maximize similarity to prompt

        if delta.grad is not None:
            delta.grad = None
        loss.backward()

        with torch.no_grad():
            delta += alpha * delta.grad.sign()                       # Gradient ascent
            delta.clamp_(-eps, eps)                                  # Project onto L-infinity ball
            delta.copy_(torch.clamp(img01 + delta, 0, 1) - img01)    # Keep pixels valid in [0,1]

        steps_run = step + 1
        if progress is not None:
            progress(steps_run, steps, float(sim.item()))

    adv01 = torch.clamp(img01 + delta.detach(), 0.0, 1.0)
    out_pil = to_pil_image(adv01.squeeze(0).cpu())                   # Quantize to 8-bit
    linf = float((adv01 - img01).abs().max().item())
    after = evaluate(bundle, out_pil, prompt, contrast_prompt)

    return ProtectResult(
        image=out_pil,
        before=before,
        after=after,
        linf=linf,
        steps_run=steps_run,
        prompt=prompt,
        contrast_prompt=contrast_prompt,
    )

# ---- CLI command --------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="OpenCLIP adversarial image protector.")
    p.add_argument("-i", "--input", required=True, help="path to input image")
    p.add_argument("-o", "--output", required=True, help="path to save protected image (use .png)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenCLIP model name")
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED, help="LAION pretrained tag")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="target text prompt to steer toward")
    p.add_argument("--contrast-prompt", default=DEFAULT_CONTRAST_PROMPT,
                   help="reference prompt used only for the success report")
    p.add_argument("--eps", type=float, default=8 / 255,
                   help="L-infinity perturbation budget in [0,1] pixel space (controls subtlety)")
    p.add_argument("--alpha", type=float, default=1 / 255, help="PGD step size")
    p.add_argument("--steps", type=int, default=250, help="number of gradient-ascent steps")
    p.add_argument("--device", default="auto", help="cuda | cpu | auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    bundle = load_model(args.model, args.pretrained, args.device, log=lambda m: print(f"[*] {m}"))
    pil = Image.open(args.input).convert("RGB")

    log_every = max(1, args.steps // 10)

    def progress(step, total, sim):
        if step % log_every == 0 or step == total:
            print(f"    step {step:4d}/{total}  cos sim -> '{args.prompt}' = {sim:.4f}")

    print(f"[*] optimizing: eps={args.eps:.4f} (~{round(args.eps * 255)}/255), "
          f"alpha={args.alpha:.4f}, steps={args.steps}")
    res = protect_image(
        bundle, pil,
        prompt=args.prompt, contrast_prompt=args.contrast_prompt,
        eps=args.eps, alpha=args.alpha, steps=args.steps, seed=args.seed,
        progress=progress,
    )
    res.image.save(args.output)
    print(f"[+] saved protected image -> {args.output}")

    print("\n==== result (using the model's real preprocessing) ====")
    print(f"  perturbation L-inf  : {res.linf:.4f}  (~{round(res.linf * 255)}/255)")
    print(f"  cos sim '{args.prompt}' : {res.before['cos_target']:.4f}  ->  {res.after['cos_target']:.4f}")
    print(f"  P(target) 2-way     : {res.before['prob_target'] * 100:5.1f}%  ->  "
          f"{res.after['prob_target'] * 100:5.1f}%   (vs '{args.contrast_prompt}')")

if __name__ == "__main__":
    main()
