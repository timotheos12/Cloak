#!/usr/bin/env python3
"""
adversarial_watermark.py  (v2 — imperceptible)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Adds an adversarial perturbation to an image so that CLIP-style watermark
classifiers (the LAION pwatermark filter used by img2dataset / datacomp)
tag it as 'watermark' — causing dataset-curation pipelines to EXCLUDE the
image from AI training sets. A self-protection tool for image owners.

What changed from v1 (why it was visible, and the fixes):

  1. FULL-RESOLUTION OPTIMISATION
     v1 optimised at 224x224 then upscaled the delta — bilinear upsampling
     smears the perturbation into broad low-frequency blotches that the eye
     catches easily. v2 keeps a full-res delta and only downsamples to feed
     the classifier (downsampling is smooth and differentiable), so the
     perturbation lives at the native pixel scale where it hides better.

  2. PERCEPTUAL MASKING
     The human eye tolerates far more change in busy/high-texture regions
     than in flat ones (luminance + contrast masking). v2 computes a local
     gradient-magnitude map and scales the allowed perturbation by it, so
     noise concentrates in edges/texture and stays out of smooth skies and
     skin where it would be obvious.

  3. LPIPS-STYLE SMOOTHNESS PENALTY
     A total-variation term in the loss suppresses isolated speckle (the
     salt-and-pepper look) in favour of smooth, structured perturbations
     that read as natural texture.

  4. YUV-WEIGHTED UPDATES
     The eye is least sensitive to chroma. v2 lets the optimiser push harder
     on chroma than on luminance, gaining classifier signal for less visible
     change.

Install:
    pip install torch torchvision transformers pillow numpy

Usage:
    python adversarial_watermark.py photo.jpg protected.png
    python adversarial_watermark.py photo.jpg protected.png --epsilon 6 --steps 300
    python adversarial_watermark.py photo.jpg protected.png --target 0.9 --device cuda

    --epsilon   Max luminance Δ per channel, 0-255  (default: 6)
                This is now a CEILING on the *masked* perturbation; the
                average change is much lower, so 6 is already very subtle.
    --steps     Optimisation iterations               (default: 250)
    --target    Stop early once this watermark prob is reached (default: 0.85)
    --tv        Smoothness penalty weight             (default: 0.08)
    --chroma    Chroma-vs-luma push ratio, >=1        (default: 2.0)
    --device    auto | cpu | cuda | mps               (default: auto)
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "amrul-hzz/watermark_detector"


# ── Device ─────────────────────────────────────────────────────────────────

def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Image helpers ───────────────────────────────────────────────────────────

def img_to_tensor(pil_img: Image.Image, device) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_img(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def normalize(x, mean, std, device):
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std,  device=device).view(1, 3, 1, 1)
    return (x - m) / s


def model_input_size(processor):
    size = processor.size
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            e = int(size["shortest_edge"]); return e, e
    if isinstance(size, int):
        return size, size
    return 224, 224


# ── Perceptual masking ──────────────────────────────────────────────────────

def perceptual_mask(img: torch.Tensor) -> torch.Tensor:
    """
    Per-pixel tolerance map in [0,1]. High where the eye is forgiving
    (textured/edge regions), low in flat regions. Based on local gradient
    magnitude of luminance, smoothed and normalised.
    """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Sobel gradients
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(lum, kx, padding=1)
    gy = F.conv2d(lum, ky, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-8)

    # Smooth so the mask isn't itself a high-freq pattern
    blur = torch.ones(1, 1, 5, 5, device=img.device, dtype=img.dtype) / 25.0
    grad = F.conv2d(grad, blur, padding=2)

    # Normalise to [0,1] and give flat regions a small floor so they still
    # receive a little perturbation (the classifier needs global signal)
    grad = grad / (grad.amax() + 1e-8)
    return 0.15 + 0.85 * grad     # floor 0.15, ceiling 1.0


def total_variation(delta: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV — penalises high-frequency speckle, rewards smooth texture."""
    dh = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
    dw = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
    return dh + dw


# ── YUV channel weighting ───────────────────────────────────────────────────
# Push chroma harder than luma since the eye is least sensitive to colour error.

def yuv_weight_map(chroma_ratio: float, device) -> torch.Tensor:
    # Approx per-RGB-channel visibility weights; lower weight = optimiser allowed
    # to move it more. Blue carries least luminance, so it tolerates the most.
    w_luma = torch.tensor([0.6, 1.0, 0.45], device=device).view(1, 3, 1, 1)
    return w_luma / chroma_ratio + (1 - 1 / chroma_ratio)


# ── Inference ───────────────────────────────────────────────────────────────

@torch.no_grad()
def watermark_prob(model, processor, img_full, wm_idx, mh, mw, device):
    small = F.interpolate(img_full, size=(mh, mw), mode="area")
    nx = normalize(small, processor.image_mean, processor.image_std, device)
    return F.softmax(model(pixel_values=nx).logits, dim=-1)[0, wm_idx].item()


# ── Model ───────────────────────────────────────────────────────────────────

def load_model(device):
    print(f"[*] Loading {MODEL_ID} …")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(device).eval()
    labels = model.config.id2label
    print(f"    Labels: {labels}")
    wm_idx = next((k for k, v in labels.items() if "watermark" in v.lower()), 1)
    print(f"    Watermark → index {wm_idx} ('{labels[wm_idx]}')")
    return processor, model, wm_idx


# ── Optimisation ────────────────────────────────────────────────────────────

def optimise(model, processor, orig, wm_idx, mh, mw,
             eps, steps, target, tv_weight, chroma_ratio, device):
    """
    Full-resolution adversarial optimisation with perceptual masking,
    TV smoothness, and chroma-weighted, mask-scaled L∞ projection.
    """
    mask = perceptual_mask(orig)              # [1,1,H,W]
    chan_w = yuv_weight_map(chroma_ratio, device)   # [1,3,1,1]

    # Effective per-pixel, per-channel epsilon ball (in [0,1] scale)
    eps_map = eps * mask * chan_w             # broadcast → [1,3,H,W]

    delta = torch.zeros_like(orig).uniform_(-1e-3, 1e-3)
    delta = (orig + delta).clamp(0, 1) - orig
    best_delta, best_prob = delta.clone(), 0.0

    mean, std = processor.image_mean, processor.image_std
    # Adam adapts step size per pixel → smoother convergence than sign-SGD
    delta.requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=eps.mean().item() / 12 if torch.is_tensor(eps) else eps / 12)

    for step in range(steps):
        opt.zero_grad()
        perturbed = (orig + delta).clamp(0, 1)
        small = F.interpolate(perturbed, size=(mh, mw), mode="area")
        logits = model(pixel_values=normalize(small, mean, std, device)).logits
        probs = F.softmax(logits, dim=-1)

        loss = -torch.log(probs[0, wm_idx] + 1e-8) + tv_weight * total_variation(delta)
        loss.backward()
        opt.step()

        with torch.no_grad():
            # Project onto the per-pixel masked epsilon ball, then valid pixels
            delta.clamp_(-eps_map, eps_map)
            delta.copy_((orig + delta).clamp(0, 1) - orig)

            prob = probs[0, wm_idx].item()
            if prob > best_prob:
                best_prob, best_delta = prob, delta.detach().clone()

            if step % 25 == 0 or step == steps - 1:
                bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
                print(f"    step {step+1:4d}/{steps}  [{bar}]  {prob:.4f}")

            if best_prob >= target:
                print(f"    target {target} reached at step {step+1}")
                break

    return best_delta, best_prob


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Imperceptible adversarial watermark injection (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--epsilon", type=float, default=6.0)
    ap.add_argument("--steps",   type=int,   default=250)
    ap.add_argument("--target",  type=float, default=0.85)
    ap.add_argument("--tv",      type=float, default=0.08)
    ap.add_argument("--chroma",  type=float, default=2.0)
    ap.add_argument("--device",  default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[*] Device: {device}")
    eps = args.epsilon / 255.0

    print(f"[*] Loading {args.input}")
    try:
        orig_pil = Image.open(args.input).convert("RGB")
    except FileNotFoundError:
        sys.exit(f"[!] File not found: {args.input}")
    w, h = orig_pil.size
    print(f"    Size: {w}×{h}")

    processor, model, wm_idx = load_model(device)
    mh, mw = model_input_size(processor)

    orig = img_to_tensor(orig_pil, device)

    base = watermark_prob(model, processor, orig, wm_idx, mh, mw, device)
    print(f"\n[*] Baseline watermark confidence: {base:.4f}")

    print(f"\n[*] Optimising  ε≤{args.epsilon}/255 (masked)  "
          f"tv={args.tv}  chroma×{args.chroma}  {args.steps} steps\n")
    delta, final = optimise(model, processor, orig, wm_idx, mh, mw,
                            eps, args.steps, args.target,
                            args.tv, args.chroma, device)

    result = (orig + delta).clamp(0, 1)
    out_img = tensor_to_img(result)

    verified = watermark_prob(model, processor,
                              img_to_tensor(out_img, device),
                              wm_idx, mh, mw, device)

    d = (delta.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
    mse = (d ** 2).mean()
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")

    print(f"\n[*] Confidence: {base:.4f} → {final:.4f} (verified {verified:.4f})")
    print(f"\n── Imperceptibility ──────────────────────")
    print(f"   Max Δ  : {np.abs(d).max():.2f}/255")
    print(f"   Mean Δ : {np.abs(d).mean():.3f}/255")
    print(f"   PSNR   : {psnr:.1f} dB  (>40 = imperceptible, >45 = excellent)")

    out_img.save(args.output)
    print(f"\n[✓] Saved → {args.output}")
    if verified < 0.5:
        print("[!] Below 0.5 — try --epsilon 8 --steps 400 --target 0.9")
    elif verified < args.target:
        print("[~] Below target but may still clear filters; raise --epsilon slightly if needed.")
    else:
        print("[✓] Clears typical pwatermark thresholds (0.3–0.5) with margin.")


if __name__ == "__main__":
    main()
