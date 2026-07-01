#!/usr/bin/env python3
"""
adversarial_watermark.py

Add subtle adversarial noise to an image so that an OpenCLIP model (LAION weights)
embeds it close to a target text prompt (default: "watermark").

This is the same family of technique used by artist-protection tools like Glaze /
Nightshade: a bounded (L-infinity) perturbation is optimized with projected gradient
ascent (PGD) to steer the CLIP *image* embedding toward a chosen *text* embedding.
Because web-scale dataset pipelines (e.g. LAION-style) use CLIP to score / filter
images and commonly drop anything flagged as watermarked, pushing your image to read
as "watermark" acts as an opt-out / poisoning signal against unconsented scraping.

Requires (first run downloads weights -> needs internet):
    pip install open_clip_torch torch torchvision pillow

Example:
    python adversarial_watermark.py -i photo.jpg -o photo_protected.png --eps 0.031 --steps 250
"""

import argparse

import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from torchvision.transforms import Normalize, Resize, CenterCrop
from torchvision.transforms.functional import to_tensor, to_pil_image


def pick_device(requested: str) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    if mean is None:  # OpenAI / LAION default normalization
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


def main():
    p = argparse.ArgumentParser(description="OpenCLIP adversarial image protector.")
    p.add_argument("-i", "--input", required=True, help="path to input image")
    p.add_argument("-o", "--output", required=True, help="path to save protected image")
    p.add_argument("--model", default="ViT-B-32", help="OpenCLIP model name")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k", help="LAION pretrained tag")
    p.add_argument("--prompt", default="watermark", help="target text prompt to steer toward")
    p.add_argument("--contrast-prompt", default="a clean photo without a watermark",
                   help="reference prompt used only for the success report")
    p.add_argument("--eps", type=float, default=8 / 255,
                   help="L-infinity perturbation budget in [0,1] pixel space (controls subtlety)")
    p.add_argument("--alpha", type=float, default=1 / 255, help="PGD step size")
    p.add_argument("--steps", type=int, default=250, help="number of gradient-ascent steps")
    p.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"[*] device: {device}")

    print(f"[*] loading OpenCLIP {args.model} / {args.pretrained} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).float().eval()
    for param in model.parameters():
        param.requires_grad_(False)  # we only need gradients w.r.t. the perturbation

    mean, std, resize_size, crop_size = extract_preprocess_params(preprocess)
    mean, std = mean.to(device), std.to(device)
    print(f"[*] model input: {crop_size}px (resize shortest side -> {resize_size}px)")

    # ----- fixed text embeddings -----
    with torch.no_grad():
        target_emb = F.normalize(
            model.encode_text(tokenizer([args.prompt]).to(device)).float(), dim=-1
        )
        contrast_emb = F.normalize(
            model.encode_text(tokenizer([args.contrast_prompt]).to(device)).float(), dim=-1
        )
        logit_scale = model.logit_scale.exp().item()

    # ----- load image into a [0,1] tensor at full resolution -----
    pil = Image.open(args.input).convert("RGB")
    img01 = to_tensor(pil).unsqueeze(0).to(device)  # (1, 3, H, W) in [0, 1]

    def report_real(pil_img):
        """Honest before/after numbers using the model's *real* preprocessing on the
        8-bit image. Returns (cos_sim_target, cos_sim_contrast, P(watermark) 2-way)."""
        with torch.no_grad():
            x = preprocess(pil_img).unsqueeze(0).to(device)
            emb = F.normalize(model.encode_image(x).float(), dim=-1)
            sim_t = (emb @ target_emb.T).item()
            sim_c = (emb @ contrast_emb.T).item()
            logits = torch.tensor([sim_t, sim_c]) * logit_scale
            prob = torch.softmax(logits, dim=0)[0].item()
        return sim_t, sim_c, prob

    # ----- projected gradient ascent on the pixels -----
    delta = torch.zeros_like(img01, requires_grad=True)
    print(f"[*] optimizing: eps={args.eps:.4f} (~{round(args.eps * 255)}/255), "
          f"alpha={args.alpha:.4f}, steps={args.steps}")
    log_every = max(1, args.steps // 10)
    for step in range(args.steps):
        adv01 = torch.clamp(img01 + delta, 0.0, 1.0)
        feats = differentiable_preprocess(adv01, resize_size, crop_size, mean, std)
        emb = F.normalize(model.encode_image(feats).float(), dim=-1)
        sim = (emb @ target_emb.T).squeeze()        # cosine similarity (both normalized)
        loss = sim                                  # we MAXIMIZE similarity to the prompt

        if delta.grad is not None:
            delta.grad = None
        loss.backward()

        with torch.no_grad():
            delta += args.alpha * delta.grad.sign()                 # gradient ASCENT
            delta.clamp_(-args.eps, args.eps)                       # project onto L-inf ball
            delta.copy_(torch.clamp(img01 + delta, 0, 1) - img01)   # keep pixels valid in [0,1]

        if (step + 1) % log_every == 0:
            print(f"    step {step + 1:4d}/{args.steps}  "
                  f"cos sim -> '{args.prompt}' = {sim.item():.4f}")

    # ----- save and verify -----
    adv01 = torch.clamp(img01 + delta.detach(), 0.0, 1.0)
    out_pil = to_pil_image(adv01.squeeze(0).cpu())
    out_pil.save(args.output)
    print(f"[+] saved protected image -> {args.output}")

    b_t, _, b_p = report_real(pil)
    a_t, _, a_p = report_real(out_pil)
    linf = (adv01 - img01).abs().max().item()
    print("\n==== result (using the model's real preprocessing) ====")
    print(f"  perturbation L-inf  : {linf:.4f}  (~{round(linf * 255)}/255)")
    print(f"  cos sim '{args.prompt}' : {b_t:.4f}  ->  {a_t:.4f}")
    print(f"  P(watermark) 2-way  : {b_p * 100:5.1f}%  ->  {a_p * 100:5.1f}%   "
          f"(vs '{args.contrast_prompt}')")


if __name__ == "__main__":
    main()
