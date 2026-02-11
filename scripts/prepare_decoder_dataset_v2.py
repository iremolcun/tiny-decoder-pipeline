"""
DINOv2 Feature Extraction & Pair Construction Pipeline
Author: İrem Ölçün

Description:
    - Extracts spatial feature maps from DINOv2
    - Uses AnyUp to upsample features to high-resolution
    - Constructs paired feature–target dataset
    - Saves hr_features.pt aligned with target_rgb.png
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math, re, gc
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ID_PAT = re.compile(r"(\d+_\d+)")

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])

def get_group_id(p: Path) -> str:
    m = ID_PAT.search(p.stem)
    return m.group(1) if m else "unknown"

def get_wh(p: Path) -> Tuple[int,int]:
    try:
        with Image.open(p) as im:
            return im.size
    except Exception:
        return (0,0)

def pick_largest(paths: List[Path]) -> Path:
    best, best_area = None, -1
    for p in paths:
        w,h = get_wh(p)
        area = w*h
        if area > best_area:
            best_area = area
            best = p
    return best

def resize_by_mpx(pil: Image.Image, max_mpx: float):
    if max_mpx <= 0:
        return pil, 1.0
    w, h = pil.size
    mpx = (w*h)/1e6
    if mpx <= max_mpx:
        return pil, 1.0
    scale = (max_mpx / mpx) ** 0.5
    new_w, new_h = max(1, int(w*scale)), max(1, int(h*scale))
    return pil.resize((new_w, new_h), Image.BICUBIC), scale

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./")
    parser.add_argument("--out_dir", type=str, default="decoder_pairs_v2")
    parser.add_argument("--model_id", type=str, default="facebook/dinov2-small")
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--q_chunk_size", type=int, default=64)
    parser.add_argument("--fallback_cpu", action="store_true")
    parser.add_argument("--max_ref_mpx", type=float, default=0.0)
    args = parser.parse_args()

    root = Path(args.data_root)
    out_root = root / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    print(f"[INFO] Loading DINO encoder: {args.model_id}")
    processor = AutoImageProcessor.from_pretrained(args.model_id, use_fast=True)
    encoder   = Dinov2Model.from_pretrained(args.model_id).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print("[INFO] Loading AnyUp (torch.hub:wimmerth/anyup)")
    anyup = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()
    for p in anyup.parameters():
        p.requires_grad = False

    to_tensor = T.ToTensor()
    imagenet_norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    def to_imagenet_norm_tensor(pil_img: Image.Image) -> torch.Tensor:
        return imagenet_norm(to_tensor(pil_img)).to(torch.float32)

    meta = {
        "model_id": args.model_id,
        "save_float16": bool(args.float16),
        "q_chunk_size": args.q_chunk_size,
        "fallback_cpu": bool(args.fallback_cpu),
        "max_ref_mpx": args.max_ref_mpx,
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))

    def cuda_cleanup():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    for split in ["train","test"]:
        in_dir = root / split
        split_out = out_root / split
        split_out.mkdir(parents=True, exist_ok=True)

        imgs = list_images(in_dir)
        print(f"[{split}] {len(imgs)} images found.")

        groups: Dict[str, List[Path]] = {}
        for p in imgs:
            gid = get_group_id(p)
            groups.setdefault(gid, []).append(p)

        for gid, paths in tqdm(groups.items(), desc=f"[{split}] groups"):
            g_out = split_out / gid
            g_out.mkdir(parents=True, exist_ok=True)

            ref_path = pick_largest(paths)
            if ref_path is None:
                continue

            try:
                ref_img_full = Image.open(ref_path).convert("RGB")
            except Exception:
                continue

            ref_img, scale_used = resize_by_mpx(ref_img_full, args.max_ref_mpx)
            ref_img.save(g_out / "target_rgb.png")

            hr_image_tensor = to_imagenet_norm_tensor(ref_img).unsqueeze(0).to(device)

            for src_path in paths:
                sdir = g_out / src_path.stem
                hr_out = sdir / "hr_features.pt"

                if hr_out.exists() and hr_out.stat().st_size > 0:
                    continue

                sdir.mkdir(parents=True, exist_ok=True)

                try:
                    src_img = Image.open(src_path).convert("RGB")
                except Exception:
                    continue

                inputs = processor(images=src_img, do_resize=False, return_tensors="pt")
                pix = inputs["pixel_values"].to(device, dtype=torch.float32)

                with torch.no_grad():
                    out = encoder(pixel_values=pix)
                    toks = out.last_hidden_state[:,1:,:]
                    B,N,C = toks.shape
                    hw = int(math.sqrt(N))
                    lr_feats = toks.reshape(B,hw,hw,C).permute(0,3,1,2).contiguous()

                with torch.no_grad():
                    hr_feats = anyup(hr_image_tensor, lr_feats, q_chunk_size=args.q_chunk_size)[0]

                if args.float16:
                    hr_feats = hr_feats.half()

                torch.save(hr_feats.cpu(), hr_out)

                (sdir / "info.json").write_text(json.dumps({
                    "group_id": gid,
                    "split": split,
                    "source": src_path.name,
                    "reference": ref_path.name,
                    "saved_dtype": "float16" if args.float16 else "float32",
                    "ref_downscale_used": scale_used,
                }, indent=2))

                del lr_feats, hr_feats
                cuda_cleanup()

        print(f"[OK] {split} → {split_out}")

    print("\n[OK] Dataset preparation complete.")

if __name__ == "__main__":
    main()

