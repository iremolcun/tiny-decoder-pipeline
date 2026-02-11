"""
DINOv2 Feature → RGB Reconstruction
Robust Inference & Evaluation Script

Author: İrem Olcun

Description:
    - Loads trained decoder weights
    - Reconstructs RGB images from hr_features.pt tensors
    - Computes MAE and PSNR metrics
    - Optionally saves side-by-side comparison grids
    - Produces per-sample info.json and global summary.json
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms as T
from contextlib import nullcontext

from train_decoder import SmallDecoder, DecoderPairsDataset, autodetect_c_in


def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target, reduction="mean").item()
    if mse == 0:
        return 99.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def safe_load_tensor(pt_path: Path):
    try:
        t = torch.load(pt_path, map_location="cpu")
        if t.dim() == 4 and t.shape[0] == 1:
            t = t[0]
        if t.dim() != 3:
            raise RuntimeError(f"bad dim {tuple(t.shape)}")
        return t.float()
    except Exception as e:
        raise RuntimeError(f"{pt_path} → {e}")


def main():
    ap = argparse.ArgumentParser(description="Robust decoder inference script.")
    ap.add_argument("--pairs_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out_dir", default="./preds_safe")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_samples", type=int, default=12)
    ap.add_argument("--save_side_by_side", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    root = Path(args.pairs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    c_in = autodetect_c_in(root)
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    model = SmallDecoder(c_in=c_in).to(device).eval()
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    ds = DecoderPairsDataset(root, args.split)
    to_pil = T.ToPILImage()

    stats = []
    done = 0
    attempted = 0
    N = min(args.max_samples, len(ds))

    print(f"[INFO] Evaluating {N} samples (split={args.split})")

    use_fp16 = (device == "cuda" and args.fp16)
    autocast_ctx = torch.amp.autocast("cuda") if use_fp16 else nullcontext()

    i = 0
    while done < N and i < len(ds):
        hrp, trg = ds.samples[i]
        attempted += 1
        i += 1

        try:
            hr = safe_load_tensor(hrp)
            tar = T.ToTensor()(Image.open(trg).convert("RGB")).float()

            if hr.shape[-2:] != tar.shape[-2:]:
                raise RuntimeError(f"size mismatch {hr.shape} vs {tar.shape}")

            hr_ = hr.unsqueeze(0).to(device)

            with torch.no_grad():
                with autocast_ctx:
                    pred = model(hr_.float())

            pred = pred.clamp(0, 1)[0].cpu()

            mae = torch.mean(torch.abs(pred - tar)).item()
            p = psnr(pred, tar)

            gdir = out_dir / f"s{done:03d}"
            gdir.mkdir(parents=True, exist_ok=True)

            to_pil(pred).save(gdir / "pred.png")
            to_pil(tar).save(gdir / "target.png")

            if args.save_side_by_side:
                grid = make_grid(torch.stack([pred, tar], dim=0), nrow=2)
                to_pil(grid).save(gdir / "compare_pred_target.png")

            with open(gdir / "info.json", "w") as f:
                json.dump(
                    {
                        "hr_path": str(hrp),
                        "target_path": str(trg),
                        "C": int(hr.shape[0]),
                        "H": int(hr.shape[1]),
                        "W": int(hr.shape[2]),
                        "MAE": mae,
                        "PSNR": p,
                    },
                    f,
                    indent=2,
                )

            stats.append(mae)
            done += 1

            print(f"[{done}/{N}] MAE={mae:.4f}  PSNR={p:.2f} dB → {gdir}")

        except Exception as e:
            print(f"[SKIP] {hrp} ({e})")
            continue

    if stats:
        mean_mae = sum(stats) / len(stats)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "mean_MAE": mean_mae,
                    "n": len(stats),
                    "attempted": attempted,
                },
                f,
                indent=2,
            )
        print(f"\n[SUMMARY] n={len(stats)} / attempted={attempted}  mean_MAE={mean_mae:.4f}")
    else:
        print("\n[SUMMARY] No valid samples processed.")


if __name__ == "__main__":
    main()

