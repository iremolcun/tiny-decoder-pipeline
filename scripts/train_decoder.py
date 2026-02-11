#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, random, math
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

# ---------------- Dataset ----------------
class DecoderPairsDataset(Dataset):
    """
    Beklenen yapı:
      <pairs_root>/<split>/<group_id>/target_rgb.png
      <pairs_root>/<split>/<group_id>/<sample_dir>/hr_features.pt
    """
    def __init__(self, root: Path, split: str):
        self.root = Path(root) / split
        if not self.root.exists():
            raise RuntimeError(f"Split yok: {self.root}")
        self.samples = []
        self.to_tensor = T.ToTensor()

        for gid_dir in sorted([d for d in self.root.iterdir() if d.is_dir()]):
            group_target = gid_dir / "target_rgb.png"
            if not group_target.exists():
                print(f"[WARN] target yok, grup atlandı: {gid_dir}")
                continue

            try:
                W, H = Image.open(group_target).convert("RGB").size
            except Exception as e:
                print(f"[WARN] target açılamadı ({gid_dir}): {e}")
                continue

            for sdir in sorted([d for d in gid_dir.iterdir() if d.is_dir()]):
                hrp = sdir / "hr_features.pt"
                if not hrp.exists():
                    continue

                try:
                    hr = torch.load(hrp, map_location="cpu")
                    if hr.dim() == 4 and hr.shape[0] == 1:
                        hr = hr[0]
                    if hr.dim() != 3:
                        raise RuntimeError(f"dim={tuple(hr.shape)}")
                    C, HH, WW = hr.shape
                    if (HH, WW) != (H, W):
                        raise RuntimeError(f"size mismatch hr={HH}x{WW} vs target={H}x{W}")
                except Exception as e:
                    print(f"[SKIP] bozuk/uyumsuz hr_features: {hrp} -> {e}")
                    continue

                self.samples.append((hrp, group_target))

        if not self.samples:
            raise RuntimeError(f"{self.root} altında geçerli örnek bulunamadı.")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _norm_shape(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[0] == 1:
            x = x[0]
        if x.dim() != 3:
            raise RuntimeError(f"hr_features shape {tuple(x.shape)}")
        return x

    def __getitem__(self, idx):
        hrp, trg = self.samples[idx]
        hr = torch.load(hrp, map_location="cpu")
        hr = self._norm_shape(hr).float()
        tar = self.to_tensor(Image.open(trg).convert("RGB")).float()
        if hr.shape[-2:] != tar.shape[-2:]:
            raise RuntimeError(f"Size mismatch: hr {tuple(hr.shape)} vs target {tuple(tar.shape)} @ {hrp}")
        return hr, tar

# --------------- Model -------------------
class SmallDecoder(nn.Module):
    def __init__(self, c_in=384, c_mid=96, n_blocks=10, dilation_every=2):
        super().__init__()
        self.feat_norm = nn.GroupNorm(num_groups=min(32, c_in), num_channels=c_in, affine=False)

        self.inp = nn.Conv2d(c_in, c_mid, 1, bias=False)
        self.inp_gn = nn.GroupNorm(8, c_mid)

        blocks = []
        for i in range(n_blocks):
            dil = 1 if (dilation_every == 0 or i % dilation_every) else 2
            blocks += [
                nn.Conv2d(c_mid, c_mid, 3, padding=dil, dilation=dil, bias=False),
                nn.GroupNorm(8, c_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_mid, c_mid, 3, padding=1, bias=False),
                nn.GroupNorm(8, c_mid),
            ]
        self.blocks = nn.ModuleList(blocks)
        self.block_stride = 5
        self.out = nn.Conv2d(c_mid, 3, 3, padding=1)

    def forward(self, x):
        x = self.feat_norm(x)
        y = self.inp_gn(self.inp(x))
        for i in range(0, len(self.blocks), self.block_stride):
            res = y
            y = self.blocks[i  ](y)
            y = self.blocks[i+1](y)
            y = self.blocks[i+2](y)
            y = self.blocks[i+3](y)
            y = self.blocks[i+4](y)
            y = F.relu(y + res, inplace=True)
        return self.out(y)

# ------------- Crops & Aug ----------------
def paired_random_crop_and_flip(hr: torch.Tensor, tar: torch.Tensor, crop: int):
    _, H, W = hr.shape
    if crop > 0 and (H > crop and W > crop):
        y = random.randint(0, H - crop)
        x = random.randint(0, W - crop)
        hr = hr[:, y:y+crop, x:x+crop]
        tar = tar[:, y:y+crop, x:x+crop]

    if random.random() < 0.5:
        hr = torch.flip(hr, dims=[2])
        tar = torch.flip(tar, dims=[2])

    if random.random() < 0.5:
        hr = torch.flip(hr, dims=[1])
        tar = torch.flip(tar, dims=[1])

    return hr, tar

# ------------- autodetect c_in ------------
def autodetect_c_in(root: Path):
    for split in ["train", "test"]:
        d = root / split
        if not d.exists(): continue
        for gid in d.iterdir():
            if not gid.is_dir(): continue
            for sdir in gid.iterdir():
                hrp = sdir / "hr_features.pt"
                if hrp.exists():
                    hr = torch.load(hrp, map_location="cpu")
                    if hr.dim()==4 and hr.shape[0]==1: hr=hr[0]
                    return hr.shape[0]
    raise RuntimeError("c_in tespit edilemedi.")

# --------------- Training -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_root", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--val_crop", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save_dir", default="./ckpts")
    args = ap.parse_args()

    root = Path(args.pairs_root)
    save_dir = Path(args.save_dir); save_dir.mkdir(exist_ok=True, parents=True)

    c_in = autodetect_c_in(root)
    device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"

    dataset = DecoderPairsDataset(root, "train")
    n_val = max(1, int(len(dataset)*0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    def collate(batch):
        out_hr, out_tar = [], []
        for hr, tar in batch:
            hr, tar = paired_random_crop_and_flip(hr, tar, args.crop)
            out_hr.append(hr); out_tar.append(tar)
        return torch.stack(out_hr), torch.stack(out_tar)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = SmallDecoder(c_in=c_in).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_loss = float("inf")
    hist = []

    for ep in range(1, args.epochs+1):
        model.train()
        tr_loss = 0
        for hr, tar in train_loader:
            hr = hr.to(device)
            tar = tar.to(device)

            opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(hr)
                loss = loss_fn(pred, tar)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item()

        tr_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for hr, tar in val_loader:
                hr = hr.to(device)
                tar = tar.to(device)
                h, w = tar.shape[-2:]
                yc = min(args.val_crop, h)
                xc = min(args.val_crop, w)
                hr = hr[..., :yc, :xc]
                tar = tar[..., :yc, :xc]
                pred = model(hr)
                val_loss += loss_fn(pred, tar).item()

        val_loss /= len(val_loader)
        hist.append(val_loss)

        print(f"[{ep:03d}/{args.epochs}] train={tr_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_dir/"decoder_best.pth")
            print("  -> yeni en iyi model kaydedildi.")

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(hist)
    plt.title("Validation Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir/"loss.png")
    print("loss.png kaydedildi.")

if __name__ == "__main__":
    main()

