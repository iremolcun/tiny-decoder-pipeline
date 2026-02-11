#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loss Curve Visualization Script
Author: İrem Ölçün

Description:
    - Reads loss_log.json generated during decoder training
    - Plots training and validation MAE curves
    - Saves loss figure as PNG
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot training and validation MAE curves.")
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to loss_log.json file"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="loss.png",
        help="Output image filename"
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"loss_log.json not found: {log_path}")

    with open(log_path, "r") as f:
        logs = json.load(f)

    epochs = [d["epoch"] for d in logs]
    train_mae = [d["train_MAE"] for d in logs]
    val_mae = [d["val_MAE"] for d in logs]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_mae, marker="o", label="Training MAE")
    plt.plot(epochs, val_mae, marker="s", label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Decoder Training vs Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

