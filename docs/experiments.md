# Experiments

This document describes the experimental setup used during the development of the Tiny Decoder Pipeline. 
It covers data preparation, training configuration, evaluation methodology, and implementation details. 
All experiments were designed to remain lightweight, reproducible, and model-agnostic.

---

## 1. Experimental Goals

The experiments were conducted to evaluate:

1. Whether DINOv2 spatial embeddings contain enough structure to enable high-resolution RGB reconstruction.
2. How effectively a compact residual convolutional decoder can invert these embeddings.
3. What reconstruction quality can be achieved under limited computational and data constraints.
4. Which architectural components contribute most to stable training.

---

## 2. Dataset Preparation

The dataset used during development contains high-resolution carpet textures and cannot be distributed due to copyright restrictions. 
Users must provide their own dataset following the required structure.

### 2.1. Input Format
Each sample contains:

- `hr_features.pt`: DINOv2-Small feature tensor 
- `target_rgb.png`: corresponding high-resolution RGB patch 

These pairs are generated automatically using:

prepare_decoder_dataset_v2.py



### 2.2. Directory Structure

data/
train/
group_001/
target_rgb.png
*/hr_features.pt
test/
group_002/
target_rgb.png
*/hr_features.pt



Each group may contain multiple feature tensors derived from the same image at different spatial scales.

---

## 3. Model Configuration

All experiments use the same lightweight decoder architecture:

- Input channels: 384 (DINOv2-Small)
- 10 residual blocks
- GroupNorm normalization
- Dilated convolutions for large receptive fields
- 1×1 → 3×3 → 3×3 convolution sequence
- Output channels: 3 (RGB)
- Loss: L1 (MAE)

The architecture prioritizes stability and efficiency while remaining compact enough for rapid experimentation.

---

## 4. Training Setup

### 4.1. Hyperparameters

| Parameter     | Value       |
|---------------|-------------|
| Epochs        | 50          |
| Learning rate | 3e-3        |
| Optimizer     | Adam        |
| Crop size     | 192         |
| Val crop      | 256         |
| Batch size    | Automatic (CUDA-dependent) |
| AMP           | Enabled     |

AMP (Automatic Mixed Precision) significantly improves throughput and memory efficiency.

### 4.2. Training Command

python3 scripts/train_decoder.py
--pairs_root decoder_pairs
--epochs 50
--lr 3e-3
--crop 192 --val_crop 256
--device cuda --amp
--out_dir ckpts



### 4.3. Outputs

- `decoder_best.pth` (best checkpoint)
- `loss_log.json` (training log)
- `loss_plot.png` (optional, generated later)

---

## 5. Evaluation Methodology

Evaluation is performed on a held-out test split using both quantitative and qualitative measures.

### 5.1. Quantitative Metrics

- MAE (L1)
- MSE (L2)
- PSNR (Peak Signal-to-Noise Ratio)

These metrics assess pixel-wise and perceptual similarity.

### 5.2. Qualitative Evaluation

For each test sample, the pipeline saves:

- reconstructed image (`pred.png`)
- ground-truth (`target.png`)
- side-by-side comparison (`compare_pred_target.png`)

These qualitative results offer insight into pattern recovery, color accuracy, and texture fidelity.

### 5.3. Inference Command

python3 scripts/inference_decoder_safe.py
--pairs_root decoder_pairs
--ckpt ckpts/decoder_best.pth
--split test
--out_dir preds
--device cuda
--max_samples 8
--save_side_by_side



---

## 6. Implementation Notes

### 6.1. Hardware

All experiments were performed on a CUDA-capable GPU using mixed precision. 
The lightweight model ensures compatibility with mid-range GPUs.

### 6.2. Reproducibility

- Fixed seeds are applied during training.
- Dataset structure is deterministic.
- Checkpoints and logs are stored in `ckpts/`.

### 6.3. Code Organization

All scripts under `scripts/` are modular:

- `prepare_decoder_dataset_v2.py` builds feature–target pairs
- `train_decoder.py` trains the decoder
- `inference_decoder_safe.py` runs evaluation
- `plot_losses.py` visualizes training curves
- `organize.py` performs optional cleanup utilities

---

## 7. Summary

The experimental setup demonstrates that:

- A lightweight residual CNN can successfully invert DINOv2 feature embeddings.
- Stable convergence can be achieved with a simple L1 objective.
- Even limited datasets yield meaningful reconstructions.
- The pipeline is reproducible, modular, and research-oriented, enabling extensions such as perceptual losses or multi-scale decoders.

This setup forms a strong foundation for future work in feature-space inversion and high-resolution texture reconstruction.
