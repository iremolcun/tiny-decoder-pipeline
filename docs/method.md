# Method

## 1. Feature Extraction

We extract high-level semantic feature maps using the **DINOv2-Small** model (HuggingFace implementation). 
Each input texture patch is processed to yield a spatial embedding tensor:

(C = 384, H/14, W/14)


These embeddings capture global color layout, structural patterns, and mid-level texture information.

For each sample, the paired dataset contains:
- `hr_features.pt` — DINOv2 spatial feature tensor 
- `target_rgb.png` — corresponding high-resolution RGB patch 

The script `prepare_decoder_dataset_v2.py` automatically:
1. Loads raw texture folders 
2. Extracts DINOv2 features 
3. Stores them in a standardized structure 
4. Generates deterministic train/validation/test splits 

This ensures reproducibility and a clean supervised reconstruction pipeline.

---

## 2. Decoder Architecture

The decoder is a compact convolutional network focused on efficiency, stability, and interpretability. 
It performs a non-linear mapping from feature space to RGB image space.

**Architecture characteristics:**
- **Input:** 384-channel DINOv2 features 
- **Output:** 3-channel RGB 
- **Core design:**
  - 1×1 projection for channel mixing 
  - 10 residual blocks with GroupNorm 
  - Dilated 3×3 convolutions for larger receptive field 
  - Skip connections for stable training 
  - Final 3×3 output layer 

### Architecture Summary

Input (384 × H/14 × W/14)
↓
1×1 Conv → GroupNorm → GELU
↓
10 × Residual Blocks:
• 3×3 Conv (with dilation)
• GroupNorm
• GELU
• skip connection
↓
3×3 Conv → RGB Output (3 channels)
↓
Upsample to target resolution



### Design Rationale
- **Residual blocks** enhance gradient flow 
- **GroupNorm** enables stable training with small batch sizes 
- **Dilated convolutions** enlarge context without extra cost 
- A **lightweight model** tests the inherent decodability of DINOv2 embeddings 

---

## 3. Training Objective

The model is optimized using **L1 loss (MAE)**:

L = ∥Pred − Target∥₁


Additional metrics:
- PSNR 
- MSE 
- Per-sample validation errors 

Mixed Precision (AMP) is used to improve training speed and reduce memory usage.

### Training Hyperparameters

| Parameter | Value |
|----------|-------|
| Epochs | 50 |
| Learning Rate | 3e-3 |
| Optimizer | Adam |
| Loss | L1 (MAE) |
| Train Crop | 192 px |
| Val Crop | 256 px |
| AMP | Enabled |

These hyperparameters correspond to the final experiment setup.

---

## 4. Experimental Setup

Experiments use:
- Paired feature–image dataset 
- Deterministic splitting into train/val/test 
- Random crop augmentation for robustness 
- Periodic validation 
- Checkpoint saving of the best-performing model 
- Training curve visualization (`loss_plot.png`) 

Inference is performed using `inference_decoder_safe.py`, producing:
- predicted RGB (`pred.png`) 
- ground truth (`target.png`) 
- side-by-side comparison (`compare_pred_target.png`) 
- per-sample metrics (`info.json`)
- aggregated summary (`summary.json`) 

---

## 5. Reconstruction Process

During inference:

1. DINOv2 hr_features are loaded 
2. The decoder generates a high-resolution RGB reconstruction 
3. Comparison figures are saved 
4. Quantitative metrics are computed 

The system enables both qualitative and quantitative evaluation of feature inversion.

---

## 6. Summary of Method

This work demonstrates that:
- DINOv2 embeddings contain enough structure to enable high-resolution inverse reconstruction 
- A small decoder can achieve stable convergence 
- Reconstruction captures global structure, texture layout, and color distribution 
- The pipeline provides a clean, research-oriented environment for future studies on feature-space inversion and texture reconstruction
