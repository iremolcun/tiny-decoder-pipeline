# Examples

This directory contains minimal usage examples demonstrating how to run inference with the trained decoder.

## Run Inference

```bash
python3 scripts/inference_decoder_safe.py \
  --pairs_root decoder_pairs \
  --ckpt ckpts/decoder_best.pth \
  --split test \
  --out_dir preds \
  --device cuda \
  --max_samples 4 \
  --save_side_by_side
This will generate:

pred.png

target.png

compare_pred_target.png

info.json (per-sample metrics)

Example Output
Below is a sample side-by-side comparison:

![Example Output](../examples/sample_compare.png)
Note: You may replace the sample image with your own inference results.
