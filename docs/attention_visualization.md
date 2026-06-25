# Attention & Feature Map Visualization

Visualize trained SuperGlue and SuperPoint models from checkpoints.
**Runs on CPU only — safe to run while GPU training is in progress.**

## Files

| File | Purpose |
|------|---------|
| `scripts/visualize_attention.py` | Main script — run this |
| `gluefactory/visualization/attention_viz.py` | Helper functions (import from here if needed) |

---

## Quick Start

```bash
python3 scripts/visualize_attention.py \
  --superglue outputs/training/superglue_slam_run/checkpoint_best.tar \
  --superpoint outputs/training/superpoint_slam_run/checkpoint_best.tar \
  --img0 data/output/sample_slam/images/rgb/1775717079.407016.png \
  --img1 data/output/sample_slam/images/rgb/1775717079.507152.png \
  --output_dir outputs/visualizations/attention_maps
```

---

## All Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--superglue` | `outputs/training/superglue_slam_run/checkpoint_best.tar` | SuperGlue checkpoint |
| `--superpoint` | `outputs/training/superpoint_slam_run/checkpoint_best.tar` | SuperPoint checkpoint |
| `--img0` | sample SLAM image | First image path |
| `--img1` | sample SLAM image | Second image path |
| `--output_dir` | `outputs/visualizations/attention_maps` | Where PNGs are saved |
| `--resize` | `512` | Max image dimension in pixels (`0` = no resize) |
| `--layers` | `all` | Which GNN layers to plot individually (see below) |

### `--layers` options
```bash
--layers all          # all 18 GNN layers → 36 per-layer PNGs (slow)
--layers none         # skip per-layer PNGs, only summary + other outputs
--layers "0,1,8,9"   # specific layer indices only
```

---

## Output Files

All saved as PNG to `--output_dir`.

### SuperGlue

| File | Description |
|------|-------------|
| `superglue_attention_lines.png` | **Paper-style figure** — lines from query keypoint to top-15 attended keypoints. Green = high attention, red = low. Left half = self-attention (within same image), right half = cross-attention (lines spanning across image pair). 5 representative layers shown. |
| `superglue_attention_overlay.png` | Gaussian heatmap overlay on image. Picks 4 query keypoints, shows 3 GNN rounds. Self-attention on same image, cross-attention on other image. |
| `superglue_match_heatmap.png` | Match confidence overlaid on both images. Shows matched (hot colormap) vs unmatched (blue) keypoints, interpolated confidence map, soft assignment matrix, and confidence histograms. |
| `superglue_attn_summary.png` | 9-round × 2-row grid (self / cross), mean over 4 heads. Quick overview of all 18 layers. |
| `superglue_attn_layer_NN_{self\|cross}_{img0\|img1}.png` | Per-layer detail: 4 heads + mean. Only saved when `--layers` is not `none`. |

### SuperPoint

| File | Description |
|------|-------------|
| `superpoint_detector_heatmap.png` | Detection confidence map overlaid on image (equivalent of SuperGlue match heatmap but for keypoint detection). |
| `superpoint_descriptor_pca.png` | Dense descriptor field compressed to 3 PCA components → RGB. Shows spatial structure learned by the descriptor head. |
| `superpoint_features_stage1.png` | Top-8 activated channels from encoder Stage 1 (64ch, H/2×W/2) |
| `superpoint_features_stage2.png` | Stage 2 (64ch, H/4×W/4) |
| `superpoint_features_stage3.png` | Stage 3 (128ch, H/8×W/8) |
| `superpoint_features_stage4.png` | Stage 4 (128ch, H/8×W/8) |

---

## Common Recipes

**Fast run — just the key figures, no per-layer plots:**
```bash
python3 scripts/visualize_attention.py \
  --img0 <path> --img1 <path> \
  --output_dir outputs/visualizations/quick \
  --layers none
# → 9 PNGs in ~30 seconds on CPU
```

**Full run — every layer:**
```bash
python3 scripts/visualize_attention.py \
  --img0 <path> --img1 <path> \
  --output_dir outputs/visualizations/full \
  --layers all
# → 43 PNGs
```

**Inspect a specific layer pair (e.g. first and last round):**
```bash
python3 scripts/visualize_attention.py \
  --img0 <path> --img1 <path> \
  --layers "0,1,16,17"
```

---

## How It Works (no code changes needed)

- **SuperGlue attention**: `MultiHeadedAttention.forward` is monkey-patched at runtime to save attention probs into the existing `layer.attn.prob` list. No model files are modified.
- **SuperPoint feature maps**: `register_forward_hook` on encoder conv stages — no model changes.
- Both checkpoints are loaded with `map_location="cpu"`, so the GPU is never touched.
