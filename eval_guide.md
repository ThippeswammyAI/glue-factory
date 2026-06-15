# Evaluation Guide — SuperPoint & SuperGlue Models

## Your Trained Models

| Model Dir | Checkpoint | What it is |
|-----------|-----------|------------|
| `outputs/training/superpoint_model_tr__run_0_force_true` | `checkpoint_best.tar` (~15 MB) | Custom-trained **SuperPoint** (detector+descriptor only, no matcher) |
| `outputs/training/superglue_model_tr_slam_ran_1` | `checkpoint_best.tar` (~144 MB) | Custom-trained **SuperGlue** matcher (trained on SLAM posed-image pairs) |

> [!IMPORTANT]
> The `--checkpoint` flag takes the **folder name** under `outputs/training/` (NOT the full path).
> The eval system looks up `outputs/training/<name>/config.yaml` and `checkpoint_best.tar` automatically.

---

## Available Standard Benchmarks

### 1. 🏆 HPatches — Homography Estimation *(Best match for your SG model)*

**What it tests:** Planar homography estimation — both illumination (`i_*`) and viewpoint (`v_*`) changes.  
**Metrics:** `H_error_dlt@1/3/5px` (AUC), `H_error_ransac@1/3/5px` (AUC)  
**Dataset size:** ~580 image pairs, auto-downloads from HuggingFace (~250 MB)  
**Best for:** Your **SuperGlue** model (it was even pre-configured with `benchmarks.hpatches` in its `config.yaml`)

```bash
# SuperGlue model on HPatches (RECOMMENDED FIRST RUN)
python3 -m gluefactory.eval.hpatches \
  --checkpoint superglue_model_tr_slam_ran_1 \
  --tag sg_slam_hpatches

# With plot to visualize the recall curve
python3 -m gluefactory.eval.hpatches \
  --checkpoint superglue_model_tr_slam_ran_1 \
  --tag sg_slam_hpatches \
  --plot
```

Results saved to: `outputs/results/hpatches/sg_slam_hpatches/`

---

### 2. 🏔️ MegaDepth-1500 — Outdoor Relative Pose

**What it tests:** Relative pose estimation (F/E matrix) on outdoor landmark scenes.  
**Metrics:** AUC @ 5°, 10°, 20° rotation error  
**Dataset size:** 1500 image pairs, auto-downloads from ETH CVG (~1.5 GB)  
**Best for:** Your **SuperGlue** model (outdoor scenes, perspective changes)

```bash
# SuperGlue model on MegaDepth-1500
python3 -m gluefactory.eval.megadepth1500 \
  --checkpoint superglue_model_tr_slam_ran_1 \
  --tag sg_slam_megadepth
```

Results saved to: `outputs/results/megadepth1500/sg_slam_megadepth/`

---

### 3. 🏠 ScanNet-1500 — Indoor Relative Pose

**What it tests:** Relative pose estimation on indoor RGB-D scenes.  
**Metrics:** AUC @ 5°, 10°, 20° rotation error  
**Dataset size:** 1500 image pairs, auto-downloads from ETH CVG (~700 MB)  
**Best for:** Your **SuperGlue SLAM model** (trained on indoor posed images — most aligned!)

```bash
# SuperGlue model on ScanNet-1500 (best match for your SLAM-trained model)
python3 -m gluefactory.eval.scannet1500 \
  --checkpoint superglue_model_tr_slam_ran_1 \
  --tag sg_slam_scannet
```

Results saved to: `outputs/results/scannet1500/sg_slam_scannet/`

---

### 4. 🌲 ETH3D — Multi-view 3D reconstruction

**What it tests:** Relative pose on high-resolution outdoor/indoor scenes.  
**Metrics:** AUC @ 5°, 10°, 20°  
**Dataset size:** Smaller, auto-downloads

```bash
python3 -m gluefactory.eval.eth3d \
  --checkpoint superglue_model_tr_slam_ran_1 \
  --tag sg_slam_eth3d
```

---

## Recommended Evaluation Order

```
Priority 1  →  HPatches      (fast, ~2 min, no GPU needed, auto-download)
Priority 2  →  ScanNet-1500  (indoor, matches your SLAM training domain)
Priority 3  →  MegaDepth-1500 (outdoor, ~20 min with GPU)
Priority 4  →  ETH3D         (optional)
```

---

## What about your SuperPoint model?

Your **SuperPoint** model (`superpoint_model_tr__run_0_force_true`) is a **detector-only** model (no matcher in its config). To evaluate it you need to pair it with a matcher. The eval pipeline needs a full `two_view_pipeline`. Options:

**Option A** — Use your custom SP + default NN matcher:
```bash
python3 -m gluefactory.eval.hpatches \
  --conf superpoint-open+NN \
  --checkpoint superpoint_model_tr__run_0_force_true \
  --tag sp_custom_hpatches
```

**Option B** — Use your custom SP + your custom SG (full pipeline):
```bash
# Requires a combined config that loads SP from your checkpoint
# and SG from the slam checkpoint
python3 -m gluefactory.eval.hpatches \
  --conf superpoint_custom+superglue_homography \
  --tag sp_sg_combined_hpatches
```

---

## Common Flags

| Flag | Description |
|------|-------------|
| `--checkpoint <name>` | Folder name under `outputs/training/` |
| `--conf <name>` | Config name from `gluefactory/configs/` (without `.yaml`) |
| `--tag <name>` | Custom name for results folder |
| `--overwrite` | Re-run predictions (re-inference) |
| `--overwrite_eval` | Re-compute metrics only (skip re-inference) |
| `--plot` | Show recall curve plots |

---

## Quick Start — Run HPatches Now

```bash
cd /home/thippe/workspaces/AiMl/glue-factory

python3 -m gluefactory.eval.hpatches \
  --checkpoint superglue_model_tr_slam_ran_1 \
  --tag sg_slam_hpatches \
  --plot
```

The dataset (~250 MB) will **auto-download** to `data/hpatches-sequences-release/` on first run.

Results will be at: `outputs/results/hpatches/sg_slam_hpatches/`
