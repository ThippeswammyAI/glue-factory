# Glue Factory — End-to-End Pipeline Reference

This document is the single source of truth for every command in the custom LiDAR
SuperPoint → SuperGlue training pipeline, from raw images to final evaluation.

---

## Table of Contents

1. [Scripts Directory](#1-scripts-directory)
2. [Dataset Preparation (from raw images)](#2-dataset-preparation-from-raw-images)
3. [Export SuperPoint Feature Cache (.h5)](#3-export-superpoint-feature-cache-h5)
4. [Train SuperPoint](#4-train-superpoint)
5. [Prepare SuperGlue Dataset (from trained SuperPoint)](#5-prepare-superglue-dataset-from-trained-superpoint)
6. [Train SuperGlue](#6-train-superglue)
7. [Evaluation Commands](#7-evaluation-commands)
8. [Monitoring & Utilities](#8-monitoring--utilities)

---

## 1. Scripts Directory

All stand-alone helper tools live in `scripts/` to keep the project root clean.

| Script | Purpose | How to Run |
| :--- | :--- | :--- |
| **`gluefactory.scripts.run_inference`** | Unified SuperPoint / SuperPoint+SuperGlue / SuperPoint+LightGlue inference and visualization (checkpoint or exported `.pt`), replacing `visualize_custom.py`, `export_interactive_matches.py`, `match_images.py`, `match_images_from_pt.py`, `infer_superpoint.py`, `infer_superglue.py`. | `python3 -m gluefactory.scripts.run_inference --backend checkpoint --matcher superglue ...` |
| **`scripts/boost_h5_scores.py`** | Scale up matching scores in an exported HDF5 feature file. | `python3 scripts/boost_h5_scores.py` |
| **`scripts/filter_h5_file.py`** | Filter keypoints in an HDF5 feature file by confidence score threshold. | `python3 scripts/filter_h5_file.py` |
| **`scripts/profile_model.py`** | Benchmark inference latency of a specific model. | `python3 scripts/profile_model.py` |
| **`scripts/profile_loop.py`** | Measure data-loading throughput of a training/validation dataset loop. | `python3 scripts/profile_loop.py` |
| **`scripts/profile_warp.py`** | Benchmark homography warping operations. | `python3 scripts/profile_warp.py` |

---

## 2. Dataset Preparation (from raw images)

The goal of this stage is to convert the raw images in
`data/output/dataset/images/{nearir,range,reflectivity,signal}/` into:
1. A text **image list** (`custom_image_list.txt`) consumed by the `homographies` dataloader.
2. An HDF5 **pseudo-label cache** (consensus keypoints) used by both SuperPoint and
   SuperGlue training.

The directory structure must match `data/output/sample_data/` exactly:

```
data/output/dataset/
├── images/
│   ├── nearir/         ← raw PNG frames
│   ├── range/
│   ├── reflectivity/   ← primary training modality
│   └── signal/
├── custom_image_list.txt   ← generated in step 2a
└── exports/
    └── pseudo_labels.h5    ← generated in step 2b
```

### 2a — Generate the Image List (manual, already done)

The image list was generated with:

```bash
# Lists all reflectivity PNGs relative to images/ and writes custom_image_list.txt
find data/output/dataset/images/reflectivity -type f -name "*.png" \
  | sed 's|data/output/dataset/images/||' \
  | sort \
  > data/output/dataset/custom_image_list.txt

# Verify (should be 344 lines)
wc -l data/output/dataset/custom_image_list.txt
head -5  data/output/dataset/custom_image_list.txt
```

> **Note:** The adaptation script (step 2b) will **also auto-write** this list — so you
> only need the `find` command if you want to regenerate it independently.

### 2b — Joint Multimodal Homographic Adaptation (generate pseudo-labels)

**Script:** `gluefactory/scripts/prepare_and_visualize_adaptation.py`

The script's `--dataset` is a **name relative to `data/`** (i.e., `DATA_PATH`).  
It auto-discovers images from `data/<dataset>/images/nearir/`, writes
`data/<dataset>/exports/pseudo_labels.h5`, and auto-generates `custom_image_list.txt`.

```bash
# Actual CLI args (from argparse in the script):
#   --dataset           name under data/          (default: custom_dataset1)
#   --num_warps         warps per modality         (default: 200)
#   --thresh            keypoint threshold         (default: 0.02)
#   --nms               NMS radius                 (default: 5)
#   --warp_mode         2d | 3d                    (default: 3d)
#   --camera_info       path to camera.info        (default: plan/camera.info)
#   --num_threads       parallel workers           (default: 4)
#   --image_list_modality  nearir|range|reflectivity|signal (default: reflectivity)
#   --weights           custom SP checkpoint (optional)
#   --use_gpu / --no_gpu
#   --save_detailed_warps   save per-warp side-by-side plots

# --- Full custom dataset (output/dataset) ---
python3 -m gluefactory.scripts.prepare_and_visualize_adaptation \
    --dataset              output/dataset \
    --image_list_modality  reflectivity \
    --num_warps            14 \
    --num_threads          14 \
    --use_gpu \
    --warp_mode            3d
# Output H5 → data/output/dataset/exports/pseudo_labels.h5
# Image list → data/output/dataset/custom_image_list.txt

# --- Small sample dataset (output/sample_data) ---
python3 -m gluefactory.scripts.prepare_and_visualize_adaptation \
    --dataset              output/sample_data \
    --image_list_modality  reflectivity \
    --num_warps            14 \
    --num_threads          8 \
    --use_gpu \
    --warp_mode            3d


# --- With custom-trained SuperPoint weights &  dataset (output/dataset) ---
python3 -m gluefactory.scripts.prepare_and_visualize_adaptation \
    --dataset              output/dataset \
    --weights              outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --image_list_modality  reflectivity \
    --num_warps            14 \
    --num_threads          14 \
    --use_gpu \
    --warp_mode            3d
```

> **Camera intrinsics:** For `--warp_mode 3d` the script reads `plan/camera.info`
> (default path). If it doesn't exist, use `--camera_info <path>` or fall back to
> `--warp_mode 2d`.

Sanity-check the output:
```bash
python3 -c "
import h5py
f = h5py.File('data/output/dataset/exports/pseudo_labels.h5', 'r')
keys = list(f.keys())
print(f'Total scenes: {len(keys)}')
print('Sample key:', keys[0])
k0 = keys[0]
print('  keypoints:', f[k0]['keypoints'].shape)
print('  scores:   ', f[k0]['keypoint_scores'].shape)
"

---

## 3. Export SuperPoint Feature Cache (.h5)

Use `gluefactory/scripts/export_local_features.py` to export keypoints + descriptors.

**CLI** (from script source):
```
python3 -m gluefactory.scripts.export_local_features <dataset> [--method sp|sp_custom|sift|disk] [--num_workers N]
```
- `<dataset>` = name under `data/`  →  e.g. `output/dataset`
- Reads `data/<dataset>/custom_image_list.txt` automatically when present
- Output: `data/exports/<method_name>.h5`

### 3a — Official SuperPoint (`--method sp`)

```bash
# output: data/exports/r1600_SP-k2048-nms3.h5
python3 -m gluefactory.scripts.export_local_features output/dataset \
    --method      sp \
    --num_workers 8
```

### 3b — Custom-Trained SuperPoint (`--method sp_custom`)

The `sp_custom` config in the script hardcodes:
- weights: `outputs/training/superpoint_custom_run/checkpoint_best.tar`
- nms_radius=4, max_num_keypoints=2048, detection_threshold=0.005, **no resize**

```bash
# output: data/exports/custom_SP-k2048-nms4.h5
python3 -m gluefactory.scripts.export_local_features output/dataset \
    --method      sp_custom \
    --num_workers 8
```

### 3c — Extract Descriptors at Consensus Keypoints (SuperGlue prep)

Use `gluefactory/scripts/export_consensus_features.py`:

```bash
# CLI args: --dataset, --pseudo_labels_h5, --weights, --output_h5, --modality
python3 -m gluefactory.scripts.export_consensus_features \
    --dataset          output/dataset \
    --pseudo_labels_h5 data/output/dataset/exports/pseudo_labels.h5 \
    --weights          outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --output_h5        data/output/dataset/exports/custom_dataset_consensus_SP.h5 \
    --modality         reflectivity
```

### Inspect any H5 file

```bash
python3 -c "
import h5py, sys
f = h5py.File(sys.argv[1], 'r')
keys = list(f.keys())
print(f'Total entries: {len(keys)},  first key: {keys[0]}')
grp = f[keys[0]]
for name, ds in grp.items():
    print(f'  {name}: {ds.shape} {ds.dtype}')
" data/output/dataset/exports/custom_SP-k2048-nms4.h5

# Post-processing helpers
python3 scripts/boost_h5_scores.py \
    --input  data/output/dataset/exports/custom_SP-k2048-nms4.h5 \
    --output data/output/dataset/exports/custom_SP-k2048-nms4-boosted.h5 --scale 2.0

python3 scripts/filter_h5_file.py \
    --input     data/output/dataset/exports/custom_SP-k2048-nms4.h5 \
    --output    data/output/dataset/exports/custom_SP-k2048-nms4-filtered.h5 \
    --threshold 0.02
```

---

## 4. Train SuperPoint

SuperPoint is trained with the **homography adaptation** self-supervised objective using
the pre-computed pseudo-label `.h5` cache as keypoint targets.

### 4a — Train on Full Custom Dataset

Config: `gluefactory/configs/superpoint_custom_homography.yaml`  
Data dir: `output/dataset` · H5: `output/dataset/exports/pseudo_labels.h5`

```bash
# --- Fresh training ---
python3 -m gluefactory.train superpoint_custom_run \
    --conf gluefactory/configs/superpoint_custom_homography.yaml

# --- Resume interrupted training ---
python3 -m gluefactory.train superpoint_custom_run \
    --conf    gluefactory/configs/superpoint_custom_homography.yaml \
    --restore

# --- Fine-tune from official Magic Leap weights ---
python3 -m gluefactory.train superpoint_finetune_run \
    --conf gluefactory/configs/superpoint_custom_homography.yaml \
    train.load_experiment=superpoint_open
```

### 4b — Train on Small Sample Dataset (fast iteration / debug)

Config: same YAML but override `data_dir` and `train_size` via CLI dotlist:

```bash
python3 -m gluefactory.train superpoint_sample_run \
    --conf gluefactory/configs/superpoint_custom_homography.yaml \
    data.data_dir=output/sample_data \
    data.image_list=custom_image_list.txt \
    data.train_size=15 \
    data.val_size=4 \
    data.load_features.path=output/sample_data/exports/pseudo_labels.h5 \
    train.epochs=50
```

### 4c — Multi-GPU Distributed Training

```bash
python3 -m gluefactory.train superpoint_custom_run \
    --conf        gluefactory/configs/superpoint_custom_homography.yaml \
    --distributed
```

### 4d — Key Training CLI Flags (from `gluefactory/train.py`)

| Flag | Effect |
| :--- | :--- |
| `--restore` | Resume from the last checkpoint of `<experiment>` |
| `--distributed` | Enable multi-GPU DDP training |
| `--overfit` | Train and eval on a single batch (debug mode) |
| `--mixed_precision float16` | Enable FP16 mixed precision |
| `--compile default` | Use `torch.compile` for extra speed |
| `--print_arch` | Print the full model architecture before training |
| `--no_eval_0` | Skip the validation pass at iteration 0 |
| `--run_benchmarks` | Run benchmark suites (HPatches etc.) at each test epoch |
| `dotlist` | Override any config key inline, e.g. `train.lr=5e-5` |

---

## 4e. SuperPoint Inference (after training)

### Visualise Keypoints on Images  (`gluefactory.scripts.run_inference --matcher none`)

Runs the trained model alone (no matcher) on every image under `--input` and saves
a per-image keypoint-overlay PNG (colour = detection score) plus an HTML grid.
This replaced the old `scripts/visualize_custom.py`.

```bash
# CLI args (see gluefactory/scripts/run_inference.py --help for the full list)
#   --backend         checkpoint | exported
#   --extractor_ckpt   path to .tar checkpoint
#   --matcher          none (extractor-only)
#   --input            image directory
#   --output           html | data | png | all
#   --output_dir        where to write results

# --- Sample dataset, reflectivity ---
python3 -m gluefactory.scripts.run_inference \
    --backend checkpoint --matcher none \
    --extractor_ckpt outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --input data/output/sample_data/images/reflectivity \
    --output all --output_dir data/output/sample_data/visualizations/custom_detections
# Outputs → data/output/sample_data/visualizations/custom_detections/
#            ├── images/<stem>_keypoints.png  (one per image)
#            └── images/index.html            (browsable grid)

# --- Different modality (near-IR) ---
python3 -m gluefactory.scripts.run_inference \
    --backend checkpoint --matcher none \
    --extractor_ckpt outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --input data/output/dataset/images/nearir \
    --output all --output_dir data/output/dataset/visualizations/custom_detections
```

> **How it loads weights:** `SLAMMatcher`/`load_experiment` reads `checkpoint["model"]`
> and strips the `extractor.` prefix automatically, so the standard training checkpoint
> works directly.

### Export Keypoints + Descriptors to H5  (`gluefactory/scripts/export_local_features.py`)

Batch-exports keypoints, descriptors, and scores for every image into an HDF5 file.
The checkpoint and model config come from the **`configs` dict inside the script**.
To change the checkpoint path edit `configs["sp_custom"]["conf"]["weights"]` in the file.

```bash
# CLI (from argparse — dataset is POSITIONAL, no --dataset flag)
#   <dataset>       name under data/           (positional, required)
#   --method        sp | sp_custom | sift | disk  (default: sp)
#   --export_prefix optional string prefix for the output filename
#   --num_workers   dataloader workers

# Current sp_custom config (check export_local_features.py to confirm):
#   weights:              outputs/training/superpoint_custom_run_0_force_true/checkpoint_best.tar
#   nms_radius:           6
#   max_num_keypoints:    2048
#   detection_threshold:  0.01

# --- Export custom SP features from full dataset ---
# Output: data/exports/custom_SP-k2048-nms4.h5
python3 -m gluefactory.scripts.export_local_features output/dataset \
    --method      sp_custom \
    --num_workers 8

# --- Export official SP features (for baseline comparison) ---
# Output: data/exports/r1600_SP-k2048-nms3.h5
python3 -m gluefactory.scripts.export_local_features output/dataset \
    --method      sp \
    --num_workers 8

# --- Copy the output to where the training YAML expects it ---
mkdir -p data/output/dataset/exports
cp data/exports/custom_SP-k2048-nms4.h5 \
   data/output/dataset/exports/custom_SP-k2048-nms4.h5

# Inspect the result
python3 -c "
import h5py
f = h5py.File('data/exports/custom_SP-k2048-nms4.h5', 'r')
keys = list(f.keys())
print(f'Entries: {len(keys)}')
print('First key:', keys[0])
for n, ds in f[keys[0]].items():
    print(f'  {n}: {ds.shape} {ds.dtype}')
"
```

---

## 5. Prepare SuperGlue Dataset (from trained SuperPoint)

SuperGlue requires a feature `.h5` with keypoints **and descriptors** from a fixed
SuperPoint checkpoint — it does **not** re-run the extractor at training time.

### 5a — Export SuperGlue-Ready Features

Same as §4e export but the result is the input to SuperGlue training:

```bash
# dataset is a POSITIONAL argument (no --dataset flag)
# output: data/exports/custom_SP-k2048-nms4.h5
python3 -m gluefactory.scripts.export_local_features output/dataset \
    --method      sp_custom \
    --num_workers 8

# Copy to where the YAML expects it:
mkdir -p data/output/dataset/exports
cp data/exports/custom_SP-k2048-nms4.h5 \
   data/output/dataset/exports/custom_SP-k2048-nms4.h5

# YAML reference (superpoint_custom+superglue_homography.yaml):
#   load_features.path: "output/dataset/exports/custom_SP-k2048-nms4.h5"
```

### 5b — Build Consensus Feature Cache (Adaptation with custom SP weights)

Re-run homographic adaptation using your **trained SuperPoint** checkpoint to get
the highest-quality SuperGlue training targets. Use `--weights` to load the checkpoint:

```bash
# Step 1 — re-run adaptation with trained SP; writes pseudo_labels.h5 and image list
python3 -m gluefactory.scripts.prepare_and_visualize_adaptation \
    --dataset              output/dataset \
    --weights              outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --image_list_modality  reflectivity \
    --num_warps            14 \
    --num_threads          14 \
    --use_gpu \
    --warp_mode            3d
# Output H5   → data/output/dataset/exports/pseudo_labels.h5  (overwritten)
# Image list  → data/output/dataset/custom_image_list.txt  (auto-written)

# Step 2 — attach descriptors to those consensus keypoints
python3 -m gluefactory.scripts.export_consensus_features \
    --dataset          output/dataset \
    --pseudo_labels_h5 data/output/dataset/exports/pseudo_labels.h5 \
    --weights          outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --output_h5        data/output/dataset/exports/custom_dataset_consensus_SP.h5 \
    --modality         reflectivity
```

### 5c — Verify the SuperGlue H5 and Image List

```bash
# Check image list (auto-written by the adaptation script)
wc -l data/output/dataset/custom_image_list.txt
head -3  data/output/dataset/custom_image_list.txt

# Inspect the consensus H5
python3 -c "
import h5py
f = h5py.File('data/output/dataset/exports/custom_dataset_consensus_SP.h5', 'r')
keys = list(f.keys())
print(f'Keys: {len(keys)}')
k0 = keys[0]
for n, ds in f[k0].items():
    print(f'  {n}: {ds.shape}')
"
```

---

## 6. Train SuperGlue

### 6a — Train SuperGlue on Full Custom Dataset

Config: `gluefactory/configs/superpoint_custom+superglue_homography.yaml`  
Uses: `data/output/dataset/exports/custom_SP-k2048-nms4.h5`

```bash
# --- Fresh training ---
python3 -m gluefactory.train superpoint_custom+superglue_homography \
    --conf gluefactory/configs/superpoint_custom+superglue_homography.yaml

# --- Resume interrupted training ---
python3 -m gluefactory.train superpoint_custom+superglue_homography \
    --conf    gluefactory/configs/superpoint_custom+superglue_homography.yaml \
    --restore
```

### 6b — Train SuperGlue on Small Sample Dataset (debug)

Config: `gluefactory/configs/superpoint_custom+superglue_homography_custom_dataset.yaml`  
Uses: `data/output/sample_data/exports/custom_dataset_consensus_SP.h5`

```bash
python3 -m gluefactory.train superpoint_custom+superglue_homography_custom_dataset \
    --conf gluefactory/configs/superpoint_custom+superglue_homography_custom_dataset.yaml
```

### 6c — Fine-tune from Official SuperGlue Weights

```bash
python3 -m gluefactory.train superglue_finetune_run \
    --conf gluefactory/configs/superpoint_custom+superglue_homography.yaml \
    train.load_experiment=superglue_outdoor
```

### 6d — Multi-GPU Distributed SuperGlue Training

```bash
python3 -m gluefactory.train superpoint_custom+superglue_homography \
    --conf        gluefactory/configs/superpoint_custom+superglue_homography.yaml \
    --distributed
```

---

## 7. Evaluation Commands

### A. Evaluate Custom SuperPoint — Standard RGB (HPatches)

NN matching baseline on the standard HPatches benchmark:

```bash
python3 -m gluefactory.eval.hpatches \
    --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --conf superpoint-open+NN \
    --overwrite
```

### B. Evaluate Custom SuperPoint + Matchers — Standard RGB (HPatches)

* **Custom SuperPoint + SuperGlue** (NMS=4, threshold=0.005):
  ```bash
  python3 -m gluefactory.eval.hpatches \
      --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
      --conf superpoint_custom+superglue \
      --overwrite
  ```

* **Custom SuperPoint + GlueStick** (lines + OpenCV estimator):
  ```bash
  python3 -m gluefactory.eval.hpatches \
      --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
      --conf gluefactory/configs/superpoint+lsd+gluestick.yaml \
      --overwrite
  ```

### C. Evaluate on Custom LiDAR Homography Dataset

Homography estimation accuracy on your low-contrast LiDAR reflectivity images:

* **Custom SuperPoint + NN (target model)**:
  ```bash
  python3 -m gluefactory.eval.homographies \
      --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
      --conf superpoint-open+NN \
      --overwrite
  ```

* **Official Magic Leap SuperPoint + NN (baseline)**:
  ```bash
  python3 -m gluefactory.eval.homographies \
      --conf superpoint+NN \
      --overwrite
  ```

* **Custom SuperPoint + Custom SuperGlue**:
  ```bash
  python3 -m gluefactory.eval.homographies \
      --checkpoint outputs/training/superglue_finetune_run/checkpoint_best.tar \
      --conf gluefactory/configs/superpoint_custom+superglue_homography.yaml \
      --overwrite
  ```

### D. Evaluate Trained SuperGlue — Standard HPatches

```bash
python3 -m gluefactory.eval.hpatches \
    --checkpoint outputs/training/superpoint_custom+superglue_homography/checkpoint_best.tar \
    --conf gluefactory/configs/superpoint_custom+superglue_homography.yaml \
    --overwrite
```

### E. Evaluate — Custom Dataset (explicit data_dir override)

Override the dataset path at eval time with dotlist args:

```bash
# Evaluate on FULL custom dataset
python3 -m gluefactory.eval.homographies \
    --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --conf superpoint-open+NN \
    data.data_dir=output/dataset \
    data.image_list=custom_image_list.txt \
    --overwrite

# Evaluate on SAMPLE dataset (quick smoke-test)
python3 -m gluefactory.eval.homographies \
    --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --conf superpoint-open+NN \
    data.data_dir=output/sample_data \
    data.image_list=custom_image_list.txt \
    --overwrite
```

---

## 8. Monitoring & Utilities

### Launch TensorBoard

```bash
# Monitor all experiments at once
tensorboard --logdir outputs/training/

# Monitor a specific experiment
tensorboard --logdir outputs/training/superpoint_custom_run/
```

### Inspect HDF5 Cache Files

```bash
# Print all top-level keys and tensor shapes
python3 -c "
import h5py, sys
path = sys.argv[1]
f = h5py.File(path, 'r')
keys = list(f.keys())
print(f'Total entries: {len(keys)}')
k = keys[0]
print(f'Sample key: {k}')
for name, ds in f[k].items():
    print(f'  {name}: {ds.shape} {ds.dtype}')
" data/output/dataset/exports/custom_SP-k2048-nms4.h5
```

### Visualize Keypoints on Custom Images

```bash
python3 -m gluefactory.scripts.run_inference \
    --backend checkpoint --matcher none \
    --extractor_ckpt outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --input data/output/dataset/images/reflectivity \
    --output all --output_dir data/output/visualizations/keypoints
```

### Generate Interactive Match Dashboard

```bash
python3 -m gluefactory.scripts.run_inference \
    --backend checkpoint --matcher superglue \
    --extractor_ckpt outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --matcher_ckpt   outputs/training/superglue_custom_run/checkpoint_best.tar \
    --input data/output/dataset/images/reflectivity \
    --filter_threshold 0.02 \
    --output all --output_dir data/output/visualizations/matches
```

---

## Pipeline Summary

```
data/output/dataset/images/{nearir,range,reflectivity,signal}/
          │
          ▼  Step 2a — generate image list
  custom_image_list.txt
          │
          ▼  Step 2b — Multimodal Homographic Adaptation
  exports/pseudo_labels.h5        ← SuperPoint pseudo-labels
          │
          ▼  Step 4 — Train SuperPoint
  outputs/training/superpoint_custom_run/checkpoint_best.tar
          │
          ▼  Step 5 — Re-export features with trained SP
  exports/custom_SP-k2048-nms4.h5 (or consensus_SP.h5)
          │
          ▼  Step 6 — Train SuperGlue
  outputs/training/superpoint_custom+superglue_homography/checkpoint_best.tar
          │
          ▼  Step 7 — Evaluate
  eval/   HPatches + Custom LiDAR Homography benchmarks
```