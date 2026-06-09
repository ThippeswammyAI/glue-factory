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
| **`scripts/visualize_custom.py`** | Run custom SuperPoint inference and overlay detected keypoints on a directory of images. | `python3 scripts/visualize_custom.py` |
| **`scripts/export_interactive_matches.py`** | Run SuperPoint/SuperGlue matching on image pairs and generate an interactive HTML match dashboard. | `python3 scripts/export_interactive_matches.py` |
| **`scripts/match_images.py`** | Batch-match image pairs using a two-view matching config. | `python3 scripts/match_images.py` |
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

### 2a — Generate the Image List

Run from the **workspace root** (`glue-factory/`).  
Generates `data/output/dataset/custom_image_list.txt` listing every reflectivity frame
(relative paths from `images/`):

```bash
# List all reflectivity PNGs relative to the images/ directory
find data/output/dataset/images/reflectivity -type f -name "*.png" \
  | sed 's|data/output/dataset/images/||' \
  | sort \
  > data/output/dataset/custom_image_list.txt

# Verify
wc -l data/output/dataset/custom_image_list.txt
head -5  data/output/dataset/custom_image_list.txt
```

> **Tip:** To include *all four* modalities in the list (for multimodal adaptation),
> replace `reflectivity` with `*` in the `find` pattern.

### 2b — Joint Multimodal Homographic Adaptation (generate pseudo-labels)

This step runs **Homographic Adaptation** across all four pixel-aligned modalities using
the official (or custom-trained) SuperPoint to produce consensus pseudo-ground-truth
keypoints, and saves them into an HDF5 cache.

```bash
# Full-GPU multimodal adaptation — writes data/output/dataset/exports/pseudo_labels.h5
python3 -m gluefactory.scripts.prepare_and_visualize_adaptation \
    --data_dir         data/output/dataset \
    --image_list       data/output/dataset/custom_image_list.txt \
    --image_list_modality reflectivity \
    --output_h5        data/output/dataset/exports/pseudo_labels.h5 \
    --num_warps        15 \
    --num_threads      8 \
    --use_gpu \
    --warp_mode        3d

# Quick sanity-check — print the first 5 keys in the HDF5 file
python3 -c "
import h5py
f = h5py.File('data/output/dataset/exports/pseudo_labels.h5', 'r')
print('Keys:', list(f.keys())[:5])
print('Keypoints shape for first key:', f[list(f.keys())[0]]['keypoints'].shape)
"
```

> **Note:** For the **small sample dataset** (`data/output/sample_data/`) the same
> commands apply — just replace `data/output/dataset` with `data/output/sample_data`.
> The sample exports already exist at
> `data/output/sample_data/exports/pseudo_labels.h5`.

---

## 3. Export SuperPoint Feature Cache (.h5)

After training (or using the official Magic Leap weights), re-export clean descriptors
and keypoints from your images into a new HDF5 file.  This `.h5` is then fed to
SuperGlue training via `load_features`.

### 3a — Using the Official (pre-trained) SuperPoint Weights

```bash
# Export features from the full dataset using the official SuperPoint
python3 -m gluefactory.scripts.export_features \
    --conf        superpoint-open+NN \
    --image_dir   data/output/dataset/images \
    --image_list  data/output/dataset/custom_image_list.txt \
    --export_dir  data/output/dataset/exports \
    --output_name custom_SP-official-k2048-nms4.h5 \
    --max_num_keypoints 2048 \
    --nms_radius  4

# Verify
python3 -c "
import h5py
f = h5py.File('data/output/dataset/exports/custom_SP-official-k2048-nms4.h5', 'r')
keys = list(f.keys())
print(f'Total images: {len(keys)}')
print('Sample key:', keys[0])
print('keypoints:', f[keys[0]]['keypoints'].shape)
print('descriptors:', f[keys[0]]['descriptors'].shape)
"
```

### 3b — Using Your Custom-Trained SuperPoint Weights

```bash
# Export features using the best checkpoint from your custom SuperPoint run
python3 -m gluefactory.scripts.export_features \
    --conf        superpoint-open+NN \
    --checkpoint  outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --image_dir   data/output/dataset/images \
    --image_list  data/output/dataset/custom_image_list.txt \
    --export_dir  data/output/dataset/exports \
    --output_name custom_SP-k2048-nms4.h5 \
    --max_num_keypoints 2048 \
    --nms_radius  4

# Optional — boost scores if match precision is too low
python3 scripts/boost_h5_scores.py \
    --input   data/output/dataset/exports/custom_SP-k2048-nms4.h5 \
    --output  data/output/dataset/exports/custom_SP-k2048-nms4-boosted.h5 \
    --scale   2.0

# Optional — filter out low-confidence keypoints
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

## 5. Prepare SuperGlue Dataset (from trained SuperPoint)

SuperGlue requires a feature `.h5` with keypoints **and descriptors** from a fixed
SuperPoint checkpoint — it does **not** re-run the extractor at training time.

### 5a — Export SuperGlue-Ready Features

```bash
# Generate the h5 used in superpoint_custom+superglue_homography.yaml
python3 -m gluefactory.scripts.export_features \
    --conf        superpoint-open+NN \
    --checkpoint  outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --image_dir   data/output/dataset/images \
    --image_list  data/output/dataset/custom_image_list.txt \
    --export_dir  data/output/dataset/exports \
    --output_name custom_SP-k2048-nms4.h5 \
    --max_num_keypoints 512 \
    --nms_radius  4 \
    --detection_threshold 0.005
```

### 5b — Build Consensus Feature Cache (Homographic Adaptation with custom SP)

For the highest-quality SuperGlue training targets, re-run homographic adaptation
**with your trained SuperPoint** checkpoint:

```bash
python3 -m gluefactory.scripts.prepare_and_visualize_adaptation \
    --data_dir         data/output/dataset \
    --image_list       data/output/dataset/custom_image_list.txt \
    --image_list_modality reflectivity \
    --checkpoint       outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --output_h5        data/output/dataset/exports/custom_dataset_consensus_SP.h5 \
    --num_warps        15 \
    --num_threads      8 \
    --use_gpu \
    --warp_mode        3d

# Inspect result
python3 -c "
import h5py
f = h5py.File('data/output/dataset/exports/custom_dataset_consensus_SP.h5', 'r')
print('Keys:', len(list(f.keys())))
k0 = list(f.keys())[0]
print('keypoints:', f[k0]['keypoints'].shape)
print('descriptors:', f[k0]['descriptors'].shape)
print('scores:', f[k0]['keypoint_scores'].shape)
"
```

### 5c — Generate the SuperGlue Image List

The `homographies` dataloader needs an image list relative to `images/`:

```bash
find data/output/dataset/images/reflectivity -type f -name "*.png" \
  | sed 's|data/output/dataset/images/||' \
  | sort \
  > data/output/dataset/custom_image_list.txt
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
python3 scripts/visualize_custom.py \
    --image_dir  data/output/dataset/images/reflectivity \
    --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --output_dir data/output/visualizations/keypoints
```

### Generate Interactive Match Dashboard

```bash
python3 scripts/export_interactive_matches.py \
    --image_dir   data/output/dataset/images/reflectivity \
    --h5_path     data/output/dataset/exports/custom_SP-k2048-nms4.h5 \
    --score_threshold 0.02 \
    --output_dir  data/output/visualizations/matches
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