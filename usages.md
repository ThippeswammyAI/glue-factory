# Glue Factory Usages & Project Structure

This document describes the structure of the utility scripts (moved to `scripts/` to clean the project root) and lists the exact commands to run the training and evaluation pipelines.

---

## 1. Restructured Scripts Directory (`scripts/`)

To keep the repository root clean, all stand-alone python tools and helpers have been moved to the `scripts/` directory:

| Script Path | Purpose | How to Run |
| :--- | :--- | :--- |
| **`scripts/visualize_custom.py`** | Runs custom SuperPoint inference and overlays detected keypoints on a directory of images. | `python3 scripts/visualize_custom.py` |
| **`scripts/export_interactive_matches.py`** | Runs SuperPoint/SuperGlue matching on image pairs and creates an interactive HTML match dashboard. | `python3 scripts/export_interactive_matches.py` |
| **`scripts/match_images.py`** | Performs batch matching of image pairs using a two-view matching configuration. | `python3 scripts/match_images.py` |
| **`scripts/boost_h5_scores.py`** | Preprocessing tool to scale up matching scores in an exported HDF5 feature file. | `python3 scripts/boost_h5_scores.py` |
| **`scripts/filter_h5_file.py`** | Filters keypoints in an exported HDF5 feature file by confidence score threshold. | `python3 scripts/filter_h5_file.py` |
| **`scripts/profile_model.py`** | Benchmarks the inference execution speed (latency) of a specific model. | `python3 scripts/profile_model.py` |
| **`scripts/profile_loop.py`** | Measures the data loading throughput of a training/validation dataset loop. | `python3 scripts/profile_loop.py` |
| **`scripts/profile_warp.py`** | Benchmarks homography warping operations. | `python3 scripts/profile_warp.py` |

---

## 2. Training Commands

### Train SuperGlue on Custom LiDAR Dataset
To start training SuperGlue using your custom dataset configuration (loading pre-extracted features):
```bash
python3 -m gluefactory.train superpoint_custom+superglue_homography_custom_dataset \
  --conf gluefactory/configs/superpoint_custom+superglue_homography_custom_dataset.yaml
```

### Launch TensorBoard
To monitor training loss, learning rate, and validation metrics:
```bash
tensorboard --logdir outputs/training/
```

---

## 3. Evaluation Commands

### A. Evaluating Custom SuperPoint on Standard RGB (HPatches)
To evaluate your custom SuperPoint model using Nearest Neighbor (NN) matching on HPatches:
```bash
python3 -m gluefactory.eval.hpatches \
  --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
  --conf superpoint-open+NN \
  --overwrite
```

### B. Evaluating Custom SuperPoint + Matchers on RGB (HPatches)
Evaluating how your custom model behaves with SuperGlue and GlueStick:

* **Custom SuperPoint + SuperGlue** (using NMS=4 and detection threshold=0.005):
  ```bash
  python3 -m gluefactory.eval.hpatches \
    --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --conf superpoint_custom+superglue \
    --overwrite
  ```

* **Custom SuperPoint + GlueStick** (utilizing lines and OpenCV estimator):
  ```bash
  python3 -m gluefactory.eval.hpatches \
    --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --conf gluefactory/configs/superpoint+lsd+gluestick.yaml \
    --overwrite
  ```

### C. Evaluating on Custom LiDAR Homography Dataset
Evaluating homography estimation performance on your low-contrast LiDAR reflectivity images:

* **Custom SuperPoint + NN (Target Model)**:
  ```bash
  python3 -m gluefactory.eval.homographies \
    --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
    --conf superpoint-open+NN \
    --overwrite
  ```

* **Official Magic Leap SuperPoint + NN (Baseline)**:
  ```bash
  python3 -m gluefactory.eval.homographies \
    --conf superpoint+NN \
    --overwrite
  ```