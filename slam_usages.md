# End-to-End Pose-Aware SLAM Training & Evaluation Pipeline

This guide outlines the workflows for training and evaluating **SuperPoint** and **SuperGlue** on your custom pose-aware SLAM RGB-D dataset (`data/output/slam`).

---

## Pipeline Overview

```
Raw SLAM Dataset (images, depth, poses, calib)
      │
      ▼  Step 1: Run Pseudo-Label Generation
Exports/pseudo_labels_slam.h5 (SuperPoint targets)
      │
      ▼  Step 2: Train SuperPoint on SLAM images
Outputs/training/superpoint_slam_run/checkpoint_best.tar
      │
      ▼  Step 3: Export features with trained SP
Exports/sp_features_slam.h5
      │
      ▼  Step 4: Generate matches & pairs for SuperGlue
Pairs_train.txt, Pairs_val.txt, sp_features_slam.h5
      │
      ▼  Step 5: Train SuperGlue on pose-based pairings
Outputs/training/superglue_slam_run/checkpoint_best.tar
      │
      ▼  Step 6: Evaluate models on SLAM Evaluation Pipeline
Metrics (mreproj_prec, mepi_prec, mrel_pose_error)
```

---

## 1. SuperPoint Dataset Creation (Pseudo-Labels)

Generate ground-truth keypoint pseudo-labels by applying homographic and 3D projective adaptation to your SLAM frames.

```bash
# --- On Sample Dataset (quick test) ---
python3 -m gluefactory.scripts.prepare_slam_superpoint \
    --data_dir data/output/sample_slam \
    --num_warps 50

# --- On Full Dataset ---
python3 -m gluefactory.scripts.prepare_slam_superpoint \
    --data_dir data/output/slam \
    --num_warps 50
```
*   **Input**: RGB-D frames, camera intrinsics, and pose file (`poses_odom_RGBD_slam.txt`).
*   **Output**: `data/output/slam/exports/pseudo_labels_slam.h5` and image lists.

---

## 2. Visualize SuperPoint Dataset (Pseudo-Labels)

Create an interactive HTML dashboard to visualize the generated pseudo-labels overlaid on the RGB frames.

```bash
# --- On Sample Dataset ---
python3 -m gluefactory.scripts.visualize_slam_labels \
    --data_dir data/output/sample_slam \
    --max_images 50

# --- On Full Dataset ---
python3 -m gluefactory.scripts.visualize_slam_labels \
    --data_dir data/output/slam \
    --max_images 100
```
*   **Outputs**: `data/output/slam/visualizations/kpt_labels/index.html` (viewable in browser).

---

## 3. Train SuperPoint on SLAM Dataset

SuperPoint is trained self-supervised using the `homographies` dataset configuration by warping the RGB SLAM images on-the-fly and matching them to the pre-generated pseudo-labels.

```bash
# --- Train on Full Dataset ---
python3 -m gluefactory.train superpoint_slam_run \
    --conf gluefactory/configs/superpoint_custom_homography_tranning.yaml \
    data.data_dir=output/slam \
    data.image_dir=images/rgb \
    data.image_list=image_list_train.txt \
    data.load_features.path=output/slam/exports/pseudo_labels_slam.h5

# --- Overfit check on Sample Dataset ---
python3 -m gluefactory.train superpoint_sample_run \
    --conf gluefactory/configs/superpoint_custom_homography_tranning.yaml \
    --overfit \
    data.data_dir=output/sample_slam \
    data.image_dir=images/rgb \
    data.image_list=image_list_train.txt \
    data.load_features.path=output/sample_slam/exports/pseudo_labels_slam.h5 \
    data.num_workers=0
```

---

## 4. SuperPoint Inference & Feature Export

Export trained SuperPoint keypoints and descriptors to a cached `.h5` file, which will serve as the input for training SuperGlue.

### Step 4a — Run the export tool
```bash
# --- Export from Full Dataset ---
python3 -m gluefactory.scripts.export_local_features output/slam \
    --method sp_custom \
    --export_prefix slam_ \
    --num_workers 4
```
*   *Note*: To load your specific checkpoint, update the `weights` path under the `sp_custom` block in `gluefactory/scripts/export_local_features.py` before running.

### Step 4b — Move export to destination directory
The training YAML expects the features to be saved in the dataset directory:
```bash
mkdir -p data/output/slam/exports
cp data/exports/slam_custom_SP-k2048-nms4.h5 data/output/slam/exports/sp_features_slam.h5
```

---

## 5. Evaluate/Validate SuperPoint

Validate your trained SuperPoint model on keypoint repeatability, depth reprojection precision, and pose estimation error using the custom SLAM evaluation pipeline.

```bash
# --- Evaluate on Sample Dataset ---
python3 -m gluefactory.eval.slam \
    --conf superpoint-open+NN \
    data.data_dir=output/sample_slam \
    --overwrite

# --- Evaluate on Full Dataset ---
python3 -m gluefactory.eval.slam \
    --conf superpoint-open+NN \
    data.data_dir=output/slam \
    --overwrite
```

---

## 6. SuperGlue Dataset Creation (Pair Matches)

Generate co-visible image pairs and pre-extract SuperPoint descriptors using your trained model weights.

```bash
# --- On Sample Dataset ---
python3 -m gluefactory.scripts.generate_slam_pairs \
    --data_dir data/output/sample_slam \
    --extract_features \
    --min_dist 0.0 \
    --sp_weights outputs/training/superpoint_slam_run/checkpoint_best.tar

# --- On Full Dataset ---
python3 -m gluefactory.scripts.generate_slam_pairs \
    --data_dir data/output/slam \
    --extract_features \
    --sp_weights outputs/training/superpoint_slam_run/checkpoint_best.tar
```
*   **Outputs**: `pairs_train.txt`, `pairs_val.txt`, and `exports/sp_features_slam.h5`.

---

## 7. Visualize SuperGlue Match Pairs

Visualize the overlapping match pairings side-by-side to verify alignment quality.

```bash
# --- On Sample Dataset ---
python3 -m gluefactory.scripts.visualize_slam_pairs \
    --data_dir data/output/sample_slam \
    --max_pairs 50

# --- On Full Dataset ---
python3 -m gluefactory.scripts.visualize_slam_pairs \
    --data_dir data/output/slam \
    --max_pairs 50
```
*   **Outputs**: `data/output/slam/visualizations/pair_matches/index.html`.

---

## 8. Train SuperGlue on SLAM Pairs

Train the SuperGlue attention-based GNN using the pose-based pairs and the pre-extracted features.

```bash
# --- Train on Full Dataset ---
python3 -m gluefactory.train superglue_slam_run \
    --conf gluefactory/configs/superpoint+superglue_slam.yaml

# --- Overfit check on Sample Dataset ---
python3 -m gluefactory.train superglue_sample_run \
    --conf gluefactory/configs/superpoint+superglue_slam.yaml \
    --overfit \
    data.data_dir=output/sample_slam \
    data.load_features.path=output/sample_slam/exports/sp_features_slam.h5 \
    data.num_workers=0
```

---

## 9. Evaluate/Validate SuperGlue

Evaluate SuperGlue on the custom evaluation pipeline.

```bash
# --- Evaluate on Sample Dataset ---
python3 -m gluefactory.eval.slam \
    --conf gluefactory/configs/superpoint+superglue_slam.yaml \
    data.data_dir=output/sample_slam \
    data.load_features.path=output/sample_slam/exports/sp_features_slam.h5 \
    --overwrite

# --- Evaluate on Full Dataset ---
python3 -m gluefactory.eval.slam \
    --conf gluefactory/configs/superpoint+superglue_slam.yaml \
    data.data_dir=output/slam \
    data.load_features.path=output/slam/exports/sp_features_slam.h5 \
    --overwrite
```
*   **Key Metrics tracked**:
    *   `mreproj_prec@3px`: Mean percentage of inlier keypoints with reprojection error < 3px.
    *   `mgt_match_recall@3px`: Mean match recall compared to depth-based ground truth.
    *   `mrel_pose_error`: Mean relative pose error (rotation and translation combined).
