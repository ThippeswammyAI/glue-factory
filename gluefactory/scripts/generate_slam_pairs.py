#!/usr/bin/env python3
"""
Generate training/validation image pair lists for the SLAM dataset.

Uses camera poses from the SLAM trajectory to find geometrically overlapping
image pairs: two frames are paired if their translational distance falls within
[--min_dist, --max_dist] and their co-visibility (estimated from depth overlap)
exceeds a threshold. Writes pairs_train.txt and pairs_val.txt in the dataset
directory, one pair per line formatted as "rgb/frame_a.png rgb/frame_b.png".

Usage:
    python -m gluefactory.scripts.generate_slam_pairs \\
        --data_dir    data/output/slam \\
        --poses_file  data/output/slam/poses.txt \\
        --modality    rgb \\
        --min_dist    0.05 \\
        --max_dist    1.0  \\
        --val_ratio   0.15
"""

import argparse
import logging
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from gluefactory.models import get_model
from gluefactory.slam.geometry import (
    compute_covisibility,
    load_camera_intrinsics,
    parse_poses,
)
from gluefactory.slam.io import write_h5_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_thread_local = threading.local()


def _get_thread_model(args, device):
    if not hasattr(_thread_local, "model"):
        model_conf = {
            "nms_radius": args.nms_radius,
            "max_num_keypoints": args.max_keypoints,
            "detection_threshold": 0.0,
            "trainable": False,
        }
        if hasattr(args, "sp_weights") and args.sp_weights:
            model_conf["weights"] = args.sp_weights
        _thread_local.model = get_model("superpoint_open")(model_conf).to(device).eval()
    return _thread_local.model


def _extract_features(image_name, dataset_dir, model, device):
    rgb_path = dataset_dir / "images/rgb" / image_name
    img_rgb = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
    if img_rgb is None:
        return None, None, None
    img_t = torch.from_numpy(img_rgb.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        pred = model({"image": img_t})
    return (
        pred["keypoints"][0].cpu().numpy(),
        pred["keypoint_scores"][0].cpu().numpy(),
        pred["descriptors"][0].cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Pose-Based Pairs for SLAM")
    parser.add_argument("--data_dir", type=str, default="data/output/sample_slam")
    parser.add_argument("--max_dist", type=float, default=2.0)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--max_angle", type=float, default=30.0)
    parser.add_argument("--min_overlap", type=float, default=0.1)
    parser.add_argument("--max_pairs", type=int, default=10)
    parser.add_argument("--split_ratio", type=float, default=0.83)
    parser.add_argument("--extract_features", action="store_true", default=False)
    parser.add_argument("--sp_weights", type=str, default=None)
    parser.add_argument("--nms_radius", type=int, default=3)
    parser.add_argument("--max_keypoints", type=int, default=512)
    parser.add_argument("--num_threads", type=int, default=14)
    args = parser.parse_args()

    dataset_dir = Path(args.data_dir)
    poses_path = dataset_dir / "poses_odom_RGBD_slam.txt"
    if not poses_path.exists():
        logger.error(f"Poses file not found: {poses_path}")
        return

    rgb_dir = dataset_dir / "images/rgb"
    depth_dir = dataset_dir / "images/depth"
    calib_dir = dataset_dir / "images/calib"

    image_names = sorted(
        [p.name for p in rgb_dir.glob("*.png")] + [p.name for p in rgb_dir.glob("*.jpg")]
    )
    if not image_names:
        logger.error("No images found.")
        return

    poses = parse_poses(poses_path)

    first_ts = Path(image_names[0]).stem
    calib_path = calib_dir / f"{first_ts}.yaml"
    if not calib_path.exists():
        calib_path = list(calib_dir.glob("*.yaml"))[0]
    K = load_camera_intrinsics(calib_path)

    logger.info("Computing pairwise distances and covisibility...")
    pairs = []
    name_to_ts = {n: Path(n).stem for n in image_names}
    valid_names = [n for n in image_names if name_to_ts[n] in poses]

    for i, name_i in enumerate(tqdm(valid_names)):
        ts_i = name_to_ts[name_i]
        R_i, t_i = poses[ts_i]

        candidates = []
        for j, name_j in enumerate(valid_names):
            if i == j:
                continue
            ts_j = name_to_ts[name_j]
            R_j, t_j = poses[ts_j]
            dist = np.linalg.norm(t_i - t_j)
            if dist < args.min_dist or dist > args.max_dist:
                continue
            R_rel = R_i.T @ R_j
            trace = np.trace(R_rel)
            angle = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)))
            if angle > args.max_angle:
                continue
            candidates.append((j, name_j, ts_j, R_j, t_j, dist))

        candidates.sort(key=lambda x: x[-1])
        candidates = candidates[: args.max_pairs * 2]

        valid_candidates = []
        depth_path_i = depth_dir / name_i
        for j, name_j, ts_j, R_j, t_j, dist in candidates:
            overlap = compute_covisibility(depth_path_i, R_i, t_i, R_j, t_j, K)
            if overlap >= args.min_overlap:
                valid_candidates.append((name_i, name_j))
                if len(valid_candidates) >= args.max_pairs:
                    break
        pairs.extend(valid_candidates)

    logger.info(f"Found {len(pairs)} pairs.")

    np.random.seed(42)
    np.random.shuffle(pairs)
    num_train = int(len(pairs) * args.split_ratio)
    train_pairs = pairs[:num_train]
    val_pairs = pairs[num_train:]

    with open(dataset_dir / "pairs_train.txt", "w") as f:
        for pi, pj in train_pairs:
            f.write(f"rgb/{pi} rgb/{pj}\n")
    with open(dataset_dir / "pairs_val.txt", "w") as f:
        for pi, pj in val_pairs:
            f.write(f"rgb/{pi} rgb/{pj}\n")

    if args.extract_features:
        logger.info("Extracting features...")
        exports_dir = dataset_dir / "exports"
        exports_dir.mkdir(exist_ok=True)
        h5_path = exports_dir / "sp_features_slam.h5"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        unique_images = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))

        with h5py.File(h5_path, "w") as f:
            with tqdm(total=len(unique_images)) as pbar:
                def worker(name):
                    local_model = _get_thread_model(args, device)
                    kpts, scores, desc = _extract_features(name, dataset_dir, local_model, device)
                    return name, kpts, scores, desc

                if args.num_threads > 1:
                    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                        for name, kpts, scores, desc in executor.map(worker, unique_images):
                            if kpts is not None:
                                write_h5_features(f, name, kpts, scores, desc)
                            pbar.update(1)
                else:
                    for n in unique_images:
                        name, kpts, scores, desc = worker(n)
                        if kpts is not None:
                            write_h5_features(f, name, kpts, scores, desc)
                        pbar.update(1)

    logger.info("Done generating pairs and features.")


if __name__ == "__main__":
    main()
