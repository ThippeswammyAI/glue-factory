#!/usr/bin/env python3
"""
Prepare SuperPoint pseudo-labels for SLAM image sequences.

Runs SuperPoint on every image in the SLAM dataset and writes raw detections
(keypoints, scores, descriptors) to an HDF5 cache. Supports multi-threaded
inference with one model instance per thread. Optionally applies pose-guided
hybrid adaptation to suppress dynamic-object keypoints using depth and camera
poses from the SLAM trajectory.

Output H5 structure (per image):
    <image_stem>/keypoints        float32 [N, 2]
    <image_stem>/keypoint_scores  float32 [N]

Usage:
    python -m gluefactory.scripts.prepare_slam_superpoint \\
        --data_dir data/output/slam \\
        --output_h5 data/output/slam/exports/pseudo_labels_slam.h5 \\
        --weights outputs/training/superpoint_slam_run/checkpoint_best.tar \\
        --modality rgb \\
        --max_keypoints 512 \\
        --num_workers 4
"""

import argparse
import logging
import numpy as np
from pathlib import Path

import torch

from gluefactory.slam.extractor import SLAMFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SLAM Homographic Adaptation")
    parser.add_argument("--data_dir", type=str, default="data/output/sample_slam")
    parser.add_argument("--output_h5", type=str, default=None,
                        help="Output H5 path (default: <data_dir>/exports/pseudo_labels_slam.h5)")
    parser.add_argument("--num_warps", type=int, default=50)
    parser.add_argument("--thresh", type=float, default=0.015)
    parser.add_argument("--nms", type=int, default=4)
    parser.add_argument("--max_keypoints", type=int, default=512)
    parser.add_argument("--warp_mode", type=str, default="3d", choices=["2d", "3d"])
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--num_threads", type=int, default=14)
    parser.add_argument("--split_ratio", type=float, default=0.83)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--modality", type=str, default="rgb")
    parser.add_argument("--pose_ratio", type=float, default=0.6)
    parser.add_argument("--poses_file", type=str, default="poses_odom_RGBD_slam.txt")
    parser.add_argument("--max_dist", type=float, default=2.0)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--max_angle", type=float, default=30.0)
    parser.add_argument("--min_overlap", type=float, default=0.1)
    parser.add_argument("--max_neighbors", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_h5 = args.output_h5 or str(data_dir / "exports/pseudo_labels_slam.h5")
    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"

    conf = {
        "nms_radius": args.nms,
        "max_num_keypoints": args.max_keypoints,
        "detection_threshold": 0.0,
    }
    extractor = SLAMFeatureExtractor(args.weights, conf, device)
    extractor.process_dataset(
        data_dir=data_dir,
        output_h5=output_h5,
        num_workers=args.num_threads,
        num_warps=args.num_warps,
        mode=args.warp_mode,
        modality=args.modality,
        poses_file=args.poses_file,
        pose_ratio=args.pose_ratio,
        detection_threshold=args.thresh,
        nms_radius=args.nms,
        max_dist=args.max_dist,
        min_dist=args.min_dist,
        max_angle=args.max_angle,
        min_overlap=args.min_overlap,
        max_neighbors=args.max_neighbors,
    )

    # Write train/val image list splits
    rgb_dir = data_dir / "images" / args.modality
    image_names = sorted(
        [p.name for p in rgb_dir.glob("*.png")] + [p.name for p in rgb_dir.glob("*.jpg")]
    )
    logger.info("Generating split files...")
    shuffled = list(image_names)
    np.random.RandomState(42).shuffle(shuffled)
    num_train = int(len(shuffled) * args.split_ratio)
    train_names = sorted(shuffled[:num_train])
    val_names = sorted(shuffled[num_train:])
    with open(data_dir / "image_list_train.txt", "w") as f:
        f.writelines([f"{args.modality}/{n}\n" for n in train_names])
    with open(data_dir / "image_list_val.txt", "w") as f:
        f.writelines([f"{args.modality}/{n}\n" for n in val_names])
    logger.info("All processing complete!")


if __name__ == "__main__":
    main()
