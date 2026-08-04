#!/usr/bin/env python3
"""
Visualize SLAM data cached in an H5 file — either image pairs (--source
pairs) or per-image pseudo-labels (--source labels) — with SuperPoint
keypoints overlaid, and generate a browsable HTML grid of the result.

Merges the previously duplicated gluefactory/scripts/visualize_slam_pairs.py
and visualize_slam_labels.py, whose H5-load / keypoint-scatter / HTML-grid
code was ~90% identical, differing only in 1-panel vs 2-panel figures and
whether the driving loop is a pairs file or H5 keys.

Usage:
    # Pairs: two SLAM images per figure, keypoints from a per-image H5 cache
    python -m gluefactory.scripts.visualize_slam_dataset --source pairs \\
        --data_dir data/output/slam --pairs_file pairs_train.txt \\
        --h5_path data/output/slam/exports/sp_features_slam.h5 \\
        --max_items 100

    # Labels: one SLAM image per figure, pseudo-label keypoints from an H5 cache
    python -m gluefactory.scripts.visualize_slam_dataset --source labels \\
        --data_dir data/output/slam \\
        --h5_path data/output/slam/exports/pseudo_labels_slam.h5 \\
        --max_items 50
"""

import argparse
import logging
from pathlib import Path

import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from gluefactory.visualization import dashboard, viz2d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visualize_slam_dataset")


def _load_h5_kpts(f_h5, name, score_thresh):
    kpts = f_h5[name]["keypoints"][:]
    scores = f_h5[name]["keypoint_scores"][:]
    mask = scores > score_thresh
    return kpts[mask], scores[mask]


def _visualize_pairs(args, dataset_dir, h5_path, out_dir):
    pairs_path = dataset_dir / args.pairs_file
    if not pairs_path.exists():
        logger.error(f"Pairs file not found: {pairs_path}")
        return []

    pairs = []
    with open(pairs_path) as f:
        for line in f:
            p1, p2 = line.strip().split()
            pairs.append((p1, p2))
    if not pairs:
        logger.warning("No pairs found.")
        return []
    if args.max_items > 0 and len(pairs) > args.max_items:
        np.random.seed(42)
        np.random.shuffle(pairs)
        pairs = pairs[: args.max_items]

    f_h5 = h5py.File(h5_path, "r") if h5_path.exists() else None
    if f_h5 is None:
        logger.warning(f"Could not load H5 features at {h5_path}. Only drawing images.")

    saved = []
    for p1, p2 in tqdm(pairs, desc="Visualizing pairs"):
        n1, n2 = Path(p1).name, Path(p2).name
        img1 = cv2.imread(str(dataset_dir / "images" / p1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(dataset_dir / "images" / p2), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue

        kpts1, scores1 = (
            _load_h5_kpts(f_h5, n1, args.score_thresh) if f_h5 is not None and n1 in f_h5
            else (np.zeros((0, 2)), np.zeros(0))
        )
        kpts2, scores2 = (
            _load_h5_kpts(f_h5, n2, args.score_thresh) if f_h5 is not None and n2 in f_h5
            else (np.zeros((0, 2)), np.zeros(0))
        )
        colors1 = plt.get_cmap("plasma")(scores1).tolist() if len(kpts1) else []
        colors2 = plt.get_cmap("plasma")(scores2).tolist() if len(kpts2) else []

        viz2d.plot_images([img1, img2], titles=[n1, n2], cmaps="gray")
        viz2d.plot_keypoints([kpts1, kpts2], colors=[colors1, colors2], ps=15, a=0.8)

        out_path = out_dir / f"{Path(n1).stem}_{Path(n2).stem}.png"
        viz2d.save_plot(out_path, facecolor="#121212")
        plt.close()
        saved.append(out_path)

    if f_h5 is not None:
        f_h5.close()
    return saved


def _visualize_labels(args, dataset_dir, h5_path, out_dir):
    if not h5_path.exists():
        logger.error(f"H5 file not found: {h5_path}")
        return []
    rgb_dir = dataset_dir / "images" / args.modality

    with h5py.File(h5_path, "r") as f:
        image_names = list(f.keys())
    if args.max_items > 0:
        np.random.seed(42)
        np.random.shuffle(image_names)
        image_names = image_names[: args.max_items]

    saved = []
    with h5py.File(h5_path, "r") as f:
        for name in tqdm(image_names, desc="Visualizing labels"):
            rgb_path = rgb_dir / name
            if not rgb_path.exists():
                logger.warning(f"Image {name} not found in {rgb_dir}, skipping.")
                continue
            img = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
            kpts, scores = _load_h5_kpts(f, name, args.score_thresh)

            out_path = out_dir / name
            viz2d.plot_keypoints_overlay(
                img, kpts, scores, out_path,
                title=f"{name} ({len(kpts)} keypoints)", colorbar=True,
            )
            saved.append(out_path)
    return saved


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", choices=["pairs", "labels"], required=True)
    parser.add_argument("--data_dir", type=str, default="data/output/sample_slam",
                         help="Path to the SLAM dataset directory")
    parser.add_argument("--h5_path", type=str, default=None,
                         help="Path to the features (--source pairs) or pseudo-labels "
                              "(--source labels) H5 file; defaults under <data_dir>/exports/")
    parser.add_argument("--pairs_file", type=str, default="pairs_train.txt",
                         help="Pairs file name (--source pairs only)")
    parser.add_argument("--modality", type=str, default="rgb",
                         help="Image modality subdirectory (--source labels only)")
    parser.add_argument("--max_items", type=int, default=50,
                         help="Max pairs/images to visualize (0 = all)")
    parser.add_argument("--score_thresh", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.data_dir)
    default_h5 = ("exports/sp_features_slam.h5" if args.source == "pairs"
                  else "exports/pseudo_labels_slam.h5")
    h5_path = Path(args.h5_path) if args.h5_path else dataset_dir / default_h5

    default_out = ("visualizations/pair_matches" if args.source == "pairs"
                   else "visualizations/kpt_labels")
    out_dir = Path(args.output_dir) if args.output_dir else dataset_dir / default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Visualizing {args.source} from {h5_path} → {out_dir}")
    if args.source == "pairs":
        saved = _visualize_pairs(args, dataset_dir, h5_path, out_dir)
        title = "SLAM Pairs Visualization"
    else:
        saved = _visualize_labels(args, dataset_dir, h5_path, out_dir)
        title = "SLAM Pseudo-Labels Visualization"

    if saved:
        html_path = dashboard.render_image_grid(
            out_dir, saved, title=title,
            grid_min_width=800 if args.source == "pairs" else 600,
        )
        logger.info(f"Generated HTML grid at {html_path}")


if __name__ == "__main__":
    main()
