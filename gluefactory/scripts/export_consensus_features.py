"""
Extract SuperPoint descriptors at consensus keypoint locations and save to H5.

Takes a pseudo-labels H5 file (containing consensus keypoints) and a trained
SuperPoint checkpoint, re-runs the dense descriptor backbone on each image, then
samples descriptors at the consensus keypoint coordinates. The result is an H5
file with keypoints, keypoint_scores, and descriptors ready for SuperGlue
training or evaluation.

Usage:
    python -m gluefactory.scripts.export_consensus_features \\
        --dataset         output/slam \\
        --pseudo_labels_h5 data/output/slam/exports/pseudo_labels_slam.h5 \\
        --weights         outputs/training/superpoint_slam_run/checkpoint_best.tar \\
        --output_h5       data/output/slam/exports/sp_features_slam.h5 \\
        --modality        rgb
"""

import argparse
import logging
from pathlib import Path

import torch

from gluefactory.settings import DATA_PATH
from gluefactory.slam.labeler import SLAMLabelGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract descriptors for consensus keypoints")
    parser.add_argument("--dataset", type=str, default="custom_dataset",
                        help="Dataset name under data/")
    parser.add_argument("--pseudo_labels_h5", type=str, required=True,
                        help="Path to pseudo labels H5 file")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to custom SuperPoint weights")
    parser.add_argument("--output_h5", type=str, required=True,
                        help="Path to save final H5 file")
    parser.add_argument("--modality", type=str, default="reflectivity",
                        help="Modality prefix")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    pseudo_labels_path = Path(args.pseudo_labels_h5)
    if not pseudo_labels_path.exists():
        logger.error(f"Pseudo labels file not found: {pseudo_labels_path}")
        return

    dataset_dir = DATA_PATH / args.dataset
    images_dir = dataset_dir / "images" / args.modality

    labeler = SLAMLabelGenerator(args.weights, {}, device)
    labeler.generate(
        pseudo_labels_h5=pseudo_labels_path,
        images_dir=images_dir,
        output_h5=args.output_h5,
        modality=args.modality,
    )
    logger.info(f"Successfully saved features to {args.output_h5}")


if __name__ == "__main__":
    main()
