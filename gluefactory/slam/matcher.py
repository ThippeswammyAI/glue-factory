"""SLAMMatcher: SuperPoint + SuperGlue inference pipeline for SLAM image pairs."""

import logging
from pathlib import Path

import numpy as np
import torch

from gluefactory.utils.experiments import load_experiment
from gluefactory.slam.io import get_image_pairs, load_image

logger = logging.getLogger(__name__)


class SLAMMatcher:
    """Run SuperPoint + SuperGlue inference on image pairs.

    Loads both models from training checkpoints via a single
    :func:`~gluefactory.utils.experiments.load_experiment` call.  Used by
    both the interactive HTML dashboard (``scripts/match_images.py``) and the
    raw-data exporter (``scripts/export_interactive_matches.py``).

    Example::

        matcher = SLAMMatcher(
            sg_ckpt="outputs/training/superglue_slam_run/checkpoint_best.tar",
            sp_ckpt="outputs/training/superpoint_slam_run/checkpoint_best.tar",
            device="cuda",
        )
        results = matcher.match_directory("data/output/slam/images/rgb", max_pairs=20)
    """

    def __init__(self, sg_ckpt, sp_ckpt, device=None, conf=None):
        """
        Args:
            sg_ckpt:  Path to the SuperGlue training checkpoint (.tar).
            sp_ckpt:  Path to the SuperPoint training checkpoint (.tar).
            device:   "cuda" or "cpu".  Auto-detected when None.
            conf:     Optional dict of inference overrides:
                      nms_radius, max_num_keypoints, detection_threshold,
                      filter_threshold, remove_borders.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        conf = conf or {}
        self._filter_threshold = conf.get("filter_threshold", 0.01)

        override_conf = {
            "extractor": {
                "name": "superpoint_open",
                "weights": str(sp_ckpt),
                "nms_radius": conf.get("nms_radius", 3),
                "max_num_keypoints": conf.get("max_num_keypoints", 512),
                "detection_threshold": conf.get("detection_threshold", 0.005),
                "remove_borders": conf.get("remove_borders", 4),
                "trainable": False,
            },
            "matcher": {
                "name": "gluefactory_nonfree.superglue",
                "filter_threshold": self._filter_threshold,
            },
        }
        self._model = load_experiment(str(sg_ckpt), override_conf).to(device).eval()
        logger.info(f"SLAMMatcher loaded on {device}")

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def match_pair(self, img0_gray, img1_gray):
        """Run SP+SG on a single image pair.

        Args:
            img0_gray, img1_gray: uint8 np.ndarray [H, W] grayscale images.

        Returns:
            dict with:
                keypoints0, keypoints1    np.ndarray [N, 2]
                scores0, scores1          np.ndarray [N]
                matches0                  np.ndarray [N] (−1 = unmatched)
                mscores0                  np.ndarray [N] (match confidence)
        """
        device = self.device
        t0 = torch.from_numpy(img0_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        t1 = torch.from_numpy(img1_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            pred = self._model({"view0": {"image": t0}, "view1": {"image": t1}})
        return {
            "keypoints0": pred["keypoints0"][0].cpu().numpy(),
            "keypoints1": pred["keypoints1"][0].cpu().numpy(),
            "scores0": pred["keypoint_scores0"][0].cpu().numpy(),
            "scores1": pred["keypoint_scores1"][0].cpu().numpy(),
            "matches0": pred["matches0"][0].cpu().numpy(),
            "mscores0": pred["matching_scores0"][0].cpu().numpy(),
        }

    # ------------------------------------------------------------------
    # Batch directory inference
    # ------------------------------------------------------------------

    def match_directory(self, input_dir, max_pairs=None, resize=0):
        """Match all consecutive image pairs in a directory.

        Args:
            input_dir:  Path to image directory (or "img0.png,img1.png" pair).
            max_pairs:  Cap on the number of pairs processed.  None = all.
            resize:     Resize longest side to this value before inference.
                        0 = no resize (only pad to multiples of 8).

        Returns:
            List of result dicts, one per pair.  Each dict contains:
                idx, name0, name1, path0, path1,
                img0_bgr, img1_bgr  (np.ndarray uint8)
                keypoints0, keypoints1, scores0, scores1, matches0, mscores0,
                metrics (dict: total_kpts0/1, num_matches, match_ratio, avg_mscore).
        """
        pairs = get_image_pairs(input_dir)
        if not pairs:
            logger.error(f"No image pairs found in: {input_dir}")
            return []
        if max_pairs is not None:
            pairs = pairs[:max_pairs]

        results = []
        for idx, (p0, p1) in enumerate(pairs):
            try:
                img0_bgr, img0_gray = load_image(p0, resize)
                img1_bgr, img1_gray = load_image(p1, resize)
                out = self.match_pair(img0_gray, img1_gray)

                valid = (out["matches0"] != -1) & (out["mscores0"] >= self._filter_threshold)
                n_match = int(valid.sum())
                n0, n1 = len(out["keypoints0"]), len(out["keypoints1"])
                avg_s = float(out["mscores0"][valid].mean()) if n_match > 0 else 0.0

                results.append({
                    "idx": idx,
                    "path0": p0,
                    "path1": p1,
                    "name0": p0.name,
                    "name1": p1.name,
                    "img0_bgr": img0_bgr,
                    "img1_bgr": img1_bgr,
                    **out,
                    "metrics": {
                        "total_kpts0": n0,
                        "total_kpts1": n1,
                        "num_matches": n_match,
                        "match_ratio": n_match / max(1, min(n0, n1)),
                        "avg_mscore": avg_s,
                    },
                })
            except Exception as e:
                logger.error(f"Error on pair {idx} ({p0.name} ↔ {p1.name}): {e}", exc_info=True)
        return results
