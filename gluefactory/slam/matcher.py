"""SLAMMatcher: SuperPoint + SuperGlue/LightGlue inference pipeline for SLAM
image pairs."""

import logging
from pathlib import Path

import numpy as np
import torch

from gluefactory.utils.experiments import load_experiment
from gluefactory.slam.io import get_image_pairs, load_image

logger = logging.getLogger(__name__)

# Matcher-specific config overrides. Both output "matches0"/"matches1"/
# "matching_scores0"/"matching_scores1" from the model's pred dict — that
# naming is shared between the two architectures already (see also
# gluefactory/scripts/export_model.py, which relies on the same convention).
# "none" disables the matcher entirely (TwoViewPipeline skips it when
# matcher.name is falsy) for extractor-only (SuperPoint-alone) inference.
_MATCHER_CONFS = {
    "none": lambda filter_threshold: {"name": None},
    "superglue": lambda filter_threshold: {
        "name": "gluefactory_nonfree.superglue",
        "filter_threshold": filter_threshold,
    },
    "lightglue": lambda filter_threshold: {
        "name": "matchers.lightglue",
        "filter_threshold": filter_threshold,
        "input_dim": 256,
        "flash": False,
        # Explicitly None: the checkpoint's state_dict is loaded by
        # load_experiment() right after construction, so LightGlue must not
        # also try to download/read its own default pretrained weights.
        "weights": None,
    },
}


class SLAMMatcher:
    """Run SuperPoint alone, or SuperPoint + SuperGlue/LightGlue, on image pairs.

    Loads the model(s) from training checkpoints via a single
    :func:`~gluefactory.utils.experiments.load_experiment` call.  Used by
    ``gluefactory.scripts.run_inference`` (checkpoint backend) for any
    combination of extractor-only / +SuperGlue / +LightGlue inference.

    Example::

        matcher = SLAMMatcher(
            matcher_ckpt="outputs/training/lightglue_slam_run/checkpoint_best.tar",
            extractor_ckpt="outputs/training/superpoint_slam_run/checkpoint_best.tar",
            matcher="lightglue",
            device="cuda",
        )
        results = matcher.match_directory("data/output/slam/images/rgb", max_pairs=20)

        # Extractor-only (no matcher_ckpt needed):
        sp_only = SLAMMatcher(None, extractor_ckpt="...", matcher="none")
    """

    def __init__(self, matcher_ckpt, extractor_ckpt, device=None, conf=None,
                 matcher="superglue"):
        """
        Args:
            matcher_ckpt:   Path to the matcher (SuperGlue or LightGlue)
                            training checkpoint (.tar). Ignored when
                            `matcher` is "none"/None.
            extractor_ckpt: Path to the SuperPoint training checkpoint (.tar).
            device:         "cuda" or "cpu".  Auto-detected when None.
            conf:           Optional dict of inference overrides:
                            nms_radius, max_num_keypoints, detection_threshold,
                            filter_threshold, remove_borders.
            matcher:        "none", "superglue" or "lightglue" — which matcher
                            architecture `matcher_ckpt` was trained with.
                            "none" runs SuperPoint alone (extractor-only).
        """
        matcher = matcher or "none"
        if matcher not in _MATCHER_CONFS:
            raise ValueError(
                f"Unknown matcher {matcher!r}, expected one of {list(_MATCHER_CONFS)}"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.has_matcher = matcher != "none"
        conf = conf or {}
        self._filter_threshold = conf.get("filter_threshold", 0.01)

        override_conf = {
            "extractor": {
                "name": "superpoint_open",
                "weights": str(extractor_ckpt),
                "nms_radius": conf.get("nms_radius", 3),
                "max_num_keypoints": conf.get("max_num_keypoints", 512),
                "detection_threshold": conf.get("detection_threshold", 0.005),
                "remove_borders": conf.get("remove_borders", 4),
                "trainable": False,
            },
            "matcher": _MATCHER_CONFS[matcher](self._filter_threshold),
        }
        # Extractor-only mode has no matcher checkpoint to load from, so use
        # the extractor checkpoint as the load_experiment() entry point.
        ckpt = matcher_ckpt if self.has_matcher else extractor_ckpt
        self._model = load_experiment(str(ckpt), override_conf).to(device).eval()
        logger.info(f"SLAMMatcher ({matcher}) loaded on {device}")

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def match_pair(self, img0_gray, img1_gray):
        """Run the extractor (+ matcher, if configured) on a single image pair.

        Args:
            img0_gray, img1_gray: uint8 np.ndarray [H, W] grayscale images.

        Returns:
            dict with:
                keypoints0, keypoints1    np.ndarray [N, 2]
                scores0, scores1          np.ndarray [N]
                matches0                  np.ndarray [N] (−1 = unmatched);
                                          all −1 when `matcher="none"`.
                mscores0                  np.ndarray [N] (match confidence);
                                          all 0 when `matcher="none"`.
        """
        device = self.device
        t0 = torch.from_numpy(img0_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        t1 = torch.from_numpy(img1_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            pred = self._model({"view0": {"image": t0}, "view1": {"image": t1}})
        kpts0 = pred["keypoints0"][0].cpu().numpy()
        kpts1 = pred["keypoints1"][0].cpu().numpy()
        if self.has_matcher:
            matches0 = pred["matches0"][0].cpu().numpy()
            mscores0 = pred["matching_scores0"][0].cpu().numpy()
        else:
            matches0 = np.full(len(kpts0), -1, dtype=np.int64)
            mscores0 = np.zeros(len(kpts0), dtype=np.float32)
        return {
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "scores0": pred["keypoint_scores0"][0].cpu().numpy(),
            "scores1": pred["keypoint_scores1"][0].cpu().numpy(),
            "matches0": matches0,
            "mscores0": mscores0,
        }

    def extract(self, img_gray):
        """Run the extractor alone on a single image (no matcher forward).

        Args:
            img_gray: uint8 np.ndarray [H, W] grayscale image.

        Returns:
            dict with "keypoints" np.ndarray [N, 2] and "scores" np.ndarray [N].
        """
        device = self.device
        t = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            pred = self._model.extractor({"image": t})
        return {
            "keypoints": pred["keypoints"][0].cpu().numpy(),
            "scores": pred["keypoint_scores"][0].cpu().numpy(),
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
        return run_match_directory(self, self._filter_threshold, input_dir, max_pairs, resize)


def run_match_directory(backend, filter_threshold, input_dir, max_pairs=None, resize=0):
    """Run `backend.match_pair(img0_gray, img1_gray)` over all pairs in a
    directory, loading images via :func:`~gluefactory.slam.io.load_image` and
    computing the same per-pair metrics as :meth:`SLAMMatcher.match_directory`.

    Shared between `SLAMMatcher` (checkpoint backend) and
    `gluefactory.scripts.run_inference`'s exported-model backend so both
    produce identically-shaped result dicts.
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
            out = backend.match_pair(img0_gray, img1_gray)

            valid = (out["matches0"] != -1) & (out["mscores0"] >= filter_threshold)
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
