"""SLAMFeatureExtractor: SuperPoint with homographic adaptation for SLAM datasets."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from gluefactory.models import get_model
from gluefactory.slam.geometry import (
    compute_covisibility,
    get_neighbor_relative_poses,
    load_camera_intrinsics,
    parse_poses,
    precompute_3d_points,
    sample_random_3d_transform,
    sample_random_homography,
    warp_3d_projective,
    warp_perspective,
)
from gluefactory.slam.io import write_h5_features

logger = logging.getLogger(__name__)

_thread_local = threading.local()


class SLAMFeatureExtractor:
    """SuperPoint feature extractor with homographic adaptation.

    Supports both 2D random homography warps and 3D pose-guided projective
    warps for generating pseudo-labels on SLAM image sequences.

    Example::

        extractor = SLAMFeatureExtractor(
            "outputs/training/superpoint_slam_run/checkpoint_best.tar", {}, "cuda"
        )
        extractor.process_dataset(
            "data/output/slam", "data/output/slam/exports/pseudo_labels_slam.h5",
            num_workers=8, num_warps=50, mode="3d",
        )
    """

    def __init__(self, sp_weights, conf=None, device=None):
        """
        Args:
            sp_weights: Path to a SuperPoint checkpoint (.tar or .pth), or None
                        to use the default pretrained weights.
            conf:       Optional dict overriding model defaults.  Recognised keys:
                        nms_radius, max_num_keypoints, detection_threshold.
            device:     "cuda" or "cpu".  Auto-detected when None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.sp_weights = str(sp_weights) if sp_weights else None
        self.conf = conf or {}
        self._model = self._build_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_conf(self):
        return {
            "nms_radius": self.conf.get("nms_radius", 4),
            "max_num_keypoints": self.conf.get("max_num_keypoints", 512),
            "detection_threshold": self.conf.get("detection_threshold", 0.0),
            "trainable": False,
            **({"weights": self.sp_weights} if self.sp_weights else {}),
        }

    def _build_model(self):
        return get_model("superpoint_open")(self._model_conf()).to(self.device).eval()

    def _get_thread_model(self):
        """Return (and lazily create) a thread-local copy of the SP model."""
        if not hasattr(_thread_local, "model"):
            _thread_local.model = get_model("superpoint_open")(
                self._model_conf()
            ).to(self.device).eval()
        return _thread_local.model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image_path):
        """Run SuperPoint on a single image.

        Returns:
            dict with keypoints [N,2], keypoint_scores [N], descriptors [N,256].
        """
        img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Could not load: {image_path}")
        img_t = (
            torch.from_numpy(img_gray.astype(np.float32))
            .unsqueeze(0).unsqueeze(0)
            .to(self.device) / 255.0
        )
        with torch.no_grad():
            pred = self._model({"image": img_t})
        return {
            "keypoints": pred["keypoints"][0].cpu().numpy(),
            "keypoint_scores": pred["keypoint_scores"][0].cpu().numpy(),
            "descriptors": pred["descriptors"][0].cpu().numpy(),
        }

    def run_adaptation(
        self,
        image_name,
        dataset_dir,
        num_warps=50,
        mode="3d",
        K=None,
        neighbors=None,
        detection_threshold=0.015,
        nms_radius=4,
        pose_ratio=0.6,
        modality="rgb",
    ):
        """Run homographic adaptation for one image and return consensus keypoints.

        Performs ``num_warps`` random (and optionally pose-guided) warps, accumulates
        keypoint score votes into a heatmap, normalises, then applies NMS.

        Args:
            image_name:          Filename of the image (basename only).
            dataset_dir:         Root of the SLAM dataset (must contain images/<modality>/).
            num_warps:           Total warp iterations.
            mode:                "3d" for pose-guided projective warps (falls back to "2d"
                                 if depth not found); "2d" for random homographies only.
            K:                   Camera intrinsics [3,3] (required for mode="3d").
            neighbors:           List of (R_rel, t_rel) from get_neighbor_relative_poses().
            detection_threshold: Minimum heatmap value to keep a keypoint.
            nms_radius:          NMS half-window size.
            pose_ratio:          Fraction of warps to be pose-guided (0–1).
            modality:            Subdirectory name under images/.

        Returns:
            (keypoints [N,2], scores [N]) sorted by score descending.
        """
        dataset_dir = Path(dataset_dir)
        model = self._get_thread_model()
        device = self.device

        rgb_path = dataset_dir / "images" / modality / image_name
        img_rgb = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
        if img_rgb is None:
            raise ValueError(f"Could not load image: {rgb_path}")
        h, w = img_rgb.shape[:2]

        # Try to load depth for 3D mode
        effective_mode = mode
        depth_m = None
        if mode == "3d":
            depth_path = dataset_dir / "images/depth" / image_name
            if depth_path.exists():
                depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth_img is not None:
                    depth_m = depth_img.astype(np.float32) / 1000.0
                else:
                    effective_mode = "2d"
            else:
                effective_mode = "2d"

        img_t = torch.from_numpy(img_rgb.astype(np.float32)).to(device)
        acc_t = torch.zeros((h, w), dtype=torch.float32, device=device)
        trials_t = torch.zeros((h, w), dtype=torch.float32, device=device)

        precomputed_3d = None
        if effective_mode == "3d" and depth_m is not None and K is not None:
            K_inv_t = torch.from_numpy(np.linalg.inv(K)).float().to(device)
            depth_t = torch.from_numpy(depth_m).to(device)
            precomputed_3d = precompute_3d_points(depth_t, K_inv_t, device)

        # --- Base detection on original image ---
        with torch.no_grad():
            pred = model({"image": img_t.unsqueeze(0).unsqueeze(0) / 255.0})
            kpts_t = pred["keypoints"][0] - 0.5
            scores_t = pred["keypoint_scores"][0]
        ix = torch.round(kpts_t[:, 0]).long()
        iy = torch.round(kpts_t[:, 1]).long()
        ib = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        acc_t.index_put_((iy[ib], ix[ib]), scores_t[ib], accumulate=True)
        trials_t += 1.0

        # --- Build warp schedule ---
        n_pose = int(num_warps * pose_ratio) if (neighbors and effective_mode == "3d") else 0
        n_rand = num_warps - n_pose
        schedule = [("random", None)] * n_rand
        if neighbors:
            schedule += [("pose", neighbors[k % len(neighbors)]) for k in range(n_pose)]

        for warp_type, warp_data in schedule:
            if effective_mode == "3d":
                R_np, t_np = warp_data if warp_type == "pose" else sample_random_3d_transform(0.7)
                R_t = torch.from_numpy(R_np).float().to(device)
                t_t = torch.from_numpy(t_np).float().to(device)
                K_t = torch.from_numpy(K).float().to(device)

                if precomputed_3d is not None:
                    warped_t, valid_mask_t, coord_map_t = warp_3d_projective(
                        img_t, precomputed_3d, R_t, t_t, K_t, device
                    )
                else:
                    warped_t = torch.zeros((h, w), dtype=img_t.dtype, device=device)
                    valid_mask_t = torch.zeros((h, w), dtype=torch.bool, device=device)
                    coord_map_t = torch.zeros((h, w, 2), dtype=torch.float32, device=device)
                H_inv_t = None
            else:
                H_np = sample_random_homography((h, w), difficulty=0.7)
                H_inv_np = np.linalg.inv(H_np)
                H_t = torch.from_numpy(H_np).float().to(device)
                H_inv_t = torch.from_numpy(H_inv_np).float().to(device)
                warped_t = warp_perspective(img_t, H_t, device)
                valid_mask_t = None
                coord_map_t = None

            with torch.no_grad():
                pred_w = model({"image": warped_t.unsqueeze(0).unsqueeze(0) / 255.0})
                kpts_w_t = pred_w["keypoints"][0] - 0.5
                scores_w_t = pred_w["keypoint_scores"][0]

            if len(kpts_w_t) == 0:
                continue

            if effective_mode == "3d":
                ix_p = torch.round(kpts_w_t[:, 0]).long()
                iy_p = torch.round(kpts_w_t[:, 1]).long()
                ib = (ix_p >= 0) & (ix_p < w) & (iy_p >= 0) & (iy_p < h)
                ix_p, iy_p, sv = ix_p[ib], iy_p[ib], scores_w_t[ib]
                if len(sv) > 0 and valid_mask_t is not None:
                    mv = valid_mask_t[iy_p, ix_p]
                    coords = coord_map_t[iy_p[mv], ix_p[mv]]
                    sp = sv[mv]
                    ix_o = torch.round(coords[:, 0]).long()
                    iy_o = torch.round(coords[:, 1]).long()
                    ib2 = (ix_o >= 0) & (ix_o < w) & (iy_o >= 0) & (iy_o < h)
                    acc_t.index_put_((iy_o[ib2], ix_o[ib2]), sp[ib2], accumulate=True)
                if valid_mask_t is not None:
                    trials_t += valid_mask_t.float()
            else:
                ones = torch.ones((len(kpts_w_t), 1), device=device)
                kpts_homo = torch.cat([kpts_w_t, ones], dim=1)
                kpts_back = (H_inv_t @ kpts_homo.T).T
                kpts_back = kpts_back[:, :2] / torch.clamp(kpts_back[:, 2:], min=1e-6)
                ix = torch.round(kpts_back[:, 0]).long()
                iy = torch.round(kpts_back[:, 1]).long()
                ib = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
                acc_t.index_put_((iy[ib], ix[ib]), scores_w_t[ib], accumulate=True)
                ones_mask = torch.ones((h, w), device=device)
                trials_t += warp_perspective(ones_mask, H_inv_t, device)

        # --- NMS on normalised heatmap ---
        acc = acc_t.cpu().numpy()
        trials = trials_t.cpu().numpy()
        heatmap = acc / np.maximum(trials, 1.0)
        kernel = np.ones((nms_radius * 2 + 1,) * 2, dtype=np.uint8)
        dilated = cv2.dilate(heatmap, kernel)
        mask = (heatmap == dilated) & (heatmap > detection_threshold)
        ys, xs = np.where(mask)
        kpts = np.stack([xs, ys], axis=1).astype(np.float32)
        scores = heatmap[ys, xs].astype(np.float32)
        idx = np.argsort(-scores)
        return kpts[idx], scores[idx]

    def process_dataset(
        self,
        data_dir,
        output_h5,
        num_workers=8,
        num_warps=50,
        mode="3d",
        modality="rgb",
        poses_file="poses_odom_RGBD_slam.txt",
        pose_ratio=0.6,
        detection_threshold=0.015,
        nms_radius=4,
        max_dist=2.0,
        min_dist=0.1,
        max_angle=30.0,
        min_overlap=0.1,
        max_neighbors=10,
    ):
        """Run adaptation over an entire SLAM dataset and write pseudo-labels to H5.

        Output H5 structure: ``<image_name>/keypoints`` and ``<image_name>/keypoint_scores``.

        Also writes ``image_list_train.txt`` and ``image_list_val.txt`` (83/17 split)
        next to the output H5 file.
        """
        data_dir = Path(data_dir)
        rgb_dir = data_dir / "images" / modality
        image_names = sorted(
            [p.name for p in rgb_dir.glob("*.png")] + [p.name for p in rgb_dir.glob("*.jpg")]
        )
        if not image_names:
            raise ValueError(f"No images found in {rgb_dir}")

        K = None
        if mode == "3d":
            calib_dir = data_dir / "images/calib"
            first_ts = Path(image_names[0]).stem
            calib_path = calib_dir / f"{first_ts}.yaml"
            if not calib_path.exists():
                calib_files = list(calib_dir.glob("*.yaml"))
                calib_path = calib_files[0] if calib_files else None
            if calib_path:
                K = load_camera_intrinsics(calib_path)
                logger.info(f"Loaded K from {calib_path}")

        poses = None
        if pose_ratio > 0.0 and mode == "3d":
            poses_path = data_dir / poses_file
            if poses_path.exists():
                poses = parse_poses(poses_path)
                logger.info(f"Loaded {len(poses)} poses from {poses_path}")
            else:
                logger.warning(f"Poses file not found: {poses_path}; using pure random warps")

        Path(output_h5).parent.mkdir(parents=True, exist_ok=True)

        def worker(name):
            neighbors = []
            if poses is not None and K is not None:
                neighbors = get_neighbor_relative_poses(
                    name, data_dir, poses, image_names, K,
                    max_dist=max_dist, min_dist=min_dist,
                    max_angle=max_angle, min_overlap=min_overlap,
                    max_neighbors=max_neighbors,
                )
            kpts, scores = self.run_adaptation(
                name, data_dir,
                num_warps=num_warps, mode=mode, K=K,
                neighbors=neighbors,
                detection_threshold=detection_threshold,
                nms_radius=nms_radius,
                pose_ratio=pose_ratio,
                modality=modality,
            )
            return name, kpts, scores

        logger.info(f"Processing {len(image_names)} images → {output_h5}")
        with h5py.File(output_h5, "w") as f:
            with tqdm(total=len(image_names), desc="Homographic Adaptation") as pbar:
                if num_workers > 1:
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        for name, kpts, scores in executor.map(worker, image_names):
                            write_h5_features(f, name, kpts, scores)
                            pbar.update(1)
                else:
                    for n in image_names:
                        name, kpts, scores = worker(n)
                        write_h5_features(f, name, kpts, scores)
                        pbar.update(1)

        logger.info("Pseudo-label generation complete.")
