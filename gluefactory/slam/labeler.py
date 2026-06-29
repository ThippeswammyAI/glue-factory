"""SLAMLabelGenerator: extract SuperPoint descriptors at consensus keypoint locations."""

import logging
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from gluefactory.models import get_model
from gluefactory.models.extractors.superpoint_open import sample_descriptors

logger = logging.getLogger(__name__)


class SLAMLabelGenerator:
    """Extract dense SuperPoint descriptors sampled at consensus keypoint positions.

    Takes pseudo-labels produced by :class:`SLAMFeatureExtractor` (which contain
    only keypoints + scores) and re-runs the SuperPoint backbone to attach
    256-D descriptors.  The output H5 is ready for SuperGlue training.

    Example::

        labeler = SLAMLabelGenerator(
            "outputs/training/superpoint_slam_run/checkpoint_best.tar", {}, "cuda"
        )
        labeler.generate(
            pseudo_labels_h5="data/output/slam/exports/pseudo_labels_slam.h5",
            images_dir="data/output/slam/images/rgb",
            output_h5="data/output/slam/exports/sp_features_slam.h5",
        )
    """

    def __init__(self, sp_weights, conf=None, device=None):
        """
        Args:
            sp_weights: Path to a SuperPoint checkpoint (.tar or .pth).
            conf:       Optional dict (nms_radius, max_num_keypoints).
            device:     "cuda" or "cpu".  Auto-detected when None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        conf = conf or {}
        model_conf = {
            "nms_radius": conf.get("nms_radius", 4),
            "max_num_keypoints": conf.get("max_num_keypoints", 2048),
            "detection_threshold": 0.0,
            "trainable": False,
            **({"weights": str(sp_weights)} if sp_weights else {}),
        }
        self._model = get_model("superpoint_open")(model_conf).to(device).eval()

    def generate(self, pseudo_labels_h5, images_dir, output_h5, modality="rgb"):
        """Sample descriptors at pseudo-label keypoints and write a features H5.

        Output H5 structure (under a ``<modality>/`` root group):
            ``<modality>/<image_name>/keypoints``        float32 [N, 2]
            ``<modality>/<image_name>/keypoint_scores``  float32 [N]
            ``<modality>/<image_name>/descriptors``      float32 [N, 256]

        Args:
            pseudo_labels_h5: Path to pseudo-labels H5 (from SLAMFeatureExtractor).
            images_dir:       Directory containing the images (flat, or relative root).
            output_h5:        Output path for the features H5.
            modality:         Subgroup name written inside the output H5.
        """
        device = self.device
        model = self._model
        images_dir = Path(images_dir)
        Path(output_h5).parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(pseudo_labels_h5, "r") as f_in, h5py.File(output_h5, "w") as f_out:
            root_grp = f_out.create_group(modality)
            image_names = list(f_in.keys())
            logger.info(f"Extracting descriptors for {len(image_names)} images → {output_h5}")

            for img_name in tqdm(image_names, desc="Extracting descriptors"):
                kpts = f_in[img_name]["keypoints"][...]
                scores = f_in[img_name]["keypoint_scores"][...]

                img_path = images_dir / img_name
                if not img_path.exists():
                    logger.warning(f"Image not found, skipping: {img_path}")
                    continue
                img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img_gray is None:
                    logger.warning(f"Could not read image, skipping: {img_path}")
                    continue

                img_t = (
                    torch.from_numpy(img_gray).float()
                    .unsqueeze(0).unsqueeze(0)
                    .to(device) / 255.0
                )
                with torch.no_grad():
                    features = model.backbone(img_t)
                    desc_dense = torch.nn.functional.normalize(
                        model.descriptor(features), p=2, dim=1
                    )
                    if len(kpts) > 0:
                        kpts_t = torch.from_numpy(kpts).float().unsqueeze(0).to(device)
                        desc = sample_descriptors(kpts_t, desc_dense, model.stride)
                        desc = desc.squeeze(0).transpose(0, 1).cpu().numpy()  # (N, 256)
                    else:
                        desc = np.zeros((0, 256), dtype=np.float32)

                grp = root_grp.create_group(img_name)
                grp.create_dataset("keypoints", data=kpts + 0.5)
                grp.create_dataset("keypoint_scores", data=scores)
                grp.create_dataset("descriptors", data=desc)

        logger.info(f"Features saved to {output_h5}")
