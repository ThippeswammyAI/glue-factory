"""Shared I/O utilities for the SLAM pipeline: image loading, pair discovery, H5 read/write."""

import logging
from pathlib import Path

import cv2
import h5py
import numpy as np

logger = logging.getLogger(__name__)


def load_image(path, resize_max=0):
    """Load an image as BGR + grayscale; pad dims to multiples of 8.

    Args:
        path: Path to image file.
        resize_max: If > 0, scale the longest side to this value. If 0, only
                    pad to the nearest multiple of 8.

    Returns:
        (img_bgr, img_gray) as np.ndarray uint8.
    """
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape[:2]

    if resize_max > 0:
        scale = resize_max / max(h, w)
        w_new = int(round(w * scale / 8) * 8)
        h_new = int(round(h * scale / 8) * 8)
        img_gray = cv2.resize(img_gray, (w_new, h_new), interpolation=cv2.INTER_AREA)
        img_bgr = cv2.resize(img_bgr, (w_new, h_new), interpolation=cv2.INTER_AREA)
    else:
        w_new = (w // 8) * 8
        h_new = (h // 8) * 8
        if w_new != w or h_new != h:
            img_gray = cv2.resize(img_gray, (w_new, h_new), interpolation=cv2.INTER_AREA)
            img_bgr = cv2.resize(img_bgr, (w_new, h_new), interpolation=cv2.INTER_AREA)

    return img_bgr, img_gray


_IMG_EXTS = {"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.PNG", "*.JPG", "*.JPEG"}


def _target_dirs(path):
    """Directory itself, or its immediate subdirectories if any (skips "outputs")."""
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    dirs = subdirs if subdirs and path.name != "outputs" else [path]
    return [d for d in dirs if d.name != "outputs"]


def list_images(input_path):
    """Return sorted image file paths for a single file, an explicit
    "img0.png,img1.png" pair, or a directory (recursing one level into
    subdirectories, skipping "outputs").

    Returns:
        list of Path.
    """
    if "," in str(input_path):
        return [Path(p.strip()) for p in str(input_path).split(",")]

    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        logger.error(f"Input is not a file or directory: {input_path}")
        return []

    files = []
    for t_dir in _target_dirs(path):
        files.extend(sorted({f for ext in _IMG_EXTS for f in t_dir.glob(ext)}))
    return files


def get_image_pairs(input_path):
    """Return consecutive (path0, path1) pairs from a directory or explicit pair.

    Accepts:
      - "img0.png,img1.png"  → one pair
      - a directory          → all consecutive sorted image pairs inside it;
                               recurses one level into subdirectories (skips "outputs")

    Returns:
        list of (Path, Path) tuples.
    """
    if "," in str(input_path):
        parts = list_images(input_path)
        return [(parts[0], parts[1])] if len(parts) == 2 else []

    path = Path(input_path)
    if not path.is_dir():
        logger.error(f"Input is not a directory: {input_path}")
        return []

    pairs = []
    for t_dir in _target_dirs(path):
        files = sorted({f for ext in _IMG_EXTS for f in t_dir.glob(ext)})
        if len(files) >= 2:
            if len(files) == 2:
                pairs.append((files[0], files[1]))
            else:
                for i in range(len(files) - 1):
                    pairs.append((files[i], files[i + 1]))

    return pairs


def write_h5_features(h5_file, name, keypoints, scores, descriptors=None):
    """Write per-image features into an open h5py.File under the key ``name``.

    Creates datasets: keypoints [N,2], keypoint_scores [N], descriptors [N,D].
    """
    grp = h5_file.create_group(name)
    grp.create_dataset("keypoints", data=keypoints)
    grp.create_dataset("keypoint_scores", data=scores)
    if descriptors is not None:
        grp.create_dataset("descriptors", data=descriptors)


def read_h5_features(h5_path, name):
    """Read keypoints/scores and optionally descriptors for one image from H5.

    Returns:
        dict with "keypoints", "keypoint_scores", and optionally "descriptors".
    """
    with h5py.File(h5_path, "r") as f:
        grp = f[name]
        result = {
            "keypoints": grp["keypoints"][...],
            "keypoint_scores": grp["keypoint_scores"][...],
        }
        if "descriptors" in grp:
            result["descriptors"] = grp["descriptors"][...]
    return result
