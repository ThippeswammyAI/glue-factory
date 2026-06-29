"""Camera geometry utilities shared across the SLAM pipeline.

Covers: pose parsing, camera intrinsics loading, covisibility estimation,
GPU-accelerated 3D projective warping, and 2D homography warping.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Pose and intrinsics
# ---------------------------------------------------------------------------

def parse_poses(poses_path):
    """Parse a SLAM trajectory file.

    Expected format (one entry per line, lines starting with '#' are skipped):
        ``#timestamp x y z qx qy qz qw``

    Returns:
        dict mapping timestamp string → (R_mat [3,3], t_vec [3]).
    """
    poses = {}
    with open(poses_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            ts = parts[0]
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            poses[ts] = (rot, t)
    return poses


def load_camera_intrinsics(path):
    """Load a 3×3 camera matrix K from a YAML file.

    Expects a ``camera_matrix.data`` flat-list field (OpenCV YAML format).

    Returns:
        K as float32 np.ndarray [3, 3].
    """
    with open(path, "r") as f:
        lines = f.readlines()
        if lines and lines[0].startswith("%YAML"):
            lines = lines[1:]
        data = yaml.unsafe_load("".join(lines))
    k_flat = data["camera_matrix"]["data"]
    return np.array(k_flat, dtype=np.float32).reshape(3, 3)


# ---------------------------------------------------------------------------
# Covisibility
# ---------------------------------------------------------------------------

def compute_covisibility(depth_path, R_i, t_i, R_j, t_j, K):
    """Estimate the fraction of visible depth pixels of frame i that project
    inside the image of frame j.

    Args:
        depth_path: Path to depth image of frame i (16-bit PNG, mm units).
        R_i, t_i:   Rotation [3,3] and translation [3] of frame i (world → cam).
        R_j, t_j:   Same for frame j.
        K:           Camera intrinsics [3,3].

    Returns:
        float in [0, 1]; 0.0 if depth is unavailable.
    """
    if not Path(depth_path).exists():
        return 0.0
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return 0.0
    depth_m = depth_img.astype(np.float32) / 1000.0
    H, W = depth_m.shape

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = depth_m > 0
    if not np.any(valid):
        return 0.0

    u_v, v_v, d_v = u[valid], v[valid], depth_m[valid]
    K_inv = np.linalg.inv(K)
    pixels_homo = np.stack([u_v, v_v, np.ones_like(u_v)], axis=0)
    P_c_i = d_v * (K_inv @ pixels_homo)
    P_w = R_i @ P_c_i + t_i[:, None]
    P_c_j = R_j.T @ P_w + (-R_j.T @ t_j)[:, None]
    z_j = P_c_j[2, :]
    valid_z = z_j > 1e-3
    if not np.any(valid_z):
        return 0.0
    projected = K @ P_c_j[:, valid_z]
    u_j = projected[0, :] / z_j[valid_z]
    v_j = projected[1, :] / z_j[valid_z]
    in_bounds = (u_j >= 0) & (u_j < W) & (v_j >= 0) & (v_j < H)
    return float(np.sum(in_bounds) / len(d_v))


def get_neighbor_relative_poses(
    image_name, dataset_dir, poses, image_names, K,
    max_dist=2.0, min_dist=0.1, max_angle=30.0,
    min_overlap=0.1, max_neighbors=10,
):
    """Find neighboring frames of ``image_name`` by pose distance + covisibility.

    Returns:
        list of (R_rel [3,3], t_rel [3]) in the coordinate frame of image_name,
        sorted by translation distance (closest first).  Empty list if depth or
        pose data is unavailable.
    """
    ts_i = Path(image_name).stem
    if ts_i not in poses:
        return []
    R_i, t_i = poses[ts_i]
    depth_path_i = Path(dataset_dir) / "images/depth" / image_name
    if not depth_path_i.exists():
        return []
    depth_img = cv2.imread(str(depth_path_i), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return []
    depth_m = depth_img.astype(np.float32) / 1000.0

    candidates = []
    for name_j in image_names:
        if name_j == image_name:
            continue
        ts_j = Path(name_j).stem
        if ts_j not in poses:
            continue
        R_j, t_j = poses[ts_j]
        dist = np.linalg.norm(t_i - t_j)
        if dist < min_dist or dist > max_dist:
            continue
        R_rel = R_i.T @ R_j
        trace = np.trace(R_rel)
        angle = np.degrees(np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0)))
        if angle > max_angle:
            continue
        candidates.append((name_j, R_j, t_j, dist))

    candidates.sort(key=lambda x: x[-1])
    candidates = candidates[: max_neighbors * 2]

    valid_neighbors = []
    for name_j, R_j, t_j, _ in candidates:
        overlap = compute_covisibility(depth_path_i, R_i, t_i, R_j, t_j, K)
        if overlap >= min_overlap:
            R_rel = R_j.T @ R_i
            t_rel = R_j.T @ (t_i - t_j)
            valid_neighbors.append((R_rel, t_rel))
            if len(valid_neighbors) >= max_neighbors:
                break
    return valid_neighbors


# ---------------------------------------------------------------------------
# Warp primitives
# ---------------------------------------------------------------------------

def precompute_3d_points(depth_map_t, K_inv_t, device):
    """Unproject all valid depth pixels to 3D (GPU tensor version).

    Returns:
        dict with keys P_3D, u_valid, v_valid, H, W; or None if depth is empty.
    """
    H, W = depth_map_t.shape
    u, v = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="xy",
    )
    valid = depth_map_t > 0
    if not torch.any(valid):
        return None
    u_v, v_v, d_v = u[valid], v[valid], depth_map_t[valid]
    pixels_homo = torch.stack(
        [u_v.float(), v_v.float(), torch.ones_like(u_v, dtype=torch.float32)], dim=0
    )
    P_3D = d_v.float() * (K_inv_t @ pixels_homo)
    return {"P_3D": P_3D, "u_valid": u_v, "v_valid": v_v, "H": H, "W": W}


def warp_3d_projective(img_t, precomputed, R_t, t_t, K_t, device):
    """Apply a 3D projective (pose) warp to a grayscale image tensor.

    Uses the precomputed 3D point cloud from ``precompute_3d_points`` and a
    Z-buffer sort to handle occlusions.

    Returns:
        (warped_img, valid_mask, coord_map) — all GPU tensors.
        coord_map[v, u] = (u_orig, v_orig) for the source pixel that landed at (u, v).
    """
    H, W = precomputed["H"], precomputed["W"]
    P_3D, u_valid, v_valid = precomputed["P_3D"], precomputed["u_valid"], precomputed["v_valid"]

    P_prime = R_t @ P_3D + t_t[:, None]
    z_prime = P_prime[2, :]
    valid_z = z_prime > 1e-3

    def _zeros(dtype):
        return torch.zeros((H, W), dtype=dtype, device=device)

    if not torch.any(valid_z):
        return _zeros(img_t.dtype), _zeros(torch.bool), torch.zeros((H, W, 2), dtype=torch.float32, device=device)

    projected = K_t @ P_prime[:, valid_z]
    u_p = projected[0, :] / torch.clamp(z_prime[valid_z], min=1e-6)
    v_p = projected[1, :] / torch.clamp(z_prime[valid_z], min=1e-6)
    z_p = z_prime[valid_z]
    u_orig, v_orig = u_valid[valid_z], v_valid[valid_z]

    u_idx, v_idx = torch.round(u_p).long(), torch.round(v_p).long()
    in_bounds = (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)
    if not torch.any(in_bounds):
        return _zeros(img_t.dtype), _zeros(torch.bool), torch.zeros((H, W, 2), dtype=torch.float32, device=device)

    u_idx, v_idx, z_p = u_idx[in_bounds], v_idx[in_bounds], z_p[in_bounds]
    u_orig, v_orig = u_orig[in_bounds], v_orig[in_bounds]
    sort_idx = torch.argsort(-z_p)
    u_idx, v_idx = u_idx[sort_idx], v_idx[sort_idx]
    u_orig, v_orig = u_orig[sort_idx], v_orig[sort_idx]

    warped = _zeros(img_t.dtype)
    warped[v_idx, u_idx] = img_t[v_orig, u_orig]
    valid_mask = _zeros(torch.bool)
    valid_mask[v_idx, u_idx] = True
    coord_map = torch.zeros((H, W, 2), dtype=torch.float32, device=device)
    coord_map[v_idx, u_idx, 0] = u_orig.float()
    coord_map[v_idx, u_idx, 1] = v_orig.float()
    return warped, valid_mask, coord_map


def warp_perspective(img_t, H_t, device):
    """Apply a 2D homography warp to a grayscale image tensor (GPU, bilinear)."""
    H_px, W_px = img_t.shape[-2:]
    grid_u, grid_v = torch.meshgrid(
        torch.arange(W_px, device=device),
        torch.arange(H_px, device=device),
        indexing="xy",
    )
    coords = torch.stack(
        [grid_u.float(), grid_v.float(), torch.ones_like(grid_u, dtype=torch.float32)],
        dim=-1,
    )
    coords_proj = coords @ H_t.T
    coords_proj = coords_proj[:, :, :2] / torch.clamp(coords_proj[:, :, 2:], min=1e-6)
    grid_sample = torch.stack(
        [
            2.0 * coords_proj[:, :, 0] / (W_px - 1) - 1.0,
            2.0 * coords_proj[:, :, 1] / (H_px - 1) - 1.0,
        ],
        dim=-1,
    ).unsqueeze(0)
    warped = torch.nn.functional.grid_sample(
        img_t.float().unsqueeze(0).unsqueeze(0),
        grid_sample,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped.squeeze(0).squeeze(0).to(img_t.dtype)


# ---------------------------------------------------------------------------
# Random transform samplers
# ---------------------------------------------------------------------------

def sample_random_homography(shape, difficulty=0.7):
    """Sample a random planar homography by perturbing the four image corners."""
    h, w = shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    max_offset = min(h, w) * 0.15 * difficulty
    offsets = np.random.uniform(-max_offset, max_offset, size=(4, 2)).astype(np.float32)
    return cv2.getPerspectiveTransform(corners, corners + offsets)


def sample_random_3d_transform(difficulty=0.7):
    """Sample a small random 3D rotation (R) and translation (t) for adaptation.

    Returns:
        (R [3,3] float32, t [3] float32).
    """
    tx = np.random.uniform(-0.3 * difficulty, 0.3 * difficulty)
    ty = np.random.uniform(-0.05 * difficulty, 0.05 * difficulty)
    tz = np.random.uniform(-0.5 * difficulty, 0.5 * difficulty)
    t = np.array([tx, ty, tz], dtype=np.float32)

    def rot_x(a):
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], dtype=np.float32)

    def rot_y(a):
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], dtype=np.float32)

    def rot_z(a):
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=np.float32)

    pitch = np.random.uniform(-np.deg2rad(7.0 * difficulty), np.deg2rad(7.0 * difficulty))
    yaw = np.random.uniform(-np.deg2rad(10.0 * difficulty), np.deg2rad(10.0 * difficulty))
    roll = np.random.uniform(-np.deg2rad(6.0 * difficulty), np.deg2rad(6.0 * difficulty))
    return rot_z(roll) @ rot_y(yaw) @ rot_x(pitch), t
