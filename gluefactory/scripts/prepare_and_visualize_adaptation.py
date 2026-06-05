#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from gluefactory.models import get_model
from gluefactory.settings import DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODALITIES = ["nearir", "range", "reflectivity", "signal"]
import threading
from concurrent.futures import ThreadPoolExecutor

thread_local = threading.local()

def get_thread_model(args, device):
    if not hasattr(thread_local, "model"):
        logger.info(f"Instantiating thread-local SuperPoint model on thread {threading.current_thread().name}...")
        model_conf = {
            "nms_radius": args.nms,
            "max_num_keypoints": 2048,
            "detection_threshold": 0.0,
            "trainable": False
        }
        if hasattr(args, "weights") and args.weights:
            model_conf["weights"] = args.weights
        thread_local.model = get_model("superpoint_open")(model_conf).to(device).eval()
    return thread_local.model


def sample_random_homography(shape, difficulty=0.7):
    """
    Generate a random homography by perturbing the four corners of the image.
    """
    h, w = shape[:2]
    corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)
    
    max_offset = min(h, w) * 0.15 * difficulty
    offsets = np.random.uniform(-max_offset, max_offset, size=(4, 2)).astype(np.float32)
    perturbed_corners = corners + offsets
    
    H = cv2.getPerspectiveTransform(corners, perturbed_corners)
    return H

def load_camera_intrinsics(info_path):
    """
    Parse camera intrinsics from camera.info file.
    """
    import yaml
    with open(info_path, "r") as f:
        data = yaml.unsafe_load(f)
    k_flat = data["k"]
    K = np.array(k_flat, dtype=np.float32).reshape(3, 3)
    return K

def sample_random_3d_transform(difficulty=0.7):
    """
    Generate a random 3D rigid transform [R|t] scaled by difficulty.
    """
    max_tx = 0.3 * difficulty
    max_ty = 0.05 * difficulty
    max_tz = 0.5 * difficulty
    
    max_pitch = np.deg2rad(7.0 * difficulty)
    max_yaw = np.deg2rad(10.0 * difficulty)
    max_roll = np.deg2rad(6.0 * difficulty)
    
    tx = np.random.uniform(-max_tx, max_tx)
    ty = np.random.uniform(-max_ty, max_ty)
    tz = np.random.uniform(-max_tz, max_tz)
    t = np.array([tx, ty, tz], dtype=np.float32)
    
    pitch = np.random.uniform(-max_pitch, max_pitch)
    yaw = np.random.uniform(-max_yaw, max_yaw)
    roll = np.random.uniform(-max_roll, max_roll)
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ], dtype=np.float32)
    
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ], dtype=np.float32)
    
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    R = Rz @ Ry @ Rx
    return R, t

def warp_perspective_torch(img_t, H_t, device):
    """
    GPU-accelerated PyTorch 2D homography warping using grid_sample.
    """
    H, W = img_t.shape[-2:]
    grid = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    coords = torch.stack([grid[0].float(), grid[1].float(), torch.ones_like(grid[0], dtype=torch.float32)], dim=-1)
    coords_proj = coords @ H_t.T
    coords_proj = coords_proj[:, :, :2] / torch.clamp(coords_proj[:, :, 2:], min=1e-6)
    
    grid_sample_coords = torch.stack([
        2.0 * coords_proj[:, :, 0] / (W - 1) - 1.0,
        2.0 * coords_proj[:, :, 1] / (H - 1) - 1.0
    ], dim=-1).unsqueeze(0)
    
    img_feed = img_t.float().unsqueeze(0).unsqueeze(0)
    warped_t = torch.nn.functional.grid_sample(img_feed, grid_sample_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped_t.squeeze(0).squeeze(0).to(img_t.dtype)

def precompute_3d_points_torch(depth_map_t, K_inv_t, device):
    """
    Precompute 3D coordinates from the range/depth map for faster warping.
    """
    H, W = depth_map_t.shape
    u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    
    valid_depth_mask = (depth_map_t > 0)
    if not torch.any(valid_depth_mask):
        return None
        
    u_valid = u[valid_depth_mask]
    v_valid = v[valid_depth_mask]
    d_valid = depth_map_t[valid_depth_mask]
    
    pixels_homo = torch.stack([u_valid.float(), v_valid.float(), torch.ones_like(u_valid, dtype=torch.float32)], dim=0)
    P_3D = d_valid.float() * (K_inv_t @ pixels_homo)
    
    return {
        "P_3D": P_3D,
        "u_valid": u_valid,
        "v_valid": v_valid,
        "H": H,
        "W": W
    }

def warp_3d_projective_torch_fast(loaded_imgs, precomputed, R_t, t_t, K_t, device):
    """
    Fast 3D projective warping using precomputed 3D points.
    """
    H = precomputed["H"]
    W = precomputed["W"]
    P_3D = precomputed["P_3D"]
    u_valid = precomputed["u_valid"]
    v_valid = precomputed["v_valid"]
    
    P_prime_3D = R_t @ P_3D + t_t[:, None]
    z_prime = P_prime_3D[2, :]
    
    valid_z_mask = (z_prime > 1e-3)
    if not torch.any(valid_z_mask):
        warped_imgs = {mod: torch.zeros((H, W), dtype=img.dtype, device=device) for mod, img in loaded_imgs.items()}
        return warped_imgs, torch.zeros((H, W), dtype=torch.bool, device=device), torch.zeros((H, W, 2), dtype=torch.float32, device=device)
        
    projected = K_t @ P_prime_3D[:, valid_z_mask]
    u_prime = projected[0, :] / torch.clamp(z_prime[valid_z_mask], min=1e-6)
    v_prime = projected[1, :] / torch.clamp(z_prime[valid_z_mask], min=1e-6)
    z_prime = z_prime[valid_z_mask]
    
    u_orig = u_valid[valid_z_mask]
    v_orig = v_valid[valid_z_mask]
    
    u_idx = torch.round(u_prime).long()
    v_idx = torch.round(v_prime).long()
    in_bounds = (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)
    
    if not torch.any(in_bounds):
        warped_imgs = {mod: torch.zeros((H, W), dtype=img.dtype, device=device) for mod, img in loaded_imgs.items()}
        return warped_imgs, torch.zeros((H, W), dtype=torch.bool, device=device), torch.zeros((H, W, 2), dtype=torch.float32, device=device)
        
    u_idx = u_idx[in_bounds]
    v_idx = v_idx[in_bounds]
    z_prime = z_prime[in_bounds]
    u_orig = u_orig[in_bounds]
    v_orig = v_orig[in_bounds]
    
    sort_idx = torch.argsort(-z_prime)
    u_idx_sorted = u_idx[sort_idx]
    v_idx_sorted = v_idx[sort_idx]
    u_orig_sorted = u_orig[sort_idx]
    v_orig_sorted = v_orig[sort_idx]
    
    warped_imgs = {}
    for mod, img in loaded_imgs.items():
        warped_img = torch.zeros((H, W), dtype=img.dtype, device=device)
        colors = img[v_orig_sorted, u_orig_sorted]
        warped_img[v_idx_sorted, u_idx_sorted] = colors
        warped_imgs[mod] = warped_img
        
    valid_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    valid_mask[v_idx_sorted, u_idx_sorted] = True
    
    warped_coord_map = torch.zeros((H, W, 2), dtype=torch.float32, device=device)
    warped_coord_map[v_idx_sorted, u_idx_sorted, 0] = u_orig_sorted.float()
    warped_coord_map[v_idx_sorted, u_idx_sorted, 1] = v_orig_sorted.float()
    
    return warped_imgs, valid_mask, warped_coord_map

def warp_3d_projective_torch(loaded_imgs, depth_map_t, R_t, t_t, K_t, device):

    """
    Fully GPU-accelerated PyTorch implementation of 3D projective warping.
    """
    H, W = depth_map_t.shape
    u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    
    valid_depth_mask = (depth_map_t > 0)
    if not torch.any(valid_depth_mask):
        warped_imgs = {mod: torch.zeros((H, W), dtype=img.dtype, device=device) for mod, img in loaded_imgs.items()}
        return warped_imgs, torch.zeros((H, W), dtype=torch.bool, device=device), torch.zeros((H, W, 2), dtype=torch.float32, device=device)
        
    u_valid = u[valid_depth_mask]
    v_valid = v[valid_depth_mask]
    d_valid = depth_map_t[valid_depth_mask]
    
    K_inv = torch.linalg.inv(K_t)
    pixels_homo = torch.stack([u_valid.float(), v_valid.float(), torch.ones_like(u_valid, dtype=torch.float32)], dim=0)
    P_3D = d_valid.float() * (K_inv @ pixels_homo)
    
    P_prime_3D = R_t @ P_3D + t_t[:, None]
    z_prime = P_prime_3D[2, :]
    
    valid_z_mask = (z_prime > 1e-3)
    if not torch.any(valid_z_mask):
        warped_imgs = {mod: torch.zeros((H, W), dtype=img.dtype, device=device) for mod, img in loaded_imgs.items()}
        return warped_imgs, torch.zeros((H, W), dtype=torch.bool, device=device), torch.zeros((H, W, 2), dtype=torch.float32, device=device)
        
    projected = K_t @ P_prime_3D[:, valid_z_mask]
    u_prime = projected[0, :] / torch.clamp(z_prime[valid_z_mask], min=1e-6)
    v_prime = projected[1, :] / torch.clamp(z_prime[valid_z_mask], min=1e-6)
    z_prime = z_prime[valid_z_mask]
    
    u_orig = u_valid[valid_z_mask]
    v_orig = v_valid[valid_z_mask]
    
    u_idx = torch.round(u_prime).long()
    v_idx = torch.round(v_prime).long()
    in_bounds = (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)
    
    if not torch.any(in_bounds):
        warped_imgs = {mod: torch.zeros((H, W), dtype=img.dtype, device=device) for mod, img in loaded_imgs.items()}
        return warped_imgs, torch.zeros((H, W), dtype=torch.bool, device=device), torch.zeros((H, W, 2), dtype=torch.float32, device=device)
        
    u_idx = u_idx[in_bounds]
    v_idx = v_idx[in_bounds]
    z_prime = z_prime[in_bounds]
    u_orig = u_orig[in_bounds]
    v_orig = v_orig[in_bounds]
    
    sort_idx = torch.argsort(-z_prime)
    u_idx_sorted = u_idx[sort_idx]
    v_idx_sorted = v_idx[sort_idx]
    u_orig_sorted = u_orig[sort_idx]
    v_orig_sorted = v_orig[sort_idx]
    
    warped_imgs = {}
    for mod, img in loaded_imgs.items():
        warped_img = torch.zeros((H, W), dtype=img.dtype, device=device)
        colors = img[v_orig_sorted, u_orig_sorted]
        warped_img[v_idx_sorted, u_idx_sorted] = colors
        warped_imgs[mod] = warped_img
        
    valid_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    valid_mask[v_idx_sorted, u_idx_sorted] = True
    
    warped_coord_map = torch.zeros((H, W, 2), dtype=torch.float32, device=device)
    warped_coord_map[v_idx_sorted, u_idx_sorted, 0] = u_orig_sorted.float()
    warped_coord_map[v_idx_sorted, u_idx_sorted, 1] = v_orig_sorted.float()
    
    return warped_imgs, valid_mask, warped_coord_map

def warp_3d_projective(loaded_imgs, depth_map, R, t, K):
    """
    Perform 3D-aware camera projection warping for all modalities using depth Z-buffering.
    loaded_imgs: dict of {mod_name: img} where img is (H, W)
    depth_map: (H, W) depth map in meters (d > 0 is valid)
    R: (3, 3) rotation matrix
    t: (3,) translation vector
    K: (3, 3) camera intrinsics
    
    Returns:
        warped_imgs: dict of {mod_name: warped_img}
        valid_mask: (H, W) boolean mask indicating valid warped pixels
        warped_coord_map: (H, W, 2) warped coordinate lookup map (containing original [u, v] for each target pixel)
    """
    H, W = depth_map.shape
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    valid_depth_mask = (depth_map > 0)
    if not np.any(valid_depth_mask):
        warped_imgs = {mod: np.zeros((H, W), dtype=img.dtype) for mod, img in loaded_imgs.items()}
        return warped_imgs, np.zeros((H, W), dtype=bool), np.zeros((H, W, 2), dtype=np.float32)
        
    u_valid = u[valid_depth_mask]
    v_valid = v[valid_depth_mask]
    d_valid = depth_map[valid_depth_mask]
    
    K_inv = np.linalg.inv(K)
    pixels_homo = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0) # (3, N)
    P_3D = d_valid * (K_inv @ pixels_homo) # (3, N)
    
    P_prime_3D = R @ P_3D + t[:, None] # (3, N)
    z_prime = P_prime_3D[2, :] # (N,)
    
    valid_z_mask = (z_prime > 1e-3)
    if not np.any(valid_z_mask):
        warped_imgs = {mod: np.zeros((H, W), dtype=img.dtype) for mod, img in loaded_imgs.items()}
        return warped_imgs, np.zeros((H, W), dtype=bool), np.zeros((H, W, 2), dtype=np.float32)
        
    projected = K @ P_prime_3D[:, valid_z_mask]
    u_prime = projected[0, :] / z_prime[valid_z_mask]
    v_prime = projected[1, :] / z_prime[valid_z_mask]
    z_prime = z_prime[valid_z_mask]
    
    u_orig = u_valid[valid_z_mask]
    v_orig = v_valid[valid_z_mask]
    
    u_idx = np.round(u_prime).astype(np.int32)
    v_idx = np.round(v_prime).astype(np.int32)
    in_bounds = (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)
    
    if not np.any(in_bounds):
        warped_imgs = {mod: np.zeros((H, W), dtype=img.dtype) for mod, img in loaded_imgs.items()}
        return warped_imgs, np.zeros((H, W), dtype=bool), np.zeros((H, W, 2), dtype=np.float32)
        
    u_idx = u_idx[in_bounds]
    v_idx = v_idx[in_bounds]
    z_prime = z_prime[in_bounds]
    u_orig = u_orig[in_bounds]
    v_orig = v_orig[in_bounds]
    
    sort_idx = np.argsort(-z_prime)
    u_idx_sorted = u_idx[sort_idx]
    v_idx_sorted = v_idx[sort_idx]
    u_orig_sorted = u_orig[sort_idx]
    v_orig_sorted = v_orig[sort_idx]
    
    warped_imgs = {}
    for mod, img in loaded_imgs.items():
        warped_img = np.zeros((H, W), dtype=img.dtype)
        colors = img[v_orig_sorted, u_orig_sorted]
        warped_img[v_idx_sorted, u_idx_sorted] = colors
        warped_imgs[mod] = warped_img
        
    valid_mask = np.zeros((H, W), dtype=bool)
    valid_mask[v_idx_sorted, u_idx_sorted] = True
    
    warped_coord_map = np.zeros((H, W, 2), dtype=np.float32)
    warped_coord_map[v_idx_sorted, u_idx_sorted, 0] = u_orig_sorted
    warped_coord_map[v_idx_sorted, u_idx_sorted, 1] = v_orig_sorted
    
    return warped_imgs, valid_mask, warped_coord_map

def multimodal_homographic_adaptation(image_name, images_dir, model, num_warps=15, detection_threshold=0.015, nms_radius=4, warp_mode="3d", K=None, return_all_warps=False, use_gpu=True):
    """
    Perform Joint Multimodal Homographic Adaptation.
    Reads synchronized images from 'nearir', 'range', 'reflectivity', and 'signal'.
    Accumulates stable keypoint detections across all modalities and warps.
    Supports either '2d' Homography or '3d' Z-buffered projection warping.
    Also returns a few warped intermediate images and their corresponding keypoints for visualization.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load all available modality images
    loaded_imgs = {}
    loaded_imgs_feed = {}
    h, w = None, None
    for mod in MODALITIES:
        img_path = images_dir / mod / image_name
        if img_path.exists():
            if mod == "range":
                img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img_raw is not None:
                    loaded_imgs["range"] = img_raw
                    # Create 8-bit version for model feed detection
                    if img_raw.max() > 0:
                        img_feed = ((img_raw.astype(np.float32) / img_raw.max()) * 255.0).astype(np.uint8)
                    else:
                        img_feed = np.zeros_like(img_raw, dtype=np.uint8)
                    loaded_imgs_feed["range"] = img_feed
                    if h is None:
                        h, w = img_raw.shape[:2]
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    loaded_imgs[mod] = img
                    loaded_imgs_feed[mod] = img
                    if h is None:
                        h, w = img.shape[:2]
                    
    if not loaded_imgs:
        raise ValueError(f"Could not load any synchronized image modalities for {image_name}")
        
    # Check for warp mode compatibility
    if warp_mode == "3d":
        if "range" not in loaded_imgs:
            logger.warning(f"Range channel not found for {image_name}. Falling back to 2D Homography warping!")
            warp_mode = "2d"
        elif K is None:
            raise ValueError("Intrinsics matrix K is required for 3D warp mode.")
            
    # Joint accumulator heatmap and trials map
    if use_gpu:
        loaded_imgs_t = {mod: torch.from_numpy(img.astype(np.float32)).to(device) for mod, img in loaded_imgs.items()}
        loaded_imgs_feed_t = {mod: torch.from_numpy(img.astype(np.float32)).to(device) for mod, img in loaded_imgs_feed.items()}
        joint_accumulator_t = torch.zeros((h, w), dtype=torch.float32, device=device)
        global_trials_t = torch.zeros((h, w), dtype=torch.float32, device=device)
        
        # Precompute 3D points once per scene to avoid massive redundant calculation
        if warp_mode == "3d":
            K_inv = np.linalg.inv(K)
            K_inv_t = torch.from_numpy(K_inv).float().to(device)
            depth_map_t = loaded_imgs_t["range"] / 1000.0
            precomputed_3d = precompute_3d_points_torch(depth_map_t, K_inv_t, device)
        else:
            precomputed_3d = None
    else:
        joint_accumulator = np.zeros((h, w), dtype=np.float32)
        global_trials = np.zeros((h, w), dtype=np.float32)
    
    # For saving intermediate warped samples
    warped_samples = []
    all_warps_info = [] if return_all_warps else None
    
    # Process each modality
    for mod_name, img in loaded_imgs.items():
        if mod_name == "range":
            continue
        # 1. Base detection on original image
        if use_gpu:
            img_feed_t = loaded_imgs_feed_t[mod_name]
            img_tensor = img_feed_t.unsqueeze(0).unsqueeze(0) / 255.0
            with torch.no_grad():
                pred = model({"image": img_tensor})
                kpts_t = pred["keypoints"][0] - 0.5
                scores_t = pred["keypoint_scores"][0]
            
            ix = torch.round(kpts_t[:, 0]).long()
            iy = torch.round(kpts_t[:, 1]).long()
            in_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            joint_accumulator_t.index_put_((iy[in_bounds], ix[in_bounds]), scores_t[in_bounds], accumulate=True)
            global_trials_t += 1.0
        else:
            img_feed = loaded_imgs_feed[mod_name]
            img_tensor = torch.from_numpy(img_feed).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                pred = model({"image": img_tensor})
                kpts = pred["keypoints"][0].cpu().numpy() - 0.5
                scores = pred["keypoint_scores"][0].cpu().numpy()
            
            for (x, y), score in zip(kpts, scores):
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < w and 0 <= iy < h:
                    joint_accumulator[iy, ix] += score
            global_trials += 1.0
        
        # 2. Homographic/Projective adaptations for this modality
        for warp_idx in range(num_warps):
            if warp_mode == "3d":
                R, t = sample_random_3d_transform(difficulty=0.7)
                
                if use_gpu:
                    R_t = torch.from_numpy(R).float().to(device)
                    t_t = torch.from_numpy(t).float().to(device)
                    K_t = torch.from_numpy(K).float().to(device)
                    
                    if precomputed_3d is not None:
                        warped_dict_t, valid_mask_t, warped_coord_map_t = warp_3d_projective_torch_fast(
                            {mod_name: loaded_imgs_t[mod_name]}, precomputed_3d, R_t, t_t, K_t, device
                        )
                    else:
                        warped_dict_t = {mod_name: torch.zeros((h, w), dtype=loaded_imgs_t[mod_name].dtype, device=device)}
                        valid_mask_t = torch.zeros((h, w), dtype=torch.bool, device=device)
                        warped_coord_map_t = torch.zeros((h, w, 2), dtype=torch.float32, device=device)
                        
                    warped_img_t = warped_dict_t[mod_name]
                    
                    if mod_name == "range":
                        mx = warped_img_t.max()
                        if mx > 0:
                            warped_feed_t = (warped_img_t / mx) * 255.0
                        else:
                            warped_feed_t = torch.zeros_like(warped_img_t)
                    else:
                        warped_feed_t = warped_img_t
                        
                    warped_tensor = warped_feed_t.unsqueeze(0).unsqueeze(0) / 255.0
                    
                    if return_all_warps:
                        warped_img = warped_img_t.cpu().numpy().astype(np.uint8)
                        valid_mask = valid_mask_t.cpu().numpy()
                    else:
                        warped_img = None
                        valid_mask = None
                else:
                    depth_m = loaded_imgs["range"].astype(np.float32) / 1000.0
                    warped_dict, valid_mask, warped_coord_map = warp_3d_projective(
                        {mod_name: img}, depth_m, R, t, K
                    )
                    warped_img = warped_dict[mod_name]
                    if mod_name == "range":
                        if warped_img.max() > 0:
                            warped_feed = ((warped_img.astype(np.float32) / warped_img.max()) * 255.0).astype(np.uint8)
                        else:
                            warped_feed = np.zeros_like(warped_img, dtype=np.uint8)
                    else:
                        warped_feed = warped_img
                    warped_tensor = torch.from_numpy(warped_feed).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            else:
                H = sample_random_homography((h, w), difficulty=0.7)
                H_inv = np.linalg.inv(H)
                
                if use_gpu:
                    H_t = torch.from_numpy(H).float().to(device)
                    H_inv_t = torch.from_numpy(H_inv).float().to(device)
                    
                    warped_img_t = warp_perspective_torch(loaded_imgs_t[mod_name], H_t, device)
                    warped_tensor = warped_img_t.unsqueeze(0).unsqueeze(0) / 255.0
                    
                    if return_all_warps:
                        warped_img = warped_img_t.cpu().numpy().astype(np.uint8)
                    else:
                        warped_img = None
                else:
                    warped_img = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
                    warped_tensor = torch.from_numpy(warped_img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                
            with torch.no_grad():
                pred_warped = model({"image": warped_tensor})
                kpts_warped_t = pred_warped["keypoints"][0] - 0.5
                scores_warped_t = pred_warped["keypoint_scores"][0]

                
            # If this is nearir and we want to keep a few visual samples
            if mod_name == "nearir" and len(warped_samples) < 3 and len(kpts_warped_t) > 0:
                if warped_img is None:
                    warped_img = warped_img_t.cpu().numpy().astype(np.uint8)
                warped_samples.append({
                    "image": warped_img,
                    "keypoints": kpts_warped_t.cpu().numpy(),
                    "scores": scores_warped_t.cpu().numpy()
                })
                
            # If we want to capture all detailed warps for visualization (CPU only)
            if return_all_warps:
                kpts_warped = kpts_warped_t.cpu().numpy()
                scores_warped = scores_warped_t.cpu().numpy()
                if len(kpts_warped) > 0:
                    if warp_mode == "3d":
                        kpts_back_list = []
                        scores_back_list = []
                        warped_coord_map = warped_coord_map_t.cpu().numpy()
                        for (x_prime, y_prime), score in zip(kpts_warped, scores_warped):
                            ix_prime = int(round(x_prime))
                            iy_prime = int(round(y_prime))
                            if 0 <= ix_prime < w and 0 <= iy_prime < h:
                                if valid_mask[iy_prime, ix_prime]:
                                    x_orig = warped_coord_map[iy_prime, ix_prime, 0]
                                    y_orig = warped_coord_map[iy_prime, ix_prime, 1]
                                    kpts_back_list.append([x_orig, y_orig])
                                    scores_back_list.append(score)
                        kpts_back = np.array(kpts_back_list, dtype=np.float32) if kpts_back_list else np.zeros((0, 2), dtype=np.float32)
                        scores_back = np.array(scores_back_list, dtype=np.float32) if scores_back_list else np.zeros((0,), dtype=np.float32)
                    else:
                        ones = np.ones((len(kpts_warped), 1), dtype=np.float32)

                        kpts_homo = np.concatenate([kpts_warped, ones], axis=1)
                        kpts_back = (H_inv @ kpts_homo.T).T
                        kpts_back = kpts_back[:, :2] / kpts_back[:, 2:]
                        scores_back = scores_warped
                else:
                    kpts_back = np.zeros((0, 2), dtype=np.float32)
                    scores_back = np.zeros((0,), dtype=np.float32)
                
                all_warps_info.append({
                    "modality": mod_name,
                    "warp_idx": warp_idx,
                    "image_orig": img,
                    "image_warped": warped_img,
                    "keypoints_warped": kpts_warped,
                    "scores_warped": scores_warped,
                    "keypoints_reprojected": kpts_back,
                    "scores_reprojected": scores_back,
                })

            # Inverse warp detected keypoints and splat
            if len(kpts_warped_t) > 0:
                if use_gpu:
                    if warp_mode == "3d":
                        ix_prime = torch.round(kpts_warped_t[:, 0]).long()
                        iy_prime = torch.round(kpts_warped_t[:, 1]).long()
                        
                        in_bounds = (ix_prime >= 0) & (ix_prime < w) & (iy_prime >= 0) & (iy_prime < h)
                        ix_prime = ix_prime[in_bounds]
                        iy_prime = iy_prime[in_bounds]
                        scores_valid = scores_warped_t[in_bounds]
                        
                        if len(scores_valid) > 0:
                            mask_valid_proj = valid_mask_t[iy_prime, ix_prime]
                            coords = warped_coord_map_t[iy_prime, ix_prime]
                            
                            coords = coords[mask_valid_proj]
                            scores_proj = scores_valid[mask_valid_proj]
                            
                            ix_orig = torch.round(coords[:, 0]).long()
                            iy_orig = torch.round(coords[:, 1]).long()
                            
                            in_bounds_orig = (ix_orig >= 0) & (ix_orig < w) & (iy_orig >= 0) & (iy_orig < h)
                            joint_accumulator_t.index_put_((iy_orig[in_bounds_orig], ix_orig[in_bounds_orig]), scores_proj[in_bounds_orig], accumulate=True)
                        
                        global_trials_t += valid_mask_t.float()
                    else:
                        ones = torch.ones((len(kpts_warped_t), 1), device=device)
                        kpts_homo = torch.cat([kpts_warped_t, ones], dim=1)
                        kpts_back = (H_inv_t @ kpts_homo.T).T
                        kpts_back = kpts_back[:, :2] / torch.clamp(kpts_back[:, 2:], min=1e-6)
                        
                        ix = torch.round(kpts_back[:, 0]).long()
                        iy = torch.round(kpts_back[:, 1]).long()
                        in_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
                        
                        joint_accumulator_t.index_put_((iy[in_bounds], ix[in_bounds]), scores_warped_t[in_bounds], accumulate=True)
                        
                        ones_mask = torch.ones((h, w), device=device)
                        valid_back_t = warp_perspective_torch(ones_mask, H_inv_t, device)
                        global_trials_t += valid_back_t
                else:
                    kpts_warped = kpts_warped_t.cpu().numpy()
                    scores_warped = scores_warped_t.cpu().numpy()
                    if warp_mode == "3d":
                        for (x_prime, y_prime), score in zip(kpts_warped, scores_warped):
                            ix_prime = int(round(x_prime))
                            iy_prime = int(round(y_prime))
                            if 0 <= ix_prime < w and 0 <= iy_prime < h:
                                if valid_mask[iy_prime, ix_prime]:
                                    x_orig = warped_coord_map[iy_prime, ix_prime, 0]
                                    y_orig = warped_coord_map[iy_prime, ix_prime, 1]
                                    ix_orig, iy_orig = int(round(x_orig)), int(round(y_orig))
                                    if 0 <= ix_orig < w and 0 <= iy_orig < h:
                                        joint_accumulator[iy_orig, ix_orig] += score
                        global_trials += valid_mask.astype(np.float32)
                    else:
                        ones = np.ones((len(kpts_warped), 1), dtype=np.float32)
                        kpts_homo = np.concatenate([kpts_warped, ones], axis=1)
                        kpts_back = (H_inv @ kpts_homo.T).T
                        kpts_back = kpts_back[:, :2] / kpts_back[:, 2:]
                        
                        for (x, y), score in zip(kpts_back, scores_warped):
                            ix, iy = int(round(x)), int(round(y))
                            if 0 <= ix < w and 0 <= iy < h:
                                joint_accumulator[iy, ix] += score
                        
                        ones_mask = np.ones((h, w), dtype=np.float32)
                        valid_back = cv2.warpPerspective(ones_mask, H_inv, (w, h), flags=cv2.INTER_NEAREST)
                        global_trials += valid_back
            
    if use_gpu:
        joint_accumulator = joint_accumulator_t.cpu().numpy()
        global_trials = global_trials_t.cpu().numpy()
            
    # Normalize by the total number of validation trials across all modalities and warps
    normalized_heatmap = joint_accumulator / np.maximum(global_trials, 1.0)
    
    # 3. NMS to find clean local maxima
    kernel_size = nms_radius * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(normalized_heatmap, kernel)
    keypoints_mask = (normalized_heatmap == dilated) & (normalized_heatmap > detection_threshold)
    
    # Extract coordinates and scores
    ys, xs = np.where(keypoints_mask)
    kpts_res = np.stack([xs, ys], axis=1).astype(np.float32)
    scores_res = normalized_heatmap[ys, xs].astype(np.float32)
    
    # Sort by score descending
    idx = np.argsort(-scores_res)
    kpts_res = kpts_res[idx]
    scores_res = scores_res[idx]
    
    if return_all_warps:
        return kpts_res, scores_res, loaded_imgs, warped_samples, all_warps_info
    return kpts_res, scores_res, loaded_imgs, warped_samples

def save_visualizations(scene_name, kpts, scores, loaded_imgs, warped_samples, output_dir):
    """
    Save two high-quality plots for each scene:
    1. A 5-panel joint consensus plot.
    2. A 4-panel intermediate warped samples plot showing the homography effects.
    """
    # 1. Joint Consensus Visualization
    fig, axes = plt.subplots(1, 5, figsize=(25, 6), dpi=150)
    fig.suptitle(f"Multimodal Adaptation Consensus\nScene: {scene_name} | Found: {len(kpts)} stable consensus keypoints", fontsize=18, color="white", weight="bold")
    
    fig.patch.set_facecolor("#0b0c10")
    for ax in axes:
        ax.set_facecolor("#0b0c10")
        ax.axis("off")
        
    for idx, mod in enumerate(MODALITIES):
        if mod in loaded_imgs:
            img_to_show = loaded_imgs[mod]
            if mod == "range":
                if img_to_show.max() > 0:
                    img_to_show = ((img_to_show.astype(np.float32) / img_to_show.max()) * 255.0).astype(np.uint8)
                else:
                    img_to_show = np.zeros_like(img_to_show, dtype=np.uint8)
            axes[idx].imshow(img_to_show, cmap="gray")
            axes[idx].set_title(f"Modality: {mod.upper()}", color="#66fcf1", fontsize=12, pad=10)
            
    # Draw Joint Multimodal Consensus overlay
    bg_img = loaded_imgs.get("nearir", list(loaded_imgs.values())[0])
    if bg_img is loaded_imgs.get("range"):
        if bg_img.max() > 0:
            bg_img = ((bg_img.astype(np.float32) / bg_img.max()) * 255.0).astype(np.uint8)
        else:
            bg_img = np.zeros_like(bg_img, dtype=np.uint8)
            
    axes[4].imshow(bg_img, cmap="gray")
    if len(kpts) > 0:
        sc = axes[4].scatter(kpts[:, 0], kpts[:, 1], c=scores, cmap="plasma", s=18, edgecolors="none", alpha=0.9)
    else:
        # Avoid scatter error if keypoints are empty
        sc = axes[4].scatter([], [], c=[], cmap="plasma", s=18, edgecolors="none", alpha=0.9)
    axes[4].set_title("JOINT CONSENSUS PSEUDO-LABELS", color="#fc4445", fontsize=12, pad=10)
    
    # Add a premium colorbar for score confidence
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Repeatability Consensus Score (Across all 4 Spatially-Synchronized Modalities & Warps)", color="white", fontsize=11, labelpad=5)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "xticklabels"), color="white")
    
    consensus_dir = output_dir / "consensus"
    consensus_dir.mkdir(exist_ok=True, parents=True)
    consensus_path = consensus_dir / f"{Path(scene_name).stem}_consensus.png"
    plt.savefig(consensus_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    
    # 2. Intermediate Warped Detections Visualization
    if warped_samples:
        fig_w, axes_w = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
        fig_w.suptitle(f"Intermediate Warped Keypoint Detections (Near-IR)\nScene: {scene_name}", fontsize=16, color="white", weight="bold")
        fig_w.patch.set_facecolor("#0b0c10")
        
        for ax in axes_w:
            ax.set_facecolor("#0b0c10")
            ax.axis("off")
            
        # Column 1: Original Image
        axes_w[0].imshow(bg_img, cmap="gray")
        axes_w[0].set_title("Original (Near-IR)", color="#66fcf1", fontsize=11, pad=10)
        
        # Columns 2-4: Warped views
        for idx, sample in enumerate(warped_samples[:3]):
            ax = axes_w[idx + 1]
            ax.imshow(sample["image"], cmap="gray")
            w_kpts = sample["keypoints"]
            w_scores = sample["scores"]
            if len(w_kpts) > 0:
                ax.scatter(w_kpts[:, 0], w_kpts[:, 1], c=w_scores, cmap="plasma", s=10, edgecolors="none", alpha=0.8)
            ax.set_title(f"Intermediate Warp {idx+1}", color="#fc4445", fontsize=11, pad=10)
            
        warps_dir = output_dir / "warped_samples"
        warps_dir.mkdir(exist_ok=True, parents=True)
        warps_path = warps_dir / f"{Path(scene_name).stem}_warped_samples.png"
        plt.savefig(warps_path, bbox_inches="tight", facecolor=fig_w.get_facecolor(), edgecolor="none")
        plt.close()

def save_detailed_warp_visualizations(scene_name, all_warps_info, output_dir):
    """
    Save detailed side-by-side plots for all warped samples of a single scene:
    Left: Warped image with detected keypoints marked.
    Right: Original image with reprojected (back-projected) keypoints marked.
    """
    detailed_dir = output_dir / "detailed_warps"
    detailed_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Saving detailed side-by-side warp visualizations to {detailed_dir}...")
    
    for item in all_warps_info:
        mod = item["modality"]
        w_idx = item["warp_idx"]
        img_orig = item["image_orig"]
        img_warped = item["image_warped"]
        kpts_w = item["keypoints_warped"]
        kpts_rep = item["keypoints_reprojected"]
        
        # Normalize range maps if applicable
        if mod == "range":
            if img_orig.max() > 0:
                img_orig = ((img_orig.astype(np.float32) / img_orig.max()) * 255.0).astype(np.uint8)
            if img_warped.max() > 0:
                img_warped = ((img_warped.astype(np.float32) / img_warped.max()) * 255.0).astype(np.uint8)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
        fig.suptitle(f"3D Projective Warp Details | Modality: {mod.upper()} | Warp Index: {w_idx}\nScene: {scene_name}", fontsize=14, color="white", weight="bold")
        fig.patch.set_facecolor("#0b0c10")
        
        for ax in axes:
            ax.set_facecolor("#0b0c10")
            ax.axis("off")
            
        # Left: Warped Image with features marked
        axes[0].imshow(img_warped, cmap="gray")
        if len(kpts_w) > 0:
            axes[0].scatter(kpts_w[:, 0], kpts_w[:, 1], c="#66fcf1", s=15, edgecolors="none", alpha=0.9, label="Detected Features")
        axes[0].set_title(f"Warped Image (Features Detected: {len(kpts_w)})", color="#66fcf1", fontsize=11, pad=8)
        
        # Right: Original Image with reprojected features marked
        axes[1].imshow(img_orig, cmap="gray")
        if len(kpts_rep) > 0:
            axes[1].scatter(kpts_rep[:, 0], kpts_rep[:, 1], c="#fc4445", s=15, edgecolors="none", alpha=0.9, label="Reprojected Features")
        axes[1].set_title(f"Original Image (Reprojected Features: {len(kpts_rep)})", color="#fc4445", fontsize=11, pad=8)
        
        plt.tight_layout()
        out_path = detailed_dir / f"{Path(scene_name).stem}_{mod}_warp_{w_idx:02d}.png"
        plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Multimodal Homographic Adaptation & Visualizer")
    parser.add_argument("--num_warps", type=int, default=200, help="Number of homography warps per modality")
    parser.add_argument("--thresh", type=float, default=0.02, help="Keypoint threshold")
    parser.add_argument("--nms", type=int, default=5, help="NMS radius")
    parser.add_argument("--warp_mode", type=str, default="3d", choices=["2d", "3d"], help="Warp mode: 2d homography or 3d projective")
    parser.add_argument("--camera_info", type=str, default="plan/camera.info", help="Path to camera info file for 3d intrinsics")
    parser.add_argument("--save_detailed_warps", action="store_true", help="Save detailed side-by-side warp visualizations for a chosen scene")
    parser.add_argument("--viz_timestamp", type=str, default="", help="Specific image filename to visualize detailed warps (defaults to first image if empty)")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU-accelerated splatting & warping.")
    parser.add_argument("--no_gpu", dest="use_gpu", action="store_false", help="Disable GPU-accelerated splatting & warping.")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of parallel thread workers for image dataset processing.")
    parser.add_argument("--dataset", type=str, default="custom_dataset1", help="Name of the dataset directory under data/")
    parser.add_argument("--image_list_modality", type=str, default="reflectivity", choices=MODALITIES, help="Modality directory prefix for custom_image_list.txt")
    parser.add_argument("--weights", type=str, default=None, help="Path to custom SuperPoint model weights")
    args = parser.parse_args()
    
    dataset_dir = DATA_PATH / args.dataset
    images_dir = dataset_dir / "images"
    exports_dir = dataset_dir / "exports"
    exports_dir.mkdir(exist_ok=True, parents=True)
    
    # Directory to save individual visualization files
    visualizations_dir = dataset_dir / "visualizations"
    visualizations_dir.mkdir(exist_ok=True, parents=True)
    
    output_h5 = exports_dir / "pseudo_labels.h5"
    
    # Find all images in 'nearir' subdirectory
    nearir_dir = images_dir / "nearir"
    if not nearir_dir.exists():
        logger.error(f"Could not find nearir directory under {images_dir}")
        return
        
    image_names = sorted([p.name for p in nearir_dir.glob("*.png")] + [p.name for p in nearir_dir.glob("*.jpg")])
    if not image_names:
        logger.error(f"No images found in {nearir_dir}")
        return
        
    logger.info(f"Found {len(image_names)} synchronized image scenes for multimodal pseudo-label generation.")
    
    # Load camera intrinsics if 3D warp is selected
    K = None
    if args.warp_mode == "3d":
        info_path = Path(args.camera_info)
        if not info_path.exists():
            # Try workspace absolute path
            ws_path = Path("/home/tippeswamy/ws/thippeswamy/glue-factory/plan/camera.info")
            if ws_path.exists():
                info_path = ws_path
            else:
                script_dir = Path(__file__).resolve().parent
                alt_path = script_dir / "../../plan/camera.info"
                if alt_path.exists():
                    info_path = alt_path
        if not info_path.exists():
            raise FileNotFoundError(f"Could not find camera.info file at {args.camera_info}")
        logger.info(f"Loading camera intrinsics from {info_path}...")
        K = load_camera_intrinsics(info_path)
        logger.info(f"Loaded camera matrix K:\n{K}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Open H5 file to write pseudo labels
    logger.info(f"Generating joint multimodal pseudo-labels and saving to {output_h5}...")
    
    viz_name = args.viz_timestamp if args.viz_timestamp else (image_names[0] if image_names else "")
    
    lock = threading.Lock()
    
    with h5py.File(output_h5, "w") as f:
        with tqdm(total=len(image_names), desc="Multimodal Adaptation") as pbar:
            def worker(name):
                should_save_detailed = args.save_detailed_warps and (name == viz_name)
                
                local_model = get_thread_model(args, device)
                
                if should_save_detailed:
                    kpts, scores, loaded_imgs, warped_samples, all_warps_info = multimodal_homographic_adaptation(
                        name, images_dir, local_model, num_warps=args.num_warps, detection_threshold=args.thresh, nms_radius=args.nms,
                        warp_mode=args.warp_mode, K=K, return_all_warps=True, use_gpu=args.use_gpu
                    )
                    save_detailed_warp_visualizations(name, all_warps_info, visualizations_dir)
                else:
                    kpts, scores, loaded_imgs, warped_samples = multimodal_homographic_adaptation(
                        name, images_dir, local_model, num_warps=args.num_warps, detection_threshold=args.thresh, nms_radius=args.nms,
                        warp_mode=args.warp_mode, K=K, return_all_warps=False, use_gpu=args.use_gpu
                    )
                
                with lock:
                    grp = f.create_group(name)
                    grp.create_dataset("keypoints", data=kpts)
                    grp.create_dataset("keypoint_scores", data=scores)
                    
                    # Save premium individual visualizations for the first 3 scenes to save time & disk space
                    if len(f.keys()) <= 3:
                        save_visualizations(name, kpts, scores, loaded_imgs, warped_samples, visualizations_dir)
                    
                    logger.info(f"Scene {name}: Extracted {len(kpts)} keypoints.")
                    pbar.update(1)

            if args.num_threads > 1:
                with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                    list(executor.map(worker, image_names))
            else:
                for name in image_names:
                    worker(name)
            
    # Copy the first scene visualization as the default overview
    sample_stem = Path(image_names[0]).stem
    default_consensus = visualizations_dir / "consensus" / f"{sample_stem}_consensus.png"
    default_overview = dataset_dir / "adaptation_visualization.png"
    if default_consensus.exists():
        import shutil
        shutil.copy(default_consensus, default_overview)
        
    # Write custom_image_list.txt listing relative paths under the chosen modality (default: reflectivity)
    image_list_path = dataset_dir / "custom_image_list.txt"
    logger.info(f"Saving custom image list to {image_list_path} using modality prefix '{args.image_list_modality}'...")
    with open(image_list_path, "w") as img_list_f:
        for name in sorted(image_names):
            img_list_f.write(f"{args.image_list_modality}/{name}\n")
            
    logger.info(f"All processing complete! Individual visualization files saved in {visualizations_dir}")

if __name__ == "__main__":
    main()
