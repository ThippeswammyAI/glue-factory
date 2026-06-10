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
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
import yaml

from gluefactory.models import get_model
from gluefactory.settings import DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

thread_local = threading.local()

def get_thread_model(args, device):
    if not hasattr(thread_local, "model"):
        logger.info(f"Instantiating thread-local SuperPoint model on thread {threading.current_thread().name}...")
        model_conf = {
            "nms_radius": args.nms,
            "max_num_keypoints": args.max_keypoints,
            "detection_threshold": 0.0,
            "trainable": False
        }
        if hasattr(args, "weights") and args.weights:
            model_conf["weights"] = args.weights
        thread_local.model = get_model("superpoint_open")(model_conf).to(device).eval()
    return thread_local.model

def sample_random_homography(shape, difficulty=0.7):
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

def load_camera_intrinsics_yaml(info_path):
    with open(info_path, "r") as f:
        lines = f.readlines()
        if lines and lines[0].startswith("%YAML"):
            lines = lines[1:]
        data = yaml.unsafe_load("".join(lines))
    k_flat = data["camera_matrix"]["data"]
    K = np.array(k_flat, dtype=np.float32).reshape(3, 3)
    return K

def sample_random_3d_transform(difficulty=0.7):
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

def warp_3d_projective_torch_fast(img_t, precomputed, R_t, t_t, K_t, device):
    H = precomputed["H"]
    W = precomputed["W"]
    P_3D = precomputed["P_3D"]
    u_valid = precomputed["u_valid"]
    v_valid = precomputed["v_valid"]
    
    P_prime_3D = R_t @ P_3D + t_t[:, None]
    z_prime = P_prime_3D[2, :]
    
    valid_z_mask = (z_prime > 1e-3)
    if not torch.any(valid_z_mask):
        return torch.zeros((H, W), dtype=img_t.dtype, device=device), torch.zeros((H, W), dtype=torch.bool, device=device), torch.zeros((H, W, 2), dtype=torch.float32, device=device)
        
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
        return torch.zeros((H, W), dtype=img_t.dtype, device=device), torch.zeros((H, W), dtype=torch.bool, device=device), torch.zeros((H, W, 2), dtype=torch.float32, device=device)
        
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
    
    warped_img = torch.zeros((H, W), dtype=img_t.dtype, device=device)
    colors = img_t[v_orig_sorted, u_orig_sorted]
    warped_img[v_idx_sorted, u_idx_sorted] = colors
        
    valid_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    valid_mask[v_idx_sorted, u_idx_sorted] = True
    
    warped_coord_map = torch.zeros((H, W, 2), dtype=torch.float32, device=device)
    warped_coord_map[v_idx_sorted, u_idx_sorted, 0] = u_orig_sorted.float()
    warped_coord_map[v_idx_sorted, u_idx_sorted, 1] = v_orig_sorted.float()
    
    return warped_img, valid_mask, warped_coord_map

def run_homographic_adaptation(image_name, dataset_dir, model, num_warps=50, detection_threshold=0.015, nms_radius=4, warp_mode="3d", K=None, use_gpu=True):
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    
    rgb_path = dataset_dir / "images/rgb" / image_name
    img_rgb = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
    if img_rgb is None:
        raise ValueError(f"Could not load RGB image {image_name}")
    h, w = img_rgb.shape[:2]
    
    depth_m = None
    if warp_mode == "3d":
        depth_path = dataset_dir / "images/depth" / image_name
        if not depth_path.exists():
            logger.warning(f"Depth map not found for {image_name}. Falling back to 2D Homography warping!")
            warp_mode = "2d"
        else:
            img_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if img_depth is not None:
                depth_m = img_depth.astype(np.float32) / 1000.0
            else:
                logger.warning(f"Could not load Depth map for {image_name}. Falling back to 2D Homography warping!")
                warp_mode = "2d"
                
    if use_gpu:
        img_rgb_t = torch.from_numpy(img_rgb.astype(np.float32)).to(device)
        joint_accumulator_t = torch.zeros((h, w), dtype=torch.float32, device=device)
        global_trials_t = torch.zeros((h, w), dtype=torch.float32, device=device)
        
        if warp_mode == "3d" and depth_m is not None:
            K_inv = np.linalg.inv(K)
            K_inv_t = torch.from_numpy(K_inv).float().to(device)
            depth_map_t = torch.from_numpy(depth_m).to(device)
            precomputed_3d = precompute_3d_points_torch(depth_map_t, K_inv_t, device)
        else:
            precomputed_3d = None
    else:
        # Not implementing CPU fallback for simplicity since you have GPU
        raise NotImplementedError("CPU mode not implemented in this version")
    
    # 1. Base detection on original image
    img_tensor = img_rgb_t.unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        pred = model({"image": img_tensor})
        kpts_t = pred["keypoints"][0] - 0.5
        scores_t = pred["keypoint_scores"][0]
    
    ix = torch.round(kpts_t[:, 0]).long()
    iy = torch.round(kpts_t[:, 1]).long()
    in_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    joint_accumulator_t.index_put_((iy[in_bounds], ix[in_bounds]), scores_t[in_bounds], accumulate=True)
    global_trials_t += 1.0
    
    # 2. Adaptations
    for warp_idx in range(num_warps):
        if warp_mode == "3d":
            R, t = sample_random_3d_transform(difficulty=0.7)
            R_t = torch.from_numpy(R).float().to(device)
            t_t = torch.from_numpy(t).float().to(device)
            K_t = torch.from_numpy(K).float().to(device)
            
            if precomputed_3d is not None:
                warped_img_t, valid_mask_t, warped_coord_map_t = warp_3d_projective_torch_fast(
                    img_rgb_t, precomputed_3d, R_t, t_t, K_t, device
                )
            else:
                warped_img_t = torch.zeros((h, w), dtype=img_rgb_t.dtype, device=device)
                valid_mask_t = torch.zeros((h, w), dtype=torch.bool, device=device)
                warped_coord_map_t = torch.zeros((h, w, 2), dtype=torch.float32, device=device)
            
            warped_tensor = warped_img_t.unsqueeze(0).unsqueeze(0) / 255.0
            H_inv_t = None
        else:
            H = sample_random_homography((h, w), difficulty=0.7)
            H_inv = np.linalg.inv(H)
            H_t = torch.from_numpy(H).float().to(device)
            H_inv_t = torch.from_numpy(H_inv).float().to(device)
            
            warped_img_t = warp_perspective_torch(img_rgb_t, H_t, device)
            warped_tensor = warped_img_t.unsqueeze(0).unsqueeze(0) / 255.0
            
        with torch.no_grad():
            pred_warped = model({"image": warped_tensor})
            kpts_warped_t = pred_warped["keypoints"][0] - 0.5
            scores_warped_t = pred_warped["keypoint_scores"][0]
            
        # Inverse warp detected keypoints and splat
        if len(kpts_warped_t) > 0:
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
                
    joint_accumulator = joint_accumulator_t.cpu().numpy()
    global_trials = global_trials_t.cpu().numpy()
        
    normalized_heatmap = joint_accumulator / np.maximum(global_trials, 1.0)
    
    # 3. NMS
    kernel_size = nms_radius * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(normalized_heatmap, kernel)
    keypoints_mask = (normalized_heatmap == dilated) & (normalized_heatmap > detection_threshold)
    
    ys, xs = np.where(keypoints_mask)
    kpts_res = np.stack([xs, ys], axis=1).astype(np.float32)
    scores_res = normalized_heatmap[ys, xs].astype(np.float32)
    
    idx = np.argsort(-scores_res)
    kpts_res = kpts_res[idx]
    scores_res = scores_res[idx]
    
    return kpts_res, scores_res

def main():
    parser = argparse.ArgumentParser(description="SLAM Homographic Adaptation")
    parser.add_argument("--data_dir", type=str, default="data/output/sample_slam", help="Path to slam dir")
    parser.add_argument("--num_warps", type=int, default=50, help="Number of homography warps per image")
    parser.add_argument("--thresh", type=float, default=0.015, help="Keypoint threshold")
    parser.add_argument("--nms", type=int, default=4, help="NMS radius")
    parser.add_argument("--max_keypoints", type=int, default=512, help="Max keypoints")
    parser.add_argument("--warp_mode", type=str, default="3d", choices=["2d", "3d"], help="Warp mode")
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--num_threads", type=int, default=14)
    parser.add_argument("--split_ratio", type=float, default=0.83)
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()
    
    dataset_dir = Path(args.data_dir)
    images_dir = dataset_dir / "images"
    rgb_dir = images_dir / "rgb"
    calib_dir = images_dir / "calib"
    exports_dir = dataset_dir / "exports"
    exports_dir.mkdir(exist_ok=True, parents=True)
    
    if not rgb_dir.exists():
        logger.error(f"Could not find rgb directory under {images_dir}")
        return
        
    image_names = sorted([p.name for p in rgb_dir.glob("*.png")] + [p.name for p in rgb_dir.glob("*.jpg")])
    if not image_names:
        logger.error(f"No images found in {rgb_dir}")
        return
        
    logger.info(f"Found {len(image_names)} synchronized image scenes for multimodal pseudo-label generation.")
    
    K = None
    if args.warp_mode == "3d":
        first_ts = Path(image_names[0]).stem
        calib_path = calib_dir / f"{first_ts}.yaml"
        if not calib_path.exists():
            logger.warning(f"Calib file {calib_path} not found. Searching for any YAML...")
            calib_files = list(calib_dir.glob("*.yaml"))
            if calib_files:
                calib_path = calib_files[0]
            else:
                raise FileNotFoundError(f"Could not find any camera calibration YAMLs in {calib_dir}")
        logger.info(f"Loading camera intrinsics from {calib_path}...")
        K = load_camera_intrinsics_yaml(calib_path)
        logger.info(f"Loaded camera matrix K:\n{K}")
        
    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    output_h5 = exports_dir / "pseudo_labels_slam.h5"
    logger.info(f"Generating joint multimodal pseudo-labels and saving to {output_h5}...")
    
    with h5py.File(output_h5, "w") as f:
        with tqdm(total=len(image_names), desc="Homographic Adaptation") as pbar:
            def worker(name):
                local_model = get_thread_model(args, device)
                kpts, scores = run_homographic_adaptation(
                    name, dataset_dir, local_model, num_warps=args.num_warps, detection_threshold=args.thresh, nms_radius=args.nms,
                    warp_mode=args.warp_mode, K=K, use_gpu=args.use_gpu
                )
                return name, kpts, scores

            if args.num_threads > 1:
                with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                    for name, kpts, scores in executor.map(worker, image_names):
                        grp = f.create_group(name)
                        grp.create_dataset("keypoints", data=kpts)
                        grp.create_dataset("keypoint_scores", data=scores)
                        pbar.update(1)
            else:
                for n in image_names:
                    name, kpts, scores = worker(n)
                    grp = f.create_group(name)
                    grp.create_dataset("keypoints", data=kpts)
                    grp.create_dataset("keypoint_scores", data=scores)
                    pbar.update(1)
                    
    # Generate splits
    logger.info("Generating split files...")
    np.random.RandomState(42).shuffle(image_names)
    num_train = int(len(image_names) * args.split_ratio)
    train_names = sorted(image_names[:num_train])
    val_names = sorted(image_names[num_train:])
    
    with open(dataset_dir / "image_list_train.txt", "w") as f:
        f.writelines([f"rgb/{name}\n" for name in train_names])
    with open(dataset_dir / "image_list_val.txt", "w") as f:
        f.writelines([f"rgb/{name}\n" for name in val_names])
        
    logger.info("All processing complete!")

if __name__ == "__main__":
    main()
