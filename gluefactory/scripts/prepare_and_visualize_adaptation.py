#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

from gluefactory.models import get_model
from gluefactory.settings import DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODALITIES = ["nearir", "range", "reflectivity", "signal"]

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
    
    max_pitch = np.deg2rad(5.0 * difficulty)
    max_yaw = np.deg2rad(10.0 * difficulty)
    max_roll = np.deg2rad(3.0 * difficulty)
    
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

def multimodal_homographic_adaptation(image_name, images_dir, model, num_warps=15, detection_threshold=0.015, nms_radius=4, warp_mode="3d", K=None, return_all_warps=False):
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
    joint_accumulator = np.zeros((h, w), dtype=np.float32)
    global_trials = np.zeros((h, w), dtype=np.float32)
    
    # For saving intermediate warped samples
    warped_samples = []
    all_warps_info = [] if return_all_warps else None
    
    # Process each modality
    for mod_name, img in loaded_imgs.items():
        # 1. Base detection on original image
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
        
        global_trials += 1.0 # Original is always a valid trial for this modality
        
        # 2. Homographic/Projective adaptations for this modality
        for warp_idx in range(num_warps):
            if warp_mode == "3d":
                R, t = sample_random_3d_transform(difficulty=0.7)
                depth_m = loaded_imgs["range"].astype(np.float32) / 1000.0
                warped_dict, valid_mask, warped_coord_map = warp_3d_projective(
                    {mod_name: img}, depth_m, R, t, K
                )
                warped_img = warped_dict[mod_name]
                
                # Create feed-able warped image
                if mod_name == "range":
                    if warped_img.max() > 0:
                        warped_feed = ((warped_img.astype(np.float32) / warped_img.max()) * 255.0).astype(np.uint8)
                    else:
                        warped_feed = np.zeros_like(warped_img, dtype=np.uint8)
                else:
                    warped_feed = warped_img
            else:
                H = sample_random_homography((h, w), difficulty=0.7)
                H_inv = np.linalg.inv(H)
                warped_img = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
                warped_feed = warped_img
                
            warped_tensor = torch.from_numpy(warped_feed).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            
            with torch.no_grad():
                pred_warped = model({"image": warped_tensor})
                kpts_warped = pred_warped["keypoints"][0].cpu().numpy() - 0.5
                scores_warped = pred_warped["keypoint_scores"][0].cpu().numpy()
                
            # If this is nearir and we want to keep a few visual samples
            if mod_name == "nearir" and len(warped_samples) < 3 and len(kpts_warped) > 0:
                warped_samples.append({
                    "image": warped_img,
                    "keypoints": kpts_warped,
                    "scores": scores_warped
                })
                
            # If we want to capture all detailed warps for visualization
            if return_all_warps:
                if len(kpts_warped) > 0:
                    if warp_mode == "3d":
                        kpts_back_list = []
                        scores_back_list = []
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
            if len(kpts_warped) > 0:
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
                else:
                    ones = np.ones((len(kpts_warped), 1), dtype=np.float32)
                    kpts_homo = np.concatenate([kpts_warped, ones], axis=1)
                    kpts_back = (H_inv @ kpts_homo.T).T
                    kpts_back = kpts_back[:, :2] / kpts_back[:, 2:]
                    
                    for (x, y), score in zip(kpts_back, scores_warped):
                        ix, iy = int(round(x)), int(round(y))
                        if 0 <= ix < w and 0 <= iy < h:
                            joint_accumulator[iy, ix] += score
                            
            # Track valid region trials for this warp
            if warp_mode == "3d":
                global_trials += valid_mask.astype(np.float32)
            else:
                ones_mask = np.ones((h, w), dtype=np.float32)
                valid_back = cv2.warpPerspective(ones_mask, H_inv, (w, h), flags=cv2.INTER_NEAREST)
                global_trials += valid_back
            
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
    
    consensus_path = output_dir / f"{Path(scene_name).stem}_consensus.png"
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
            
        warps_path = output_dir / f"{Path(scene_name).stem}_warped_samples.png"
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
    parser.add_argument("--num_warps", type=int, default=15, help="Number of homography warps per modality")
    parser.add_argument("--thresh", type=float, default=0.015, help="Keypoint threshold")
    parser.add_argument("--nms", type=int, default=4, help="NMS radius")
    parser.add_argument("--warp_mode", type=str, default="3d", choices=["2d", "3d"], help="Warp mode: 2d homography or 3d projective")
    parser.add_argument("--camera_info", type=str, default="plan/camera.info", help="Path to camera info file for 3d intrinsics")
    parser.add_argument("--save_detailed_warps", action="store_true", help="Save detailed side-by-side warp visualizations for a chosen scene")
    parser.add_argument("--viz_timestamp", type=str, default="", help="Specific image filename to visualize detailed warps (defaults to first image if empty)")
    args = parser.parse_args()
    
    dataset_dir = DATA_PATH / "custom_dataset"
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
        
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Instantiating SuperPoint base detector on {device}...")
    model = get_model("superpoint_open")({
        "nms_radius": args.nms,
        "max_num_keypoints": 2048,
        "detection_threshold": 0.0,
        "trainable": False
    }).to(device).eval()
    
    # Open H5 file to write pseudo labels
    logger.info(f"Generating joint multimodal pseudo-labels and saving to {output_h5}...")
    
    viz_name = args.viz_timestamp if args.viz_timestamp else (image_names[0] if image_names else "")
    
    with h5py.File(output_h5, "w") as f:
        for name in tqdm(image_names, desc="Multimodal Adaptation"):
            should_save_detailed = args.save_detailed_warps and (name == viz_name)
            
            if should_save_detailed:
                kpts, scores, loaded_imgs, warped_samples, all_warps_info = multimodal_homographic_adaptation(
                    name, images_dir, model, num_warps=args.num_warps, detection_threshold=args.thresh, nms_radius=args.nms,
                    warp_mode=args.warp_mode, K=K, return_all_warps=True
                )
                save_detailed_warp_visualizations(name, all_warps_info, visualizations_dir)
            else:
                kpts, scores, loaded_imgs, warped_samples = multimodal_homographic_adaptation(
                    name, images_dir, model, num_warps=args.num_warps, detection_threshold=args.thresh, nms_radius=args.nms,
                    warp_mode=args.warp_mode, K=K, return_all_warps=False
                )
            
            grp = f.create_group(name)
            grp.create_dataset("keypoints", data=kpts)
            grp.create_dataset("keypoint_scores", data=scores)
            
            # Save premium individual visualizations
            # save_visualizations(name, kpts, scores, loaded_imgs, warped_samples, visualizations_dir)
            
            logger.info(f"Scene {name}: Extracted {len(kpts)} keypoints and saved visual plots.")
            
    # Copy the first scene visualization as the default overview
    sample_stem = Path(image_names[0]).stem
    default_consensus = visualizations_dir / f"{sample_stem}_consensus.png"
    default_overview = dataset_dir / "adaptation_visualization.png"
    if default_consensus.exists():
        import shutil
        shutil.copy(default_consensus, default_overview)
        
    logger.info(f"All processing complete! Individual visualization files saved in {visualizations_dir}")

if __name__ == "__main__":
    main()
