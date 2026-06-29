#!/usr/bin/env python3
"""
Generate training/validation image pair lists for the SLAM dataset.

Uses camera poses from the SLAM trajectory to find geometrically overlapping
image pairs: two frames are paired if their translational distance falls within
[--min_dist, --max_dist] and their co-visibility (estimated from depth overlap)
exceeds a threshold. Writes pairs_train.txt and pairs_val.txt in the dataset
directory, one pair per line formatted as "rgb/frame_a.png rgb/frame_b.png".

Usage:
    python -m gluefactory.scripts.generate_slam_pairs \\
        --data_dir    data/output/slam \\
        --poses_file  data/output/slam/poses.txt \\
        --modality    rgb \\
        --min_dist    0.05 \\
        --max_dist    1.0  \\
        --val_ratio   0.15
"""

import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import h5py
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch
import threading
from concurrent.futures import ThreadPoolExecutor

from gluefactory.models import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

thread_local = threading.local()

def get_thread_model(args, device):
    if not hasattr(thread_local, "model"):
        model_conf = {
            "nms_radius": args.nms_radius,
            "max_num_keypoints": args.max_keypoints,
            "detection_threshold": 0.0,
            "trainable": False
        }
        if hasattr(args, "sp_weights") and args.sp_weights:
            model_conf["weights"] = args.sp_weights
        thread_local.model = get_model("superpoint_open")(model_conf).to(device).eval()
    return thread_local.model

def parse_poses(poses_path):
    # #timestamp x y z qx qy qz qw
    poses = {}
    with open(poses_path, "r") as f:
        for line in f:
            if line.startswith("#"):
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

def load_camera_intrinsics_yaml(info_path):
    import yaml
    with open(info_path, "r") as f:
        lines = f.readlines()
        if lines and lines[0].startswith("%YAML"):
            lines = lines[1:]
        data = yaml.unsafe_load("".join(lines))
    k_flat = data["camera_matrix"]["data"]
    K = np.array(k_flat, dtype=np.float32).reshape(3, 3)
    return K

def compute_covisibility_overlap(depth_path_i, R_i, t_i, R_j, t_j, K):
    if not Path(depth_path_i).exists():
        return 0.0
    
    depth_img = cv2.imread(str(depth_path_i), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        return 0.0
    
    depth_m = depth_img.astype(np.float32) / 1000.0
    H, W = depth_m.shape
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = depth_m > 0
    if not np.any(valid):
        return 0.0
        
    u_valid = u[valid]
    v_valid = v[valid]
    d_valid = depth_m[valid]
    
    # K_inv
    K_inv = np.linalg.inv(K)
    pixels_homo = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0)
    
    # 3D points in camera i
    P_c_i = d_valid * (K_inv @ pixels_homo)
    
    # 3D points in world
    P_w = R_i @ P_c_i + t_i[:, None]
    
    # 3D points in camera j
    # P_w = R_j * P_c_j + t_j  => P_c_j = R_j.T * (P_w - t_j)
    R_j_inv = R_j.T
    t_j_inv = -R_j.T @ t_j
    P_c_j = R_j_inv @ P_w + t_j_inv[:, None]
    
    # Project to image j
    z_j = P_c_j[2, :]
    valid_z = z_j > 1e-3
    if not np.any(valid_z):
        return 0.0
        
    projected = K @ P_c_j[:, valid_z]
    u_j = projected[0, :] / z_j[valid_z]
    v_j = projected[1, :] / z_j[valid_z]
    
    in_bounds = (u_j >= 0) & (u_j < W) & (v_j >= 0) & (v_j < H)
    
    overlap = np.sum(in_bounds) / len(d_valid)
    return overlap

def extract_features(image_name, dataset_dir, model, device):
    rgb_path = dataset_dir / "images/rgb" / image_name
    img_rgb = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
    if img_rgb is None:
        return None, None, None
        
    img_tensor = torch.from_numpy(img_rgb.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        pred = model({"image": img_tensor})
        kpts = pred["keypoints"][0].cpu().numpy()
        scores = pred["keypoint_scores"][0].cpu().numpy()
        desc = pred["descriptors"][0].cpu().numpy()
        
    return kpts, scores, desc

def main():
    parser = argparse.ArgumentParser(description="Generate Pose-Based Pairs for SLAM")
    parser.add_argument("--data_dir", type=str, default="data/output/sample_slam", help="Path to slam dir")
    parser.add_argument("--max_dist", type=float, default=2.0, help="Max translation distance")
    parser.add_argument("--min_dist", type=float, default=0.1, help="Min translation distance")
    parser.add_argument("--max_angle", type=float, default=30.0, help="Max rotation angle (deg)")
    parser.add_argument("--min_overlap", type=float, default=0.1, help="Min covisibility overlap")
    parser.add_argument("--max_pairs", type=int, default=10, help="Max pairs per frame")
    parser.add_argument("--split_ratio", type=float, default=0.83, help="Train/val split")
    parser.add_argument("--extract_features", action="store_true", default=False)
    parser.add_argument("--sp_weights", type=str, default=None)
    parser.add_argument("--nms_radius", type=int, default=3)
    parser.add_argument("--max_keypoints", type=int, default=512)
    parser.add_argument("--num_threads", type=int, default=14)
    args = parser.parse_args()
    
    dataset_dir = Path(args.data_dir)
    poses_path = dataset_dir / "poses_odom_RGBD_slam.txt"
    if not poses_path.exists():
        logger.error(f"Poses file not found: {poses_path}")
        return
        
    rgb_dir = dataset_dir / "images/rgb"
    depth_dir = dataset_dir / "images/depth"
    calib_dir = dataset_dir / "images/calib"
    
    image_names = sorted([p.name for p in rgb_dir.glob("*.png")] + [p.name for p in rgb_dir.glob("*.jpg")])
    if not image_names:
        logger.error("No images found.")
        return
        
    poses = parse_poses(poses_path)
    
    # Get K
    first_ts = Path(image_names[0]).stem
    calib_path = calib_dir / f"{first_ts}.yaml"
    if not calib_path.exists():
        calib_path = list(calib_dir.glob("*.yaml"))[0]
    K = load_camera_intrinsics_yaml(calib_path)
    
    logger.info("Computing pairwise distances and covisibility...")
    pairs = []
    
    # Map names to timestamps
    name_to_ts = {name: Path(name).stem for name in image_names}
    valid_names = [name for name in image_names if name_to_ts[name] in poses]
    
    for i, name_i in enumerate(tqdm(valid_names)):
        ts_i = name_to_ts[name_i]
        R_i, t_i = poses[ts_i]
        
        candidates = []
        for j, name_j in enumerate(valid_names):
            if i == j:
                continue
            ts_j = name_to_ts[name_j]
            R_j, t_j = poses[ts_j]
            
            dist = np.linalg.norm(t_i - t_j)
            if dist < args.min_dist or dist > args.max_dist:
                continue
                
            # Angle
            R_rel = R_i.T @ R_j
            trace = np.trace(R_rel)
            angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
            if np.degrees(angle) > args.max_angle:
                continue
                
            candidates.append((j, name_j, ts_j, R_j, t_j, dist))
            
        # Sort by distance
        candidates.sort(key=lambda x: x[-1])
        candidates = candidates[:args.max_pairs * 2] # Check a bit more for covisibility
        
        valid_candidates = []
        depth_path_i = depth_dir / name_i
        for j, name_j, ts_j, R_j, t_j, dist in candidates:
            overlap = compute_covisibility_overlap(depth_path_i, R_i, t_i, R_j, t_j, K)
            if overlap >= args.min_overlap:
                valid_candidates.append((name_i, name_j))
                if len(valid_candidates) >= args.max_pairs:
                    break
        
        pairs.extend(valid_candidates)
        
    logger.info(f"Found {len(pairs)} pairs.")
    
    np.random.seed(42)
    np.random.shuffle(pairs)
    num_train = int(len(pairs) * args.split_ratio)
    train_pairs = pairs[:num_train]
    val_pairs = pairs[num_train:]
    
    with open(dataset_dir / "pairs_train.txt", "w") as f:
        for pi, pj in train_pairs:
            f.write(f"rgb/{pi} rgb/{pj}\n")
            
    with open(dataset_dir / "pairs_val.txt", "w") as f:
        for pi, pj in val_pairs:
            f.write(f"rgb/{pi} rgb/{pj}\n")
            
    if args.extract_features:
        logger.info("Extracting features...")
        exports_dir = dataset_dir / "exports"
        exports_dir.mkdir(exist_ok=True)
        h5_path = exports_dir / "sp_features_slam.h5"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        unique_images = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
        
        with h5py.File(h5_path, "w") as f:
            with tqdm(total=len(unique_images)) as pbar:
                def worker(name):
                    local_model = get_thread_model(args, device)
                    kpts, scores, desc = extract_features(name, dataset_dir, local_model, device)
                    return name, kpts, scores, desc
                if args.num_threads > 1:
                    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                        for name, kpts, scores, desc in executor.map(worker, unique_images):
                            if kpts is not None:
                                grp = f.create_group(name)
                                grp.create_dataset("keypoints", data=kpts)
                                grp.create_dataset("keypoint_scores", data=scores)
                                grp.create_dataset("descriptors", data=desc)
                            pbar.update(1)
                else:
                    for n in unique_images:
                        name, kpts, scores, desc = worker(n)
                        if kpts is not None:
                            grp = f.create_group(name)
                            grp.create_dataset("keypoints", data=kpts)
                            grp.create_dataset("keypoint_scores", data=scores)
                            grp.create_dataset("descriptors", data=desc)
                        pbar.update(1)

    logger.info("Done generating pairs and features.")

if __name__ == "__main__":
    main()
