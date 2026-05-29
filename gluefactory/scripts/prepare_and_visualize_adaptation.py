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

def multimodal_homographic_adaptation(image_name, images_dir, model, num_warps=15, detection_threshold=0.015, nms_radius=4):
    """
    Perform Joint Multimodal Homographic Adaptation.
    Reads synchronized images from 'nearir', 'range', 'reflectivity', and 'signal'.
    Accumulates stable keypoint detections across all modalities and warps.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load all available modality images
    loaded_imgs = {}
    h, w = None, None
    for mod in MODALITIES:
        img_path = images_dir / mod / image_name
        if img_path.exists():
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                loaded_imgs[mod] = img
                if h is None:
                    h, w = img.shape
                    
    if not loaded_imgs:
        raise ValueError(f"Could not load any synchronized image modalities for {image_name}")
        
    # Joint accumulator heatmap and trials map
    joint_accumulator = np.zeros((h, w), dtype=np.float32)
    global_trials = np.zeros((h, w), dtype=np.float32)
    
    # Process each modality
    for mod_name, img in loaded_imgs.items():
        # 1. Base detection on original image
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            pred = model({"image": img_tensor})
            kpts = pred["keypoints"][0].cpu().numpy() - 0.5
            scores = pred["keypoint_scores"][0].cpu().numpy()
            
        for (x, y), score in zip(kpts, scores):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < h:
                joint_accumulator[iy, ix] += score
        
        global_trials += 1.0 # Original is always a valid trial for this modality
        
        # 2. Homographic adaptations for this modality
        for _ in range(num_warps):
            H = sample_random_homography((h, w), difficulty=0.7)
            H_inv = np.linalg.inv(H)
            
            # Warp image
            warped_img = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
            warped_tensor = torch.from_numpy(warped_img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            
            with torch.no_grad():
                pred_warped = model({"image": warped_tensor})
                kpts_warped = pred_warped["keypoints"][0].cpu().numpy() - 0.5
                scores_warped = pred_warped["keypoint_scores"][0].cpu().numpy()
                
            # Inverse warp detected keypoints and splat
            if len(kpts_warped) > 0:
                ones = np.ones((len(kpts_warped), 1), dtype=np.float32)
                kpts_homo = np.concatenate([kpts_warped, ones], axis=1)
                kpts_back = (H_inv @ kpts_homo.T).T
                kpts_back = kpts_back[:, :2] / kpts_back[:, 2:]
                
                for (x, y), score in zip(kpts_back, scores_warped):
                    ix, iy = int(round(x)), int(round(y))
                    if 0 <= ix < w and 0 <= iy < h:
                        joint_accumulator[iy, ix] += score
                        
            # Track valid region trials for this warp
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
    
    return kpts_res, scores_res, loaded_imgs

def main():
    parser = argparse.ArgumentParser(description="Multimodal Homographic Adaptation & Visualizer")
    parser.add_argument("--num_warps", type=int, default=15, help="Number of homography warps per modality")
    parser.add_argument("--thresh", type=float, default=0.015, help="Keypoint threshold")
    parser.add_argument("--nms", type=int, default=4, help="NMS radius")
    args = parser.parse_args()
    
    dataset_dir = DATA_PATH / "custom_dataset"
    images_dir = dataset_dir / "images"
    exports_dir = dataset_dir / "exports"
    exports_dir.mkdir(exist_ok=True, parents=True)
    
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
    sample_imgs_dict = None
    sample_name = image_names[0]
    
    with h5py.File(output_h5, "w") as f:
        for name in tqdm(image_names, desc="Multimodal Homographic Adaptation"):
            kpts, scores, loaded_imgs = multimodal_homographic_adaptation(
                name, images_dir, model, num_warps=args.num_warps, detection_threshold=args.thresh, nms_radius=args.nms
            )
            
            if name == sample_name:
                sample_imgs_dict = loaded_imgs
            
            grp = f.create_group(name)
            grp.create_dataset("keypoints", data=kpts)
            grp.create_dataset("keypoint_scores", data=scores)
            
            logger.info(f"Scene {name}: Extracted {len(kpts)} multimodal consensus pseudo-ground-truth keypoints.")
            
    # 4. Generate premium visual validation image showing all 4 modalities and consensus
    logger.info("Generating premium multimodal dataset visual validation samples...")
    with h5py.File(output_h5, "r") as f:
        kpts = f[sample_name]["keypoints"][:]
        scores = f[sample_name]["keypoint_scores"][:]
        
    fig, axes = plt.subplots(1, 5, figsize=(25, 6), dpi=150)
    fig.suptitle(f"Multimodal Homographic Adaptation Consensus Verification\nScene: {sample_name} | Found: {len(kpts)} stable consensus keypoints", fontsize=18, color="white", weight="bold")
    
    fig.patch.set_facecolor("#0b0c10")
    for ax in axes:
        ax.set_facecolor("#0b0c10")
        ax.axis("off")
        
    # Draw individual modalities
    for idx, mod in enumerate(MODALITIES):
        if mod in sample_imgs_dict:
            axes[idx].imshow(sample_imgs_dict[mod], cmap="gray")
            axes[idx].set_title(f"Modality: {mod.upper()}", color="#66fcf1", fontsize=12, pad=10)
        else:
            axes[idx].text(0.5, 0.5, f"{mod.upper()}\nNot Found", color="red", ha="center", va="center")
            
    # Draw Joint Multimodal Consensus overlay
    bg_img = sample_imgs_dict.get("nearir", list(sample_imgs_dict.values())[0])
    axes[4].imshow(bg_img, cmap="gray")
    sc = axes[4].scatter(kpts[:, 0], kpts[:, 1], c=scores, cmap="plasma", s=18, edgecolors="none", alpha=0.9)
    axes[4].set_title("JOINT CONSENSUS PSEUDO-LABELS", color="#fc4445", fontsize=12, pad=10)
    
    # Add a premium colorbar for score confidence
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Repeatability Consensus Score (Across all 4 Spatially-Synchronized Modalities & Warps)", color="white", fontsize=11, labelpad=5)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "xticklabels"), color="white")
    
    viz_output = dataset_dir / "adaptation_visualization.png"
    plt.savefig(viz_output, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    
    logger.info(f"Visual validation saved to {viz_output} successfully!")

if __name__ == "__main__":
    main()
