import os
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from gluefactory.models import get_model
from gluefactory.settings import DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visualize_custom")

def main():
    parser = argparse.ArgumentParser(description="Visualize trained SuperPoint detections.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/training/superpoint_custom_run/checkpoint_best.tar",
        help="Path to trained checkpoint."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="output/sample_data",
        help="Dataset name (e.g. output/sample_data)."
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="reflectivity",
        help="Modality prefix (e.g. reflectivity, nearir, signal)."
    )
    args = parser.parse_args()
    
    from gluefactory.settings import DATA_PATH
    dataset_dir = DATA_PATH / args.dataset
    checkpoint_path = Path(args.checkpoint)
    images_dir = dataset_dir / "images" / args.modality
    output_dir = dataset_dir / "visualizations/custom_detections"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}!")
        return
        
    if not images_dir.exists():
        logger.error(f"Images directory not found at {images_dir}!")
        return

    # Find all images
    image_paths = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")))
    if not image_paths:
        logger.error(f"No images found in {images_dir}")
        return
        
    logger.info(f"Found {len(image_paths)} images to test on.")

    # 1. Load Trained SuperPoint Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Configure SuperPoint open model
    model_conf = {
        "name": "superpoint_open",
        "nms_radius": 3,
        "max_num_keypoints": 512,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "trainable": False
    }
    
    model = get_model("superpoint_open")(model_conf).to(device)
    
    # Load state dict and strip 'extractor.' prefix from weights
    logger.info(f"Loading trained weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"]
    
    # Strip the prefix 'extractor.' from model state dict
    extractor_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("extractor."):
            extractor_state_dict[k.replace("extractor.", "")] = v
        else:
            extractor_state_dict[k] = v
            
    model.load_state_dict(extractor_state_dict)
    model.eval()
    logger.info("Model loaded successfully!")

    # For collage plotting
    collage_imgs = []
    collage_titles = []
    
    # 2. Run inference and plot
    for idx, img_path in enumerate(tqdm(image_paths, desc="Inference")):
        # Read image
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            logger.warning(f"Could not read image {img_path}")
            continue
            
        h, w = img_gray.shape[:2]
        
        # Prepare input tensor
        img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        
        with torch.no_grad():
            pred = model({"image": img_tensor})
            # Substract 0.5 coordinate offset (pixel center convention)
            kpts = pred["keypoints"][0].cpu().numpy() - 0.5
            scores = pred["keypoint_scores"][0].cpu().numpy()
            
        logger.info(f"Image {img_path.name}: Extracted {len(kpts)} keypoints.")
        
        # Plotting the result
        fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
        fig.patch.set_facecolor("#0b0c10")
        ax.set_facecolor("#0b0c10")
        ax.axis("off")
        
        ax.imshow(img_gray, cmap="gray")
        if len(kpts) > 0:
            ax.scatter(kpts[:, 0], kpts[:, 1], c=scores, cmap="plasma", s=15, edgecolors="none", alpha=0.9)
            
        ax.set_title(f"Trained SuperPoint Detections ({len(kpts)} kpts)\nImage: {img_path.name}", color="#66fcf1", fontsize=14, pad=12)
        
        # Save individual result
        out_filename = output_dir / f"{img_path.stem}_detections.png"
        plt.savefig(out_filename, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        
        # Add to collage list for the first 6 images
        if len(collage_imgs) < 6:
            # Create RGB overlay for collage
            overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            for kp in kpts[:150]: # Show top 150 keypoints for clear visibility
                x, y = int(round(kp[0])), int(round(kp[1]))
                cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1) # Green dots
            collage_imgs.append(overlay)
            collage_titles.append(f"{img_path.name} ({len(kpts)} kpts)")

    # 3. Create a beautiful grid collage
    if collage_imgs:
        logger.info("Creating summary collage...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
        fig.patch.set_facecolor("#0b0c10")
        axes = axes.flatten()
        
        for ax, img, title in zip(axes, collage_imgs, collage_titles):
            ax.imshow(img)
            ax.set_title(title, color="#66fcf1", fontsize=12, pad=8)
            ax.axis("off")
            
        fig.suptitle("Trained SuperPoint Custom Detections Collage (Top 150 Keypoints shown in Green)", color="#fc4445", fontsize=18, weight="bold", y=0.96)
        
        collage_out = output_dir / "collage.png"
        plt.savefig(collage_out, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        logger.info(f"Collage saved to {collage_out}")
        
    logger.info(f"All processing complete! Keypoint images saved in: {output_dir}")

if __name__ == "__main__":
    main()
