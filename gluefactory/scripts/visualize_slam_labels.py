#!/usr/bin/env python3
"""
Visualize SuperPoint pseudo-labels stored in an HDF5 file on SLAM images.

Overlays keypoints (coloured by detection score) onto the corresponding raw
images and saves side-by-side PNGs. Useful for verifying that the pseudo-label
generation step produced sensible detections before starting SuperPoint training.

Usage:
    python -m gluefactory.scripts.visualize_slam_labels \\
        --data_dir  data/output/slam \\
        --h5_labels data/output/slam/exports/pseudo_labels_slam.h5 \\
        --modality  rgb \\
        --max_imgs  50 \\
        --output_dir data/output/slam/visualizations/slam_labels
"""

import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_html_grid(image_paths, output_html):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>SLAM Pseudo-Labels Visualization</title>
        <style>
            body { font-family: sans-serif; background-color: #121212; color: #ffffff; padding: 20px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(600px, 1fr)); gap: 20px; }
            .item { background: #1e1e1e; padding: 15px; border-radius: 8px; text-align: center; }
            img { max-width: 100%; height: auto; border-radius: 4px; }
            h2 { color: #bb86fc; font-size: 1.2em; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <h1>SLAM Pseudo-Labels Visualization</h1>
        <div class="grid">
    """
    for img_path in image_paths:
        html_content += f"""
            <div class="item">
                <h2>{img_path.name}</h2>
                <img src="{img_path.name}" alt="{img_path.name}">
            </div>
        """
    html_content += """
        </div>
    </body>
    </html>
    """
    with open(output_html, "w") as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description="Visualize SLAM Pseudo-Labels")
    parser.add_argument("--data_dir", type=str, default="data/output/sample_slam", help="Path to slam dir")
    parser.add_argument("--h5_path", type=str, default=None, help="Path to pseudo_labels_slam.h5")
    parser.add_argument("--max_images", type=int, default=100, help="Max images to visualize")
    parser.add_argument("--score_thresh", type=float, default=0.01, help="Min score for display")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir")
    args = parser.parse_args()
    
    dataset_dir = Path(args.data_dir)
    rgb_dir = dataset_dir / "images/rgb"
    
    h5_path = args.h5_path if args.h5_path else dataset_dir / "exports/pseudo_labels_slam.h5"
    h5_path = Path(h5_path)
    
    out_dir = args.output_dir if args.output_dir else dataset_dir / "visualizations/kpt_labels"
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    if not h5_path.exists():
        logger.error(f"H5 file not found: {h5_path}")
        return
        
    with h5py.File(h5_path, "r") as f:
        image_names = list(f.keys())
        
    if args.max_images > 0:
        np.random.seed(42)
        np.random.shuffle(image_names)
        image_names = image_names[:args.max_images]
        
    logger.info(f"Visualizing {len(image_names)} images to {out_dir}...")
    saved_images = []
    
    with h5py.File(h5_path, "r") as f:
        for name in tqdm(image_names):
            rgb_path = rgb_dir / name
            if not rgb_path.exists():
                logger.warning(f"Image {name} not found in {rgb_dir}, skipping.")
                continue
                
            img = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
            kpts = f[name]["keypoints"][:]
            scores = f[name]["keypoint_scores"][:]
            
            mask = scores > args.score_thresh
            kpts = kpts[mask]
            scores = scores[mask]
            
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
            ax.imshow(img, cmap="gray")
            if len(kpts) > 0:
                sc = ax.scatter(kpts[:, 0], kpts[:, 1], c=scores, cmap="plasma", s=15, alpha=0.8, edgecolors="none")
                plt.colorbar(sc, ax=ax, label="Score")
            
            ax.set_title(f"{name} ({len(kpts)} keypoints)")
            ax.axis("off")
            
            out_path = out_dir / name
            plt.savefig(out_path, bbox_inches="tight", facecolor="#121212")
            plt.close(fig)
            saved_images.append(out_path)
            
    if saved_images:
        html_path = out_dir / "index.html"
        generate_html_grid(saved_images, html_path)
        logger.info(f"Generated HTML dashboard at {html_path}")

if __name__ == "__main__":
    main()
