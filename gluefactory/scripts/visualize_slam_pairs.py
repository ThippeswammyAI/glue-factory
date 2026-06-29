#!/usr/bin/env python3
"""
Visualize SLAM image pairs with SuperPoint keypoints from an H5 feature cache.

For each pair in a pairs_train.txt / pairs_val.txt file, renders the two images
side-by-side with detected keypoints overlaid (coloured by score) and saves a
PNG. Also generates a browsable HTML index of all saved images.

Usage:
    python -m gluefactory.scripts.visualize_slam_pairs \\
        --data_dir    data/output/slam \\
        --pairs_file  pairs_train.txt \\
        --h5_features data/output/slam/exports/sp_features_slam.h5 \\
        --max_pairs   100 \\
        --score_thresh 0.01
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

def generate_html_dashboard(image_paths, output_html):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>SLAM Pairs Visualization</title>
        <style>
            body { font-family: sans-serif; background-color: #121212; color: #ffffff; padding: 20px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(800px, 1fr)); gap: 20px; }
            .item { background: #1e1e1e; padding: 15px; border-radius: 8px; text-align: center; }
            img { max-width: 100%; height: auto; border-radius: 4px; }
            h2 { color: #bb86fc; font-size: 1.2em; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <h1>SLAM Pairs Visualization</h1>
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
    parser = argparse.ArgumentParser(description="Visualize SLAM Pairs")
    parser.add_argument("--data_dir", type=str, default="data/output/sample_slam", help="Path to slam dir")
    parser.add_argument("--pairs_file", type=str, default="pairs_train.txt", help="Pairs file name")
    parser.add_argument("--h5_features", type=str, default=None, help="Path to features h5")
    parser.add_argument("--max_pairs", type=int, default=50, help="Max pairs to visualize")
    parser.add_argument("--score_thresh", type=float, default=0.01, help="Min score")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir")
    args = parser.parse_args()
    
    dataset_dir = Path(args.data_dir)
    pairs_path = dataset_dir / args.pairs_file
    h5_path = args.h5_features if args.h5_features else dataset_dir / "exports/sp_features_slam.h5"
    h5_path = Path(h5_path)
    
    out_dir = args.output_dir if args.output_dir else dataset_dir / "visualizations/pair_matches"
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    if not pairs_path.exists():
        logger.error(f"Pairs file not found: {pairs_path}")
        return
        
    pairs = []
    with open(pairs_path, "r") as f:
        for line in f:
            p1, p2 = line.strip().split()
            pairs.append((p1, p2))
            
    if not pairs:
        logger.warning("No pairs found.")
        return
        
    if args.max_pairs > 0 and len(pairs) > args.max_pairs:
        np.random.seed(42)
        np.random.shuffle(pairs)
        pairs = pairs[:args.max_pairs]
        
    logger.info(f"Visualizing {len(pairs)} pairs to {out_dir}...")
    saved_images = []
    
    f_h5 = h5py.File(h5_path, "r") if h5_path.exists() else None
    if f_h5 is None:
        logger.warning(f"Could not load H5 features at {h5_path}. Only drawing images.")
    
    for i, (p1, p2) in enumerate(tqdm(pairs)):
        n1 = Path(p1).name
        n2 = Path(p2).name
        
        rgb1_path = dataset_dir / "images" / p1
        rgb2_path = dataset_dir / "images" / p2
        
        img1 = cv2.imread(str(rgb1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(rgb2_path), cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            continue
            
        fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
        ax[0].imshow(img1, cmap="gray")
        ax[1].imshow(img2, cmap="gray")
        
        if f_h5 is not None and n1 in f_h5 and n2 in f_h5:
            kpts1 = f_h5[n1]["keypoints"][:]
            scores1 = f_h5[n1]["keypoint_scores"][:]
            mask1 = scores1 > args.score_thresh
            kpts1 = kpts1[mask1]
            scores1 = scores1[mask1]
            
            if len(kpts1) > 0:
                ax[0].scatter(kpts1[:, 0], kpts1[:, 1], c=scores1, cmap="plasma", s=15, alpha=0.8, edgecolors="none")
                
            kpts2 = f_h5[n2]["keypoints"][:]
            scores2 = f_h5[n2]["keypoint_scores"][:]
            mask2 = scores2 > args.score_thresh
            kpts2 = kpts2[mask2]
            scores2 = scores2[mask2]
            
            if len(kpts2) > 0:
                ax[1].scatter(kpts2[:, 0], kpts2[:, 1], c=scores2, cmap="plasma", s=15, alpha=0.8, edgecolors="none")
        
        ax[0].axis("off")
        ax[1].axis("off")
        ax[0].set_title(n1)
        ax[1].set_title(n2)
        
        out_name = f"{Path(n1).stem}_{Path(n2).stem}.png"
        out_path = out_dir / out_name
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", facecolor="#121212")
        plt.close(fig)
        saved_images.append(out_path)
        
    if f_h5 is not None:
        f_h5.close()
        
    if saved_images:
        html_path = out_dir / "index.html"
        generate_html_dashboard(saved_images, html_path)
        logger.info(f"Generated HTML dashboard at {html_path}")

if __name__ == "__main__":
    main()
