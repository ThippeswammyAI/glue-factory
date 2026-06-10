import os
import argparse
import logging
import shutil
from pathlib import Path
import json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from gluefactory.utils.experiments import load_experiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("superglue_inference")

# Default paths
DEFAULT_SUPERGLUE_CKPT = "/home/thippe/workspaces/AiMl/glue-factory/outputs/training/superglue_slam_run/checkpoint_best.tar"
DEFAULT_SUPERPOINT_CKPT = "outputs/training/superpoint_custom_run_0_force_true/checkpoint_best.tar"
DEFAULT_INPUT_DIR = "data/output/sample_data/images/reflectivity"
DEFAULT_OUTPUT_DIR = "data/output/sample_data/visualizations/superglue_inference"

def parse_args():
    parser = argparse.ArgumentParser(description="Test SuperGlue model inference on image pairs.")
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Path to an image directory, a single image pair (comma-separated), or a folder containing image folders.",
    )
    parser.add_argument(
        "--checkpoint_superglue",
        type=str,
        default=DEFAULT_SUPERGLUE_CKPT,
        help="Path to the trained SuperGlue checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_superpoint",
        type=str,
        default=DEFAULT_SUPERPOINT_CKPT,
        help="Path to the trained SuperPoint checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the visualizations and HTML dashboard.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=640,
        help="Resize image max dimension to this value. Set to 0 to keep original size (multiples of 8).",
    )
    parser.add_argument(
        "--nms_radius",
        type=int,
        default=3,
        help="NMS radius for SuperPoint keypoint extraction.",
    )
    parser.add_argument(
        "--max_num_keypoints",
        type=int,
        default=512,
        help="Maximum number of keypoints to extract.",
    )
    parser.add_argument(
        "--detection_threshold",
        type=float,
        default=0.005,
        help="SuperPoint detection score threshold.",
    )
    parser.add_argument(
        "--filter_threshold",
        type=float,
        default=0.01,
        help="SuperGlue match confidence filtering threshold.",
    )
    return parser.parse_args()

def load_image(path, resize_max=0):
    """Load image as BGR and grayscale, resizes, and ensures multiples of 8."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at {path}")
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

def get_image_pairs(input_path):
    """Finds image pairs to match based on input path."""
    path = Path(input_path)
    pairs = []
    
    if "," in input_path:
        parts = [Path(p.strip()) for p in input_path.split(",")]
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
            return pairs
            
    if path.is_file():
        logger.error(f"Input path {input_path} is a file, but an image pair or directory is expected.")
        return pairs
        
    if path.is_dir():
        # Check if contains subdirectories with images (like lidar nearir, reflectivity, etc.)
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        img_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.PNG", "*.JPG", "*.JPEG"]
        
        # If there are subdirectories (excluding outputs), search inside them
        target_dirs = subdirs if len(subdirs) > 0 and path.name != "outputs" else [path]
        
        for t_dir in target_dirs:
            if t_dir.name == "outputs":
                continue
            img_files = []
            for ext in img_extensions:
                img_files.extend(list(t_dir.glob(ext)))
            img_files = sorted(list(set(img_files)))
            
            if len(img_files) >= 2:
                # If there are exactly two files, match them
                if len(img_files) == 2:
                    pairs.append((img_files[0], img_files[1]))
                else:
                    # Match consecutive pairs
                    for i in range(len(img_files) - 1):
                        pairs.append((img_files[i], img_files[i+1]))
                        
    return pairs

def plot_matches(img0, img1, kpts0, kpts1, matches, mscores, output_path, title="SuperGlue Matches"):
    """Plot static matches side-by-side."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    
    # Create side-by-side canvas
    canvas_h = max(h0, h1)
    canvas_w = w0 + w1
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0+w1] = img1
    
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("#0b0c10")
    ax.set_facecolor("#0b0c10")
    ax.axis("off")
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    
    # Draw keypoints
    ax.scatter(kpts0[:, 0], kpts0[:, 1], c="#06b6d4", s=10, edgecolors="none", alpha=0.8, label="View 0 Keypoints")
    ax.scatter(kpts1[:, 0] + w0, kpts1[:, 1], c="#d946ef", s=10, edgecolors="none", alpha=0.8, label="View 1 Keypoints")
    
    # Draw matched lines
    count = 0
    for idx0, idx1 in enumerate(matches):
        if idx1 == -1:
            continue
        p0 = kpts0[idx0]
        p1 = kpts1[idx1]
        score = mscores[idx0]
        
        # Color line based on match confidence
        color = plt.cm.plasma(score)
        ax.plot([p0[0], p1[0] + w0], [p0[1], p1[1]], color=color, linewidth=1.2, alpha=0.85)
        count += 1
        
    ax.set_title(f"{title}\n{count} matches found", color="#66fcf1", fontsize=16, weight="bold", pad=12)
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

def main():
    args = parse_args()
    
    # 1. Verify Checkpoints
    sg_ckpt = Path(args.checkpoint_superglue)
    sp_ckpt = Path(args.checkpoint_superpoint)
    
    if not sg_ckpt.exists():
        logger.error(f"SuperGlue checkpoint not found at {sg_ckpt}!")
        return
    if not sp_ckpt.exists():
        logger.error(f"SuperPoint checkpoint not found at {sp_ckpt}!")
        return
        
    # 2. Find Image Pairs
    pairs = get_image_pairs(args.input)
    if not pairs:
        logger.error(f"No image pairs found in input: {args.input}")
        return
        
    logger.info(f"Found {len(pairs)} image pairs to process.")
    
    # 3. Load Joint Model (SuperPoint + SuperGlue)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading models on device: {device}")
    
    conf = {
        "extractor": {
            "name": "superpoint_open",
            "weights": str(sp_ckpt),
            "nms_radius": args.nms_radius,
            "max_num_keypoints": args.max_num_keypoints,
            "detection_threshold": args.detection_threshold,
            "remove_borders": 4,
            "trainable": False,
        },
        "matcher": {
            "name": "gluefactory_nonfree.superglue",
            "filter_threshold": args.filter_threshold,
        }
    }
    
    try:
        model = load_experiment(str(sg_ckpt), conf).to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load joint model: {e}")
        return
        
    # 4. Prepare Output Directories
    out_dir = Path(args.output_dir)
    images_out_dir = out_dir / "images"
    plots_out_dir = out_dir / "plots"
    
    images_out_dir.mkdir(parents=True, exist_ok=True)
    plots_out_dir.mkdir(parents=True, exist_ok=True)
    
    dashboard_data = []
    
    # 5. Run Inference on Pairs
    for idx, (p0, p1) in enumerate(tqdm(pairs, desc="SuperGlue Matching Inference")):
        try:
            # Load images
            img0_bgr, img0_gray = load_image(p0, args.resize)
            img1_bgr, img1_gray = load_image(p1, args.resize)
            
            # Save copied images to output images directory
            img0_name = f"pair_{idx}_view0.png"
            img1_name = f"pair_{idx}_view1.png"
            cv2.imwrite(str(images_out_dir / img0_name), img0_bgr)
            cv2.imwrite(str(images_out_dir / img1_name), img1_bgr)
            
            # Prepare PyTorch Tensors
            t0 = torch.from_numpy(img0_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            t1 = torch.from_numpy(img1_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            
            data = {
                "view0": {"image": t0},
                "view1": {"image": t1}
            }
            
            # Run model
            with torch.no_grad():
                pred = model(data)
                
            # Parse outputs
            kpts0 = pred["keypoints0"][0].cpu().numpy()
            kpts1 = pred["keypoints1"][0].cpu().numpy()
            scores0 = pred["keypoint_scores0"][0].cpu().numpy()
            scores1 = pred["keypoint_scores1"][0].cpu().numpy()
            matches0 = pred["matches0"][0].cpu().numpy()
            mscores0 = pred["matching_scores0"][0].cpu().numpy()
            
            # Calculate metrics
            valid_mask = (matches0 != -1) & (mscores0 >= args.filter_threshold)
            num_matches = int(valid_mask.sum())
            total_kpts0 = len(kpts0)
            total_kpts1 = len(kpts1)
            match_ratio = num_matches / max(1, min(total_kpts0, total_kpts1))
            avg_mscore = float(mscores0[valid_mask].mean()) if num_matches > 0 else 0.0
            
            logger.info(
                f"Pair {idx}: {p0.name} & {p1.name} -> "
                f"Keypoints: ({total_kpts0} / {total_kpts1}), "
                f"Matches: {num_matches} (Ratio: {match_ratio:.1%}, Avg Score: {avg_mscore:.2f})"
            )
            
            # Save static matplotlib plot
            plot_name = f"pair_{idx}_matches.png"
            plot_path = plots_out_dir / plot_name
            plot_matches(
                img0_bgr,
                img1_bgr,
                kpts0,
                kpts1,
                matches0,
                mscores0,
                plot_path,
                title=f"Matches: {p0.name} ↔ {p1.name}"
            )
            
            # Compile dashboard data object
            # Matches array format: [[idx0, idx1, score], ...]
            matches_list = []
            for i0, i1 in enumerate(matches0):
                if i1 != -1:
                    matches_list.append([int(i0), int(i1), float(mscores0[i0])])
                    
            pair_info = {
                "idx": idx,
                "name0": p0.name,
                "name1": p1.name,
                "image0_url": f"images/{img0_name}",
                "image1_url": f"images/{img1_name}",
                "plot_url": f"plots/{plot_name}",
                "keypoints0": kpts0.tolist(),
                "keypoints1": kpts1.tolist(),
                "scores0": scores0.tolist(),
                "scores1": scores1.tolist(),
                "matches": matches_list,
                "metrics": {
                    "total_kpts0": total_kpts0,
                    "total_kpts1": total_kpts1,
                    "num_matches": num_matches,
                    "match_ratio": match_ratio,
                    "avg_mscore": avg_mscore
                },
                "image_width": img0_bgr.shape[1],
                "image_height": img0_bgr.shape[0]
            }
            
            dashboard_data.append(pair_info)
            
        except Exception as e:
            logger.error(f"Error processing pair {idx} ({p0.name} <-> {p1.name}): {e}", exc_info=True)
            continue
            
    # 6. Write JS data file
    js_data_path = out_dir / "matches_data.js"
    with open(js_data_path, "w") as f:
        f.write("const MATCHES_DATA = ")
        json.dump(dashboard_data, f, indent=2)
        f.write(";\n")
    logger.info(f"Dashboard matches data saved to {js_data_path}")
    
    # 7. Write index.html dashboard file
    html_path = out_dir / "index.html"
    html_content = get_html_template()
    with open(html_path, "w") as f:
        f.write(html_content)
    logger.info(f"Interactive dashboard generated successfully at {html_path}")
    
    print("\n" + "="*80)
    print("INFERENCE AND VISUALIZATION COMPLETE!")
    print(f"Total processed pairs: {len(dashboard_data)}")
    print(f"Results directory: {out_dir.resolve()}")
    print("To view the interactive dashboard, open the following file in your browser:")
    print(f"file://{html_path.resolve()}")
    print("="*80 + "\n")

def get_html_template():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SuperGlue Custom Inference Visualizer</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --bg-card: rgba(15, 23, 42, 0.7);
            --border-color: rgba(6, 182, 212, 0.15);
            --border-glow: rgba(6, 182, 212, 0.3);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-cyan: #06b6d4;
            --accent-purple: #d946ef;
            --accent-green: #10b981;
            --accent-orange: #f97316;
            --accent-red: #ef4444;
            --font-main: 'Outfit', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-dark);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
            display: flex;
        }

        /* Sidebar Styling */
        .sidebar {
            width: 380px;
            background-color: rgba(9, 13, 22, 0.95);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            height: 100%;
            flex-shrink: 0;
            backdrop-filter: blur(20px);
            z-index: 10;
        }

        .sidebar-header {
            padding: 24px;
            border-bottom: 1px solid var(--border-color);
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(217, 70, 239, 0.1) 100%);
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }

        .sidebar-header h1 {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 6px;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .sidebar-header p {
            font-size: 13px;
            color: var(--text-secondary);
        }

        .pair-list-title {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            padding: 16px 24px 8px 24px;
        }

        .pair-list {
            flex-grow: 1;
            overflow-y: auto;
            padding: 0 16px 24px 16px;
        }

        .pair-list::-webkit-scrollbar {
            width: 6px;
        }

        .pair-list::-webkit-scrollbar-track {
            background: transparent;
        }

        .pair-list::-webkit-scrollbar-thumb {
            background: rgba(6, 182, 212, 0.2);
            border-radius: 3px;
        }

        .pair-list::-webkit-scrollbar-thumb:hover {
            background: rgba(6, 182, 212, 0.4);
        }

        .pair-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .pair-item:hover {
            background: rgba(6, 182, 212, 0.04);
            border-color: var(--border-glow);
            transform: translateY(-2px);
        }

        .pair-item.active {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(217, 70, 239, 0.1) 100%);
            border-color: rgba(6, 182, 212, 0.5);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        .pair-item.active::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            width: 4px;
            background: linear-gradient(to bottom, var(--accent-cyan), var(--accent-purple));
        }

        .pair-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }

        .pair-idx {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--accent-cyan);
            font-weight: 500;
        }

        .badge {
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            padding: 2px 8px;
            border-radius: 9999px;
            letter-spacing: 0.5px;
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
        }

        .pair-name {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 6px;
        }

        .pair-stats-row {
            display: flex;
            gap: 12px;
            font-size: 11px;
            color: var(--text-secondary);
        }

        .pair-stat span.val {
            color: var(--text-primary);
            font-family: var(--font-mono);
            font-weight: 500;
        }

        /* Main Workspace Styling */
        .workspace {
            flex-grow: 1;
            height: 100%;
            overflow-y: auto;
            padding: 32px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-title h2 {
            font-size: 22px;
            font-weight: 600;
            letter-spacing: -0.5px;
            margin-bottom: 4px;
        }

        .header-title p {
            font-size: 13px;
            color: var(--text-secondary);
            font-family: var(--font-mono);
        }

        /* Dashboard Overview Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 16px 20px;
            backdrop-filter: blur(12px);
            display: flex;
            flex-direction: column;
            gap: 4px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }

        .stat-card::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            height: 3px;
            width: 100%;
            background: linear-gradient(to right, var(--accent-cyan), var(--accent-purple));
            opacity: 0.5;
        }

        .stat-card .label {
            font-size: 12px;
            color: var(--text-secondary);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .stat-card .value {
            font-size: 24px;
            font-weight: 700;
            font-family: var(--font-mono);
            color: var(--text-primary);
        }

        /* Control Panel */
        .control-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 18px 24px;
            backdrop-filter: blur(12px);
            display: flex;
            flex-wrap: wrap;
            gap: 32px;
            align-items: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .control-group label {
            font-size: 11px;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.75px;
        }

        .slider-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .slider-container input[type="range"] {
            -webkit-appearance: none;
            width: 150px;
            height: 5px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.1);
            outline: none;
        }

        .slider-container input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: var(--accent-cyan);
            box-shadow: 0 0 10px var(--accent-cyan);
            cursor: pointer;
            transition: transform 0.1s, background-color 0.2s;
        }

        .slider-container input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.25);
            background-color: var(--text-primary);
        }

        .slider-val {
            font-family: var(--font-mono);
            font-size: 13px;
            min-width: 36px;
            color: var(--accent-cyan);
            font-weight: 500;
        }

        .toggle-group {
            display: flex;
            gap: 8px;
        }

        .btn-toggle {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-secondary);
            padding: 8px 14px;
            border-radius: 10px;
            font-family: var(--font-main);
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .btn-toggle:hover {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-primary);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .btn-toggle.active {
            background: rgba(6, 182, 212, 0.12);
            border-color: var(--accent-cyan);
            color: var(--text-primary);
            box-shadow: 0 0 10px rgba(6, 182, 212, 0.2);
        }

        .btn-toggle.active::before {
            content: '●';
            color: var(--accent-cyan);
            font-size: 8px;
        }

        /* Visualization Display Box */
        .visualization-box {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 24px;
            backdrop-filter: blur(12px);
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            min-height: 480px;
            box-shadow: inset 0 0 40px rgba(0,0,0,0.5), 0 12px 40px rgba(0,0,0,0.3);
            overflow: hidden;
            cursor: grab;
        }

        .visualization-box:active {
            cursor: grabbing;
        }

        .image-pair-wrapper {
            position: relative;
            display: flex;
            flex-direction: column; /* Stack view0 on top, view1 on bottom! */
            gap: 24px;
            align-items: center;
            justify-content: center;
            padding: 10px;
            border-radius: 12px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255,255,255,0.03);
            max-width: 95%;
            transform-origin: center center;
            transition: transform 0.05s ease-out;
            user-select: none;
        }

        .image-container {
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
            transition: border-color 0.3s;
        }

        .image-container img {
            display: block;
            max-width: 100%;
            max-height: 38vh; /* Scale images nicely for vertical stack */
            height: auto;
            user-select: none;
            -webkit-user-drag: none;
        }

        .view-label {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(9, 13, 22, 0.85);
            backdrop-filter: blur(4px);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            z-index: 5;
            pointer-events: none;
        }

        #matches-canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: auto; /* Enable mouse events over the canvas! */
            z-index: 8;
        }

        /* Hover/Select Detail Popup */
        .info-popup {
            position: absolute;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(9, 13, 22, 0.95);
            border: 1px solid var(--accent-cyan);
            border-radius: 12px;
            padding: 12px 24px;
            display: flex;
            gap: 24px;
            backdrop-filter: blur(16px);
            z-index: 15;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7), 0 0 20px rgba(6, 182, 212, 0.15);
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s ease, transform 0.2s ease;
        }

        .info-popup.show {
            opacity: 1;
            transform: translate(-50%, -5px);
        }

        .info-item {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .info-item .lbl {
            font-size: 10px;
            color: var(--text-secondary);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .info-item .val {
            font-size: 14px;
            font-family: var(--font-mono);
            font-weight: 600;
        }

        .info-item.accent-cyan .val { color: var(--accent-cyan); }
        .info-item.accent-purple .val { color: var(--accent-purple); }
        .info-item.accent-green .val { color: var(--accent-green); }
        .info-item.accent-orange .val { color: var(--accent-orange); }

        .legend {
            position: absolute;
            top: 24px;
            right: 24px;
            display: flex;
            gap: 16px;
            background: rgba(0,0,0,0.4);
            padding: 8px 16px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            font-size: 12px;
            z-index: 9;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .legend-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .legend-line {
            width: 16px;
            height: 2px;
        }

        /* Overlay loader */
        .loader {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--bg-dark);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 100;
            font-size: 18px;
            font-weight: 500;
            letter-spacing: 1px;
            color: var(--accent-cyan);
            transition: opacity 0.5s ease;
        }
    </style>
</head>
<body>
    <!-- Loader -->
    <div id="loader" class="loader">Loading Dashboard Data...</div>

    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-header">
            <h1>SuperGlue Visualizer</h1>
            <p>Custom Training Model Inference</p>
        </div>
        
        <div class="pair-list-title">Image Pairs</div>
        <div class="pair-list" id="pair-list">
            <!-- Dynamic pairs will be rendered here -->
        </div>
    </div>

    <!-- Main Workspace -->
    <div class="workspace">
        <div class="header">
            <div class="header-title">
                <h2 id="active-filename">Select an image pair</h2>
                <p id="active-original-image">Matching view0 ↔ view1</p>
            </div>
        </div>

        <!-- Statistics Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <span class="label">Total Keypoints (V0 / V1)</span>
                <span class="value" id="stat-kpts">- / -</span>
            </div>
            <div class="stat-card">
                <span class="label">Filtered Matches</span>
                <span class="value" id="stat-matches">-</span>
            </div>
            <div class="stat-card">
                <span class="label">Match Percentage</span>
                <span class="value" id="stat-pct">-</span>
            </div>
            <div class="stat-card">
                <span class="label">Avg Match Score</span>
                <span class="value" id="stat-score">-</span>
            </div>
        </div>

        <!-- Controls -->
        <div class="control-panel">
            <div class="control-group">
                <label>Match Confidence Threshold</label>
                <div class="slider-container">
                    <input type="range" id="score-thresh" min="0.0" max="1.0" step="0.01" value="0.01" oninput="updateScoreThresh(this.value)">
                    <span class="slider-val" id="score-thresh-val">0.01</span>
                </div>
            </div>
            
            <div class="control-group">
                <label>Keypoint Threshold</label>
                <div class="slider-container">
                    <input type="range" id="kpt-thresh" min="0.0" max="1" step="0.01" value="0.10" oninput="updateKptThresh(this.value)">
                    <span class="slider-val" id="kpt-thresh-val">0.10</span>
                </div>
            </div>

            <div class="control-group">
                <label>Display Layers</label>
                <div class="toggle-group">
                    <button class="btn-toggle active" id="btn-show-kpts" onclick="toggleLayer('kpts')">Keypoints</button>
                    <button class="btn-toggle active" id="btn-show-matches" onclick="toggleLayer('matches')">Match Lines</button>
                    <button class="btn-toggle" id="btn-show-selected-only" onclick="toggleSelectedOnly()">Selected Match Only</button>
                </div>
            </div>

            <div class="control-group">
                <label>Zoom & Pan</label>
                <div class="toggle-group">
                    <button class="btn-toggle" onclick="zoomIn()">Zoom In</button>
                    <button class="btn-toggle" onclick="zoomOut()">Zoom Out</button>
                    <button class="btn-toggle" onclick="resetZoom()">Reset</button>
                </div>
            </div>

            <div class="control-group" id="selected-info" style="display: none; align-items: flex-start;">
                <label style="color: var(--accent-cyan); margin-bottom: 0;">Selected Match</label>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span id="selected-match-details" style="font-family: var(--font-mono); font-size: 12px; color: var(--accent-cyan);">-</span>
                    <button class="btn-toggle" style="padding: 2px 8px; font-size: 11px;" onclick="clearSelection()">Clear</button>
                </div>
            </div>
        </div>

        <!-- Visualization Box -->
        <div class="visualization-box">
            <!-- Legend -->
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-dot" style="background-color: var(--accent-cyan);"></div>
                    <span>V0 Pt</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background-color: var(--accent-purple);"></div>
                    <span>V1 Pt</span>
                </div>
                <div class="legend-item">
                    <div class="legend-line" style="background: linear-gradient(to right, var(--accent-cyan), var(--accent-purple));"></div>
                    <span>Match Line</span>
                </div>
            </div>

            <!-- Image wrapper -->
            <div class="image-pair-wrapper" id="image-pair-wrapper">
                <div class="image-container" id="img-container-0">
                    <span class="view-label">View 0</span>
                    <img id="img-0" src="" alt="View 0">
                </div>
                <div class="image-container" id="img-container-1">
                    <span class="view-label">View 1</span>
                    <img id="img-1" src="" alt="View 1">
                </div>
                <canvas id="matches-canvas"></canvas>
            </div>

            <!-- Detail overlay -->
            <div class="info-popup" id="info-popup">
                <div class="info-item accent-cyan">
                    <span class="lbl">V0 Coordinate</span>
                    <span class="val" id="pop-pt0">-</span>
                </div>
                <div class="info-item accent-cyan">
                    <span class="lbl">V0 Score</span>
                    <span class="val" id="pop-score0">-</span>
                </div>
                <div class="info-item accent-purple">
                    <span class="lbl">V1 Coordinate</span>
                    <span class="val" id="pop-pt1">-</span>
                </div>
                <div class="info-item accent-purple">
                    <span class="lbl">V1 Score</span>
                    <span class="val" id="pop-score1">-</span>
                </div>
                <div class="info-item accent-green" id="pop-match-container">
                    <span class="lbl">Match Confidence</span>
                    <span class="val" id="pop-status">-</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Load data script -->
    <script src="matches_data.js"></script>
    <script>
        let activeIdx = 0;
        let activeData = null;
        let scoreThresh = 0.01;
        let kptThresh = 0.00;
        
        let settings = {
            showKpts: true,
            showMatches: true
        };

        let hoveredPoint = null; // { view: 0/1, index: idx }
        let selectedPoint = null; // { view: 0/1, index: idx }
        let showSelectedOnly = false;

        // Zoom & Pan variables
        let zoom = 1.0;
        let panX = 0;
        let panY = 0;
        let isMouseDown = false;
        let hasDragged = false;
        let dragStartX = 0;
        let dragStartY = 0;
        let initialPanX = 0;
        let initialPanY = 0;
        
        const loader = document.getElementById("loader");
        const pairList = document.getElementById("pair-list");
        const canvas = document.getElementById("matches-canvas");
        const ctx = canvas.getContext("2d");
        
        const img0 = document.getElementById("img-0");
        const img1 = document.getElementById("img-1");
        const container = document.getElementById("image-pair-wrapper");
        const vizBox = document.querySelector(".visualization-box");

        window.onload = function() {
            if (typeof MATCHES_DATA === 'undefined' || MATCHES_DATA.length === 0) {
                loader.innerText = "Error: No MATCHES_DATA loaded! Please run the inference script first.";
                return;
            }
            loader.style.opacity = 0;
            setTimeout(() => loader.style.display = "none", 500);
            
            buildPairList();
            selectPair(0);
            
            // Set up resize handler
            window.addEventListener('resize', draw);
            
            // Set up canvas event listeners for mouse interactions
            canvas.addEventListener('mousemove', handleMouseMove);
            canvas.addEventListener('mouseleave', handleMouseLeave);

            // Zoom & Pan Mouse Events on the visualization-box
            vizBox.addEventListener('mousedown', function(e) {
                if (e.button !== 0) return; // Only left click
                isMouseDown = true;
                hasDragged = false;
                dragStartX = e.clientX;
                dragStartY = e.clientY;
                initialPanX = panX;
                initialPanY = panY;
            });

            window.addEventListener('mousemove', function(e) {
                if (!isMouseDown) return;
                const dx = e.clientX - dragStartX;
                const dy = e.clientY - dragStartY;
                
                if (Math.hypot(dx, dy) > 3) {
                    hasDragged = true;
                    panX = initialPanX + dx;
                    panY = initialPanY + dy;
                    updateTransform();
                }
            });

            window.addEventListener('mouseup', function(e) {
                if (isMouseDown) {
                    isMouseDown = false;
                    if (!hasDragged) {
                        // Click event! check if inside canvas
                        const canvasRect = canvas.getBoundingClientRect();
                        if (e.clientX >= canvasRect.left && e.clientX <= canvasRect.right &&
                            e.clientY >= canvasRect.top && e.clientY <= canvasRect.bottom) {
                            // Compute unscaled canvas coordinate mapping
                            const x = (e.clientX - canvasRect.left) * (canvas.width / canvasRect.width);
                            const y = (e.clientY - canvasRect.top) * (canvas.height / canvasRect.height);
                            handleCanvasClick(x, y);
                        }
                    }
                }
            });

            vizBox.addEventListener('wheel', function(e) {
                e.preventDefault();
                const zoomSpeed = 0.05;
                if (e.deltaY < 0) {
                    zoom = Math.min(zoom + zoomSpeed, 5.0);
                } else {
                    zoom = Math.max(zoom - zoomSpeed, 0.4);
                }
                updateTransform();
            }, { passive: false });
        };

        function buildPairList() {
            pairList.innerHTML = "";
            MATCHES_DATA.forEach((pair, idx) => {
                const item = document.createElement("div");
                item.className = "pair-item" + (idx === 0 ? " active" : "");
                item.id = `pair-item-${idx}`;
                item.onclick = () => selectPair(idx);
                
                item.innerHTML = `
                    <div class="pair-meta">
                        <span class="pair-idx">PAIR #${idx}</span>
                        <span class="badge">${pair.metrics.num_matches} matches</span>
                    </div>
                    <div class="pair-name">${pair.name0} ↔ ${pair.name1}</div>
                    <div class="pair-stats-row">
                        <div class="pair-stat">Kpts: <span class="val">${pair.metrics.total_kpts0}/${pair.metrics.total_kpts1}</span></div>
                        <div class="pair-stat">Ratio: <span class="val">${Math.round(pair.metrics.match_ratio * 100)}%</span></div>
                    </div>
                `;
                pairList.appendChild(item);
            });
        }

        function selectPair(idx) {
            const prevActive = document.querySelector(".pair-item.active");
            if (prevActive) prevActive.classList.remove("active");
            
            const newActive = document.getElementById(`pair-item-${idx}`);
            if (newActive) newActive.classList.add("active");
            
            activeIdx = idx;
            activeData = MATCHES_DATA[idx];
            
            document.getElementById("active-filename").innerText = `${activeData.name0} ↔ ${activeData.name1}`;
            document.getElementById("active-original-image").innerText = `Image size: ${activeData.image_width}x${activeData.image_height}`;
            
            // Clear hover/selection states
            hoveredPoint = null;
            clearSelection();
            resetZoom();

            let loadedCount = 0;
            const onImageLoad = () => {
                loadedCount++;
                if (loadedCount === 2) {
                    resizeCanvas();
                    draw();
                }
            };
            
            img0.onload = onImageLoad;
            img1.onload = onImageLoad;
            
            img0.src = activeData.image0_url;
            img1.src = activeData.image1_url;
        }

        function resizeCanvas() {
            canvas.width = container.offsetWidth;
            canvas.height = container.offsetHeight;
        }

        function getCoordinateMapping() {
            const container0 = document.getElementById("img-container-0");
            const container1 = document.getElementById("img-container-1");
            
            return {
                offset0: { x: container0.offsetLeft, y: container0.offsetTop },
                offset1: { x: container1.offsetLeft, y: container1.offsetTop },
                scale0: { x: container0.offsetWidth / activeData.image_width, y: container0.offsetHeight / activeData.image_height },
                scale1: { x: container1.offsetWidth / activeData.image_width, y: container1.offsetHeight / activeData.image_height }
            };
        }

        function updateScoreThresh(val) {
            scoreThresh = parseFloat(val);
            document.getElementById("score-thresh-val").innerText = scoreThresh.toFixed(2);
            draw();
        }

        function updateKptThresh(val) {
            kptThresh = parseFloat(val);
            document.getElementById("kpt-thresh-val").innerText = kptThresh.toFixed(3);
            draw();
        }

        function toggleLayer(layer) {
            if (layer === 'kpts') {
                settings.showKpts = !settings.showKpts;
                document.getElementById("btn-show-kpts").classList.toggle("active", settings.showKpts);
            } else if (layer === 'matches') {
                settings.showMatches = !settings.showMatches;
                document.getElementById("btn-show-matches").classList.toggle("active", settings.showMatches);
            }
            draw();
        }

        function toggleSelectedOnly() {
            showSelectedOnly = !showSelectedOnly;
            document.getElementById("btn-show-selected-only").classList.toggle("active", showSelectedOnly);
            draw();
        }

        function updateTransform() {
            container.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
        }

        function zoomIn() {
            zoom = Math.min(zoom + 0.25, 5.0);
            updateTransform();
        }

        function zoomOut() {
            zoom = Math.max(zoom - 0.25, 0.4);
            updateTransform();
        }

        function resetZoom() {
            zoom = 1.0;
            panX = 0;
            panY = 0;
            updateTransform();
        }

        function clearSelection() {
            selectedPoint = null;
            document.getElementById("selected-info").style.display = "none";
            draw();
        }

        function drawMatchLine(m, isHighlighted, opacityOverride) {
            const map = getCoordinateMapping();
            const kpts0 = activeData.keypoints0;
            const kpts1 = activeData.keypoints1;
            
            const idx0 = m[0];
            const idx1 = m[1];
            const conf = m[2];
            
            const p0 = kpts0[idx0];
            const p1 = kpts1[idx1];
            
            const x0 = map.offset0.x + p0[0] * map.scale0.x;
            const y0 = map.offset0.y + p0[1] * map.scale0.y;
            const x1 = map.offset1.x + p1[0] * map.scale1.x;
            const y1 = map.offset1.y + p1[1] * map.scale1.y;
            
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            
            if (isHighlighted) {
                const gradient = ctx.createLinearGradient(x0, y0, x1, y1);
                gradient.addColorStop(0, "var(--accent-cyan)");
                gradient.addColorStop(1, "var(--accent-purple)");
                ctx.strokeStyle = gradient;
                ctx.lineWidth = 3.5;
                ctx.shadowColor = conf >= scoreThresh ? "var(--accent-green)" : "var(--accent-red)";
                ctx.shadowBlur = 8;
            } else {
                const opacity = opacityOverride !== undefined ? opacityOverride : (0.15 + conf * 0.55);
                const gradient = ctx.createLinearGradient(x0, y0, x1, y1);
                gradient.addColorStop(0, `rgba(6, 182, 212, ${opacity})`);
                gradient.addColorStop(1, `rgba(217, 70, 239, ${opacity})`);
                ctx.strokeStyle = gradient;
                ctx.lineWidth = 1 + conf * 1.5;
            }
            ctx.stroke();
            ctx.shadowBlur = 0; // Reset shadow
        }

        function draw() {
            if (!activeData || !img0.complete || !img1.complete) return;
            
            resizeCanvas();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const map = getCoordinateMapping();
            
            const kpts0 = activeData.keypoints0;
            const kpts1 = activeData.keypoints1;
            const scores0 = activeData.scores0;
            const scores1 = activeData.scores1;
            const matches = activeData.matches;
            
            // Calculate filtered matches count
            const filteredMatches = matches.filter(m => m[2] >= scoreThresh && scores0[m[0]] >= kptThresh && scores1[m[1]] >= kptThresh);
            
            // Update stats grid
            document.getElementById("stat-kpts").innerText = `${kpts0.length} / ${kpts1.length}`;
            document.getElementById("stat-matches").innerText = filteredMatches.length;
            const matchRate = (filteredMatches.length / Math.max(1, Math.min(kpts0.length, kpts1.length))) * 100;
            document.getElementById("stat-pct").innerText = `${Math.round(matchRate)}%`;
            
            const sumScore = filteredMatches.reduce((acc, m) => acc + m[2], 0);
            const avgScore = filteredMatches.length > 0 ? (sumScore / filteredMatches.length).toFixed(2) : "0.00";
            document.getElementById("stat-score").innerText = avgScore;
            
            const focusPoint = hoveredPoint || selectedPoint;
            
            // If a point or line is selected, make ALL OTHER keypoints and match lines invisible!
            if (selectedPoint) {
                const isView0 = selectedPoint.view === 0;
                const idx = selectedPoint.index;
                const match = matches.find(m => isView0 ? m[0] === idx : m[1] === idx);
                
                // Draw ONLY the selected keypoints and match line
                const pt = isView0 ? kpts0[idx] : kpts1[idx];
                const offset = isView0 ? map.offset0 : map.offset1;
                const scale = isView0 ? map.scale0 : map.scale1;
                const hx = offset.x + pt[0] * scale.x;
                const hy = offset.y + pt[1] * scale.y;
                
                ctx.beginPath();
                ctx.arc(hx, hy, 9, 0, 2 * Math.PI);
                ctx.strokeStyle = isView0 ? "var(--accent-cyan)" : "var(--accent-purple)";
                ctx.lineWidth = 3;
                ctx.stroke();
                
                ctx.beginPath();
                ctx.arc(hx, hy, 4, 0, 2 * Math.PI);
                ctx.fillStyle = isView0 ? "var(--accent-cyan)" : "var(--accent-purple)";
                ctx.fill();
                
                if (match && match[2] >= scoreThresh && scores0[match[0]] >= kptThresh && scores1[match[1]] >= kptThresh) {
                    const partnerIdx = isView0 ? match[1] : match[0];
                    const partnerPt = isView0 ? kpts1[partnerIdx] : kpts0[partnerIdx];
                    const partnerOffset = isView0 ? map.offset1 : map.offset0;
                    const partnerScale = isView0 ? map.scale1 : map.scale0;
                    const px = partnerOffset.x + partnerPt[0] * partnerScale.x;
                    const py = partnerOffset.y + partnerPt[1] * partnerScale.y;
                    
                    ctx.beginPath();
                    ctx.arc(px, py, 8, 0, 2 * Math.PI);
                    ctx.strokeStyle = isView0 ? "var(--accent-purple)" : "var(--accent-cyan)";
                    ctx.lineWidth = 3;
                    ctx.stroke();
                    
                    ctx.beginPath();
                    ctx.arc(px, py, 4, 0, 2 * Math.PI);
                    ctx.fillStyle = isView0 ? "var(--accent-purple)" : "var(--accent-cyan)";
                    ctx.fill();
                    
                    // Draw match line in bold, solid glowing style
                    drawMatchLine(match, true);
                    
                    showPopup(
                        isView0 ? pt : partnerPt,
                        isView0 ? scores0[idx] : scores0[partnerIdx],
                        isView0 ? partnerPt : pt,
                        isView0 ? scores1[partnerIdx] : scores1[idx],
                        match[2],
                        "Valid Match (Selected)"
                    );
                } else {
                    showPopup(
                        isView0 ? pt : null,
                        isView0 ? scores0[idx] : null,
                        isView0 ? null : pt,
                        isView0 ? null : scores1[idx],
                        null,
                        "Unmatched Keypoint (Selected)"
                    );
                }
            } else {
                // Normal view (no selection)
                
                // 1. Draw All Keypoints if enabled
                if (settings.showKpts) {
                    kpts0.forEach((pt, i) => {
                        if (scores0[i] < kptThresh) return;
                        const cx = map.offset0.x + pt[0] * map.scale0.x;
                        const cy = map.offset0.y + pt[1] * map.scale0.y;
                        
                        ctx.beginPath();
                        ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
                        ctx.fillStyle = "rgba(6, 182, 212, 0.4)";
                        ctx.fill();
                    });
                    
                    kpts1.forEach((pt, i) => {
                        if (scores1[i] < kptThresh) return;
                        const cx = map.offset1.x + pt[0] * map.scale1.x;
                        const cy = map.offset1.y + pt[1] * map.scale1.y;
                        
                        ctx.beginPath();
                        ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
                        ctx.fillStyle = "rgba(217, 70, 239, 0.4)";
                        ctx.fill();
                    });
                }
                
                // 2. Draw Match Lines if enabled
                if (settings.showMatches) {
                    if (showSelectedOnly) {
                        if (focusPoint) {
                            const isView0 = focusPoint.view === 0;
                            const idx = focusPoint.index;
                            const match = matches.find(m => isView0 ? m[0] === idx : m[1] === idx);
                            if (match && match[2] >= scoreThresh && scores0[match[0]] >= kptThresh && scores1[match[1]] >= kptThresh) {
                                drawMatchLine(match, true);
                            }
                        }
                    } else {
                        if (focusPoint) {
                            // Draw other lines very faintly
                            filteredMatches.forEach(m => {
                                const isFocus = (focusPoint.view === 0 && m[0] === focusPoint.index) || 
                                                (focusPoint.view === 1 && m[1] === focusPoint.index);
                                if (!isFocus) {
                                    drawMatchLine(m, false, 0.04);
                                }
                            });
                            // Highlight focus match line
                            const isView0 = focusPoint.view === 0;
                            const idx = focusPoint.index;
                            const match = matches.find(m => isView0 ? m[0] === idx : m[1] === idx);
                            if (match && match[2] >= scoreThresh && scores0[match[0]] >= kptThresh && scores1[match[1]] >= kptThresh) {
                                drawMatchLine(match, true);
                            }
                        } else {
                            // Draw all normally
                            filteredMatches.forEach(m => {
                                drawMatchLine(m, false);
                            });
                        }
                    }
                }
                
                // 3. Highlight hovered point circles
                if (hoveredPoint) {
                    const isView0 = hoveredPoint.view === 0;
                    const idx = hoveredPoint.index;
                    const pt = isView0 ? kpts0[idx] : kpts1[idx];
                    const offset = isView0 ? map.offset0 : map.offset1;
                    const scale = isView0 ? map.scale0 : map.scale1;
                    const hx = offset.x + pt[0] * scale.x;
                    const hy = offset.y + pt[1] * scale.y;
                    
                    ctx.beginPath();
                    ctx.arc(hx, hy, 8, 0, 2 * Math.PI);
                    ctx.strokeStyle = isView0 ? "var(--accent-cyan)" : "var(--accent-purple)";
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    ctx.beginPath();
                    ctx.arc(hx, hy, 3, 0, 2 * Math.PI);
                    ctx.fillStyle = isView0 ? "var(--accent-cyan)" : "var(--accent-purple)";
                    ctx.fill();
                    
                    let match = matches.find(m => isView0 ? m[0] === idx : m[1] === idx);
                    if (match && match[2] >= scoreThresh && scores0[match[0]] >= kptThresh && scores1[match[1]] >= kptThresh) {
                        const partnerIdx = isView0 ? match[1] : match[0];
                        const partnerPt = isView0 ? kpts1[partnerIdx] : kpts0[partnerIdx];
                        const partnerOffset = isView0 ? map.offset1 : map.offset0;
                        const partnerScale = isView0 ? map.scale1 : map.scale0;
                        const px = partnerOffset.x + partnerPt[0] * partnerScale.x;
                        const py = partnerOffset.y + partnerPt[1] * partnerScale.y;
                        
                        ctx.beginPath();
                        ctx.arc(px, py, 7, 0, 2 * Math.PI);
                        ctx.strokeStyle = isView0 ? "var(--accent-purple)" : "var(--accent-cyan)";
                        ctx.lineWidth = 2;
                        ctx.stroke();
                        
                        ctx.beginPath();
                        ctx.arc(px, py, 3, 0, 2 * Math.PI);
                        ctx.fillStyle = isView0 ? "var(--accent-purple)" : "var(--accent-cyan)";
                        ctx.fill();
                    }
                    
                    if (match) {
                        const partnerIdx = isView0 ? match[1] : match[0];
                        const partnerPt = isView0 ? kpts1[partnerIdx] : kpts0[partnerIdx];
                        showPopup(
                            isView0 ? pt : partnerPt,
                            isView0 ? scores0[idx] : scores0[partnerIdx],
                            isView0 ? partnerPt : pt,
                            isView0 ? scores1[partnerIdx] : scores1[idx],
                            match[2],
                            match[2] >= scoreThresh ? "Valid Match" : "Below Threshold"
                        );
                    } else {
                        showPopup(
                            isView0 ? pt : null,
                            isView0 ? scores0[idx] : null,
                            isView0 ? null : pt,
                            isView0 ? null : scores1[idx],
                            null,
                            "Unmatched Keypoint"
                        );
                    }
                } else {
                    hidePopup();
                }
            }
        }

        function handleMouseMove(e) {
            if (!activeData || isMouseDown) return;
            
            const rect = canvas.getBoundingClientRect();
            const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
            const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
            
            const map = getCoordinateMapping();
            const container0 = document.getElementById("img-container-0");
            const container1 = document.getElementById("img-container-1");
            
            const inImg0 = mouseX >= map.offset0.x && mouseX <= map.offset0.x + container0.offsetWidth &&
                          mouseY >= map.offset0.y && mouseY <= map.offset0.y + container0.offsetHeight;
                           
            const inImg1 = mouseX >= map.offset1.x && mouseX <= map.offset1.x + container1.offsetWidth &&
                          mouseY >= map.offset1.y && mouseY <= map.offset1.y + container1.offsetHeight;
            
            let closestPt = null;
            let minDist = 15; // Search radius in pixels
            
            if (inImg0) {
                activeData.keypoints0.forEach((pt, idx) => {
                    if (activeData.scores0[idx] < kptThresh) return;
                    const px = map.offset0.x + pt[0] * map.scale0.x;
                    const py = map.offset0.y + pt[1] * map.scale0.y;
                    const dist = Math.hypot(mouseX - px, mouseY - py);
                    if (dist < minDist) {
                        minDist = dist;
                        closestPt = { view: 0, index: idx };
                    }
                });
            } else if (inImg1) {
                activeData.keypoints1.forEach((pt, idx) => {
                    if (activeData.scores1[idx] < kptThresh) return;
                    const px = map.offset1.x + pt[0] * map.scale1.x;
                    const py = map.offset1.y + pt[1] * map.scale1.y;
                    const dist = Math.hypot(mouseX - px, mouseY - py);
                    if (dist < minDist) {
                        minDist = dist;
                        closestPt = { view: 1, index: idx };
                    }
                });
            }
            
            if (closestPt) {
                if (!hoveredPoint || hoveredPoint.view !== closestPt.view || hoveredPoint.index !== closestPt.index) {
                    hoveredPoint = closestPt;
                    draw();
                }
            } else if (hoveredPoint) {
                hoveredPoint = null;
                draw();
            }
        }

        function handleMouseLeave() {
            if (hoveredPoint) {
                hoveredPoint = null;
                draw();
            }
        }

        function distToSegment(p, v, w) {
            const l2 = Math.pow(v.x - w.x, 2) + Math.pow(v.y - w.y, 2);
            if (l2 === 0) return Math.hypot(p.x - v.x, p.y - v.y);
            let t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
            t = Math.max(0, Math.min(1, t));
            return Math.hypot(p.x - (v.x + t * (w.x - v.x)), p.y - (v.y + t * (w.y - v.y)));
        }

        function handleCanvasClick(mouseX, mouseY) {
            if (!activeData) return;
            
            const map = getCoordinateMapping();
            const container0 = document.getElementById("img-container-0");
            const container1 = document.getElementById("img-container-1");
            
            const kpts0 = activeData.keypoints0;
            const kpts1 = activeData.keypoints1;
            const scores0 = activeData.scores0;
            const scores1 = activeData.scores1;
            const matches = activeData.matches;
            
            const filteredMatches = matches.filter(m => m[2] >= scoreThresh && scores0[m[0]] >= kptThresh && scores1[m[1]] >= kptThresh);
            
            const inImg0 = mouseX >= map.offset0.x && mouseX <= map.offset0.x + container0.offsetWidth &&
                          mouseY >= map.offset0.y && mouseY <= map.offset0.y + container0.offsetHeight;
                           
            const inImg1 = mouseX >= map.offset1.x && mouseX <= map.offset1.x + container1.offsetWidth &&
                          mouseY >= map.offset1.y && mouseY <= map.offset1.y + container1.offsetHeight;
            
            let closestPt = null;
            let minPtDist = 20; // Search radius in pixels
            
            if (inImg0) {
                activeData.keypoints0.forEach((pt, idx) => {
                    if (activeData.scores0[idx] < kptThresh) return;
                    const px = map.offset0.x + pt[0] * map.scale0.x;
                    const py = map.offset0.y + pt[1] * map.scale0.y;
                    const dist = Math.hypot(mouseX - px, mouseY - py);
                    if (dist < minPtDist) {
                        minPtDist = dist;
                        closestPt = { view: 0, index: idx };
                    }
                });
            } else if (inImg1) {
                activeData.keypoints1.forEach((pt, idx) => {
                    if (activeData.scores1[idx] < kptThresh) return;
                    const px = map.offset1.x + pt[0] * map.scale1.x;
                    const py = map.offset1.y + pt[1] * map.scale1.y;
                    const dist = Math.hypot(mouseX - px, mouseY - py);
                    if (dist < minPtDist) {
                        minPtDist = dist;
                        closestPt = { view: 1, index: idx };
                    }
                });
            }
            
            // Search for closest match line
            let closestMatch = null;
            let minMatchDist = 15;
            
            filteredMatches.forEach(m => {
                const p0 = kpts0[m[0]];
                const p1 = kpts1[m[1]];
                const x0 = map.offset0.x + p0[0] * map.scale0.x;
                const y0 = map.offset0.y + p0[1] * map.scale0.y;
                const x1 = map.offset1.x + p1[0] * map.scale1.x;
                const y1 = map.offset1.y + p1[1] * map.scale1.y;
                
                const dist = distToSegment(
                    { x: mouseX, y: mouseY },
                    { x: x0, y: y0 },
                    { x: x1, y: y1 }
                );
                
                if (dist < minMatchDist) {
                    minMatchDist = dist;
                    closestMatch = m;
                }
            });
            
            if (closestPt) {
                const idx = closestPt.index;
                const isV0 = closestPt.view === 0;
                const match = matches.find(m => isV0 ? m[0] === idx : m[1] === idx);
                
                if (selectedPoint && selectedPoint.view === closestPt.view && selectedPoint.index === closestPt.index) {
                    clearSelection();
                } else {
                    selectedPoint = closestPt;
                    updateSelectionUI(match, closestPt);
                }
            } else if (closestMatch) {
                const matchPt = { view: 0, index: closestMatch[0] };
                if (selectedPoint && selectedPoint.view === 0 && selectedPoint.index === closestMatch[0]) {
                    clearSelection();
                } else {
                    selectedPoint = matchPt;
                    updateSelectionUI(closestMatch, matchPt);
                }
            } else {
                clearSelection();
            }
            draw();
        }

        function updateSelectionUI(match, pt) {
            let detailsText = "";
            if (match) {
                detailsText = `V0 #${match[0]} ↔ V1 #${match[1]} (${(match[2]*100).toFixed(1)}% conf)`;
            } else {
                detailsText = `${pt.view === 0 ? 'View 0' : 'View 1'} #${pt.index} (unmatched)`;
            }
            document.getElementById("selected-match-details").innerText = detailsText;
            document.getElementById("selected-info").style.display = "inline-flex";
        }

        function showPopup(pt0, score0, pt1, score1, conf, status) {
            const popup = document.getElementById("info-popup");
            
            if (pt0) {
                document.getElementById("pop-pt0").innerText = `(${Math.round(pt0[0])}, ${Math.round(pt0[1])})`;
                document.getElementById("pop-score0").innerText = score0.toFixed(3);
            } else {
                document.getElementById("pop-pt0").innerText = "-";
                document.getElementById("pop-score0").innerText = "-";
            }
            
            if (pt1) {
                document.getElementById("pop-pt1").innerText = `(${Math.round(pt1[0])}, ${Math.round(pt1[1])})`;
                document.getElementById("pop-score1").innerText = score1.toFixed(3);
            } else {
                document.getElementById("pop-pt1").innerText = "-";
                document.getElementById("pop-score1").innerText = "-";
            }
            
            const matchContainer = document.getElementById("pop-match-container");
            if (conf !== null) {
                document.getElementById("pop-status").innerText = `${(conf * 100).toFixed(1)}% (${status})`;
                matchContainer.style.display = "flex";
                if (status.includes("Valid")) {
                    matchContainer.className = "info-item accent-green";
                } else {
                    matchContainer.className = "info-item accent-orange";
                }
            } else {
                document.getElementById("pop-status").innerText = status;
                matchContainer.className = "info-item accent-red";
            }
            
            popup.classList.add("show");
        }

        function hidePopup() {
            document.getElementById("info-popup").classList.remove("show");
        }
    </script>
</body>
</html>
"""
    
if __name__ == "__main__":
    main()
