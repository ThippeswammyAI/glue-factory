import os
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from omegaconf import OmegaConf
from gluefactory.datasets.homographies import HomographyDataset
from gluefactory.geometry.gt_generation import gt_matches_from_homography

def main():
    # Define directories
    output_dir = Path("data/custom_dataset/visualizations/matches_interactive")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    conf_path = "gluefactory/configs/superpoint_custom+superglue_homography_custom_dataset.yaml"
    conf = OmegaConf.load(conf_path)
    data_conf = conf.data
    
    # Initialize dataset
    print("Initializing dataset...")
    dataset = HomographyDataset(data_conf)
    
    all_data = []
    
    # Process both train and val splits
    for split in ["train", "val"]:
        sub_dataset = dataset.get_dataset(split)
        print(f"Processing split '{split}' containing {len(sub_dataset)} samples...")
        
        for idx in range(len(sub_dataset)):
            item = sub_dataset[idx]
            name = item["name"]
            
            # Extract images and save them
            # Images are float tensors [3, H, W] in [0, 1]
            img0_np = item["view0"]["image"].permute(1, 2, 0).numpy()
            img1_np = item["view1"]["image"].permute(1, 2, 0).numpy()
            
            # Convert float [0, 1] RGB to uint8 BGR for cv2
            img0_bgr = (img0_np * 255.0).astype(np.uint8)[:, :, ::-1]
            img1_bgr = (img1_np * 255.0).astype(np.uint8)[:, :, ::-1]
            
            img0_filename = f"{split}_{idx}_view0.jpg"
            img1_filename = f"{split}_{idx}_view1.jpg"
            
            cv2.imwrite(str(images_dir / img0_filename), img0_bgr)
            cv2.imwrite(str(images_dir / img1_filename), img1_bgr)
            
            # Extract keypoints and compute matches
            kp0 = item["view0"]["cache"]["keypoints"]
            kp1 = item["view1"]["cache"]["keypoints"]
            scores0 = item["view0"]["cache"]["keypoint_scores"]
            scores1 = item["view1"]["cache"]["keypoint_scores"]
            H = torch.from_numpy(item["H_0to1"]).unsqueeze(0)
            
            # Compute matches
            matches_dict = gt_matches_from_homography(
                kp0.unsqueeze(0),
                kp1.unsqueeze(0),
                H,
                pos_th=conf.model.ground_truth.th_positive,
                neg_th=conf.model.ground_truth.th_negative
            )
            
            matches0 = matches_dict["matches0"][0].numpy()
            
            # Convert tensors/numpy to native python lists for JSON serialization
            kp0_list = kp0.tolist()
            kp1_list = kp1.tolist()
            scores0_list = scores0.tolist()
            scores1_list = scores1.tolist()
            
            # Create list of matches: [[idx0, idx1], ...]
            matches_list = []
            for idx0, idx1 in enumerate(matches0):
                if idx1 >= 0:
                    matches_list.append([idx0, int(idx1)])
                    
            H_list = item["H_0to1"].tolist()
            
            sample_info = {
                "idx": idx,
                "name": name,
                "split": split,
                "image0_url": f"images/{img0_filename}",
                "image1_url": f"images/{img1_filename}",
                "keypoints0": kp0_list,
                "keypoints1": kp1_list,
                "scores0": scores0_list,
                "scores1": scores1_list,
                "matches": matches_list,
                "H_0to1": H_list,
                "image_width": img0_bgr.shape[1],
                "image_height": img0_bgr.shape[0]
            }
            
            all_data.append(sample_info)
            print(f"  Sample {idx}: {len(kp0_list)} / {len(kp1_list)} kpts, {len(matches_list)} matches.")
            
    # Save matches data as a js file
    output_js_path = output_dir / "matches_data.js"
    with open(output_js_path, "w") as f:
        f.write("const MATCHES_DATA = ")
        json.dump(all_data, f, indent=2)
        f.write(";\n")
        
    print(f"Data saved to {output_js_path}")

    # Save index.html file
    output_html_path = output_dir / "index.html"
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Homography Dataset Match Visualizer</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
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
            background-color: rgba(15, 23, 42, 0.95);
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
        }

        .sidebar-header h1 {
            font-size: 20px;
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

        .filter-tabs {
            display: flex;
            padding: 12px 24px;
            gap: 8px;
            border-bottom: 1px solid var(--border-color);
        }

        .tab-btn {
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid transparent;
            color: var(--text-secondary);
            padding: 8px 12px;
            border-radius: 8px;
            font-family: var(--font-main);
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .tab-btn:hover {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-primary);
        }

        .tab-btn.active {
            background: rgba(6, 182, 212, 0.15);
            border-color: rgba(6, 182, 212, 0.3);
            color: var(--accent-cyan);
        }

        .pair-list {
            flex-grow: 1;
            overflow-y: auto;
            padding: 16px 24px;
        }

        .pair-list::-webkit-scrollbar {
            width: 6px;
        }

        .pair-list::-webkit-scrollbar-track {
            background: transparent;
        }

        .pair-list::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }

        .pair-list::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .pair-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .pair-item:hover {
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }

        .pair-item.active {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.08) 0%, rgba(217, 70, 239, 0.08) 100%);
            border-color: rgba(6, 182, 212, 0.4);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
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
            margin-bottom: 8px;
        }

        .pair-idx {
            font-family: var(--font-mono);
            font-size: 12px;
            color: var(--text-secondary);
        }

        .badge {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            padding: 2px 8px;
            border-radius: 9999px;
            letter-spacing: 0.5px;
        }

        .badge-train {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
        }

        .badge-val {
            background: rgba(217, 70, 239, 0.15);
            color: var(--accent-purple);
        }

        .pair-name {
            font-size: 14px;
            font-weight: 500;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 8px;
        }

        .pair-stats-row {
            display: flex;
            gap: 16px;
            font-size: 12px;
            color: var(--text-secondary);
        }

        .pair-stat {
            display: flex;
            align-items: center;
            gap: 4px;
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
            font-size: 24px;
            font-weight: 600;
            letter-spacing: -0.5px;
            margin-bottom: 4px;
        }

        .header-title p {
            font-size: 14px;
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
            padding: 20px;
            backdrop-filter: blur(12px);
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .stat-card .label {
            font-size: 13px;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .stat-card .value {
            font-size: 28px;
            font-weight: 700;
            font-family: var(--font-mono);
            color: var(--text-primary);
        }

        /* Control Panel */
        .control-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(12px);
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
            align-items: center;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .control-group label {
            font-size: 12px;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .slider-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .slider-container input[type="range"] {
            -webkit-appearance: none;
            width: 150px;
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.1);
            outline: none;
        }

        .slider-container input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--accent-cyan);
            cursor: pointer;
            transition: transform 0.1s;
        }

        .slider-container input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }

        .slider-val {
            font-family: var(--font-mono);
            font-size: 13px;
            min-width: 32px;
        }

        .toggle-group {
            display: flex;
            gap: 12px;
        }

        .btn-toggle {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 8px 16px;
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
        }

        .btn-toggle.active {
            background: rgba(6, 182, 212, 0.15);
            border-color: var(--accent-cyan);
            color: var(--text-primary);
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
            padding: 32px;
            backdrop-filter: blur(12px);
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            min-height: 400px;
            box-shadow: inset 0 0 40px rgba(0,0,0,0.3);
        }

        .image-pair-wrapper {
            position: relative;
            display: flex;
            gap: 40px;
            align-items: center;
            justify-content: center;
            padding: 10px;
            border-radius: 12px;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255,255,255,0.03);
        }

        .image-container {
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid rgba(255,255,255,0.05);
            transition: border-color 0.3s;
        }

        .image-container:hover {
            border-color: rgba(6, 182, 212, 0.3);
        }

        .image-container img {
            display: block;
            max-width: 100%;
            height: auto;
            user-select: none;
            -webkit-user-drag: none;
        }

        .view-label {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(4px);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-primary);
            border: 1px solid rgba(255,255,255,0.1);
            z-index: 5;
            pointer-events: none;
        }

        #matches-canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 8;
        }

        /* Hover Detail Popup */
        .info-popup {
            position: absolute;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 12px 24px;
            display: flex;
            gap: 24px;
            backdrop-filter: blur(16px);
            z-index: 15;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
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
            font-size: 11px;
            color: var(--text-secondary);
            font-weight: 500;
            text-transform: uppercase;
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
            background: rgba(0,0,0,0.3);
            padding: 8px 16px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            font-size: 12px;
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
    </style>
</head>
<body>
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-header">
            <h1>Match Visualizer</h1>
            <p>SuperPoint-SuperGlue Homography Dataset</p>
        </div>
        
        <div class="filter-tabs">
            <button class="tab-btn active" onclick="setSplitFilter('all')">All (19)</button>
            <button class="tab-btn" onclick="setSplitFilter('train')">Train (15)</button>
            <button class="tab-btn" onclick="setSplitFilter('val')">Val (4)</button>
        </div>

        <div class="pair-list" id="pair-list">
            <!-- Dynamic pairs will be rendered here -->
        </div>
    </div>

    <!-- Main Workspace -->
    <div class="workspace">
        <div class="header">
            <div class="header-title">
                <h2 id="active-filename">Loading...</h2>
                <p id="active-original-image">Original: -</p>
            </div>
        </div>

        <!-- Statistics Grid -->
        <div class="stats-grid">
            <div class="stat-card">
                <span class="label">Total Keypoints (View 0 / 1)</span>
                <span class="value" id="stat-kpts">- / -</span>
            </div>
            <div class="stat-card">
                <span class="label">Valid Matches</span>
                <span class="value" id="stat-matches">-</span>
            </div>
            <div class="stat-card">
                <span class="label">Match Percentage</span>
                <span class="value" id="stat-pct">-</span>
            </div>
            <div class="stat-card">
                <span class="label">Avg Score (View 0 / 1)</span>
                <span class="value" id="stat-score">- / -</span>
            </div>
        </div>

        <!-- Controls -->
        <div class="control-panel">
            <div class="control-group">
                <label>Keypoint Threshold</label>
                <div class="slider-container">
                    <input type="range" id="score-thresh" min="0.0" max="0.5" step="0.01" value="0.0" oninput="updateScoreThresh(this.value)">
                    <span class="slider-val" id="score-thresh-val">0.00</span>
                </div>
            </div>

            <div class="control-group">
                <label>Display Layers</label>
                <div class="toggle-group">
                    <button class="btn-toggle active" id="btn-show-kpts" onclick="toggleLayer('kpts')">All Keypoints</button>
                    <button class="btn-toggle active" id="btn-show-matches" onclick="toggleLayer('matches')">All Match Lines</button>
                    <button class="btn-toggle" id="btn-show-reproj" onclick="toggleLayer('reproj')">Reprojection Hover</button>
                </div>
            </div>
        </div>

        <!-- Visualization Box -->
        <div class="visualization-box">
            <!-- Legend -->
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-dot" style="background-color: var(--accent-cyan);"></div>
                    <span>View 0 Keypoint</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background-color: var(--accent-purple);"></div>
                    <span>View 1 Keypoint</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background-color: var(--accent-green);"></div>
                    <span>Valid Match</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background-color: var(--accent-orange); border-radius: 0;"></div>
                    <span>Reprojection</span>
                </div>
            </div>

            <!-- Image wrapper -->
            <div class="image-pair-wrapper" id="image-pair-wrapper">
                <div class="image-container" id="img-container-0">
                    <span class="view-label">View 0 (Warped)</span>
                    <img id="img-0" src="" alt="View 0">
                </div>
                <div class="image-container" id="img-container-1">
                    <span class="view-label">View 1 (Warped)</span>
                    <img id="img-1" src="" alt="View 1">
                </div>
                <canvas id="matches-canvas"></canvas>
            </div>

            <!-- Detail overlay -->
            <div class="info-popup" id="info-popup">
                <div class="info-item accent-cyan">
                    <span class="lbl">View 0 Pt</span>
                    <span class="val" id="pop-pt0">-</span>
                </div>
                <div class="info-item accent-cyan">
                    <span class="lbl">Score 0</span>
                    <span class="val" id="pop-score0">-</span>
                </div>
                <div class="info-item accent-purple">
                    <span class="lbl">View 1 Pt</span>
                    <span class="val" id="pop-pt1">-</span>
                </div>
                <div class="info-item accent-purple">
                    <span class="lbl">Score 1</span>
                    <span class="val" id="pop-score1">-</span>
                </div>
                <div class="info-item accent-green" id="pop-match-container">
                    <span class="lbl">Status</span>
                    <span class="val" id="pop-status">-</span>
                </div>
                <div class="info-item accent-orange" id="pop-error-container">
                    <span class="lbl">Warp Error</span>
                    <span class="val" id="pop-error">-</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Interactive Logic Script -->
    <script src="matches_data.js"></script>
    <script>
        let currentSplit = 'all';
        let activeIdx = 0;
        let activeData = null;
        let scoreThresh = 0.0;
        
        let settings = {
            showKpts: true,
            showMatches: true,
            showReproj: false
        };

        let hoveredPoint = null; // { view: 0/1, index: idx }

        // Elements
        const pairListContainer = document.getElementById('pair-list');
        const img0 = document.getElementById('img-0');
        const img1 = document.getElementById('img-1');
        const canvas = document.getElementById('matches-canvas');
        const ctx = canvas.getContext('2d');
        const wrapper = document.getElementById('image-pair-wrapper');

        function init() {
            renderList();
            loadPair(0);
            
            // Event listeners for window resize
            window.addEventListener('resize', handleResize);
            
            // Mouse move handler for keypoint hover detection
            const imgContainer0 = document.getElementById('img-container-0');
            const imgContainer1 = document.getElementById('img-container-1');

            imgContainer0.addEventListener('mousemove', (e) => handleMouseMove(e, 0));
            imgContainer1.addEventListener('mousemove', (e) => handleMouseMove(e, 1));
            
            imgContainer0.addEventListener('mouseleave', handleMouseLeave);
            imgContainer1.addEventListener('mouseleave', handleMouseLeave);

            // Trigger resize after images load
            img0.onload = () => handleResize();
            img1.onload = () => handleResize();
        }

        function setSplitFilter(split) {
            currentSplit = split;
            document.querySelectorAll('.filter-tabs .tab-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.innerText.toLowerCase().includes(split)) {
                    btn.classList.add('active');
                }
            });
            renderList();
        }

        function renderList() {
            pairListContainer.innerHTML = '';
            MATCHES_DATA.forEach((item, idx) => {
                if (currentSplit !== 'all' && item.split !== currentSplit) return;
                
                const div = document.createElement('div');
                div.className = `pair-item ${idx === activeIdx ? 'active' : ''}`;
                div.onclick = () => loadPair(idx);

                const badgeClass = item.split === 'train' ? 'badge-train' : 'badge-val';
                
                // Calculate match pct
                const matchPct = ((item.matches.length / item.keypoints0.length) * 100).toFixed(0);

                div.innerHTML = `
                    <div class="pair-meta">
                        <span class="pair-idx">PAIR #${String(idx).padStart(2, '0')}</span>
                        <span class="badge ${badgeClass}">${item.split}</span>
                    </div>
                    <div class="pair-name" title="${item.name}">${item.name}</div>
                    <div class="pair-stats-row">
                        <div class="pair-stat">Kpts: <span class="val">${item.keypoints0.length}</span></div>
                        <div class="pair-stat">Matches: <span class="val">${item.matches.length} (${matchPct}%)</span></div>
                    </div>
                `;
                pairListContainer.appendChild(div);
            });
        }

        function loadPair(idx) {
            activeIdx = idx;
            activeData = MATCHES_DATA[idx];
            
            // Update active list style
            document.querySelectorAll('.pair-item').forEach((item, i) => {
                item.classList.remove('active');
            });
            // Find active element in list container
            const listItems = pairListContainer.children;
            let counter = 0;
            MATCHES_DATA.forEach((item, i) => {
                if (currentSplit !== 'all' && item.split !== currentSplit) return;
                if (i === idx) {
                    if (listItems[counter]) listItems[counter].classList.add('active');
                }
                counter++;
            });

            // Set titles and details
            document.getElementById('active-filename').innerText = `Pair #${idx}: ${activeData.name}`;
            document.getElementById('active-original-image').innerText = `Original Path: data/custom_dataset/images/${activeData.name}`;

            // Load images
            img0.src = activeData.image0_url;
            img1.src = activeData.image1_url;
            
            hoveredPoint = null;
            hidePopup();
            
            // Set statistics
            document.getElementById('stat-kpts').innerText = `${activeData.keypoints0.length} / ${activeData.keypoints1.length}`;
            document.getElementById('stat-matches').innerText = activeData.matches.length;
            const matchPct = ((activeData.matches.length / activeData.keypoints0.length) * 100).toFixed(1);
            document.getElementById('stat-pct').innerText = `${matchPct}%`;
            
            const avgS0 = (activeData.scores0.reduce((a, b) => a + b, 0) / activeData.scores0.length).toFixed(3);
            const avgS1 = (activeData.scores1.reduce((a, b) => a + b, 0) / activeData.scores1.length).toFixed(3);
            document.getElementById('stat-score').innerText = `${avgS0} / ${avgS1}`;
        }

        function updateScoreThresh(val) {
            scoreThresh = parseFloat(val);
            document.getElementById('score-thresh-val').innerText = scoreThresh.toFixed(2);
            draw();
        }

        function toggleLayer(layer) {
            if (layer === 'kpts') {
                settings.showKpts = !settings.showKpts;
                document.getElementById('btn-show-kpts').classList.toggle('active', settings.showKpts);
            } else if (layer === 'matches') {
                settings.showMatches = !settings.showMatches;
                document.getElementById('btn-show-matches').classList.toggle('active', settings.showMatches);
            } else if (layer === 'reproj') {
                settings.showReproj = !settings.showReproj;
                document.getElementById('btn-show-reproj').classList.toggle('active', settings.showReproj);
            }
            draw();
        }

        function handleResize() {
            if (!activeData) return;
            
            // Update canvas dimensions to match the image wrapper bounding size
            canvas.width = wrapper.clientWidth;
            canvas.height = wrapper.clientHeight;
            
            draw();
        }

        // Project a coordinate [x, y] using homography matrix H
        function projectHomography(x, y, H) {
            const w = H[2][0] * x + H[2][1] * y + H[2][2];
            const px = (H[0][0] * x + H[0][1] * y + H[0][2]) / w;
            const py = (H[1][0] * x + H[1][1] * y + H[1][2]) / w;
            return [px, py];
        }

        function handleMouseMove(e, viewIdx) {
            if (!activeData) return;

            const img = viewIdx === 0 ? img0 : img1;
            const rect = img.getBoundingClientRect();
            
            // Cursor position relative to the image
            const curX = e.clientX - rect.left;
            const curY = e.clientY - rect.top;
            
            // Scale cursor back to original feature scale (512x208)
            const scaleX = activeData.image_width / rect.width;
            const scaleY = activeData.image_height / rect.height;
            const imgSpaceX = curX * scaleX;
            const imgSpaceY = curY * scaleY;
            
            // Find nearest keypoint in viewIdx
            const kpts = viewIdx === 0 ? activeData.keypoints0 : activeData.keypoints1;
            const scores = viewIdx === 0 ? activeData.scores0 : activeData.scores1;
            
            let minDist = Infinity;
            let bestIdx = -1;
            
            for (let i = 0; i < kpts.length; i++) {
                if (scores[i] < scoreThresh) continue;
                
                const dx = kpts[i][0] - imgSpaceX;
                const dy = kpts[i][1] - imgSpaceY;
                const dist = Math.sqrt(dx*dx + dy*dy);
                
                if (dist < minDist) {
                    minDist = dist;
                    bestIdx = i;
                }
            }
            
            // Distance threshold in image space (e.g. 15 pixels)
            const pixelThreshold = 15;
            if (bestIdx !== -1 && (minDist / scaleX) < pixelThreshold) {
                if (!hoveredPoint || hoveredPoint.view !== viewIdx || hoveredPoint.index !== bestIdx) {
                    hoveredPoint = { view: viewIdx, index: bestIdx };
                    updatePopup(viewIdx, bestIdx);
                    draw();
                }
            } else {
                if (hoveredPoint !== null) {
                    hoveredPoint = null;
                    hidePopup();
                    draw();
                }
            }
        }

        function handleMouseLeave() {
            if (hoveredPoint !== null) {
                hoveredPoint = null;
                hidePopup();
                draw();
            }
        }

        function updatePopup(view, index) {
            const popup = document.getElementById('info-popup');
            
            let pt0_coords = "-";
            let s0_val = "-";
            let pt1_coords = "-";
            let s1_val = "-";
            let status = "Unmatched";
            let errorText = "-";
            
            const matchStatusContainer = document.getElementById('pop-match-container');
            const errorContainer = document.getElementById('pop-error-container');
            
            if (view === 0) {
                const kpt0 = activeData.keypoints0[index];
                pt0_coords = `(${kpt0[0].toFixed(1)}, ${kpt0[1].toFixed(1)})`;
                s0_val = activeData.scores0[index].toFixed(4);
                
                // Find matching keypoint in view1
                const match = activeData.matches.find(m => m[0] === index);
                if (match) {
                    const matchIdx = match[1];
                    const kpt1 = activeData.keypoints1[matchIdx];
                    pt1_coords = `(${kpt1[0].toFixed(1)}, ${kpt1[1].toFixed(1)})`;
                    s1_val = activeData.scores1[matchIdx].toFixed(4);
                    status = "Matched (GT)";
                    
                    // Compute homography warp error
                    const [px, py] = projectHomography(kpt0[0], kpt0[1], activeData.H_0to1);
                    const dx = px - kpt1[0];
                    const dy = py - kpt1[1];
                    const err = Math.sqrt(dx*dx + dy*dy);
                    errorText = `${err.toFixed(2)} px`;
                }
            } else {
                const kpt1 = activeData.keypoints1[index];
                pt1_coords = `(${kpt1[0].toFixed(1)}, ${kpt1[1].toFixed(1)})`;
                s1_val = activeData.scores1[index].toFixed(4);
                
                // Find matching keypoint in view0
                const match = activeData.matches.find(m => m[1] === index);
                if (match) {
                    const matchIdx = match[0];
                    const kpt0 = activeData.keypoints0[matchIdx];
                    pt0_coords = `(${kpt0[0].toFixed(1)}, ${kpt0[1].toFixed(1)})`;
                    s0_val = activeData.scores0[matchIdx].toFixed(4);
                    status = "Matched (GT)";
                    
                    // Compute homography warp error
                    const [px, py] = projectHomography(kpt0[0], kpt0[1], activeData.H_0to1);
                    const dx = px - kpt1[0];
                    const dy = py - kpt1[1];
                    const err = Math.sqrt(dx*dx + dy*dy);
                    errorText = `${err.toFixed(2)} px`;
                }
            }
            
            document.getElementById('pop-pt0').innerText = pt0_coords;
            document.getElementById('pop-score0').innerText = s0_val;
            document.getElementById('pop-pt1').innerText = pt1_coords;
            document.getElementById('pop-score1').innerText = s1_val;
            document.getElementById('pop-status').innerText = status;
            document.getElementById('pop-error').innerText = errorText;
            
            if (status.includes("Matched")) {
                matchStatusContainer.style.display = 'flex';
                document.getElementById('pop-status').style.color = 'var(--accent-green)';
            } else {
                matchStatusContainer.style.display = 'flex';
                document.getElementById('pop-status').style.color = 'var(--accent-red)';
            }
            
            errorContainer.style.display = errorText !== "-" ? 'flex' : 'none';
            popup.classList.add('show');
        }

        function hidePopup() {
            const popup = document.getElementById('info-popup');
            popup.classList.remove('show');
        }

        function getCanvasCoords(viewIdx, x, y) {
            const img = viewIdx === 0 ? img0 : img1;
            const rect = img.getBoundingClientRect();
            const wrapperRect = wrapper.getBoundingClientRect();
            
            const scaleX = rect.width / activeData.image_width;
            const scaleY = rect.height / activeData.image_height;
            
            const x_canvas = (x * scaleX) + rect.left - wrapperRect.left;
            const y_canvas = (y * scaleY) + rect.top - wrapperRect.top;
            
            return [x_canvas, y_canvas];
        }

        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (!activeData) return;

            // Draw all keypoints if enabled
            if (settings.showKpts) {
                // View 0 keypoints
                activeData.keypoints0.forEach((kp, i) => {
                    if (activeData.scores0[i] < scoreThresh) return;
                    const [cx, cy] = getCanvasCoords(0, kp[0], kp[1]);
                    ctx.beginPath();
                    ctx.arc(cx, cy, 2, 0, 2 * Math.PI);
                    ctx.fillStyle = 'rgba(6, 182, 212, 0.6)'; // Cyan
                    ctx.fill();
                });
                
                // View 1 keypoints
                activeData.keypoints1.forEach((kp, i) => {
                    if (activeData.scores1[i] < scoreThresh) return;
                    const [cx, cy] = getCanvasCoords(1, kp[0], kp[1]);
                    ctx.beginPath();
                    ctx.arc(cx, cy, 2, 0, 2 * Math.PI);
                    ctx.fillStyle = 'rgba(217, 70, 239, 0.6)'; // Purple
                    ctx.fill();
                });
            }

            // Draw matches if enabled
            if (settings.showMatches) {
                activeData.matches.forEach(match => {
                    const idx0 = match[0];
                    const idx1 = match[1];
                    
                    if (activeData.scores0[idx0] < scoreThresh || activeData.scores1[idx1] < scoreThresh) return;
                    
                    const kp0 = activeData.keypoints0[idx0];
                    const kp1 = activeData.keypoints1[idx1];
                    
                    const [cx0, cy0] = getCanvasCoords(0, kp0[0], kp0[1]);
                    const [cx1, cy1] = getCanvasCoords(1, kp1[0], kp1[1]);
                    
                    ctx.beginPath();
                    ctx.moveTo(cx0, cy0);
                    ctx.lineTo(cx1, cy1);
                    ctx.strokeStyle = 'rgba(16, 185, 129, 0.18)'; // Translucent emerald green
                    ctx.lineWidth = 1;
                    ctx.stroke();
                });
            }

            // Draw Hovered Highlight or Reprojection
            if (hoveredPoint !== null) {
                const isView0 = hoveredPoint.view === 0;
                const idx = hoveredPoint.index;
                
                const kp_hover = isView0 ? activeData.keypoints0[idx] : activeData.keypoints1[idx];
                const [hX, hY] = getCanvasCoords(hoveredPoint.view, kp_hover[0], kp_hover[1]);
                
                // Draw circle around hovered keypoint
                ctx.beginPath();
                ctx.arc(hX, hY, 8, 0, 2 * Math.PI);
                ctx.strokeStyle = isView0 ? 'var(--accent-cyan)' : 'var(--accent-purple)';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Draw matching line
                const match = isView0 
                    ? activeData.matches.find(m => m[0] === idx)
                    : activeData.matches.find(m => m[1] === idx);
                    
                if (match) {
                    const matchIdx = isView0 ? match[1] : match[0];
                    const kp_match = isView0 ? activeData.keypoints1[matchIdx] : activeData.keypoints0[matchIdx];
                    const [mX, mY] = getCanvasCoords(isView0 ? 1 : 0, kp_match[0], kp_match[1]);
                    
                    // Draw match circle
                    ctx.beginPath();
                    ctx.arc(mX, mY, 8, 0, 2 * Math.PI);
                    ctx.strokeStyle = isView0 ? 'var(--accent-purple)' : 'var(--accent-cyan)';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    // Connect with thick glowing line
                    ctx.shadowBlur = 10;
                    ctx.shadowColor = 'var(--accent-green)';
                    ctx.beginPath();
                    ctx.moveTo(hX, hY);
                    ctx.lineTo(mX, mY);
                    ctx.strokeStyle = 'var(--accent-green)';
                    ctx.lineWidth = 2.5;
                    ctx.stroke();
                    ctx.shadowBlur = 0; // Reset shadow
                }
                
                // Draw Homography Reprojection for View 0
                if (settings.showReproj && isView0) {
                    const [px, py] = projectHomography(kp_hover[0], kp_hover[1], activeData.H_0to1);
                    const [rX, rY] = getCanvasCoords(1, px, py);
                    
                    // Draw projected marker in View 1 (Neon Orange square)
                    ctx.beginPath();
                    ctx.rect(rX - 4, rY - 4, 8, 8);
                    ctx.strokeStyle = 'var(--accent-orange)';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                    ctx.fillStyle = 'rgba(249, 115, 22, 0.4)';
                    ctx.fill();
                    
                    // Draw line from source keypoint to projected position
                    ctx.setLineDash([4, 4]);
                    ctx.beginPath();
                    ctx.moveTo(hX, hY);
                    ctx.lineTo(rX, rY);
                    ctx.strokeStyle = 'var(--accent-orange)';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            }
        }

        // Initialize on load
        window.onload = init;
    </script>
</body>
</html>"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML dashboard saved to {output_html_path}")

if __name__ == "__main__":
    main()
