"""
Run SuperPoint + SuperGlue inference on image pairs and produce an interactive
HTML match-visualization dashboard.

Loads both models from training checkpoints, runs them on all consecutive image
pairs in the input directory, saves per-pair static PNG plots, and writes a
self-contained interactive dashboard (index.html + matches_data.js) to the
output directory.

See also:
  scripts/export_interactive_matches.py  — same inference, data-only output (no HTML)

Usage:
    MPLBACKEND=Agg python scripts/match_images.py \\
        --input               data/output/slam/images/rgb \\
        --checkpoint_superglue  outputs/training/superglue_slam_run/checkpoint_best.tar \\
        --checkpoint_superpoint outputs/training/superpoint_slam_run/checkpoint_best.tar \\
        --output_dir          data/output/slam/visualizations/match_inference_rgb \\
        --resize 512

    # Single pair:
    MPLBACKEND=Agg python scripts/match_images.py \\
        --input img0.png,img1.png \\
        --checkpoint_superglue  outputs/training/superglue_slam_run/checkpoint_best.tar \\
        --checkpoint_superpoint outputs/training/superpoint_slam_run/checkpoint_best.tar
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gluefactory.slam.matcher import SLAMMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("superglue_inference")

DEFAULT_SUPERGLUE_CKPT = "outputs/training/superglue_slam_run/checkpoint_best.tar"
DEFAULT_SUPERPOINT_CKPT = "outputs/training/superpoint_slam_run/checkpoint_best.tar"
DEFAULT_INPUT_DIR = "data/output/slam/images/rgb"
DEFAULT_OUTPUT_DIR = "data/output/slam/visualizations/match_inference_rgb"


def parse_args():
    parser = argparse.ArgumentParser(description="Test SuperGlue model inference on image pairs.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--checkpoint_superglue", type=str, default=DEFAULT_SUPERGLUE_CKPT)
    parser.add_argument("--checkpoint_superpoint", type=str, default=DEFAULT_SUPERPOINT_CKPT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resize", type=int, default=640)
    parser.add_argument("--nms_radius", type=int, default=3)
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--detection_threshold", type=float, default=0.005)
    parser.add_argument("--filter_threshold", type=float, default=0.01)
    parser.add_argument("--max_pairs", type=int, default=0,
                        help="Max pairs (0 = all)")
    return parser.parse_args()


def plot_matches(img0, img1, kpts0, kpts1, matches, mscores, output_path, title="SuperGlue Matches"):
    """Plot static matches side-by-side."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0 + w1] = img1

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("#0b0c10")
    ax.set_facecolor("#0b0c10")
    ax.axis("off")
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.scatter(kpts0[:, 0], kpts0[:, 1], c="#06b6d4", s=10, edgecolors="none", alpha=0.8)
    ax.scatter(kpts1[:, 0] + w0, kpts1[:, 1], c="#d946ef", s=10, edgecolors="none", alpha=0.8)
    count = 0
    for idx0, idx1 in enumerate(matches):
        if idx1 == -1:
            continue
        p0, p1 = kpts0[idx0], kpts1[idx1]
        color = plt.cm.plasma(mscores[idx0])
        ax.plot([p0[0], p1[0] + w0], [p0[1], p1[1]], color=color, linewidth=1.2, alpha=0.85)
        count += 1
    ax.set_title(f"{title}\n{count} matches found", color="#66fcf1", fontsize=16, weight="bold", pad=12)
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def main():
    args = parse_args()

    sg_ckpt = Path(args.checkpoint_superglue)
    sp_ckpt = Path(args.checkpoint_superpoint)
    if not sg_ckpt.exists():
        logger.error(f"SuperGlue checkpoint not found: {sg_ckpt}")
        return
    if not sp_ckpt.exists():
        logger.error(f"SuperPoint checkpoint not found: {sp_ckpt}")
        return

    conf = {
        "nms_radius": args.nms_radius,
        "max_num_keypoints": args.max_num_keypoints,
        "detection_threshold": args.detection_threshold,
        "filter_threshold": args.filter_threshold,
    }
    matcher = SLAMMatcher(sg_ckpt, sp_ckpt, conf=conf)
    max_pairs = args.max_pairs if args.max_pairs > 0 else None
    results = matcher.match_directory(args.input, max_pairs=max_pairs, resize=args.resize)

    if not results:
        logger.error("No pairs were matched.")
        return

    out_dir = Path(args.output_dir)
    images_out_dir = out_dir / "images"
    plots_out_dir = out_dir / "plots"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    plots_out_dir.mkdir(parents=True, exist_ok=True)

    dashboard_data = []
    for r in tqdm(results, desc="Saving plots"):
        idx = r["idx"]
        img0_name = f"pair_{idx}_view0.png"
        img1_name = f"pair_{idx}_view1.png"
        cv2.imwrite(str(images_out_dir / img0_name), r["img0_bgr"])
        cv2.imwrite(str(images_out_dir / img1_name), r["img1_bgr"])

        plot_name = f"pair_{idx}_matches.png"
        plot_matches(
            r["img0_bgr"], r["img1_bgr"],
            r["keypoints0"], r["keypoints1"],
            r["matches0"], r["mscores0"],
            plots_out_dir / plot_name,
            title=f"Matches: {r['name0']} ↔ {r['name1']}",
        )

        matches_list = [
            [int(i0), int(i1), float(r["mscores0"][i0])]
            for i0, i1 in enumerate(r["matches0"]) if i1 != -1
        ]
        m = r["metrics"]
        logger.info(
            f"Pair {idx}: {r['name0']} ↔ {r['name1']}  "
            f"kpts {m['total_kpts0']}/{m['total_kpts1']}  "
            f"matches {m['num_matches']} ({m['match_ratio']:.1%})  "
            f"avg {m['avg_mscore']:.2f}"
        )
        dashboard_data.append({
            "idx": idx,
            "name0": r["name0"],
            "name1": r["name1"],
            "image0_url": f"images/{img0_name}",
            "image1_url": f"images/{img1_name}",
            "plot_url": f"plots/{plot_name}",
            "keypoints0": r["keypoints0"].tolist(),
            "keypoints1": r["keypoints1"].tolist(),
            "scores0": r["scores0"].tolist(),
            "scores1": r["scores1"].tolist(),
            "matches": matches_list,
            "metrics": m,
            "image_width": r["img0_bgr"].shape[1],
            "image_height": r["img0_bgr"].shape[0],
        })

    js_path = out_dir / "matches_data.js"
    with open(js_path, "w") as f:
        f.write("const MATCHES_DATA = ")
        json.dump(dashboard_data, f, indent=2)
        f.write(";\n")
    logger.info(f"Dashboard data saved to {js_path}")

    html_path = out_dir / "index.html"
    with open(html_path, "w") as f:
        f.write(get_html_template())
    logger.info(f"Interactive dashboard generated at {html_path}")

    print(f"\nTotal processed pairs: {len(dashboard_data)}")
    print(f"Results: {out_dir.resolve()}")
    print(f"Dashboard: file://{html_path.resolve()}\n")


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
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: var(--font-main); background-color: var(--bg-dark); color: var(--text-primary); height: 100vh; overflow: hidden; display: flex; }
        .sidebar { width: 380px; background-color: rgba(9,13,22,0.95); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; height: 100%; flex-shrink: 0; }
        .sidebar-header { padding: 24px; border-bottom: 1px solid var(--border-color); background: linear-gradient(135deg, rgba(6,182,212,0.1) 0%, rgba(217,70,239,0.1) 100%); }
        .sidebar-header h1 { font-size: 22px; font-weight: 700; background: linear-gradient(to right, var(--accent-cyan), var(--accent-purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .sidebar-header p { font-size: 13px; color: var(--text-secondary); }
        .pair-list-title { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary); padding: 16px 24px 8px; }
        .pair-list { flex-grow: 1; overflow-y: auto; padding: 0 16px 24px; }
        .pair-item { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 14px; margin-bottom: 10px; cursor: pointer; transition: all 0.25s; position: relative; overflow: hidden; }
        .pair-item:hover { background: rgba(6,182,212,0.04); border-color: var(--border-glow); transform: translateY(-2px); }
        .pair-item.active { background: linear-gradient(135deg, rgba(6,182,212,0.1) 0%, rgba(217,70,239,0.1) 100%); border-color: rgba(6,182,212,0.5); }
        .pair-item.active::before { content:''; position:absolute; left:0; top:0; height:100%; width:4px; background:linear-gradient(to bottom,var(--accent-cyan),var(--accent-purple)); }
        .pair-meta { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; }
        .pair-idx { font-family:var(--font-mono); font-size:11px; color:var(--accent-cyan); font-weight:500; }
        .badge { font-size:10px; font-weight:600; text-transform:uppercase; padding:2px 8px; border-radius:9999px; background:rgba(16,185,129,0.15); color:var(--accent-green); }
        .pair-name { font-size:13px; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom:6px; }
        .pair-stats-row { display:flex; gap:12px; font-size:11px; color:var(--text-secondary); }
        .pair-stat span.val { color:var(--text-primary); font-family:var(--font-mono); font-weight:500; }
        .workspace { flex-grow:1; height:100%; overflow-y:auto; padding:32px; display:flex; flex-direction:column; gap:24px; }
        .header { display:flex; justify-content:space-between; align-items:center; }
        .header-title h2 { font-size:22px; font-weight:600; margin-bottom:4px; }
        .header-title p { font-size:13px; color:var(--text-secondary); font-family:var(--font-mono); }
        .stats-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
        .stat-card { background:var(--bg-card); border:1px solid var(--border-color); border-radius:16px; padding:16px 20px; display:flex; flex-direction:column; gap:4px; position:relative; overflow:hidden; }
        .stat-card::after { content:''; position:absolute; bottom:0; left:0; height:3px; width:100%; background:linear-gradient(to right,var(--accent-cyan),var(--accent-purple)); opacity:0.5; }
        .stat-card .label { font-size:12px; color:var(--text-secondary); font-weight:500; text-transform:uppercase; letter-spacing:0.5px; }
        .stat-card .value { font-size:24px; font-weight:700; font-family:var(--font-mono); }
        .control-panel { background:var(--bg-card); border:1px solid var(--border-color); border-radius:16px; padding:18px 24px; display:flex; flex-wrap:wrap; gap:32px; align-items:center; }
        .control-group { display:flex; flex-direction:column; gap:8px; }
        .control-group label { font-size:11px; color:var(--text-secondary); font-weight:600; text-transform:uppercase; letter-spacing:0.75px; }
        .slider-container { display:flex; align-items:center; gap:12px; }
        .slider-container input[type="range"] { -webkit-appearance:none; width:150px; height:5px; border-radius:3px; background:rgba(255,255,255,0.1); outline:none; }
        .slider-container input[type="range"]::-webkit-slider-thumb { -webkit-appearance:none; width:15px; height:15px; border-radius:50%; background:var(--accent-cyan); cursor:pointer; }
        .slider-val { font-family:var(--font-mono); font-size:13px; min-width:36px; color:var(--accent-cyan); font-weight:500; }
        .toggle-group { display:flex; gap:8px; }
        .btn-toggle { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); color:var(--text-secondary); padding:8px 14px; border-radius:10px; font-family:var(--font-main); font-size:13px; cursor:pointer; transition:all 0.2s; }
        .btn-toggle:hover { background:rgba(255,255,255,0.08); color:var(--text-primary); }
        .btn-toggle.active { background:rgba(6,182,212,0.12); border-color:var(--accent-cyan); color:var(--text-primary); }
        .visualization-box { background:var(--bg-card); border:1px solid var(--border-color); border-radius:20px; padding:24px; flex-grow:1; display:flex; flex-direction:column; align-items:center; justify-content:center; position:relative; min-height:480px; overflow:hidden; cursor:grab; }
        .visualization-box:active { cursor:grabbing; }
        .image-pair-wrapper { position:relative; display:flex; flex-direction:column; gap:24px; align-items:center; justify-content:center; padding:10px; border-radius:12px; background:rgba(0,0,0,0.3); max-width:95%; transform-origin:center center; transition:transform 0.05s ease-out; user-select:none; }
        .image-container { position:relative; border-radius:8px; overflow:hidden; border:1px solid rgba(255,255,255,0.08); }
        .image-container img { display:block; max-width:100%; max-height:38vh; height:auto; user-select:none; -webkit-user-drag:none; }
        .view-label { position:absolute; top:10px; left:10px; background:rgba(9,13,22,0.85); padding:4px 10px; border-radius:6px; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; border:1px solid var(--border-color); z-index:5; pointer-events:none; }
        #matches-canvas { position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:auto; z-index:8; }
        .legend { position:absolute; top:24px; right:24px; display:flex; gap:16px; background:rgba(0,0,0,0.4); padding:8px 16px; border-radius:8px; border:1px solid var(--border-color); font-size:12px; z-index:9; }
        .legend-item { display:flex; align-items:center; gap:6px; }
        .legend-dot { width:8px; height:8px; border-radius:50%; }
        .legend-line { width:16px; height:2px; }
        .loader { position:absolute; top:0; left:0; width:100%; height:100%; background:var(--bg-dark); display:flex; align-items:center; justify-content:center; z-index:100; font-size:18px; font-weight:500; letter-spacing:1px; color:var(--accent-cyan); transition:opacity 0.5s ease; }
        .info-popup { position:absolute; bottom:24px; left:50%; transform:translateX(-50%); background:rgba(9,13,22,0.95); border:1px solid var(--accent-cyan); border-radius:12px; padding:12px 24px; display:flex; gap:24px; backdrop-filter:blur(16px); z-index:15; pointer-events:none; opacity:0; transition:opacity 0.2s ease,transform 0.2s ease; }
        .info-popup.show { opacity:1; transform:translate(-50%,-5px); }
        .info-item { display:flex; flex-direction:column; gap:2px; }
        .info-item .lbl { font-size:10px; color:var(--text-secondary); font-weight:500; text-transform:uppercase; }
        .info-item .val { font-size:14px; font-family:var(--font-mono); font-weight:600; }
        .info-item.accent-cyan .val { color:var(--accent-cyan); }
        .info-item.accent-purple .val { color:var(--accent-purple); }
        .info-item.accent-green .val { color:var(--accent-green); }
    </style>
</head>
<body>
    <div id="loader" class="loader">Loading Dashboard Data...</div>
    <div class="sidebar">
        <div class="sidebar-header"><h1>SuperGlue Visualizer</h1><p>Custom Training Model Inference</p></div>
        <div class="pair-list-title">Image Pairs</div>
        <div class="pair-list" id="pair-list"></div>
    </div>
    <div class="workspace">
        <div class="header">
            <div class="header-title">
                <h2 id="active-filename">Select an image pair</h2>
                <p id="active-original-image">Matching view0 ↔ view1</p>
            </div>
        </div>
        <div class="stats-grid">
            <div class="stat-card"><span class="label">Total Keypoints (V0 / V1)</span><span class="value" id="stat-kpts">- / -</span></div>
            <div class="stat-card"><span class="label">Filtered Matches</span><span class="value" id="stat-matches">-</span></div>
            <div class="stat-card"><span class="label">Match Percentage</span><span class="value" id="stat-pct">-</span></div>
            <div class="stat-card"><span class="label">Avg Match Score</span><span class="value" id="stat-score">-</span></div>
        </div>
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
        </div>
        <div class="visualization-box">
            <div class="legend">
                <div class="legend-item"><div class="legend-dot" style="background:var(--accent-cyan)"></div><span>V0</span></div>
                <div class="legend-item"><div class="legend-dot" style="background:var(--accent-purple)"></div><span>V1</span></div>
                <div class="legend-item"><div class="legend-line" style="background:linear-gradient(to right,var(--accent-cyan),var(--accent-purple))"></div><span>Match</span></div>
            </div>
            <div class="image-pair-wrapper" id="image-pair-wrapper">
                <div class="image-container" id="img-container-0"><span class="view-label">View 0</span><img id="img-0" src="" alt="View 0"></div>
                <div class="image-container" id="img-container-1"><span class="view-label">View 1</span><img id="img-1" src="" alt="View 1"></div>
                <canvas id="matches-canvas"></canvas>
            </div>
            <div class="info-popup" id="info-popup">
                <div class="info-item accent-cyan"><span class="lbl">V0 Coord</span><span class="val" id="pop-pt0">-</span></div>
                <div class="info-item accent-cyan"><span class="lbl">V0 Score</span><span class="val" id="pop-score0">-</span></div>
                <div class="info-item accent-purple"><span class="lbl">V1 Coord</span><span class="val" id="pop-pt1">-</span></div>
                <div class="info-item accent-purple"><span class="lbl">V1 Score</span><span class="val" id="pop-score1">-</span></div>
                <div class="info-item accent-green" id="pop-match-container"><span class="lbl">Match Confidence</span><span class="val" id="pop-status">-</span></div>
            </div>
        </div>
    </div>
    <script src="matches_data.js"></script>
    <script>
        let activeIdx=0,activeData=null,scoreThresh=0.01,kptThresh=0.00;
        let settings={showKpts:true,showMatches:true};
        let hoveredPoint=null;
        let zoom=1.0,panX=0,panY=0,isMouseDown=false,hasDragged=false,dragStartX=0,dragStartY=0,initialPanX=0,initialPanY=0;
        const loader=document.getElementById("loader");
        const pairList=document.getElementById("pair-list");
        const canvas=document.getElementById("matches-canvas");
        const ctx=canvas.getContext("2d");
        const img0=document.getElementById("img-0");
        const img1=document.getElementById("img-1");
        const container=document.getElementById("image-pair-wrapper");
        const vizBox=document.querySelector(".visualization-box");
        window.onload=function(){
            if(typeof MATCHES_DATA==='undefined'||MATCHES_DATA.length===0){loader.innerText="Error: No MATCHES_DATA loaded!";return;}
            loader.style.opacity=0;setTimeout(()=>loader.style.display="none",500);
            buildPairList();selectPair(0);
            window.addEventListener('resize',draw);
            canvas.addEventListener('mousemove',handleMouseMove);
            canvas.addEventListener('mouseleave',handleMouseLeave);
            vizBox.addEventListener('mousedown',function(e){if(e.button!==0)return;isMouseDown=true;hasDragged=false;dragStartX=e.clientX;dragStartY=e.clientY;initialPanX=panX;initialPanY=panY;});
            window.addEventListener('mousemove',function(e){if(!isMouseDown)return;const dx=e.clientX-dragStartX,dy=e.clientY-dragStartY;if(Math.hypot(dx,dy)>3){hasDragged=true;panX=initialPanX+dx;panY=initialPanY+dy;updateTransform();}});
            window.addEventListener('mouseup',function(e){if(isMouseDown){isMouseDown=false;}});
            vizBox.addEventListener('wheel',function(e){e.preventDefault();const s=0.05;zoom=e.deltaY<0?Math.min(zoom+s,5.0):Math.max(zoom-s,0.4);updateTransform();},{passive:false});
        };
        function buildPairList(){pairList.innerHTML="";MATCHES_DATA.forEach((pair,idx)=>{const item=document.createElement("div");item.className="pair-item"+(idx===0?" active":"");item.id=`pair-item-${idx}`;item.onclick=()=>selectPair(idx);item.innerHTML=`<div class="pair-meta"><span class="pair-idx">PAIR #${idx}</span><span class="badge">${pair.metrics.num_matches} matches</span></div><div class="pair-name">${pair.name0} ↔ ${pair.name1}</div><div class="pair-stats-row"><div class="pair-stat">Kpts: <span class="val">${pair.metrics.total_kpts0}/${pair.metrics.total_kpts1}</span></div><div class="pair-stat">Ratio: <span class="val">${Math.round(pair.metrics.match_ratio*100)}%</span></div></div>`;pairList.appendChild(item);});}
        function selectPair(idx){const prev=document.querySelector(".pair-item.active");if(prev)prev.classList.remove("active");const n=document.getElementById(`pair-item-${idx}`);if(n)n.classList.add("active");activeIdx=idx;activeData=MATCHES_DATA[idx];document.getElementById("active-filename").innerText=`${activeData.name0} ↔ ${activeData.name1}`;document.getElementById("active-original-image").innerText=`Image size: ${activeData.image_width}x${activeData.image_height}`;hoveredPoint=null;resetZoom();let lc=0;const onLoad=()=>{lc++;if(lc===2){resizeCanvas();draw();}};img0.onload=onLoad;img1.onload=onLoad;img0.src=activeData.image0_url;img1.src=activeData.image1_url;}
        function resizeCanvas(){canvas.width=container.offsetWidth;canvas.height=container.offsetHeight;}
        function getMap(){const c0=document.getElementById("img-container-0"),c1=document.getElementById("img-container-1");return{offset0:{x:c0.offsetLeft,y:c0.offsetTop},offset1:{x:c1.offsetLeft,y:c1.offsetTop},scale0:{x:c0.offsetWidth/activeData.image_width,y:c0.offsetHeight/activeData.image_height},scale1:{x:c1.offsetWidth/activeData.image_width,y:c1.offsetHeight/activeData.image_height}};}
        function updateScoreThresh(v){scoreThresh=parseFloat(v);document.getElementById("score-thresh-val").innerText=scoreThresh.toFixed(2);draw();}
        function updateKptThresh(v){kptThresh=parseFloat(v);document.getElementById("kpt-thresh-val").innerText=kptThresh.toFixed(3);draw();}
        function toggleLayer(l){if(l==='kpts'){settings.showKpts=!settings.showKpts;document.getElementById("btn-show-kpts").classList.toggle("active",settings.showKpts);}else{settings.showMatches=!settings.showMatches;document.getElementById("btn-show-matches").classList.toggle("active",settings.showMatches);}draw();}
        function updateTransform(){container.style.transform=`translate(${panX}px,${panY}px) scale(${zoom})`;}
        function zoomIn(){zoom=Math.min(zoom+0.25,5.0);updateTransform();}
        function zoomOut(){zoom=Math.max(zoom-0.25,0.4);updateTransform();}
        function resetZoom(){zoom=1.0;panX=0;panY=0;updateTransform();}
        function draw(){
            if(!activeData||!img0.complete||!img1.complete)return;
            resizeCanvas();ctx.clearRect(0,0,canvas.width,canvas.height);
            const map=getMap(),kpts0=activeData.keypoints0,kpts1=activeData.keypoints1,scores0=activeData.scores0,scores1=activeData.scores1,matches=activeData.matches;
            const fm=matches.filter(m=>m[2]>=scoreThresh&&scores0[m[0]]>=kptThresh&&scores1[m[1]]>=kptThresh);
            document.getElementById("stat-kpts").innerText=`${kpts0.length} / ${kpts1.length}`;
            document.getElementById("stat-matches").innerText=fm.length;
            document.getElementById("stat-pct").innerText=`${Math.round(fm.length/Math.max(1,Math.min(kpts0.length,kpts1.length))*100)}%`;
            const avg=fm.length>0?(fm.reduce((a,m)=>a+m[2],0)/fm.length).toFixed(2):"0.00";
            document.getElementById("stat-score").innerText=avg;
            if(settings.showKpts){kpts0.forEach((pt,i)=>{if(scores0[i]<kptThresh)return;ctx.beginPath();ctx.arc(map.offset0.x+pt[0]*map.scale0.x,map.offset0.y+pt[1]*map.scale0.y,3,0,2*Math.PI);ctx.fillStyle="rgba(6,182,212,0.4)";ctx.fill();});kpts1.forEach((pt,i)=>{if(scores1[i]<kptThresh)return;ctx.beginPath();ctx.arc(map.offset1.x+pt[0]*map.scale1.x,map.offset1.y+pt[1]*map.scale1.y,3,0,2*Math.PI);ctx.fillStyle="rgba(217,70,239,0.4)";ctx.fill();});}
            if(settings.showMatches){fm.forEach(m=>{const p0=kpts0[m[0]],p1=kpts1[m[1]];const x0=map.offset0.x+p0[0]*map.scale0.x,y0=map.offset0.y+p0[1]*map.scale0.y;const x1=map.offset1.x+p1[0]*map.scale1.x,y1=map.offset1.y+p1[1]*map.scale1.y;const op=0.15+m[2]*0.55;const g=ctx.createLinearGradient(x0,y0,x1,y1);g.addColorStop(0,`rgba(6,182,212,${op})`);g.addColorStop(1,`rgba(217,70,239,${op})`);ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.strokeStyle=g;ctx.lineWidth=1+m[2]*1.5;ctx.stroke();});}
            if(hoveredPoint){const iv0=hoveredPoint.view===0;const idx=hoveredPoint.index;const pt=iv0?kpts0[idx]:kpts1[idx];const offset=iv0?map.offset0:map.offset1;const scale=iv0?map.scale0:map.scale1;ctx.beginPath();ctx.arc(offset.x+pt[0]*scale.x,offset.y+pt[1]*scale.y,8,0,2*Math.PI);ctx.strokeStyle=iv0?"var(--accent-cyan)":"var(--accent-purple)";ctx.lineWidth=2;ctx.stroke();}
            else{document.getElementById("info-popup").classList.remove("show");}
        }
        function handleMouseMove(e){if(!activeData||isMouseDown)return;const rect=canvas.getBoundingClientRect();const mx=(e.clientX-rect.left)*(canvas.width/rect.width),my=(e.clientY-rect.top)*(canvas.height/rect.height);const map=getMap();let closest=null,minD=15;activeData.keypoints0.forEach((pt,i)=>{if(activeData.scores0[i]<kptThresh)return;const dx=mx-(map.offset0.x+pt[0]*map.scale0.x),dy=my-(map.offset0.y+pt[1]*map.scale0.y);const d=Math.hypot(dx,dy);if(d<minD){minD=d;closest={view:0,index:i};}});activeData.keypoints1.forEach((pt,i)=>{if(activeData.scores1[i]<kptThresh)return;const dx=mx-(map.offset1.x+pt[0]*map.scale1.x),dy=my-(map.offset1.y+pt[1]*map.scale1.y);const d=Math.hypot(dx,dy);if(d<minD){minD=d;closest={view:1,index:i};}});if(closest){if(!hoveredPoint||hoveredPoint.view!==closest.view||hoveredPoint.index!==closest.index){hoveredPoint=closest;draw();}}else if(hoveredPoint){hoveredPoint=null;draw();}}
        function handleMouseLeave(){if(hoveredPoint){hoveredPoint=null;draw();}}
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
