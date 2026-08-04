"""Self-contained HTML dashboard generation for inference-visualization scripts.

Two flavours:
  1) `render_dashboard`   — interactive canvas-overlay match viewer (keypoints,
                            match lines, zoom/pan, thresholds) driven by a
                            `matches_data.js` file. Used for SP+SuperGlue/
                            LightGlue pair inference (`gluefactory.scripts.run_inference`).
  2) `render_image_grid`  — a plain browsable grid of pre-rendered PNGs. Used
                            for single-image keypoint visualizations and for
                            the cached-H5 SLAM dataset viewer
                            (`gluefactory.scripts.visualize_slam_dataset`).

Previously this HTML/CSS/JS was duplicated (and already diverging) across
scripts/match_images.py, scripts/match_images_from_pt.py,
gluefactory/scripts/visualize_slam_pairs.py and visualize_slam_labels.py.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def write_matches_data(out_dir, records, filename="matches_data.js"):
    """Write `records` as `const MATCHES_DATA = [...];` to `out_dir/filename`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "w") as f:
        f.write("const MATCHES_DATA = ")
        json.dump(records, f, indent=2)
        f.write(";\n")
    logger.info(f"Saved {len(records)} record(s) → {path}")
    return path


def render_dashboard(out_dir, records, title="Match Visualizer"):
    """Write `matches_data.js` plus an interactive canvas-overlay `index.html`.

    Each record is expected to have: idx, name0, name1, image0_url, image1_url,
    keypoints0, keypoints1, scores0, scores1, matches (list of [i0, i1, score]),
    metrics (dict), image_width, image_height.
    """
    out_dir = Path(out_dir)
    write_matches_data(out_dir, records)
    html_path = out_dir / "index.html"
    html_path.write_text(_MATCH_DASHBOARD_HTML.replace("{{TITLE}}", title))
    logger.info(f"Interactive dashboard generated at {html_path}")
    return html_path


def render_image_grid(out_dir, image_paths, title="Visualization", grid_min_width=800,
                       out_name="index.html"):
    """Write a plain browsable HTML grid of PNGs already saved under `out_dir`.

    Args:
        out_dir: directory the images live in (and where index.html is written).
        image_paths: paths to the images, used only for their basename (assumed
                     to already sit directly under `out_dir`).
        title: page title / heading.
        grid_min_width: CSS grid-template-columns minmax() width, in px.
        out_name: output HTML filename.
    """
    out_dir = Path(out_dir)
    items = "\n".join(
        f'        <div class="item"><h2>{p.name}</h2>'
        f'<img src="{p.name}" alt="{p.name}"></div>'
        for p in image_paths
    )
    html = _IMAGE_GRID_HTML.replace("{{TITLE}}", title) \
        .replace("{{GRID_MIN_WIDTH}}", str(grid_min_width)) \
        .replace("{{ITEMS}}", items)
    html_path = out_dir / out_name
    html_path.write_text(html)
    logger.info(f"Generated HTML grid at {html_path}")
    return html_path


_IMAGE_GRID_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{TITLE}}</title>
    <style>
        body { font-family: sans-serif; background-color: #121212; color: #ffffff; padding: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax({{GRID_MIN_WIDTH}}px, 1fr)); gap: 20px; }
        .item { background: #1e1e1e; padding: 15px; border-radius: 8px; text-align: center; }
        img { max-width: 100%; height: auto; border-radius: 4px; }
        h2 { color: #bb86fc; font-size: 1.2em; margin-bottom: 10px; }
    </style>
</head>
<body>
    <h1>{{TITLE}}</h1>
    <div class="grid">
{{ITEMS}}
    </div>
</body>
</html>
"""


_MATCH_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
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
        <div class="sidebar-header"><h1>{{TITLE}}</h1><p>Model Inference Visualizer</p></div>
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
