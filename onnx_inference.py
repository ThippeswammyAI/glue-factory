#!/usr/bin/env python3
"""
SuperPoint + SuperGlue inference using exported ONNX models.

Replaces the .tar checkpoint pipeline with ONNX runtime — no PyTorch or
gluefactory imports needed at inference time.

Outputs
-------
  • <out_dir>/matches.png           — side-by-side match visualization
  • <out_dir>/keypoints.png         — per-image keypoints + score heatmaps
  • <out_dir>/analysis.txt          — text metric report

Usage
-----
  # Boat demo (bundled assets)
  python onnx_inference.py

  # Custom pair
  python onnx_inference.py --img0 path/a.png --img1 path/b.png

  # Directory (consecutive pairs)
  python onnx_inference.py --input_dir data/output/slam/images/rgb --max_pairs 10

  # Different model paths
  python onnx_inference.py --sp_onnx my_sp.onnx --sg_onnx my_sg.onnx
"""

import argparse
import sys
import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import onnxruntime as ort

# ── config ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
DEFAULT_SP_ONNX   = ROOT / "superpoint.onnx"
DEFAULT_SG_ONNX   = ROOT / "superglue.onnx"
DEFAULT_OUT_DIR   = ROOT / "onnx_inference_output"

NMS_RADIUS        = 4
DETECTION_THRESH  = 0.005
MAX_KEYPOINTS     = 512
MATCH_THRESH      = 0.01
REMOVE_BORDERS    = 4
RESIZE_MAX        = 640   # long-edge resize; 0 = no resize


# ── SuperPoint ONNX extractor ──────────────────────────────────────────────────

class SPExtractor:
    """Run SuperPoint ONNX and extract sparse keypoints + descriptors."""

    def __init__(self, onnx_path: Path, providers=None):
        providers = providers or ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(str(onnx_path), providers=providers)

    def __call__(
        self,
        img_gray: np.ndarray,          # uint8 [H, W]
        max_kpts: int = MAX_KEYPOINTS,
        thresh:   float = DETECTION_THRESH,
        border:   int   = REMOVE_BORDERS,
    ) -> dict:
        """
        Returns dict:
          keypoints  [N, 2]  float32 pixel coords (x, y), pixel-center (+0.5)
          scores     [N]     float32 detection scores
          descriptors[N, 256]float32 L2-normalised
          score_map  [H, W]  float32 raw NMS heatmap (for visualization)
        """
        H, W = img_gray.shape
        # ensure divisible by 8 (SuperPoint stride)
        pH = (H + 7) // 8 * 8
        pW = (W + 7) // 8 * 8
        if pH != H or pW != W:
            img_gray = cv2.copyMakeBorder(img_gray, 0, pH - H, 0, pW - W,
                                          cv2.BORDER_REFLECT)

        img_f32 = img_gray.astype(np.float32) / 255.0
        inp = img_f32[None, None]                      # [1, 1, pH, pW]

        score_map, desc_map = self._sess.run(
            ["score_map", "desc_map"], {"image": inp}
        )
        # score_map : [1, pH, pW]  desc_map : [1, 256, pH/8, pW/8]
        score_map = score_map[0]                       # [pH, pW]

        # crop back if we padded
        score_map = score_map[:H, :W]
        desc_map  = desc_map[:, :, :H // 8, :W // 8]  # [1, 256, H/8, W/8]

        # border zeroing (same as model, but on the cropped map)
        if border > 0:
            score_map[:border]  = 0
            score_map[-border:] = 0
            score_map[:, :border]  = 0
            score_map[:, -border:] = 0

        # extract keypoints
        ys, xs = np.where(score_map > thresh)
        if len(xs) == 0:
            empty = np.zeros((0, 2), np.float32)
            return dict(keypoints=empty, scores=np.zeros(0, np.float32),
                        descriptors=np.zeros((0, 256), np.float32),
                        score_map=score_map)

        kscores = score_map[ys, xs]
        # top-K
        if max_kpts > 0 and len(kscores) > max_kpts:
            top = np.argpartition(kscores, -max_kpts)[-max_kpts:]
            xs, ys, kscores = xs[top], ys[top], kscores[top]
            order = np.argsort(kscores)[::-1]
            xs, ys, kscores = xs[order], ys[order], kscores[order]

        # pixel-center convention (match the .tar model's +0.5 offset)
        kpts = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float32)  # [N, 2]

        # sample descriptors via bilinear interpolation
        desc = _sample_desc(desc_map, kpts, stride=8)  # [N, 256]

        return dict(keypoints=kpts, scores=kscores.astype(np.float32),
                    descriptors=desc, score_map=score_map)


def _sample_desc(desc_map: np.ndarray, kpts: np.ndarray, stride: int = 8) -> np.ndarray:
    """Bilinear-interpolate desc_map at keypoint locations.

    desc_map : [1, 256, h, w]
    kpts     : [N, 2]  pixel coords (x, y) in original-resolution space
    returns  : [N, 256]  L2-normalised
    """
    _, C, h, w = desc_map.shape
    # map pixel coords → desc_map cell space, same formula as superpoint_open.py
    # (kpts + 0.5) / (w * stride)  → [0, 1]  → * 2 - 1  → [-1, 1]
    norm = np.array([w * stride, h * stride], dtype=np.float32)
    kpts_n = (kpts + 0.5) / norm * 2.0 - 1.0          # [N, 2] in [-1, 1]

    descs = np.zeros((len(kpts), C), dtype=np.float32)
    feat  = desc_map[0]                                 # [C, h, w]
    # manual bilinear sample (avoids torch dependency here)
    xn = (kpts_n[:, 0] + 1.0) / 2.0 * (w - 1)         # [N] in [0, w-1]
    yn = (kpts_n[:, 1] + 1.0) / 2.0 * (h - 1)         # [N] in [0, h-1]
    x0 = np.clip(np.floor(xn).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(yn).astype(int), 0, h - 2)
    x1, y1 = x0 + 1, y0 + 1
    wa = (x1 - xn) * (y1 - yn)
    wb = (x1 - xn) * (yn - y0)
    wc = (xn - x0) * (y1 - yn)
    wd = (xn - x0) * (yn - y0)
    descs = (feat[:, y0, x0].T * wa[:, None] +
             feat[:, y1, x0].T * wb[:, None] +
             feat[:, y0, x1].T * wc[:, None] +
             feat[:, y1, x1].T * wd[:, None])
    norms = np.linalg.norm(descs, axis=1, keepdims=True).clip(min=1e-8)
    return (descs / norms).astype(np.float32)


# ── SuperGlue ONNX matcher ─────────────────────────────────────────────────────

class SGMatcher:
    """Run SuperGlue ONNX on a keypoint pair."""

    def __init__(self, onnx_path: Path, providers=None):
        providers = providers or ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(str(onnx_path), providers=providers)

    def __call__(
        self,
        kpts0:  np.ndarray,   # [N, 2]
        kpts1:  np.ndarray,   # [M, 2]
        descs0: np.ndarray,   # [N, 256]
        descs1: np.ndarray,   # [M, 256]
        scores0: np.ndarray,  # [N]
        scores1: np.ndarray,  # [M]
        img_hw0: tuple,       # (H, W) of image 0
        img_hw1: tuple,       # (H, W) of image 1
        match_thresh: float = MATCH_THRESH,
    ) -> dict:
        H0, W0 = img_hw0
        H1, W1 = img_hw1
        size0 = np.array([[W0, H0]], dtype=np.float32)   # [1, 2]  (w, h)
        size1 = np.array([[W1, H1]], dtype=np.float32)

        m0, m1, ms0, ms1 = self._sess.run(
            ["matches0", "matches1", "mscores0", "mscores1"],
            {
                "kpts0":   kpts0[None].astype(np.float32),    # [1, N, 2]
                "kpts1":   kpts1[None].astype(np.float32),
                "descs0":  descs0[None].astype(np.float32),   # [1, N, 256]
                "descs1":  descs1[None].astype(np.float32),
                "scores0": scores0[None].astype(np.float32),  # [1, N]
                "scores1": scores1[None].astype(np.float32),
                "size0":   size0,
                "size1":   size1,
            },
        )
        m0, m1 = m0[0], m1[0]    # [N], [M]
        ms0, ms1 = ms0[0], ms1[0]

        # apply match threshold
        m0 = np.where(ms0 >= match_thresh, m0, np.full_like(m0, -1))
        m1 = np.where(ms1 >= match_thresh, m1, np.full_like(m1, -1))

        return dict(matches0=m0, matches1=m1, mscores0=ms0, mscores1=ms1)


# ── image loading ──────────────────────────────────────────────────────────────

def load_image(path: Path, resize_max: int = RESIZE_MAX):
    """Load image as uint8 grayscale, optionally resize long edge."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if resize_max > 0:
        h, w = gray.shape
        scale = resize_max / max(h, w)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img  = cv2.resize(img,  (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return gray, img   # uint8 [H,W], uint8 [H,W,3] BGR


# ── analysis ───────────────────────────────────────────────────────────────────

def compute_metrics(sp0, sp1, sg) -> dict:
    n0, n1   = len(sp0["keypoints"]), len(sp1["keypoints"])
    valid    = sg["matches0"] >= 0
    n_match  = int(valid.sum())
    ms_valid = sg["mscores0"][valid]
    return {
        "num_kpts0":    n0,
        "num_kpts1":    n1,
        "num_matches":  n_match,
        "match_ratio":  n_match / max(min(n0, n1), 1),
        "avg_mscore":   float(ms_valid.mean()) if n_match else 0.0,
        "max_mscore":   float(ms_valid.max())  if n_match else 0.0,
        "min_mscore":   float(ms_valid.min())  if n_match else 0.0,
        "avg_kpt_score0": float(sp0["scores"].mean()) if n0 else 0.0,
        "avg_kpt_score1": float(sp1["scores"].mean()) if n1 else 0.0,
    }


def format_report(name0, name1, m) -> str:
    return textwrap.dedent(f"""
    ┌─── ONNX Inference Report ───────────────────────────────────────┐
    │  Images     : {name0}
    │             : {name1}
    ├─────────────────────────────────────────────────────────────────┤
    │  Keypoints  : {m['num_kpts0']:>5d}  (view 0)   avg score {m['avg_kpt_score0']:.4f}
    │             : {m['num_kpts1']:>5d}  (view 1)   avg score {m['avg_kpt_score1']:.4f}
    ├─────────────────────────────────────────────────────────────────┤
    │  Matches    : {m['num_matches']:>5d}  ({m['match_ratio']:.1%} of min-kpts)
    │  Avg conf   : {m['avg_mscore']:.4f}
    │  Max conf   : {m['max_mscore']:.4f}
    │  Min conf   : {m['min_mscore']:.4f}
    └─────────────────────────────────────────────────────────────────┘
    """).strip()


# ── visualization ──────────────────────────────────────────────────────────────

_DARK_BG   = "#0b0c10"
_CYAN      = "#06b6d4"
_PURPLE    = "#d946ef"
_GREEN     = "#10b981"
_TITLE_CLR = "#66fcf1"


def _bgr_to_rgb(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def plot_matches(img0_bgr, img1_bgr, kpts0, kpts1, matches0, mscores0,
                 name0="", name1="", out_path=None):
    """Full-canvas side-by-side match visualization."""
    h0, w0 = img0_bgr.shape[:2]
    h1, w1 = img1_bgr.shape[:2]
    H = max(h0, h1)
    canvas = np.zeros((H, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0]       = img0_bgr
    canvas[:h1, w0:w0+w1]  = img1_bgr

    valid_mask = matches0 >= 0
    n_match    = int(valid_mask.sum())

    fig, ax = plt.subplots(figsize=(18, 9), dpi=120)
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    ax.axis("off")
    ax.imshow(_bgr_to_rgb(canvas))

    # keypoints
    ax.scatter(kpts0[:, 0], kpts0[:, 1],
               c=_CYAN, s=12, linewidths=0, alpha=0.7, zorder=3)
    ax.scatter(kpts1[:, 0] + w0, kpts1[:, 1],
               c=_PURPLE, s=12, linewidths=0, alpha=0.7, zorder=3)

    # match lines colored by confidence (plasma colormap)
    for i0, (m1, ms) in enumerate(zip(matches0, mscores0)):
        if m1 < 0:
            continue
        p0 = kpts0[i0]
        p1 = kpts1[m1]
        color = plt.cm.plasma(float(ms))
        ax.plot([p0[0], p1[0] + w0], [p0[1], p1[1]],
                color=color, linewidth=1.2, alpha=0.85, zorder=2)

    # vertical divider
    ax.axvline(w0, color="rgba(255,255,255,0.15)"
               if False else (1, 1, 1, 0.15), linewidth=1.5, zorder=4)

    ax.set_title(
        f"{name0}  ↔  {name1}\n"
        f"{n_match} matches  ·  "
        f"{len(kpts0)} kpts (V0)  ·  {len(kpts1)} kpts (V1)",
        color=_TITLE_CLR, fontsize=15, fontweight="bold", pad=12,
    )

    # colorbar for match confidence
    sm = plt.cm.ScalarMappable(cmap="plasma",
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="vertical",
                      fraction=0.015, pad=0.01)
    cb.set_label("Match confidence", color=_TITLE_CLR, fontsize=10)
    cb.ax.yaxis.set_tick_params(color=_TITLE_CLR, labelcolor=_TITLE_CLR)

    plt.tight_layout(pad=0.5)
    if out_path:
        plt.savefig(out_path, bbox_inches="tight",
                    facecolor=_DARK_BG, edgecolor="none")
        print(f"  saved {out_path}")
    plt.close()
    return fig


def plot_keypoints(img0_bgr, img1_bgr, sp0, sp1, name0="", name1="", out_path=None):
    """4-panel: keypoints overlay + score heatmaps for each view."""
    fig = plt.figure(figsize=(20, 10), facecolor=_DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.12, wspace=0.06,
                            top=0.90, bottom=0.02,
                            left=0.02, right=0.98)

    def _ax(row, col, title):
        a = fig.add_subplot(gs[row, col])
        a.axis("off")
        a.set_facecolor(_DARK_BG)
        a.set_title(title, color=_TITLE_CLR, fontsize=12, fontweight="bold", pad=6)
        return a

    # ── row 0: keypoints on image ──────────────────────────────────
    for col, (img_bgr, sp, name, color) in enumerate([
        (img0_bgr, sp0, name0, _CYAN),
        (img1_bgr, sp1, name1, _PURPLE),
    ]):
        ax = _ax(0, col, f"{name}\n{len(sp['keypoints'])} keypoints")
        ax.imshow(_bgr_to_rgb(img_bgr))
        kpts, scores = sp["keypoints"], sp["scores"]
        if len(kpts):
            sc = ax.scatter(kpts[:, 0], kpts[:, 1],
                            c=scores, cmap="plasma",
                            s=18, linewidths=0, alpha=0.9, zorder=3,
                            vmin=0, vmax=scores.max())
            fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01,
                         label="Keypoint score").ax.yaxis.set_tick_params(
                             color=_TITLE_CLR, labelcolor=_TITLE_CLR)

    # ── row 1: score heatmaps ──────────────────────────────────────
    # NMS output is very sparse; blur it so the heatmap is visible
    from scipy.ndimage import gaussian_filter
    for col, (sp, name) in enumerate([(sp0, name0), (sp1, name1)]):
        ax   = _ax(1, col, f"{name} – score heatmap (Gaussian σ=8)")
        smap = sp["score_map"]
        # overlay original image faintly under the heatmap
        img_bgr = img0_bgr if col == 0 else img1_bgr
        img_gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ax.imshow(img_gray_u8, cmap="gray", alpha=0.35)
        smap_viz = gaussian_filter(smap.astype(np.float32), sigma=8)
        im = ax.imshow(smap_viz, cmap="inferno", alpha=0.75,
                       vmin=0, vmax=smap_viz.max() * 0.6,
                       interpolation="bilinear")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01,
                     label="Density").ax.yaxis.set_tick_params(
                         color=_TITLE_CLR, labelcolor=_TITLE_CLR)

    fig.suptitle("SuperPoint ONNX — Keypoint Detections & Score Heatmaps",
                 color=_TITLE_CLR, fontsize=16, fontweight="bold", y=0.97)
    if out_path:
        plt.savefig(out_path, bbox_inches="tight",
                    facecolor=_DARK_BG, edgecolor="none")
        print(f"  saved {out_path}")
    plt.close()
    return fig


def plot_score_histogram(sp0, sp1, sg, name0="", name1="", out_path=None):
    """Match-score and keypoint-score histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=_DARK_BG)

    valid = sg["matches0"] >= 0
    ms    = sg["mscores0"][valid]

    specs = [
        (axes[0], sp0["scores"], f"{name0}\nKeypoint scores", _CYAN),
        (axes[1], sp1["scores"], f"{name1}\nKeypoint scores", _PURPLE),
        (axes[2], ms,            "SuperGlue match scores",   _GREEN),
    ]
    for ax, data, title, color in specs:
        ax.set_facecolor(_DARK_BG)
        ax.spines[:].set_color("#2a2f3a")
        ax.tick_params(colors="#94a3b8")
        if len(data):
            ax.hist(data, bins=40, color=color, alpha=0.8, edgecolor="none")
            ax.axvline(data.mean(), color="white", linewidth=1.2,
                       linestyle="--", label=f"mean={data.mean():.3f}")
            ax.legend(facecolor=_DARK_BG, edgecolor="#2a2f3a",
                      labelcolor="#94a3b8", fontsize=10)
        ax.set_title(title, color=_TITLE_CLR, fontsize=12,
                     fontweight="bold", pad=8)
        ax.set_xlabel("Score", color="#94a3b8")
        ax.set_ylabel("Count",  color="#94a3b8")

    fig.suptitle("Score Distributions", color=_TITLE_CLR,
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight",
                    facecolor=_DARK_BG, edgecolor="none")
        print(f"  saved {out_path}")
    plt.close()


# ── pipeline ───────────────────────────────────────────────────────────────────

def run_pair(sp_ext, sg_mat, img0_path, img1_path, out_dir: Path,
             prefix: str = "pair"):
    """Full SP+SG inference + visualization for one image pair."""
    gray0, bgr0 = load_image(img0_path)
    gray1, bgr1 = load_image(img1_path)
    name0 = Path(img0_path).name
    name1 = Path(img1_path).name

    print(f"\n[{prefix}] {name0} ↔ {name1}")
    print(f"  images:  {gray0.shape[::-1]}  {gray1.shape[::-1]}")

    # ── SuperPoint ───────────────────────────────────────────────
    sp0 = sp_ext(gray0)
    sp1 = sp_ext(gray1)
    print(f"  SP kpts: {len(sp0['keypoints'])}  |  {len(sp1['keypoints'])}")

    # ── SuperGlue ────────────────────────────────────────────────
    sg = sg_mat(
        sp0["keypoints"], sp1["keypoints"],
        sp0["descriptors"], sp1["descriptors"],
        sp0["scores"],     sp1["scores"],
        gray0.shape, gray1.shape,
    )
    metrics = compute_metrics(sp0, sp1, sg)
    report  = format_report(name0, name1, metrics)
    print(report)

    # ── save analysis text ───────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}_analysis.txt").write_text(report + "\n")

    # ── plots ────────────────────────────────────────────────────
    plot_matches(bgr0, bgr1,
                 sp0["keypoints"], sp1["keypoints"],
                 sg["matches0"],   sg["mscores0"],
                 name0, name1,
                 out_path=out_dir / f"{prefix}_matches.png")

    plot_keypoints(bgr0, bgr1, sp0, sp1, name0, name1,
                   out_path=out_dir / f"{prefix}_keypoints.png")

    plot_score_histogram(sp0, sp1, sg, name0, name1,
                         out_path=out_dir / f"{prefix}_histograms.png")

    return metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sp_onnx",    type=Path, default=DEFAULT_SP_ONNX)
    p.add_argument("--sg_onnx",    type=Path, default=DEFAULT_SG_ONNX)
    p.add_argument("--img0",       type=Path, default=ROOT / "assets/boat1.png",
                   help="First image (default: assets/boat1.png)")
    p.add_argument("--img1",       type=Path, default=ROOT / "assets/boat2.png",
                   help="Second image (default: assets/boat2.png)")
    p.add_argument("--input_dir",  type=Path, default=None,
                   help="Directory of images; runs consecutive pairs")
    p.add_argument("--max_pairs",  type=int,  default=5)
    p.add_argument("--out_dir",    type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--resize",     type=int,  default=RESIZE_MAX,
                   help="Long-edge resize (0 = no resize)")
    p.add_argument("--max_kpts",   type=int,  default=MAX_KEYPOINTS)
    p.add_argument("--thresh",     type=float,default=DETECTION_THRESH)
    p.add_argument("--match_thresh",type=float,default=MATCH_THRESH)
    p.add_argument("--gpu",        action="store_true",
                   help="Use CUDAExecutionProvider if available")
    return p.parse_args()


def main():
    args = parse_args()

    for f in [args.sp_onnx, args.sg_onnx]:
        if not f.exists():
            print(f"Error: ONNX file not found: {f}", file=sys.stderr)
            print("Run export_to_onnx.py first.", file=sys.stderr)
            sys.exit(1)

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.gpu else ["CPUExecutionProvider"])

    global RESIZE_MAX, MAX_KEYPOINTS, DETECTION_THRESH, MATCH_THRESH
    RESIZE_MAX       = args.resize
    MAX_KEYPOINTS    = args.max_kpts
    DETECTION_THRESH = args.thresh
    MATCH_THRESH     = args.match_thresh

    print(f"[init] Loading SP ONNX: {args.sp_onnx}")
    sp_ext = SPExtractor(args.sp_onnx, providers)
    print(f"[init] Loading SG ONNX: {args.sg_onnx}")
    sg_mat = SGMatcher(args.sg_onnx,   providers)

    if args.input_dir:
        exts   = {".png", ".jpg", ".jpeg"}
        imgs   = sorted(p for p in Path(args.input_dir).iterdir()
                        if p.suffix.lower() in exts)
        pairs  = list(zip(imgs, imgs[1:]))
        if args.max_pairs:
            pairs = pairs[:args.max_pairs]
        for i, (p0, p1) in enumerate(pairs):
            run_pair(sp_ext, sg_mat, p0, p1, args.out_dir, prefix=f"pair_{i:03d}")
    else:
        run_pair(sp_ext, sg_mat, args.img0, args.img1, args.out_dir, prefix="pair_000")

    print(f"\nDone. Results in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
