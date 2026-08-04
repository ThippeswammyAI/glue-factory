"""
Visualize SuperGlue attention maps and SuperPoint feature maps from checkpoints.

Runs entirely on CPU — safe to use while GPU training is in progress.

Usage:
    python -m gluefactory.scripts.visualize_attention \
        --superglue outputs/training/superglue_slam_run/checkpoint_best.tar \
        --superpoint outputs/training/superpoint_slam_run/checkpoint_best.tar \
        --img0 data/output/sample_slam/images/rgb/1775717079.407016.png \
        --img1 data/output/sample_slam/images/rgb/1775717079.507152.png \
        --output_dir outputs/visualizations/attention_maps
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from gluefactory.utils.experiments import load_experiment
from gluefactory.visualization.attention_viz import (
    collect_attention_maps,
    patch_superglue_attention,
    plot_attention_layer,
    plot_attention_summary,
    plot_superglue_match_heatmap,
    plot_superglue_attention_on_image,
    plot_superglue_attention_lines,
    plot_superpoint_descriptor_pca,
    plot_superpoint_detector_heatmap,
    plot_superpoint_feature_maps,
    register_superpoint_hooks,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("viz_attention")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path, size):
    """Load image as (1,1,H,W) float32 tensor and (H,W) numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if size > 0:
        h, w = img.shape
        scale = size / max(h, w)
        new_w = int(round(w * scale / 8) * 8)
        new_h = int(round(h * scale / 8) * 8)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor, img


def run_superpoint(model, image_tensor):
    """Run SuperPoint on a single image tensor (1,1,H,W).

    Calls the extractor directly if the model is a TwoViewPipeline wrapper.
    """
    sp = model.extractor if hasattr(model, "extractor") and model.extractor is not None else model
    with torch.inference_mode():
        pred = sp({"image": image_tensor})
    return pred


def _compute_dense_scores(pred):
    """Reconstruct (H, W) dense score map from prediction dict.

    Uses pred["logits"] (open-source SP) or pred["keypoint_scores"] (nonfree SP).
    """
    if "logits" in pred:
        import torch.nn.functional as F
        logits = pred["logits"].detach().cpu()  # (1, stride²+1, H', W')
        probs = F.softmax(logits, dim=1)[:, :-1]  # drop dustbin → (1, stride², H', W')
        b, c, h, w = probs.shape
        stride = int(round(c ** 0.5))
        probs = probs.permute(0, 2, 3, 1).reshape(b, h, w, stride, stride)
        probs = probs.permute(0, 1, 3, 2, 4).reshape(b, h * stride, w * stride)
        return probs[0].numpy()
    if "keypoint_scores" in pred:
        # Nonfree SP returns dense_scores as keypoint_scores before sparse extraction
        sc = pred["keypoint_scores"]
        if sc.dim() == 3:
            return sc[0].detach().cpu().numpy()
    return None


def build_superglue_data(view0_tensor, view1_tensor, pred0, pred1):
    """Build the data dict expected by SuperGlue's _forward."""
    def _stack(t):
        return t if t.dim() == 3 else t.unsqueeze(0)

    kpts0 = pred0["keypoints"]
    kpts1 = pred1["keypoints"]
    desc0 = pred0["descriptors"]   # (1, N, D) or (N, D)
    desc1 = pred1["descriptors"]
    sc0 = pred0["keypoint_scores"]
    sc1 = pred1["keypoint_scores"]

    # Ensure batch dim
    for t in [kpts0, kpts1, desc0, desc1, sc0, sc1]:
        if t.dim() == 2 and t.shape[0] != 1:
            pass  # already (N, D) — wrap below

    if kpts0.dim() == 2:
        kpts0 = kpts0.unsqueeze(0)
    if kpts1.dim() == 2:
        kpts1 = kpts1.unsqueeze(0)
    if desc0.dim() == 2:
        desc0 = desc0.unsqueeze(0)
    if desc1.dim() == 2:
        desc1 = desc1.unsqueeze(0)
    if sc0.dim() == 1:
        sc0 = sc0.unsqueeze(0)
    if sc1.dim() == 1:
        sc1 = sc1.unsqueeze(0)

    return {
        "view0": {"image": view0_tensor},
        "view1": {"image": view1_tensor},
        "keypoints0": kpts0,
        "keypoints1": kpts1,
        "descriptors0": desc0,
        "descriptors1": desc1,
        "keypoint_scores0": sc0,
        "keypoint_scores1": sc1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--superglue",
        default="outputs/training/superglue_slam_run/checkpoint_best.tar",
        help="Path to SuperGlue checkpoint (.tar)",
    )
    p.add_argument(
        "--superpoint",
        default="outputs/training/superpoint_slam_run/checkpoint_best.tar",
        help="Path to SuperPoint checkpoint (.tar)",
    )
    p.add_argument(
        "--img0",
        default="data/output/sample_slam/images/rgb/1775717079.407016.png",
        help="Path to first image",
    )
    p.add_argument(
        "--img1",
        default="data/output/sample_slam/images/rgb/1775717079.507152.png",
        help="Path to second image",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/visualizations/attention_maps",
        help="Directory to save visualizations",
    )
    p.add_argument(
        "--resize",
        type=int,
        default=512,
        help="Resize max image dimension (0 = no resize)",
    )
    p.add_argument(
        "--layers",
        type=str,
        default="all",
        help='Which GNN layers to plot individually: "all", "none", or comma-separated indices e.g. "0,1,8,9"',
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== SuperGlue + SuperPoint attention/feature-map visualizer ===")
    log.info(f"Output directory: {out_dir}")

    # -----------------------------------------------------------------------
    # 1. Load images
    # -----------------------------------------------------------------------
    log.info("Loading images …")
    img0_tensor, img0_np = load_image(args.img0, args.resize)
    img1_tensor, img1_np = load_image(args.img1, args.resize)
    log.info(f"  img0: {img0_np.shape}  img1: {img1_np.shape}")

    # -----------------------------------------------------------------------
    # 2. Load and run SuperPoint
    # -----------------------------------------------------------------------
    log.info(f"Loading SuperPoint from {args.superpoint} …")
    sp_model = load_experiment(args.superpoint).eval()

    # Hooks capture encoder stages + dense descriptors (avoids mutating locked OmegaConf)
    sp_hooks, sp_activations = register_superpoint_hooks(sp_model)

    log.info("Running SuperPoint on img0 …")
    pred0 = run_superpoint(sp_model, img0_tensor)
    # Save activations from img0 run (overwritten each forward — keep img0's)
    sp_activations_img0 = {k: v.clone() for k, v in sp_activations.items()}

    # Dense detection score map: reconstruct from logits if available
    dense_scores_img0 = _compute_dense_scores(pred0)
    # Dense descriptors: returned directly by open-source SP when dense_outputs=True
    dense_desc_img0 = pred0.get("dense_descriptors")

    log.info("Running SuperPoint on img1 …")
    pred1 = run_superpoint(sp_model, img1_tensor)

    for h in sp_hooks:
        h.remove()

    n0 = pred0["keypoints"].shape[-2]
    n1 = pred1["keypoints"].shape[-2]
    log.info(f"  Keypoints: img0={n0}  img1={n1}")

    # -----------------------------------------------------------------------
    # 3. SuperPoint visualizations
    # -----------------------------------------------------------------------
    log.info("Generating SuperPoint visualizations …")

    plot_superpoint_feature_maps(sp_activations_img0, out_dir)

    if dense_scores_img0 is not None:
        plot_superpoint_detector_heatmap(
            dense_scores_img0, img0_np, out_dir / "superpoint_detector_heatmap.png"
        )
        log.info("  Saved superpoint_detector_heatmap.png")

    if dense_desc_img0 is not None:
        plot_superpoint_descriptor_pca(
            dense_desc_img0, img0_np, out_dir / "superpoint_descriptor_pca.png"
        )
        log.info("  Saved superpoint_descriptor_pca.png")
    else:
        log.warning("  dense_descriptors not found; skipping PCA plot")

    # -----------------------------------------------------------------------
    # 4. Load and run SuperGlue
    # -----------------------------------------------------------------------
    log.info(f"Loading SuperGlue from {args.superglue} …")
    sg_model = load_experiment(args.superglue).eval()

    log.info("Patching MultiHeadedAttention to capture attention probs …")
    patch_superglue_attention(sg_model)

    data = build_superglue_data(img0_tensor, img1_tensor, pred0, pred1)
    log.info("Running SuperGlue …")
    with torch.inference_mode():
        sg_pred = sg_model(data)

    attention_maps = collect_attention_maps(sg_model)

    n_captured = sum(1 for m in attention_maps if m["prob_img0"] is not None)
    log.info(f"  Captured attention maps: {n_captured}/{len(attention_maps)} layers")

    # -----------------------------------------------------------------------
    # 5. SuperGlue per-layer attention plots
    # -----------------------------------------------------------------------
    if args.layers == "none":
        layer_indices = []
    elif args.layers == "all":
        layer_indices = list(range(len(attention_maps)))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]

    log.info(f"Generating per-layer attention plots for {len(layer_indices)} layers …")
    for i in layer_indices:
        m = attention_maps[i]
        if m["prob_img0"] is None:
            continue
        # img0 direction (self: kp0→kp0, cross: kp0→kp1)
        fname = out_dir / f"superglue_attn_layer_{i:02d}_{m['type']}_img0.png"
        plot_attention_layer(m["prob_img0"], i, m["type"], save_path=fname)
        # img1 direction
        if m["prob_img1"] is not None:
            fname1 = out_dir / f"superglue_attn_layer_{i:02d}_{m['type']}_img1.png"
            plot_attention_layer(m["prob_img1"], i, m["type"], save_path=fname1)
    if layer_indices:
        log.info(f"  Per-layer PNGs saved to {out_dir}/superglue_attn_layer_*.png")

    # -----------------------------------------------------------------------
    # 6. SuperGlue summary grid
    # -----------------------------------------------------------------------
    log.info("Generating attention summary grid …")
    plot_attention_summary(
        attention_maps, save_path=out_dir / "superglue_attn_summary.png"
    )
    log.info("  Saved superglue_attn_summary.png")

    # -----------------------------------------------------------------------
    # 7. SuperGlue match confidence heatmap (like SP detector heatmap)
    # -----------------------------------------------------------------------
    log.info("Generating SuperGlue match confidence heatmap …")
    plot_superglue_match_heatmap(
        sg_pred,
        data["keypoints0"], data["keypoints1"],
        img0_np, img1_np,
        out_dir / "superglue_match_heatmap.png",
    )
    log.info("  Saved superglue_match_heatmap.png")

    # -----------------------------------------------------------------------
    # 8. Research-paper-style attention overlaid on images
    #    (self-attention on same image, cross-attention to other image)
    # -----------------------------------------------------------------------
    log.info("Generating research-paper-style attention image overlays …")
    plot_superglue_attention_on_image(
        attention_maps,
        data["keypoints0"], data["keypoints1"],
        img0_np, img1_np,
        out_dir / "superglue_attention_overlay.png",
    )
    log.info("  Saved superglue_attention_overlay.png")

    # -----------------------------------------------------------------------
    # 9. Paper-style attention line figure (self + cross, lines on images)
    # -----------------------------------------------------------------------
    log.info("Generating paper-style attention line visualization …")
    plot_superglue_attention_lines(
        attention_maps,
        data["keypoints0"], data["keypoints1"],
        img0_np, img1_np,
        out_dir / "superglue_attention_lines.png",
        n_layers=5, top_k=15,
    )
    log.info("  Saved superglue_attention_lines.png")

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    saved = sorted(out_dir.glob("*.png"))
    log.info(f"\nDone. {len(saved)} PNG files in {out_dir}:")
    for f in saved:
        log.info(f"  {f.name}")


if __name__ == "__main__":
    main()
