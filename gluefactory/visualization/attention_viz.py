"""Attention map and feature map visualization utilities for SuperGlue and SuperPoint."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
import torch
from pathlib import Path


# ---------------------------------------------------------------------------
# SuperGlue — attention capture
# ---------------------------------------------------------------------------

def patch_superglue_attention(superglue_module):
    """Monkey-patch MultiHeadedAttention.forward to store attention probs.

    Leverages the existing `layer.attn.prob = []` reset in AttentionalGNN.forward
    (superglue.py:160) — after the patch each call appends its prob tensor to
    that list, so per GNN-layer you get [prob_for_img0, prob_for_img1].

    Call before inference; no model file is modified.
    """
    import gluefactory_nonfree.superglue as sg_module

    def patched_forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [
            proj(x).view(b, self.dim, self.h, -1)
            for proj, x in zip(self.proj, (query, key, value))
        ]
        x, prob = sg_module.attention(query, key, value)
        if hasattr(self, "prob") and isinstance(self.prob, list):
            self.prob.append(prob.detach().cpu())
        return self.merge(x.contiguous().view(b, self.dim * self.h, -1))

    sg_module.MultiHeadedAttention.forward = patched_forward


def collect_attention_maps(model):
    """Return attention maps captured after a forward pass.

    Returns list of dicts (one per GNN layer):
      {"type": "self"|"cross",
       "prob_img0": (1, 4, N, N) or (1, 4, N0, N1),
       "prob_img1": (1, 4, N, N) or (1, 4, N1, N0)}
    """
    gnn = _get_gnn(model)
    maps = []
    for layer_mod, name in zip(gnn.layers, gnn.names):
        probs = layer_mod.attn.prob
        entry = {
            "type": name,
            "prob_img0": probs[0] if len(probs) > 0 else None,
            "prob_img1": probs[1] if len(probs) > 1 else None,
        }
        maps.append(entry)
    return maps


def _get_gnn(model):
    if hasattr(model, "matcher") and hasattr(model.matcher, "gnn"):
        return model.matcher.gnn
    if hasattr(model, "gnn"):
        return model.gnn
    raise ValueError("Cannot locate AttentionalGNN in model")


# ---------------------------------------------------------------------------
# SuperGlue — plotting
# ---------------------------------------------------------------------------

def plot_attention_layer(prob, layer_idx, layer_type, save_path=None):
    """Plot 4 attention heads + mean for one GNN layer.

    prob: (1, 4, N_q, N_k) tensor
    """
    prob_np = prob[0].numpy()  # (4, N_q, N_k)
    mean_prob = prob_np.mean(0)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(
        f"SuperGlue GNN — Layer {layer_idx:02d} ({layer_type}-attention)", fontsize=13
    )
    cmaps = ["viridis", "plasma", "magma", "inferno"]
    for h in range(4):
        im = axes[h].imshow(prob_np[h], aspect="auto", cmap=cmaps[h], vmin=0)
        axes[h].set_title(f"Head {h + 1}")
        axes[h].set_xlabel("Key kpts")
        if h == 0:
            axes[h].set_ylabel("Query kpts")
        plt.colorbar(im, ax=axes[h], fraction=0.046, pad=0.04)

    im = axes[4].imshow(mean_prob, aspect="auto", cmap="hot", vmin=0)
    axes[4].set_title("Mean (all 4 heads)")
    axes[4].set_xlabel("Key kpts")
    plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close()
    else:
        plt.show()


def plot_attention_summary(attention_maps, save_path=None):
    """Summary grid: 9 rounds × 2 rows (self / cross), mean head only."""
    self_maps = [m for m in attention_maps if m["type"] == "self"]
    cross_maps = [m for m in attention_maps if m["type"] == "cross"]
    rounds = max(len(self_maps), len(cross_maps))

    fig, axes = plt.subplots(2, rounds, figsize=(rounds * 3, 7))
    fig.suptitle(
        "SuperGlue GNN Attention Summary — mean over 4 heads (img0 direction)",
        fontsize=13,
    )
    row_labels = ["Self-attention", "Cross-attention"]

    for col, m in enumerate(self_maps):
        if m["prob_img0"] is None:
            continue
        prob_np = m["prob_img0"][0].numpy().mean(0)
        axes[0, col].imshow(prob_np, aspect="auto", cmap="viridis", vmin=0)
        axes[0, col].set_title(f"Round {col + 1}", fontsize=9)
        axes[0, col].axis("off")

    for col, m in enumerate(cross_maps):
        if m["prob_img0"] is None:
            continue
        prob_np = m["prob_img0"][0].numpy().mean(0)
        axes[1, col].imshow(prob_np, aspect="auto", cmap="plasma", vmin=0)
        axes[1, col].set_title(f"Round {col + 1}", fontsize=9)
        axes[1, col].axis("off")

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# SuperPoint — hooks
# ---------------------------------------------------------------------------

def register_superpoint_hooks(model):
    """Register forward hooks on SuperPoint encoder conv stages and descriptor head.

    Handles both the nonfree SuperPoint (conv1b/2b/3b/4b) and the open-source
    variant (backbone[0..3] Sequential with VGGBlocks).

    Returns (hook_handles, activations_dict).  Remove handles when done.
    activations_dict keys: stage1..stage4, dense_descriptors.
    """
    sp = _get_superpoint(model)
    activations = {}
    handles = []

    if hasattr(sp, "conv1b"):
        # Nonfree SuperPoint: named conv layers
        stage_modules = {
            "stage1": sp.conv1b,
            "stage2": sp.conv2b,
            "stage3": sp.conv3b,
            "stage4": sp.conv4b,
        }
    elif hasattr(sp, "backbone"):
        # Open-source SuperPoint: backbone is a Sequential of 4 stage blocks.
        # Hook the last VGGBlock of each stage (before MaxPool where applicable).
        stage_modules = {
            "stage1": sp.backbone[0][1],
            "stage2": sp.backbone[1][1],
            "stage3": sp.backbone[2][1],
            "stage4": sp.backbone[3][1],
        }
    else:
        raise ValueError(
            f"Unrecognised SuperPoint architecture: {list(sp._modules.keys())}"
        )

    for name, mod in stage_modules.items():
        def _make_hook(n):
            def _hook(module, inp, out):
                # VGGBlock returns a tensor; capture the bn output (post-activation)
                activations[n] = out.detach().cpu()
            return _hook
        handles.append(mod.register_forward_hook(_make_hook(name)))

    return handles, activations


def _get_superpoint(model):
    if hasattr(model, "extractor") and model.extractor is not None:
        return model.extractor
    if hasattr(model, "conv1a") or hasattr(model, "backbone"):
        return model
    raise ValueError("Cannot locate SuperPoint encoder in model")


# ---------------------------------------------------------------------------
# SuperPoint — plotting
# ---------------------------------------------------------------------------

def plot_superpoint_feature_maps(activations, save_dir):
    """Plot 8 highest-activation channels for each encoder stage."""
    save_dir = Path(save_dir)
    stage_info = [
        ("stage1", "Stage 1  H/2×W/2", "Blues"),
        ("stage2", "Stage 2  H/4×W/4", "Greens"),
        ("stage3", "Stage 3  H/8×W/8", "Oranges"),
        ("stage4", "Stage 4  H/8×W/8", "Reds"),
    ]

    for stage_key, title, cmap in stage_info:
        if stage_key not in activations:
            continue
        feat = activations[stage_key][0]  # (C, H, W)
        means = feat.mean(dim=(1, 2))
        top_ch = means.argsort(descending=True)[:8].tolist()

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"SuperPoint Feature Maps — {title}", fontsize=12)

        for i, ch in enumerate(top_ch):
            ax = axes[i // 4, i % 4]
            fmap = feat[ch].numpy()
            im = ax.imshow(fmap, cmap=cmap)
            ax.set_title(f"ch {ch}  (μ={means[ch]:.3f})", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        out = save_dir / f"superpoint_features_{stage_key}.png"
        plt.savefig(out, bbox_inches="tight", dpi=100)
        plt.close()
        print(f"  Saved {out.name}")


def plot_superpoint_detector_heatmap(dense_scores, image_np, save_path):
    """Overlay detection score heatmap on image.

    dense_scores: (H, W) numpy array
    image_np: (H, W) or (H, W, 3) numpy array
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("SuperPoint — Detection Score Heatmap", fontsize=13)

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Input image")
    axes[0].axis("off")

    im = axes[1].imshow(dense_scores, cmap="hot")
    axes[1].set_title("Detection heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(image_np, cmap="gray", alpha=0.6)
    axes[2].imshow(dense_scores, cmap="hot", alpha=0.55,
                   extent=[0, image_np.shape[1], image_np.shape[0], 0])
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()


def plot_superpoint_descriptor_pca(dense_desc, image_np, save_path):
    """Visualize dense descriptor field via PCA → RGB.

    dense_desc: (1, C, H, W) or (C, H, W) tensor
    """
    from sklearn.decomposition import PCA

    if dense_desc.dim() == 4:
        dense_desc = dense_desc[0]
    C, H, W = dense_desc.shape
    desc_flat = dense_desc.reshape(C, -1).T.numpy()  # (H*W, C)

    pca = PCA(n_components=3)
    pca_out = pca.fit_transform(desc_flat)  # (H*W, 3)

    for i in range(3):
        mn, mx = pca_out[:, i].min(), pca_out[:, i].max()
        pca_out[:, i] = (pca_out[:, i] - mn) / (mx - mn + 1e-8)

    rgb = pca_out.reshape(H, W, 3)
    var_pct = pca.explained_variance_ratio_.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SuperPoint — Descriptor Field (PCA → RGB)", fontsize=13)

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    axes[1].set_title(f"PCA 3/{C} components  ({var_pct:.1f}% variance)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# SuperGlue — research-paper-style attention overlay on images
# ---------------------------------------------------------------------------

def plot_superglue_attention_on_image(
    attention_maps, kpts0_xy, kpts1_xy, image0_np, image1_np, save_path,
    query_indices=None, rounds_to_show=(0, 4, 8),
):
    """Visualize attention as a spatial heatmap projected back onto the image.

    Mimics SuperGlue paper figures:
      - Pick a few query keypoints (marked with a star)
      - For self-attention: render their attention weights over the SAME image
      - For cross-attention: render their attention weights over the OTHER image
      - Uses gaussian splatting at each key keypoint position, weighted by attention

    Args:
        attention_maps:  output of collect_attention_maps()
        kpts0_xy / kpts1_xy: (1, N, 2) or (N, 2) keypoint tensors
        image0_np / image1_np: (H, W) grayscale numpy arrays
        save_path:       output path
        query_indices:   list of keypoint indices to use as queries; None → auto-pick 4
        rounds_to_show:  which GNN rounds (0-based) to include (3 columns per round)
    """
    kp0 = kpts0_xy[0].cpu().numpy() if kpts0_xy.dim() == 3 else kpts0_xy.cpu().numpy()
    kp1 = kpts1_xy[0].cpu().numpy() if kpts1_xy.dim() == 3 else kpts1_xy.cpu().numpy()

    # Auto-pick 4 query keypoints spread across the image if not specified
    if query_indices is None:
        n = len(kp0)
        query_indices = [n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5]

    n_queries = len(query_indices)
    n_rounds = len(rounds_to_show)

    # Layout: n_queries rows × (self-attn image | cross-attn image | separator) cols per round
    # Simplified: 2 cols per round (self, cross), n_queries rows
    fig, axes = plt.subplots(
        n_queries, n_rounds * 2,
        figsize=(n_rounds * 2 * 5, n_queries * 4),
    )
    if n_queries == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        "SuperGlue Attention on Image\n"
        "Left col = self-attention (same image)   Right col = cross-attention (other image)",
        fontsize=13,
    )

    H0, W0 = image0_np.shape[:2]
    H1, W1 = image1_np.shape[:2]

    for col_pair, round_idx in enumerate(rounds_to_show):
        # self-attention is at even GNN layer indices, cross at odd
        self_layer_idx = round_idx * 2
        cross_layer_idx = round_idx * 2 + 1

        self_map = attention_maps[self_layer_idx] if self_layer_idx < len(attention_maps) else None
        cross_map = attention_maps[cross_layer_idx] if cross_layer_idx < len(attention_maps) else None

        for row, q_idx in enumerate(query_indices):
            col_self = col_pair * 2
            col_cross = col_pair * 2 + 1

            # --- self-attention on img0 ---
            ax_s = axes[row, col_self]
            ax_s.imshow(image0_np, cmap="gray")
            if self_map is not None and self_map["prob_img0"] is not None and q_idx < len(kp0):
                # Mean over 4 heads for this query: shape (N_key,)
                attn_weights = self_map["prob_img0"][0, :, q_idx, :].mean(0).numpy()
                _render_attention_heatmap(ax_s, kp0, attn_weights, H0, W0)
                # Mark the query keypoint
                ax_s.scatter(
                    kp0[q_idx, 0], kp0[q_idx, 1],
                    marker="*", s=300, c="lime", zorder=5, edgecolors="black", linewidths=0.5,
                )
            if row == 0:
                ax_s.set_title(f"Round {round_idx+1} — Self-attn", fontsize=9)
            ax_s.axis("off")

            # --- cross-attention on img1 ---
            ax_c = axes[row, col_cross]
            ax_c.imshow(image1_np, cmap="gray")
            if cross_map is not None and cross_map["prob_img0"] is not None and q_idx < len(kp0):
                # Cross-attention from img0 query → img1 keys
                attn_weights = cross_map["prob_img0"][0, :, q_idx, :].mean(0).numpy()
                _render_attention_heatmap(ax_c, kp1, attn_weights, H1, W1)
                ax_s.scatter(
                    kp0[q_idx, 0], kp0[q_idx, 1],
                    marker="*", s=300, c="lime", zorder=5, edgecolors="black", linewidths=0.5,
                )
            if row == 0:
                ax_c.set_title(f"Round {round_idx+1} — Cross-attn", fontsize=9)
            ax_c.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=110)
    plt.close()


def _render_attention_heatmap(ax, kp_xy, attn_weights, H, W, sigma_frac=0.04):
    """Splat gaussian blobs at each keypoint position weighted by attention.

    attn_weights: (N_key,) numpy array, already normalized (sum ~1)
    """
    sigma = max(H, W) * sigma_frac
    grid_h, grid_w = H // 4, W // 4
    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)

    # Scale keypoints to grid resolution
    scale_x = grid_w / W
    scale_y = grid_h / H
    kp_grid = kp_xy * np.array([scale_x, scale_y])

    ys, xs = np.mgrid[0:grid_h, 0:grid_w]
    sigma_g = sigma * max(scale_x, scale_y)

    for i, (x, y) in enumerate(kp_grid):
        w = attn_weights[i]
        if w < 1e-4:
            continue
        blob = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma_g ** 2))
        heatmap += w * blob

    # Normalize
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    ax.imshow(
        heatmap, cmap="hot", alpha=0.65, vmin=0, vmax=1,
        extent=[0, W, H, 0], origin="upper",
    )


# ---------------------------------------------------------------------------
# SuperGlue — match confidence heatmap (analog of SP detector heatmap)
# ---------------------------------------------------------------------------

def plot_superglue_match_heatmap(
    sg_pred, kpts0_xy, kpts1_xy, image0_np, image1_np, save_path
):
    """Visualize SuperGlue matching confidence overlaid on both images.

    Mirrors the SuperPoint detector heatmap style:
      - Matched keypoints shown as colored dots (hot colormap by confidence)
      - Unmatched keypoints shown in blue
      - Dense confidence map per image (scatter → interpolated grid)
      - Side-by-side: raw image / confidence scatter / soft assignment matrix

    Args:
        sg_pred:    output dict from SuperGlue forward (matches0, matching_scores0, etc.)
        kpts0_xy:   (N, 2) or (1, N, 2) float tensor, keypoint (x, y) in image0
        kpts1_xy:   (M, 2) or (1, M, 2) float tensor, keypoint (x, y) in image1
        image0_np:  (H, W) numpy grayscale array
        image1_np:  (H, W) numpy grayscale array
        save_path:  output path
    """
    # ---- unpack predictions ------------------------------------------------
    m0 = sg_pred["matches0"][0].cpu().numpy()           # (N,) index into kpts1, -1=unmatched
    m1 = sg_pred["matches1"][0].cpu().numpy()           # (M,)
    sc0 = sg_pred["matching_scores0"][0].cpu().numpy()  # (N,) confidence
    sc1 = sg_pred["matching_scores1"][0].cpu().numpy()  # (M,)

    kp0 = kpts0_xy[0].cpu().numpy() if kpts0_xy.dim() == 3 else kpts0_xy.cpu().numpy()
    kp1 = kpts1_xy[0].cpu().numpy() if kpts1_xy.dim() == 3 else kpts1_xy.cpu().numpy()

    matched0 = m0 >= 0
    matched1 = m1 >= 0

    H0, W0 = image0_np.shape[:2]
    H1, W1 = image1_np.shape[:2]

    # ---- soft assignment matrix (Sinkhorn scores, no dustbin) --------------
    if "sinkhorn_cost" in sg_pred:
        assign = sg_pred["sinkhorn_cost"][0].detach().cpu().numpy()  # (N, M)
    elif "log_assignment" in sg_pred:
        assign = sg_pred["log_assignment"][0, :-1, :-1].detach().cpu().exp().numpy()
    else:
        assign = None

    # ---- build figure: 3-row layout ----------------------------------------
    # Rows 1–2: per-image overlay (raw / scatter / interpolated grid)
    # Row 3: assignment matrix + confidence histograms
    n_rows = 3 if assign is not None else 2
    fig = plt.figure(figsize=(18, n_rows * 5))
    fig.suptitle("SuperGlue — Match Confidence Heatmap", fontsize=14)

    for row, (img_np, kp, matched, scores, H, W, label) in enumerate([
        (image0_np, kp0, matched0, sc0, H0, W0, "Image 0"),
        (image1_np, kp1, matched1, sc1, H1, W1, "Image 1"),
    ]):
        # col 1: raw image
        ax1 = fig.add_subplot(n_rows, 3, row * 3 + 1)
        ax1.imshow(img_np, cmap="gray")
        ax1.set_title(f"{label} — original")
        ax1.axis("off")

        # col 2: confidence scatter over image
        ax2 = fig.add_subplot(n_rows, 3, row * 3 + 2)
        ax2.imshow(img_np, cmap="gray", alpha=0.6)
        if (~matched).any():
            ax2.scatter(
                kp[~matched, 0], kp[~matched, 1],
                c="royalblue", s=8, alpha=0.5, label="Unmatched", linewidths=0,
            )
        if matched.any():
            sc_im = ax2.scatter(
                kp[matched, 0], kp[matched, 1],
                c=scores[matched], cmap="hot", s=20, vmin=0, vmax=1,
                label="Matched", linewidths=0, zorder=3,
            )
            plt.colorbar(sc_im, ax=ax2, fraction=0.046, pad=0.04, label="confidence")
        n_m = matched.sum()
        ax2.set_title(f"{label} — {n_m}/{len(kp)} matched")
        ax2.legend(loc="upper right", fontsize=7, markerscale=1.5)
        ax2.axis("off")

        # col 3: dense confidence grid via griddata interpolation
        ax3 = fig.add_subplot(n_rows, 3, row * 3 + 3)
        ax3.imshow(img_np, cmap="gray", alpha=0.5)
        if matched.any():
            from scipy.interpolate import griddata
            grid_x, grid_y = np.meshgrid(
                np.linspace(0, W - 1, W // 4),
                np.linspace(0, H - 1, H // 4),
            )
            conf_grid = griddata(
                kp[matched], scores[matched],
                (grid_x, grid_y), method="linear",
            )
            im = ax3.imshow(
                conf_grid, cmap="hot", alpha=0.6, vmin=0, vmax=1,
                extent=[0, W, H, 0], origin="upper",
            )
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="confidence")
        ax3.set_title(f"{label} — interpolated confidence map")
        ax3.axis("off")

    # ---- row 3: assignment matrix + histograms -----------------------------
    if assign is not None:
        ax_a = fig.add_subplot(n_rows, 3, (n_rows - 1) * 3 + 1)
        im = ax_a.imshow(assign, aspect="auto", cmap="plasma")
        ax_a.set_title("Soft assignment matrix (N₀ × N₁)")
        ax_a.set_xlabel("Keypoints image 1")
        ax_a.set_ylabel("Keypoints image 0")
        plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)

        ax_h0 = fig.add_subplot(n_rows, 3, (n_rows - 1) * 3 + 2)
        ax_h0.hist(sc0[matched0], bins=30, color="tomato", alpha=0.8, label="matched")
        ax_h0.hist(sc0[~matched0], bins=30, color="steelblue", alpha=0.6, label="unmatched")
        ax_h0.set_title("Image 0 — confidence distribution")
        ax_h0.set_xlabel("Confidence")
        ax_h0.set_ylabel("Count")
        ax_h0.legend(fontsize=8)

        ax_h1 = fig.add_subplot(n_rows, 3, (n_rows - 1) * 3 + 3)
        ax_h1.hist(sc1[matched1], bins=30, color="tomato", alpha=0.8, label="matched")
        ax_h1.hist(sc1[~matched1], bins=30, color="steelblue", alpha=0.6, label="unmatched")
        ax_h1.set_title("Image 1 — confidence distribution")
        ax_h1.set_xlabel("Confidence")
        ax_h1.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# SuperGlue — paper-style attention line visualization
# ---------------------------------------------------------------------------

def plot_superglue_attention_lines(
    attention_maps, kpts0_xy, kpts1_xy, image0_np, image1_np, save_path,
    n_layers=5, top_k=15,
):
    """Reproduce the SuperGlue paper figure style.

    Grid layout:
      Left half  — Self-Attention:  each row shows img0 | img1 independently,
                   with lines from one query keypoint to its top-K attended kpts
                   within the SAME image.
      Right half — Cross-Attention: each row shows img0 | img1 side-by-side,
                   with lines going ACROSS from img0 query to top-K kpts in img1
                   (drawn with ConnectionPatch so they span the panel boundary).

    Line colors: green=highest attention → red=lower (continuous RdYlGn colormap).
    Line width and alpha scale with attention weight.
    All non-query / non-attended keypoints shown as small white dots.
    """
    kp0 = kpts0_xy[0].cpu().numpy() if kpts0_xy.dim() == 3 else kpts0_xy.cpu().numpy()
    kp1 = kpts1_xy[0].cpu().numpy() if kpts1_xy.dim() == 3 else kpts1_xy.cpu().numpy()

    # Select which layers to show
    self_layers  = [i for i, m in enumerate(attention_maps) if m["type"] == "self"]
    cross_layers = [i for i, m in enumerate(attention_maps) if m["type"] == "cross"]
    sel_self  = _sample_layers(self_layers,  n_layers)
    sel_cross = _sample_layers(cross_layers, n_layers)

    img_h, img_w = image0_np.shape[:2]
    row_h = max(3.5, img_h / img_w * 4.5)

    fig = plt.figure(figsize=(22, n_layers * row_h + 0.6))
    # 1 header row + n_layers data rows; 4 columns (s_img0, s_img1, c_img0, c_img1)
    gs = gridspec.GridSpec(
        n_layers + 1, 4,
        figure=fig,
        height_ratios=[0.25] + [1.0] * n_layers,
        hspace=0.06, wspace=0.03,
    )

    # Column headers
    for col_span, title in [([0, 2], "Self-Attention"), ([2, 4], "Cross-Attention")]:
        ax_h = fig.add_subplot(gs[0, col_span[0]:col_span[1]])
        ax_h.text(0.5, 0.5, title, ha="center", va="center",
                  fontsize=15, fontweight="bold", transform=ax_h.transAxes)
        ax_h.axis("off")

    cmap = plt.get_cmap("RdYlGn")

    for row in range(n_layers):
        # ------------------------------------------------------------------ #
        #  Self-attention                                                       #
        # ------------------------------------------------------------------ #
        sl = sel_self[row]
        sm = attention_maps[sl]
        prob0 = sm["prob_img0"][0].numpy() if sm["prob_img0"] is not None else None  # (4, N, N)
        prob1 = sm["prob_img1"][0].numpy() if sm["prob_img1"] is not None else None

        ax_s0 = fig.add_subplot(gs[row + 1, 0])
        ax_s1 = fig.add_subplot(gs[row + 1, 1])

        if prob0 is not None:
            head_s, q_s0 = _pick_best_head_and_query(prob0, len(kp0))
            label_s = f"Layer {sl}\nSelf-Attention\nHead {head_s}"
            _draw_attn_lines_self(ax_s0, image0_np, kp0, prob0[head_s], q_s0, top_k, cmap, label_s)
        else:
            ax_s0.imshow(image0_np, cmap="gray"); ax_s0.axis("off")

        if prob1 is not None:
            head_s1, q_s1 = _pick_best_head_and_query(prob1, len(kp1))
            _draw_attn_lines_self(ax_s1, image1_np, kp1, prob1[head_s1], q_s1, top_k, cmap, "")
        else:
            ax_s1.imshow(image1_np, cmap="gray"); ax_s1.axis("off")

        # ------------------------------------------------------------------ #
        #  Cross-attention                                                      #
        # ------------------------------------------------------------------ #
        cl = sel_cross[row]
        cm = attention_maps[cl]
        prob_c = cm["prob_img0"][0].numpy() if cm["prob_img0"] is not None else None  # (4, N0, N1)

        ax_c0 = fig.add_subplot(gs[row + 1, 2])
        ax_c1 = fig.add_subplot(gs[row + 1, 3])

        if prob_c is not None:
            head_c, q_c = _pick_best_head_and_query(prob_c, len(kp0))
            label_c = f"Layer {cl}\nCross-Attention\nHead {head_c}"
            _draw_attn_lines_cross(
                fig, ax_c0, ax_c1,
                image0_np, image1_np,
                kp0, kp1,
                prob_c[head_c],   # (N0, N1)
                q_c, top_k, cmap, label_c,
            )
        else:
            ax_c0.imshow(image0_np, cmap="gray"); ax_c0.axis("off")
            ax_c1.imshow(image1_np, cmap="gray"); ax_c1.axis("off")

    plt.savefig(save_path, bbox_inches="tight", dpi=130)
    plt.close()


# ---- helpers ----------------------------------------------------------------

def _sample_layers(layer_list, n):
    """Pick n evenly-spaced indices from layer_list."""
    if len(layer_list) <= n:
        return layer_list
    step = len(layer_list) / n
    return [layer_list[int(i * step)] for i in range(n)]


def _pick_best_head_and_query(prob, n_kpts):
    """Return (head_idx, query_idx) with the highest single attention weight.

    prob: (H, N_q, N_k)  — already numpy
    """
    # Max attention over all (head, query, key) combinations
    # We want the query that is most "peaked" (high max weight)
    # Use the head+query that gives the single highest attention weight to any key
    flat_max = prob.reshape(prob.shape[0], prob.shape[1], -1).max(-1)  # (H, N_q)
    h_idx, q_idx = np.unravel_index(flat_max.argmax(), flat_max.shape)
    q_idx = min(q_idx, n_kpts - 1)
    return int(h_idx), int(q_idx)


def _attn_line_style(rank, total, weight):
    """Return (color, linewidth, alpha) for a line at `rank` out of `total`."""
    t = 1.0 - rank / max(total - 1, 1)  # 1=best, 0=worst
    color = plt.get_cmap("RdYlGn")(t)
    lw = 0.8 + 2.2 * t
    alpha = 0.35 + 0.65 * t
    return color, lw, alpha


def _draw_attn_lines_self(ax, image_np, kp_xy, prob_head, q_idx, top_k, cmap, label):
    """Draw self-attention lines for one image on one axes.

    prob_head: (N, N) — attention from every query to every key, one head
    q_idx:     int    — the query keypoint index
    """
    ax.imshow(image_np, cmap="gray")
    H, W = image_np.shape[:2]

    # All keypoints as white dots
    ax.scatter(kp_xy[:, 0], kp_xy[:, 1], s=6, c="white", linewidths=0,
               alpha=0.5, zorder=2)

    # Top-k attended keypoints
    attn = prob_head[q_idx]          # (N_key,)
    top_idx = np.argsort(attn)[::-1][:top_k]

    for rank, ki in enumerate(top_idx):
        if ki == q_idx:
            continue
        color, lw, alpha = _attn_line_style(rank, top_k, attn[ki])
        ax.plot(
            [kp_xy[q_idx, 0], kp_xy[ki, 0]],
            [kp_xy[q_idx, 1], kp_xy[ki, 1]],
            color=color, linewidth=lw, alpha=alpha, zorder=3,
        )
        ax.scatter(kp_xy[ki, 0], kp_xy[ki, 1], s=18, color=color,
                   linewidths=0, zorder=4)

    # Query keypoint (star)
    top_color, _, _ = _attn_line_style(0, top_k, 1.0)
    ax.scatter(kp_xy[q_idx, 0], kp_xy[q_idx, 1],
               marker="*", s=250, color=top_color, edgecolors="white",
               linewidths=0.4, zorder=5)

    if label:
        ax.text(0.01, 0.99, label, transform=ax.transAxes,
                fontsize=7, color="white", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.axis("off")


def _draw_attn_lines_cross(
    fig, ax0, ax1, image0_np, image1_np, kp0, kp1,
    prob_head, q_idx, top_k, cmap, label,
):
    """Draw cross-attention lines spanning from ax0 (img0) to ax1 (img1).

    prob_head: (N0, N1) — cross-attention from img0 queries to img1 keys, one head
    q_idx:     int      — query keypoint index in img0
    """
    H0, W0 = image0_np.shape[:2]
    H1, W1 = image1_np.shape[:2]

    ax0.imshow(image0_np, cmap="gray")
    ax1.imshow(image1_np, cmap="gray")

    # All keypoints as white dots
    ax0.scatter(kp0[:, 0], kp0[:, 1], s=6, c="white", linewidths=0, alpha=0.5, zorder=2)
    ax1.scatter(kp1[:, 0], kp1[:, 1], s=6, c="white", linewidths=0, alpha=0.5, zorder=2)

    # Top-k attended keypoints in img1
    attn = prob_head[q_idx]         # (N1,)
    top_idx = np.argsort(attn)[::-1][:top_k]

    for rank, ki in enumerate(top_idx):
        color, lw, alpha = _attn_line_style(rank, top_k, attn[ki])

        # Line from query (ax0) to key (ax1) using ConnectionPatch
        con = ConnectionPatch(
            xyA=(kp0[q_idx, 0], kp0[q_idx, 1]), coordsA=ax0.transData,
            xyB=(kp1[ki, 0],    kp1[ki, 1]),    coordsB=ax1.transData,
            color=color, linewidth=lw, alpha=alpha, zorder=3,
        )
        fig.add_artist(con)

        # Attended keypoint dot on ax1
        ax1.scatter(kp1[ki, 0], kp1[ki, 1], s=18, color=color,
                    linewidths=0, zorder=4)

    # Query keypoint star on ax0
    top_color, _, _ = _attn_line_style(0, top_k, 1.0)
    ax0.scatter(kp0[q_idx, 0], kp0[q_idx, 1],
                marker="*", s=250, color=top_color, edgecolors="white",
                linewidths=0.4, zorder=5)

    if label:
        ax0.text(0.01, 0.99, label, transform=ax0.transAxes,
                 fontsize=7, color="white", va="top",
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0))

    ax0.set_xlim(0, W0); ax0.set_ylim(H0, 0); ax0.axis("off")
    ax1.set_xlim(0, W1); ax1.set_ylim(H1, 0); ax1.axis("off")
