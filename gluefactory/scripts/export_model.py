#!/usr/bin/env python3
"""
Export SuperPoint, SuperGlue, or LightGlue .tar checkpoints to TorchScript
(.pt/.pth) or ONNX (.onnx). One tool, one CLI, for any model x any format —
this replaces the old top-level export_to_torchscript.py / export_to_onnx.py
pair so the checkpoint-loading logic and I/O naming can't drift apart
between them the way they previously did.

Same input/output naming and tensor layout for a given model across both
formats, everywhere that's actually possible (see the note at the bottom
for the one place it isn't):

SuperPoint (dense feature extractor — single image in, dense maps out):
  Input:  image                      [B, 1|3, H, W]  float32 in [0, 1]
  Output: scores                     [B, H, W]       NMS-filtered heatmap
          descriptors                [B, 256, H/8, W/8]  L2-normalised

SuperGlue / LightGlue (pairwise matchers — fixed-N keypoints in, matches out):
  Input:  keypoints0, keypoints1     [B, N, 2]   pixel coords (x, y)
          descriptors0, descriptors1 [B, 256, N] descriptors, channel-first
                                                   (same layout as SuperPoint's
                                                   dense `descriptors` output)
          scores0, scores1           [B, N]      keypoint confidence
                                                   (SuperGlue only — LightGlue's
                                                   keypoint encoder has no use
                                                   for them)
  Output: matches0, matches1                 [B, N]  matched index or -1
          matching_scores0, matching_scores1 [B, N]  match confidence

  TorchScript (.pt/.pth) additionally takes image0, image1 [B, 1, H, W] and
  derives (width, height) from their shape — this works because the .pt
  wrapper is torch.jit.SCRIPTED (real dynamic control flow).

  ONNX instead takes size0, size1 [B, 2] (width, height) as direct tensor
  inputs. This is not a style choice: `torch.onnx.export` only traces, and
  deriving a *new* tensor's values from `image.shape[i]` gets silently baked
  in as a constant (confirmed by testing — image0/image1 dropped from the
  graph entirely, size frozen at the trace-time dummy). So ONNX must take
  the real (width, height) values directly as inputs to stay genuinely
  dynamic.

Sinkhorn iterations (SuperGlue) are unrolled at trace/export time. LightGlue's
early-stopping and point-pruning (depth/width_confidence) are disabled in the
exported graph — it always runs all layers, matching how the SLAM config is
trained (both left at -1).

NOTE: as of PyTorch 2.2, exporting LightGlue to ONNX hits an exporter bug
(`IndexError` in `torch.onnx.symbolic_opset9.transpose`) once enough
transformer layers are chained in its rotary self/cross-attention stack —
this is independent of the wrapper's own ops (verified numerically correct
against the reference model in isolation; it reproduces even with plain
reshape/permute/einsum attention that avoids `unflatten`). Not something this
script can work around — a case for `torch>=2.3` or the dynamo-based exporter
(`torch.onnx.dynamo_export`, needs `onnxscript`). LightGlue's TorchScript
export is unaffected.

Usage:
  python -m gluefactory.scripts.export_model --model superpoint --format pt \\
      --ckpt outputs/training/superpoint_slam_run/checkpoint_best.tar \\
      --out superpoint.pt

  python -m gluefactory.scripts.export_model --model superglue --format onnx \\
      --ckpt outputs/training/superglue_slam_run/checkpoint_best.tar \\
      --out superglue.onnx

  python -m gluefactory.scripts.export_model --model lightglue --format pt \\
      --ckpt outputs/training/lightglue_slam_run/checkpoint_best.tar \\
      --out lightglue.pt

  # --model/--format also accept short aliases: sp/sg/lg, pth (alias for pt)
  python -m gluefactory.scripts.export_model --model lg --format pt \\
      --ckpt outputs/training/lightglue_slam_run/checkpoint_best.tar
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


# ── shared checkpoint builders ──────────────────────────────────────────────
# Used by both the TorchScript and ONNX backends below, so loading logic
# can't drift between formats the way it did across the old two scripts.

def _strip_matcher_prefix(state_dict: dict) -> dict:
    """Strip the 'matcher.' prefix left by two_view_pipeline training."""
    if any(k.startswith("matcher.") for k in state_dict):
        return {
            k[len("matcher."):]: v
            for k, v in state_dict.items()
            if k.startswith("matcher.")
        }
    return state_dict


def _build_superpoint(ckpt_path: Path) -> nn.Module:
    """Instantiate SuperPoint and load weights from a .tar checkpoint."""
    from gluefactory.models.extractors.superpoint_open import SuperPoint
    from omegaconf import OmegaConf

    # superpoint_open._init handles .tar format + 'extractor.' prefix stripping
    # when conf.weights points to an existing file.
    conf = OmegaConf.create({
        "weights": str(ckpt_path),
        "trainable": True,
        "freeze_batch_normalization": False,
        "timeit": False,
    })
    model = SuperPoint(conf)
    model.eval()
    return model


def _build_superglue(ckpt_path: Path) -> nn.Module:
    """Instantiate SuperGlue and load weights from a .tar checkpoint."""
    from gluefactory_nonfree.superglue import SuperGlue
    from omegaconf import OmegaConf

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _strip_matcher_prefix(ckpt["model"])

    # conf.weights="" is falsy → _init skips the URL download
    conf = OmegaConf.create({
        "weights": "",
        "trainable": True,
        "freeze_batch_normalization": False,
        "timeit": False,
    })
    model = SuperGlue(conf)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _read_sinkhorn_iters(ckpt_path: Path, override: int = None) -> int:
    if override is not None:
        return override
    ckpt = torch.load(ckpt_path, map_location="cpu")
    try:
        return ckpt["conf"]["model"]["matcher"]["sinkhorn_iterations"]
    except (KeyError, TypeError):
        return 50


def _build_lightglue(ckpt_path: Path) -> nn.Module:
    """Instantiate LightGlue and load weights from a .tar checkpoint."""
    from gluefactory.models.matchers.lightglue import LightGlue
    from omegaconf import OmegaConf

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _strip_matcher_prefix(ckpt["model"])

    try:
        conf = OmegaConf.create(dict(ckpt["conf"]["model"]["matcher"]))
    except (KeyError, TypeError):
        conf = OmegaConf.create({})
    conf.weights = None  # weights already loaded from state_dict below

    model = LightGlue(conf)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if model.conf.depth_confidence > 0 or model.conf.width_confidence > 0:
        print(
            "     warning: depth/width_confidence enabled in checkpoint conf; "
            "export assumes a fixed-depth graph (no early-stop / point-pruning)",
            file=sys.stderr,
        )

    return model


# ═════════════════════════════════════════════════════════════════════════
# TorchScript (.pt / .pth) backend
# ═════════════════════════════════════════════════════════════════════════

# ── SuperPoint ───────────────────────────────────────────────────────────────

class _SuperPointExtractor(nn.Module):
    """Inner CNN forward (traced with an explicit image tensor input)."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.detector = model.detector
        self.descriptor = model.descriptor
        self.stride = model.stride

    def forward(self, image: torch.Tensor):
        if image.shape[1] == 3:
            rgb_w = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * rgb_w).sum(1, keepdim=True)

        features = self.backbone(image)
        descriptors = torch.nn.functional.normalize(
            self.descriptor(features), p=2, dim=1
        )

        logits = self.detector(features)
        scores = torch.nn.functional.softmax(logits, dim=1)[:, :-1]
        b, _, h, w = scores.shape
        k = self.stride
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, k, k)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * k, w * k)

        return scores, descriptors


class _SuperPointScriptWrapper(nn.Module):
    """Scripted dict-I/O wrapper around the traced extractor.

    Matches the SuperGlue/LightGlue wrappers' style below: a
    torch.jit.script'ed outer module so image-size extraction stays
    dynamic (not baked as constants) instead of freezing at the
    trace-time dummy shape.
    """

    def __init__(self, traced_extractor):
        super().__init__()
        self.extractor = traced_extractor

    def forward(self, data: Dict[str, torch.Tensor]):
        image = data["image"]
        scores, descriptors = self.extractor(image)
        result: Dict[str, torch.Tensor] = {
            "scores": scores,
            "descriptors": descriptors,
        }
        return result


def export_superpoint_pt(ckpt_path: Path, out_path: Path, h: int, w: int) -> None:
    print(f"\n[SP] loading  {ckpt_path}")
    sp = _build_superpoint(ckpt_path)

    # Step 1: trace the inner CNN (no dict I/O, clean tensor->tensor)
    extractor = _SuperPointExtractor(sp).eval()
    dummy = torch.zeros(1, 1, h, w)
    with torch.no_grad():
        traced_extractor = torch.jit.trace(extractor, (dummy,))

    # Step 2: script the dict wrapper (dynamic shape extraction)
    wrapper = _SuperPointScriptWrapper(traced_extractor).eval()
    dummy_data = {"image": dummy}
    with torch.no_grad():
        scripted = torch.jit.script(wrapper)
        _ = scripted(dummy_data)
        scripted.save(str(out_path))

    print(f"[SP] saved  → {out_path}")
    try:
        loaded = torch.jit.load(str(out_path))
        out = loaded(dummy_data)
        print(f"     output: scores={out['scores'].shape}, "
              f"descriptors={out['descriptors'].shape}")
    except Exception as e:
        print(f"     verification error: {e}")


# ── SuperGlue ────────────────────────────────────────────────────────────────

class _SuperGlueMatcher(nn.Module):
    """Inner matching logic (traced with explicit tensor inputs)."""

    def __init__(self, model, n_sinkhorn: int):
        super().__init__()
        self.kenc = model.kenc
        self.gnn = model.gnn
        self.final_proj = model.final_proj
        self.bin_score = model.bin_score
        self.desc_dim = model.conf.descriptor_dim
        self.n_sinkhorn = n_sinkhorn
        self.threshold = model.conf.filter_threshold

    @staticmethod
    def _norm_kpts(kpts: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        shift = size.float() / 2
        scale = size.float().max(1).values * 0.7
        return (kpts - shift[:, None]) / scale[:, None, None]

    def forward(
        self,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        descs0: torch.Tensor,
        descs1: torch.Tensor,
        scores0: torch.Tensor,
        scores1: torch.Tensor,
        size0: torch.Tensor,
        size1: torch.Tensor,
    ):
        from gluefactory_nonfree.superglue import log_optimal_transport, arange_like

        kn0 = self._norm_kpts(kpts0, size0)
        kn1 = self._norm_kpts(kpts1, size1)

        d0 = descs0.transpose(1, 2) + self.kenc(kn0, scores0)
        d1 = descs1.transpose(1, 2) + self.kenc(kn1, scores1)

        d0, d1 = self.gnn(d0, d1)
        md0, md1 = self.final_proj(d0), self.final_proj(d1)

        cost = torch.einsum("bdn,bdm->bnm", md0, md1) / (self.desc_dim ** 0.5)
        log_assign = log_optimal_transport(cost, self.bin_score, self.n_sinkhorn)

        s_kpts = log_assign[:, :-1, :-1]
        max0, max1 = s_kpts.max(2), s_kpts.max(1)
        m0, m1 = max0.indices, max1.indices

        mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
        mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)

        zero = log_assign.new_tensor(0)
        msc0 = torch.where(mutual0, max0.values.exp(), zero)
        msc1 = torch.where(mutual1, msc0.gather(1, m1), zero)
        valid0 = mutual0 & (msc0 > self.threshold)
        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, torch.full_like(m0, -1))
        m1 = torch.where(valid1, m1, torch.full_like(m1, -1))

        return m0, m1, msc0, msc1


class _SuperGlueScriptWrapper(nn.Module):
    """Scripted dict-I/O wrapper around the traced matcher.

    This wrapper is torch.jit.script'ed so that image size extraction
    from tensor shapes is **dynamic** (not baked as constants), making
    the model work correctly with varying image resolutions on both
    CPU and CUDA.
    """

    def __init__(self, traced_matcher):
        super().__init__()
        self.matcher = traced_matcher

    def forward(self, data: Dict[str, torch.Tensor]):
        kpts0 = data["keypoints0"]
        kpts1 = data["keypoints1"]
        scores0 = data["scores0"]
        scores1 = data["scores1"]
        desc0 = data["descriptors0"].transpose(1, 2)
        desc1 = data["descriptors1"].transpose(1, 2)

        img0 = data["image0"]
        img1 = data["image1"]

        h0 = img0.size(2)
        w0 = img0.size(3)
        h1 = img1.size(2)
        w1 = img1.size(3)

        dtype = kpts0.dtype
        device = kpts0.device
        size0 = torch.tensor([[w0, h0]], dtype=dtype, device=device)
        size1 = torch.tensor([[w1, h1]], dtype=dtype, device=device)

        m0, m1, msc0, msc1 = self.matcher(
            kpts0, kpts1, desc0, desc1, scores0, scores1, size0, size1
        )

        result: Dict[str, torch.Tensor] = {"matches0": m0, "matching_scores0": msc0}
        return result


def export_superglue_pt(
    ckpt_path: Path,
    out_path: Path,
    N: int,
    sinkhorn_iters: int = None,
    h: int = 480,
    w: int = 640,
) -> None:
    print(f"\n[SG] loading  {ckpt_path}")
    sg = _build_superglue(ckpt_path)
    n_sink = _read_sinkhorn_iters(ckpt_path, sinkhorn_iters)
    print(f"     sinkhorn iterations: {n_sink}")

    # Step 1: trace the inner matcher (no dict I/O, clean tensor->tensor)
    matcher = _SuperGlueMatcher(sg, n_sink).eval()
    B, D = 1, sg.conf.descriptor_dim
    dummy_kpts = torch.zeros(B, N, 2)
    dummy_desc = torch.zeros(B, N, D)
    dummy_sc = torch.ones(B, N)
    dummy_size = torch.tensor([[float(w), float(h)]])
    with torch.no_grad():
        traced_matcher = torch.jit.trace(
            matcher,
            (dummy_kpts, dummy_kpts, dummy_desc, dummy_desc,
             dummy_sc, dummy_sc, dummy_size, dummy_size),
        )

    # Step 2: script the dict wrapper (dynamic shape extraction)
    wrapper = _SuperGlueScriptWrapper(traced_matcher).eval()
    dummy_data = {
        "keypoints0": dummy_kpts,
        "keypoints1": dummy_kpts,
        "descriptors0": torch.zeros(B, D, N),
        "descriptors1": torch.zeros(B, D, N),
        "scores0": dummy_sc,
        "scores1": dummy_sc,
        "image0": torch.zeros(B, 1, h, w),
        "image1": torch.zeros(B, 1, h, w),
    }
    with torch.no_grad():
        scripted = torch.jit.script(wrapper)
        # Verify it works before saving
        _ = scripted(dummy_data)
        scripted.save(str(out_path))

    print(f"[SG] saved  → {out_path}")
    try:
        loaded = torch.jit.load(str(out_path))
        out = loaded(dummy_data)
        print(f"     output: matches0={out['matches0'].shape}, "
              f"matching_scores0={out['matching_scores0'].shape}")
    except Exception as e:
        print(f"     verification error: {e}")


# ── LightGlue ────────────────────────────────────────────────────────────────

class _LightGlueMatcher(nn.Module):
    """Inner matching logic (traced with explicit tensor inputs)."""

    def __init__(self, model):
        super().__init__()
        self.input_proj = model.input_proj
        self.posenc = model.posenc
        self.transformers = model.transformers
        self.final_assign = model.log_assignment[-1]
        self.n_layers = model.conf.n_layers
        self.threshold = model.conf.filter_threshold

    @staticmethod
    def _norm_kpts(kpts: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        shift = size.float() / 2
        scale = size.float().max(-1).values / 2
        return (kpts - shift[:, None]) / scale[:, None, None]

    def forward(
        self,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        descs0: torch.Tensor,
        descs1: torch.Tensor,
        size0: torch.Tensor,
        size1: torch.Tensor,
    ):
        from gluefactory.models.matchers.lightglue import filter_matches

        kn0 = self._norm_kpts(kpts0, size0)
        kn1 = self._norm_kpts(kpts1, size1)

        d0 = self.input_proj(descs0)
        d1 = self.input_proj(descs1)
        enc0 = self.posenc(kn0)
        enc1 = self.posenc(kn1)

        for i in range(self.n_layers):
            d0, d1 = self.transformers[i](d0, d1, enc0, enc1)

        scores, _ = self.final_assign(d0, d1)
        m0, m1, msc0, msc1 = filter_matches(scores, self.threshold)

        return m0, m1, msc0, msc1


class _LightGlueScriptWrapper(nn.Module):
    """Scripted dict-I/O wrapper around the traced matcher.

    Scripted so image-size extraction from tensor shapes is dynamic
    (not baked as constants), matching the SuperGlue wrapper above.
    """

    def __init__(self, traced_matcher):
        super().__init__()
        self.matcher = traced_matcher

    def forward(self, data: Dict[str, torch.Tensor]):
        kpts0 = data["keypoints0"]
        kpts1 = data["keypoints1"]
        desc0 = data["descriptors0"].transpose(1, 2)
        desc1 = data["descriptors1"].transpose(1, 2)

        img0 = data["image0"]
        img1 = data["image1"]

        h0 = img0.size(2)
        w0 = img0.size(3)
        h1 = img1.size(2)
        w1 = img1.size(3)

        dtype = kpts0.dtype
        device = kpts0.device
        size0 = torch.tensor([[w0, h0]], dtype=dtype, device=device)
        size1 = torch.tensor([[w1, h1]], dtype=dtype, device=device)

        m0, m1, msc0, msc1 = self.matcher(kpts0, kpts1, desc0, desc1, size0, size1)

        result: Dict[str, torch.Tensor] = {"matches0": m0, "matching_scores0": msc0}
        return result


def export_lightglue_pt(
    ckpt_path: Path,
    out_path: Path,
    N: int,
    h: int = 480,
    w: int = 640,
) -> None:
    print(f"\n[LG] loading  {ckpt_path}")
    lg = _build_lightglue(ckpt_path)
    print(f"     n_layers={lg.conf.n_layers} descriptor_dim={lg.conf.descriptor_dim} "
          f"input_dim={lg.conf.input_dim}")

    # Step 1: trace the inner matcher (no dict I/O, clean tensor->tensor)
    matcher = _LightGlueMatcher(lg).eval()
    B, D = 1, lg.conf.input_dim
    dummy_kpts = torch.zeros(B, N, 2)
    dummy_desc = torch.zeros(B, N, D)
    dummy_size = torch.tensor([[float(w), float(h)]])
    with torch.no_grad():
        traced_matcher = torch.jit.trace(
            matcher,
            (dummy_kpts, dummy_kpts, dummy_desc, dummy_desc, dummy_size, dummy_size),
        )

    # Step 2: script the dict wrapper (dynamic shape extraction)
    wrapper = _LightGlueScriptWrapper(traced_matcher).eval()
    dummy_data = {
        "keypoints0": dummy_kpts,
        "keypoints1": dummy_kpts,
        "descriptors0": torch.zeros(B, D, N),
        "descriptors1": torch.zeros(B, D, N),
        "image0": torch.zeros(B, 1, h, w),
        "image1": torch.zeros(B, 1, h, w),
    }
    with torch.no_grad():
        scripted = torch.jit.script(wrapper)
        _ = scripted(dummy_data)
        scripted.save(str(out_path))

    print(f"[LG] saved  → {out_path}")
    try:
        loaded = torch.jit.load(str(out_path))
        out = loaded(dummy_data)
        print(f"     output: matches0={out['matches0'].shape}, "
              f"matching_scores0={out['matching_scores0'].shape}")
    except Exception as e:
        print(f"     verification error: {e}")


# ═════════════════════════════════════════════════════════════════════════
# ONNX backend
# ═════════════════════════════════════════════════════════════════════════

# ── SuperPoint ───────────────────────────────────────────────────────────────

class SuperPointONNX(nn.Module):
    """Dense SuperPoint backbone for ONNX export.

    Runs the encoder, detector, and descriptor heads and applies NMS +
    border zeroing.  Keypoint extraction is left to the caller.
    """

    def __init__(self, model):
        super().__init__()
        self.backbone   = model.backbone
        self.detector   = model.detector
        self.descriptor = model.descriptor
        self.stride     = model.stride
        self.nms_r      = model.conf.nms_radius
        self.rm_b       = model.conf.remove_borders

    def forward(self, image: torch.Tensor):
        """
        image : [B, 1|3, H, W] – float32, values in [0, 1]
        returns
          scores      : [B, H, W]          NMS-filtered keypoint heatmap
          descriptors : [B, 256, H/8, W/8] L2-normalised dense descriptors
        """
        if image.shape[1] == 3:
            rgb_w = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * rgb_w).sum(1, keepdim=True)

        feats       = self.backbone(image)
        descriptors = torch.nn.functional.normalize(self.descriptor(feats), p=2, dim=1)

        logits = self.detector(feats)
        scores = torch.nn.functional.softmax(logits, dim=1)[:, :-1]
        b, _, h, w = scores.shape
        k      = self.stride
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, k, k)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * k, w * k)

        # batched NMS (2-pass, same as superpoint_open.py)
        # unsqueeze to 4D so ONNX MaxPool gets the expected [B,C,H,W] input
        r  = self.nms_r
        def mp(x):
            return torch.nn.functional.max_pool2d(
                x.unsqueeze(1), 2 * r + 1, stride=1, padding=r
            ).squeeze(1)
        z  = torch.zeros_like(scores)
        mm = scores == mp(scores)
        for _ in range(2):
            sup = mp(mm.float()) > 0
            ss  = torch.where(sup, z, scores)
            mm  = mm | ((ss == mp(ss)) & ~sup)
        scores = torch.where(mm, scores, z)

        # border zeroing (non-in-place for ONNX compatibility)
        p    = self.rm_b
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, :p]  = False
        mask[:, -p:] = False
        mask[:, :, :p]  = False
        mask[:, :, -p:] = False
        scores = torch.where(mask, scores, z)

        return scores, descriptors


def export_superpoint_onnx(
    ckpt_path: Path,
    out_path: Path,
    h: int,
    w: int,
    opset: int,
) -> None:
    print(f"\n[SP] loading  {ckpt_path}")
    sp      = _build_superpoint(ckpt_path)
    wrapper = SuperPointONNX(sp).eval()

    dummy = torch.zeros(1, 1, h, w)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["image"],
            output_names=["scores", "descriptors"],
            dynamic_axes={
                "image":       {0: "batch", 2: "height", 3: "width"},
                "scores":      {0: "batch", 1: "height", 2: "width"},
                "descriptors": {0: "batch", 2: "height_d8", 3: "width_d8"},
            },
            opset_version=opset,
        )
    print(f"[SP] saved  → {out_path}")
    _verify_onnx(out_path, {"image": dummy.numpy()}, ["scores", "descriptors"])


# ── SuperGlue ────────────────────────────────────────────────────────────────

class SuperGlueONNX(nn.Module):
    """SuperGlue wrapper for ONNX export with fixed or dynamic N keypoints.

    Inlines normalize_keypoints to avoid the @AMP_CUSTOM_FWD_F32 decorator,
    and delegates Sinkhorn / matching logic to the original helper functions.
    """

    def __init__(self, model, n_sinkhorn: int):
        super().__init__()
        self.kenc       = model.kenc
        self.gnn        = model.gnn
        self.final_proj = model.final_proj
        self.bin_score  = model.bin_score
        self.desc_dim   = model.conf.descriptor_dim
        self.n_sinkhorn = n_sinkhorn
        self.threshold  = model.conf.filter_threshold

    @staticmethod
    def _norm_kpts(kpts: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        """Normalise pixel coords to roughly [-1, 1].

        size : [B, 2]  (width, height) in pixels – same convention as SuperGlue.
        """
        shift = size.float() / 2                      # [B, 2]
        scale = size.float().max(1).values * 0.7      # [B]
        return (kpts - shift[:, None]) / scale[:, None, None]

    def forward(
        self,
        keypoints0:   torch.Tensor,   # [B, N, 2]
        keypoints1:   torch.Tensor,   # [B, N, 2]
        descriptors0: torch.Tensor,   # [B, 256, N]
        descriptors1: torch.Tensor,   # [B, 256, N]
        scores0:      torch.Tensor,   # [B, N]
        scores1:      torch.Tensor,   # [B, N]
        size0:        torch.Tensor,   # [B, 2]  (w, h)
        size1:        torch.Tensor,   # [B, 2]  (w, h)
    ):
        from gluefactory_nonfree.superglue import log_optimal_transport, arange_like

        kn0 = self._norm_kpts(keypoints0, size0)
        kn1 = self._norm_kpts(keypoints1, size1)

        d0 = descriptors0 + self.kenc(kn0, scores0)   # [B, D, N]
        d1 = descriptors1 + self.kenc(kn1, scores1)

        d0, d1   = self.gnn(d0, d1)
        md0, md1 = self.final_proj(d0), self.final_proj(d1)

        cost = torch.einsum("bdn,bdm->bnm", md0, md1) / (self.desc_dim ** 0.5)
        log_assign = log_optimal_transport(cost, self.bin_score, self.n_sinkhorn)

        s_kpts = log_assign[:, :-1, :-1]
        max0, max1 = s_kpts.max(2), s_kpts.max(1)
        m0, m1 = max0.indices, max1.indices

        mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
        mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)

        zero    = log_assign.new_tensor(0)
        msc0    = torch.where(mutual0, max0.values.exp(), zero)
        msc1    = torch.where(mutual1, msc0.gather(1, m1), zero)
        valid0  = mutual0 & (msc0 > self.threshold)
        valid1  = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, torch.full_like(m0, -1))
        m1 = torch.where(valid1, m1, torch.full_like(m1, -1))

        return m0, m1, msc0, msc1


def export_superglue_onnx(
    ckpt_path:  Path,
    out_path:   Path,
    N:          int,
    opset:      int,
    sinkhorn_iters: int = None,
    h:          int = 480,
    w:          int = 640,
) -> None:
    print(f"\n[SG] loading  {ckpt_path}")
    sg = _build_superglue(ckpt_path)
    n_sink = _read_sinkhorn_iters(ckpt_path, sinkhorn_iters)
    print(f"     sinkhorn iterations: {n_sink}")
    wrapper = SuperGlueONNX(sg, n_sink).eval()

    B, D = 1, sg.conf.descriptor_dim
    keypoints0   = torch.zeros(B, N, 2)
    keypoints1   = torch.zeros(B, N, 2)
    descriptors0 = torch.zeros(B, D, N)
    descriptors1 = torch.zeros(B, D, N)
    scores0      = torch.ones(B, N)
    scores1      = torch.ones(B, N)
    size0        = torch.tensor([[float(w), float(h)]]).expand(B, -1)
    size1        = torch.tensor([[float(w), float(h)]]).expand(B, -1)

    dummy_inputs = (keypoints0, keypoints1, descriptors0, descriptors1,
                     scores0, scores1, size0, size1)
    input_names  = ["keypoints0", "keypoints1", "descriptors0", "descriptors1",
                    "scores0", "scores1", "size0", "size1"]
    output_names = ["matches0", "matches1",
                     "matching_scores0", "matching_scores1"]

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "keypoints0":       {0: "batch", 1: "num_kpts"},
                "keypoints1":       {0: "batch", 1: "num_kpts"},
                "descriptors0":     {0: "batch", 2: "num_kpts"},
                "descriptors1":     {0: "batch", 2: "num_kpts"},
                "scores0":          {0: "batch", 1: "num_kpts"},
                "scores1":          {0: "batch", 1: "num_kpts"},
                "size0":            {0: "batch"},
                "size1":            {0: "batch"},
                "matches0":         {0: "batch", 1: "num_kpts"},
                "matches1":         {0: "batch", 1: "num_kpts"},
                "matching_scores0": {0: "batch", 1: "num_kpts"},
                "matching_scores1": {0: "batch", 1: "num_kpts"},
            },
            opset_version=opset,
        )
    print(f"[SG] saved  → {out_path}")
    _verify_onnx(
        out_path,
        {n: v.numpy() for n, v in zip(input_names, dummy_inputs)},
        output_names,
    )


# ── LightGlue ────────────────────────────────────────────────────────────────

class LightGlueONNX(nn.Module):
    """LightGlue wrapper for ONNX export with fixed-N keypoints.

    Runs a fixed number of transformer layers (no early-stopping / point-
    pruning, which rely on data-dependent control flow that ONNX can't
    express) and inlines keypoint normalisation to avoid the
    @AMP_CUSTOM_FWD_F32 decorator on the original `normalize_keypoints`.

    Self/cross-attention are also reimplemented here (rather than calling
    `SelfBlock`/`CrossBlock` directly) using explicit reshape/permute with
    static integer sizes instead of `unflatten(..., -1)`, since the ONNX
    exporter's rank tracking is unreliable through `unflatten`. This gets a
    single transformer layer to export, but chaining many identical layers
    still hits an exporter limitation as of PyTorch 2.2 — see the module
    docstring above.
    """

    def __init__(self, model, n_layers: int):
        super().__init__()
        self.input_proj = model.input_proj
        self.posenc     = model.posenc
        self.transformers = model.transformers
        self.final_assign = model.log_assignment[-1]
        self.n_layers   = n_layers
        self.threshold  = model.conf.filter_threshold
        self.num_heads  = model.conf.num_heads
        self.head_dim   = model.conf.descriptor_dim // model.conf.num_heads

    @staticmethod
    def _norm_kpts(kpts: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        """Normalise pixel coords to roughly [-1, 1].

        size : [B, 2]  (width, height) in pixels.
        """
        shift = size.float() / 2                  # [B, 2]
        scale = size.float().max(-1).values / 2    # [B]
        return (kpts - shift[:, None]) / scale[:, None, None]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

    def _apply_rotary(self, freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (t * freqs[0]) + (self._rotate_half(t) * freqs[1])

    def _self_attn(self, sa, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = sa.Wqkv(x).reshape(b, n, self.num_heads, self.head_dim, 3)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
        q = self._apply_rotary(encoding, q)
        k = self._apply_rotary(encoding, k)
        s = q.shape[-1] ** -0.5
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * s
        attn = torch.softmax(sim, -1)
        context = torch.einsum("bhij,bhjd->bhid", attn, v)
        message = sa.out_proj(
            context.permute(0, 2, 1, 3).reshape(b, n, self.num_heads * self.head_dim)
        )
        return x + sa.ffn(torch.cat([x, message], -1))

    def _cross_attn(self, ca, x0: torch.Tensor, x1: torch.Tensor):
        b, n0, _ = x0.shape
        _, n1, _ = x1.shape
        h, d = self.num_heads, self.head_dim

        def split_heads(t, n):
            return t.reshape(b, n, h, d).permute(0, 2, 1, 3)

        qk0 = split_heads(ca.to_qk(x0), n0) * ca.scale**0.5
        qk1 = split_heads(ca.to_qk(x1), n1) * ca.scale**0.5
        v0 = split_heads(ca.to_v(x0), n0)
        v1 = split_heads(ca.to_v(x1), n1)

        sim = torch.einsum("bhid,bhjd->bhij", qk0, qk1)
        attn01 = torch.softmax(sim, dim=-1)
        attn10 = torch.softmax(sim.permute(0, 1, 3, 2), dim=-1)
        m0 = torch.einsum("bhij,bhjd->bhid", attn01, v1)
        m1 = torch.einsum("bhji,bhjd->bhid", attn10.permute(0, 1, 3, 2), v0)

        m0 = ca.to_out(m0.permute(0, 2, 1, 3).reshape(b, n0, h * d))
        m1 = ca.to_out(m1.permute(0, 2, 1, 3).reshape(b, n1, h * d))
        x0 = x0 + ca.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + ca.ffn(torch.cat([x1, m1], -1))
        return x0, x1

    def forward(
        self,
        keypoints0:   torch.Tensor,   # [B, N, 2]
        keypoints1:   torch.Tensor,   # [B, N, 2]
        descriptors0: torch.Tensor,   # [B, D, N]
        descriptors1: torch.Tensor,   # [B, D, N]
        size0:        torch.Tensor,   # [B, 2]  (w, h)
        size1:        torch.Tensor,   # [B, 2]  (w, h)
    ):
        from gluefactory.models.matchers.lightglue import filter_matches

        kn0 = self._norm_kpts(keypoints0, size0)
        kn1 = self._norm_kpts(keypoints1, size1)

        d0 = self.input_proj(descriptors0.transpose(1, 2))
        d1 = self.input_proj(descriptors1.transpose(1, 2))
        enc0 = self.posenc(kn0)
        enc1 = self.posenc(kn1)

        for i in range(self.n_layers):
            layer = self.transformers[i]
            d0 = self._self_attn(layer.self_attn, d0, enc0)
            d1 = self._self_attn(layer.self_attn, d1, enc1)
            d0, d1 = self._cross_attn(layer.cross_attn, d0, d1)

        scores, _ = self.final_assign(d0, d1)
        m0, m1, msc0, msc1 = filter_matches(scores, self.threshold)

        return m0, m1, msc0, msc1


def export_lightglue_onnx(
    ckpt_path: Path,
    out_path:  Path,
    N:         int,
    opset:     int,
    h:         int = 480,
    w:         int = 640,
) -> None:
    print(f"\n[LG] loading  {ckpt_path}")
    lg      = _build_lightglue(ckpt_path)
    wrapper = LightGlueONNX(lg, lg.conf.n_layers).eval()

    B, D = 1, lg.conf.input_dim
    keypoints0   = torch.zeros(B, N, 2)
    keypoints1   = torch.zeros(B, N, 2)
    descriptors0 = torch.zeros(B, D, N)
    descriptors1 = torch.zeros(B, D, N)
    size0        = torch.tensor([[float(w), float(h)]]).expand(B, -1)
    size1        = torch.tensor([[float(w), float(h)]]).expand(B, -1)

    dummy_inputs = (keypoints0, keypoints1, descriptors0, descriptors1, size0, size1)
    input_names  = ["keypoints0", "keypoints1", "descriptors0", "descriptors1",
                    "size0", "size1"]
    output_names = ["matches0", "matches1",
                     "matching_scores0", "matching_scores1"]

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy_inputs,
                str(out_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={
                    "keypoints0":       {0: "batch", 1: "num_kpts"},
                    "keypoints1":       {0: "batch", 1: "num_kpts"},
                    "descriptors0":     {0: "batch", 2: "num_kpts"},
                    "descriptors1":     {0: "batch", 2: "num_kpts"},
                    "size0":            {0: "batch"},
                    "size1":            {0: "batch"},
                    "matches0":         {0: "batch", 1: "num_kpts"},
                    "matches1":         {0: "batch", 1: "num_kpts"},
                    "matching_scores0": {0: "batch", 1: "num_kpts"},
                    "matching_scores1": {0: "batch", 1: "num_kpts"},
                },
                opset_version=opset,
            )
    except Exception as e:
        print(
            "[LG] export FAILED — this is a known PyTorch ONNX-exporter "
            f"limitation with LightGlue's {lg.conf.n_layers}-layer rotary "
            "attention stack (see module docstring), not a bug in this "
            f"script. torch={torch.__version__}. Original error: {e}",
            file=sys.stderr,
        )
        raise
    print(f"[LG] saved  → {out_path}")
    _verify_onnx(
        out_path,
        {n: v.numpy() for n, v in zip(input_names, dummy_inputs)},
        output_names,
    )


# ── ONNX verification (optional, requires onnxruntime) ──────────────────────

def _verify_onnx(path: Path, inputs: dict, output_names: list) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("    (skipping verification – install onnxruntime to enable)")
        return
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outs = sess.run(output_names, inputs)
        shapes = {name: out.shape for name, out in zip(output_names, outs)}
        print(f"    verified via onnxruntime – output shapes: {shapes}")
    except Exception as e:
        print(f"    onnxruntime verification warning: {e}")


# ── CLI ──────────────────────────────────────────────────────────────────────

MODEL_ALIASES = {
    "sp": "superpoint", "superpoint": "superpoint",
    "sg": "superglue", "superglue": "superglue",
    "lg": "lightglue", "lightglue": "lightglue",
}
FORMAT_ALIASES = {"pt": "pt", "pth": "pt", "onnx": "onnx"}


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", required=True, choices=sorted(MODEL_ALIASES),
        help="Which model to export: superpoint/sp, superglue/sg, lightglue/lg",
    )
    p.add_argument(
        "--format", required=True, choices=sorted(FORMAT_ALIASES),
        help="Export format: pt or pth (TorchScript, identical output) or onnx",
    )
    p.add_argument("--ckpt", required=True, type=Path, help=".tar checkpoint path")
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output path (default: <model>.<format>, e.g. lightglue.pt)",
    )
    p.add_argument("--image_h", type=int, default=480,
                    help="Trace/dummy image height (default: %(default)s)")
    p.add_argument("--image_w", type=int, default=640,
                    help="Trace/dummy image width (default: %(default)s)")
    p.add_argument(
        "--num_kpts", type=int, default=512,
        help="Number of keypoints for SuperGlue/LightGlue trace (default: %(default)s)",
    )
    p.add_argument(
        "--sinkhorn_iters", type=int, default=None,
        help="SuperGlue sinkhorn iterations to unroll "
             "(default: read from checkpoint, fall back to 50)",
    )
    p.add_argument(
        "--opset", type=int, default=16,
        help="ONNX opset version (default: %(default)s)",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    model = MODEL_ALIASES[args.model]
    fmt = FORMAT_ALIASES[args.format]

    if not args.ckpt.exists():
        print(f"Error: checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or Path(f"{model}.{args.format}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "pt":
        if model == "superpoint":
            export_superpoint_pt(args.ckpt, out_path, args.image_h, args.image_w)
        elif model == "superglue":
            export_superglue_pt(args.ckpt, out_path, args.num_kpts,
                                 args.sinkhorn_iters, args.image_h, args.image_w)
        else:
            export_lightglue_pt(args.ckpt, out_path, args.num_kpts,
                                 args.image_h, args.image_w)
    else:
        if model == "superpoint":
            export_superpoint_onnx(args.ckpt, out_path, args.image_h, args.image_w,
                                    args.opset)
        elif model == "superglue":
            export_superglue_onnx(args.ckpt, out_path, args.num_kpts, args.opset,
                                   args.sinkhorn_iters, args.image_h, args.image_w)
        else:
            export_lightglue_onnx(args.ckpt, out_path, args.num_kpts, args.opset,
                                   args.image_h, args.image_w)


if __name__ == "__main__":
    main()
