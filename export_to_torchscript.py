#!/usr/bin/env python3
"""
Export SuperPoint and SuperGlue .tar checkpoints to TorchScript .pt format
for use with RTAB-Map's C++ Torch integration (CPU and CUDA).

SuperPoint output (tuple):
  (score_map [1, H, W], desc_map [1, 256, H/8, W/8])

SuperGlue input/output (dicts):
  Input:  keypoints0/1 (1xNx2), descriptors0/1 (1x256xN),
          scores0/1 (1xN), image0/1 (1x1xHxW)
  Output: matches0 (1xN int64), matching_scores0 (1xN float32)

Usage:
  python export_to_torchscript.py \
      --sp_ckpt outputs/training/superpoint_slam_run/checkpoint_best.tar \
      --sg_ckpt outputs/training/superglue_slam_run/checkpoint_best.tar \
      --sp_out /path/to/superpoint.pt \
      --sg_out /path/to/superglue.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ── SuperPoint TorchScript wrapper ─────────────────────────────────────────────

class SuperPointTorchScript(nn.Module):
    """Traced wrapper: (score_map [1,H,W], desc_map [1,256,H/8,W/8])"""

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
        desc_map = torch.nn.functional.normalize(
            self.descriptor(features), p=2, dim=1
        )

        logits = self.detector(features)
        s = torch.nn.functional.softmax(logits, dim=1)[:, :-1]
        b, _, h, w = s.shape
        k = self.stride
        s = s.permute(0, 2, 3, 1).reshape(b, h, w, k, k)
        s = s.permute(0, 1, 3, 2, 4).reshape(b, h * k, w * k)

        return (s, desc_map)


def _build_superpoint(ckpt_path: Path) -> nn.Module:
    from gluefactory.models.extractors.superpoint_open import SuperPoint
    from omegaconf import OmegaConf

    conf = OmegaConf.create({
        "weights": str(ckpt_path),
        "trainable": True,
        "freeze_batch_normalization": False,
        "timeit": False,
    })
    model = SuperPoint(conf)
    model.eval()
    return model


def export_superpoint(ckpt_path: Path, out_path: Path, h: int, w: int) -> None:
    print(f"\n[SP] loading  {ckpt_path}")
    sp = _build_superpoint(ckpt_path)
    wrapper = SuperPointTorchScript(sp).eval()

    dummy = torch.zeros(1, 1, h, w)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy,))
        traced.save(str(out_path))

    print(f"[SP] saved  → {out_path}")
    try:
        loaded = torch.jit.load(str(out_path))
        out = loaded(dummy)
        print(f"     output: ({out[0].shape}, {out[1].shape})")
    except Exception as e:
        print(f"     verification error: {e}")


# ── SuperGlue TorchScript wrapper ──────────────────────────────────────────────

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
        # Inline these so they're available as attributes at script time
        self._log_opt = None
        self._arange_like = None

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


def _build_superglue(ckpt_path: Path, sinkhorn_iters: int = None):
    from gluefactory_nonfree.superglue import SuperGlue
    from omegaconf import OmegaConf

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]

    if any(k.startswith("matcher.") for k in state_dict):
        state_dict = {
            k[len("matcher."):]: v
            for k, v in state_dict.items()
            if k.startswith("matcher.")
        }

    conf = OmegaConf.create({
        "weights": "",
        "trainable": True,
        "freeze_batch_normalization": False,
        "timeit": False,
    })
    model = SuperGlue(conf)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if sinkhorn_iters is None:
        try:
            sinkhorn_iters = ckpt["conf"]["model"]["matcher"]["sinkhorn_iterations"]
        except (KeyError, TypeError):
            sinkhorn_iters = 50

    return model, sinkhorn_iters


def export_superglue(
    ckpt_path: Path,
    out_path: Path,
    N: int,
    sinkhorn_iters: int = None,
    h: int = 480,
    w: int = 640,
) -> None:
    print(f"\n[SG] loading  {ckpt_path}")
    sg, n_sink = _build_superglue(ckpt_path, sinkhorn_iters)
    print(f"     sinkhorn iterations: {n_sink}")

    # Step 1: trace the inner matcher (no dict I/O, clean tensor->tensor)
    matcher = _SuperGlueMatcher(sg, n_sink).eval()
    B, D = 1, 256
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


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sp_ckpt", type=Path, default=None)
    p.add_argument("--sg_ckpt", type=Path, default=None)
    p.add_argument("--sp_out", type=Path, default=Path("superpoint.pt"))
    p.add_argument("--sg_out", type=Path, default=Path("superglue.pt"))
    p.add_argument("--image_h", type=int, default=480)
    p.add_argument("--image_w", type=int, default=640)
    p.add_argument("--num_kpts", type=int, default=2048,
                   help="Number of keypoints for SG trace")
    p.add_argument("--sg_sinkhorn_iters", type=int, default=None)
    args = p.parse_args()

    if args.sp_ckpt is None and args.sg_ckpt is None:
        print("Error: specify at least one of --sp_ckpt or --sg_ckpt",
              file=sys.stderr)
        sys.exit(1)

    if args.sp_ckpt is not None:
        if not args.sp_ckpt.exists():
            print(f"Error: SP checkpoint not found: {args.sp_ckpt}",
                  file=sys.stderr)
            sys.exit(1)
        args.sp_out.parent.mkdir(parents=True, exist_ok=True)
        export_superpoint(args.sp_ckpt, args.sp_out,
                          args.image_h, args.image_w)

    if args.sg_ckpt is not None:
        if not args.sg_ckpt.exists():
            print(f"Error: SG checkpoint not found: {args.sg_ckpt}",
                  file=sys.stderr)
            sys.exit(1)
        args.sg_out.parent.mkdir(parents=True, exist_ok=True)
        export_superglue(args.sg_ckpt, args.sg_out,
                         args.num_kpts, args.sg_sinkhorn_iters,
                         args.image_h, args.image_w)


if __name__ == "__main__":
    main()
