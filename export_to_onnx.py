#!/usr/bin/env python3
"""
Export SuperPoint and/or SuperGlue .tar checkpoints to ONNX.

SuperPoint produces dense outputs (NMS-filtered score heatmap + descriptor map).
Keypoint extraction (thresholding / top-k) happens outside the ONNX graph.

  Input:  image        [B, 1|3, H, W]  float32 in [0, 1]
  Output: score_map    [B, H, W]
          desc_map     [B, 256, H/8, W/8]

SuperGlue uses fixed-N keypoint inputs to keep a static graph.
Sinkhorn iterations are unrolled at trace time.

  Inputs:  kpts0, kpts1   [B, N, 2]   pixel coords (x, y)
           descs0, descs1 [B, N, 256] L2-normalised descriptors
           scores0,scores1[B, N]      keypoint confidence
           size0, size1   [B, 2]      image (width, height) in pixels
  Outputs: matches0, matches1  [B, N]  matched index or -1
           mscores0, mscores1  [B, N]  match confidence

Usage examples
--------------
  # Both models with best checkpoints
  python export_to_onnx.py \\
      --sp_ckpt outputs/training/superpoint_slam_run/checkpoint_best.tar \\
      --sg_ckpt outputs/training/superglue_slam_run/checkpoint_best.tar

  # SuperPoint only with custom image size
  python export_to_onnx.py \\
      --sp_ckpt outputs/training/superpoint_slam_run/checkpoint_best.tar \\
      --image_h 480 --image_w 640

  # SuperGlue with fewer sinkhorn iterations (smaller/faster graph)
  python export_to_onnx.py \\
      --sg_ckpt outputs/training/superglue_slam_run/checkpoint_best.tar \\
      --sg_sinkhorn_iters 20 --num_kpts 512
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ── SuperPoint ─────────────────────────────────────────────────────────────────

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
          score_map : [B, H, W]          NMS-filtered keypoint heatmap
          desc_map  : [B, 256, H/8, W/8] L2-normalised dense descriptors
        """
        if image.shape[1] == 3:
            rgb_w = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * rgb_w).sum(1, keepdim=True)

        feats    = self.backbone(image)
        desc_map = torch.nn.functional.normalize(self.descriptor(feats), p=2, dim=1)

        logits  = self.detector(feats)
        s       = torch.nn.functional.softmax(logits, dim=1)[:, :-1]
        b, _, h, w = s.shape
        k       = self.stride
        s       = s.permute(0, 2, 3, 1).reshape(b, h, w, k, k)
        s       = s.permute(0, 1, 3, 2, 4).reshape(b, h * k, w * k)

        # batched NMS (2-pass, same as superpoint_open.py)
        # unsqueeze to 4D so ONNX MaxPool gets the expected [B,C,H,W] input
        r  = self.nms_r
        def mp(x):
            return torch.nn.functional.max_pool2d(
                x.unsqueeze(1), 2 * r + 1, stride=1, padding=r
            ).squeeze(1)
        z  = torch.zeros_like(s)
        mm = s == mp(s)
        for _ in range(2):
            sup = mp(mm.float()) > 0
            ss  = torch.where(sup, z, s)
            mm  = mm | ((ss == mp(ss)) & ~sup)
        s = torch.where(mm, s, z)

        # border zeroing (non-in-place for ONNX compatibility)
        p    = self.rm_b
        mask = torch.ones_like(s, dtype=torch.bool)
        mask[:, :p]  = False
        mask[:, -p:] = False
        mask[:, :, :p]  = False
        mask[:, :, -p:] = False
        s = torch.where(mask, s, z)

        return s, desc_map


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


def export_superpoint(
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
            output_names=["score_map", "desc_map"],
            dynamic_axes={
                "image":     {0: "batch", 2: "height", 3: "width"},
                "score_map": {0: "batch", 1: "height", 2: "width"},
                "desc_map":  {0: "batch", 2: "height_d8", 3: "width_d8"},
            },
            opset_version=opset,
        )
    print(f"[SP] saved  → {out_path}")
    _verify_onnx(out_path, {"image": dummy.numpy()}, ["score_map", "desc_map"])


# ── SuperGlue ──────────────────────────────────────────────────────────────────

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
        kpts0:   torch.Tensor,   # [B, N, 2]
        kpts1:   torch.Tensor,   # [B, N, 2]
        descs0:  torch.Tensor,   # [B, N, 256]
        descs1:  torch.Tensor,   # [B, N, 256]
        scores0: torch.Tensor,   # [B, N]
        scores1: torch.Tensor,   # [B, N]
        size0:   torch.Tensor,   # [B, 2]  (w, h)
        size1:   torch.Tensor,   # [B, 2]  (w, h)
    ):
        from gluefactory_nonfree.superglue import log_optimal_transport, arange_like

        kn0 = self._norm_kpts(kpts0, size0)
        kn1 = self._norm_kpts(kpts1, size1)

        d0 = descs0.transpose(1, 2) + self.kenc(kn0, scores0)   # [B, D, N]
        d1 = descs1.transpose(1, 2) + self.kenc(kn1, scores1)

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


def _build_superglue(ckpt_path: Path) -> nn.Module:
    """Instantiate SuperGlue and load weights from a .tar checkpoint."""
    from gluefactory_nonfree.superglue import SuperGlue
    from omegaconf import OmegaConf

    ckpt       = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]

    # Strip 'matcher.' prefix (two_view_pipeline training artifact)
    if any(k.startswith("matcher.") for k in state_dict):
        state_dict = {
            k[len("matcher."):]: v
            for k, v in state_dict.items()
            if k.startswith("matcher.")
        }

    # conf.weights="" is falsy → _init skips the URL download
    conf  = OmegaConf.create({
        "weights": "",
        "trainable": True,
        "freeze_batch_normalization": False,
        "timeit": False,
    })
    model = SuperGlue(conf)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def export_superglue(
    ckpt_path:  Path,
    out_path:   Path,
    N:          int,
    n_sinkhorn: int,
    opset:      int,
) -> None:
    print(f"\n[SG] loading  {ckpt_path}")
    sg      = _build_superglue(ckpt_path)
    wrapper = SuperGlueONNX(sg, n_sinkhorn).eval()

    B, D = 1, 256
    kpts0   = torch.zeros(B, N, 2)
    kpts1   = torch.zeros(B, N, 2)
    descs0  = torch.zeros(B, N, D)
    descs1  = torch.zeros(B, N, D)
    scores0 = torch.ones(B, N)
    scores1 = torch.ones(B, N)
    size0   = torch.tensor([[640.0, 480.0]]).expand(B, -1)
    size1   = torch.tensor([[640.0, 480.0]]).expand(B, -1)

    dummy_inputs = (kpts0, kpts1, descs0, descs1, scores0, scores1, size0, size1)
    input_names  = ["kpts0", "kpts1", "descs0", "descs1",
                    "scores0", "scores1", "size0", "size1"]
    output_names = ["matches0", "matches1", "mscores0", "mscores1"]

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "kpts0":    {0: "batch", 1: "num_kpts"},
                "kpts1":    {0: "batch", 1: "num_kpts"},
                "descs0":   {0: "batch", 1: "num_kpts"},
                "descs1":   {0: "batch", 1: "num_kpts"},
                "scores0":  {0: "batch", 1: "num_kpts"},
                "scores1":  {0: "batch", 1: "num_kpts"},
                "size0":    {0: "batch"},
                "size1":    {0: "batch"},
                "matches0": {0: "batch", 1: "num_kpts"},
                "matches1": {0: "batch", 1: "num_kpts"},
                "mscores0": {0: "batch", 1: "num_kpts"},
                "mscores1": {0: "batch", 1: "num_kpts"},
            },
            opset_version=opset,
        )
    print(f"[SG] saved  → {out_path}")
    _verify_onnx(
        out_path,
        {n: v.numpy() for n, v in zip(input_names, dummy_inputs)},
        output_names,
    )


# ── ONNX verification (optional, requires onnxruntime) ────────────────────────

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


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sp_ckpt", type=Path, default=None,
        help="SuperPoint .tar checkpoint (default: %(default)s)",
    )
    p.add_argument(
        "--sg_ckpt", type=Path, default=None,
        help="SuperGlue .tar checkpoint (default: %(default)s)",
    )
    p.add_argument(
        "--sp_out", type=Path, default=Path("superpoint.onnx"),
        help="Output path for SuperPoint ONNX (default: %(default)s)",
    )
    p.add_argument(
        "--sg_out", type=Path, default=Path("superglue.onnx"),
        help="Output path for SuperGlue ONNX (default: %(default)s)",
    )
    p.add_argument(
        "--image_h", type=int, default=480,
        help="Trace height for SP export – also determines desc_map height/8 (default: %(default)s)",
    )
    p.add_argument(
        "--image_w", type=int, default=640,
        help="Trace width for SP export (default: %(default)s)",
    )
    p.add_argument(
        "--num_kpts", type=int, default=512,
        help="Number of keypoints for SG trace input (default: %(default)s)",
    )
    p.add_argument(
        "--sg_sinkhorn_iters", type=int, default=None,
        help="Sinkhorn iterations to unroll in SG graph "
             "(default: read from checkpoint, fall back to 50). "
             "Fewer → smaller / faster graph.",
    )
    p.add_argument(
        "--opset", type=int, default=16,
        help="ONNX opset version (default: %(default)s)",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    if args.sp_ckpt is None and args.sg_ckpt is None:
        print("Error: specify at least one of --sp_ckpt or --sg_ckpt", file=sys.stderr)
        sys.exit(1)

    if args.sp_ckpt is not None:
        if not args.sp_ckpt.exists():
            print(f"Error: SP checkpoint not found: {args.sp_ckpt}", file=sys.stderr)
            sys.exit(1)
        args.sp_out.parent.mkdir(parents=True, exist_ok=True)
        export_superpoint(args.sp_ckpt, args.sp_out,
                          args.image_h, args.image_w, args.opset)

    if args.sg_ckpt is not None:
        if not args.sg_ckpt.exists():
            print(f"Error: SG checkpoint not found: {args.sg_ckpt}", file=sys.stderr)
            sys.exit(1)
        n_sink = args.sg_sinkhorn_iters
        if n_sink is None:
            ckpt = torch.load(args.sg_ckpt, map_location="cpu")
            try:
                n_sink = ckpt["conf"]["model"]["matcher"]["sinkhorn_iterations"]
            except (KeyError, TypeError):
                n_sink = 50
            print(f"[SG] using {n_sink} sinkhorn iterations (from checkpoint)")
        args.sg_out.parent.mkdir(parents=True, exist_ok=True)
        export_superglue(args.sg_ckpt, args.sg_out,
                         args.num_kpts, n_sink, args.opset)


if __name__ == "__main__":
    main()
