"""
Run SuperPoint alone, or SuperPoint + SuperGlue/LightGlue, on images and
visualize the result — the single entry point replacing the previously
duplicated scripts/match_images.py, scripts/match_images_from_pt.py,
scripts/export_interactive_matches.py, scripts/visualize_custom.py,
gluefactory/scripts/infer_superpoint.py and infer_superglue.py.

Two backends:
  --backend checkpoint   Load .tar training checkpoints via
                         gluefactory.slam.matcher.SLAMMatcher.
  --backend exported     Load .pt TorchScript graphs produced by
                         gluefactory.scripts.export_model, decoding
                         SuperPoint's dense heatmap with the same NMS/
                         threshold/top-k logic used at training time
                         (gluefactory.models.extractors.superpoint_open).

Three matcher choices (any of the extractor's outputs can be visualized
alone by passing --matcher none):
  --matcher none | superglue | lightglue

Four output modes, combinable via --output all:
  html  interactive canvas-overlay dashboard (pairs) or keypoint grid (--matcher none)
  data  matches_data.js only (no HTML)
  png   static per-pair or per-image PNGs
  all   all of the above

Usage:
    # SuperPoint + LightGlue, from training checkpoints, full dashboard
    MPLBACKEND=Agg python -m gluefactory.scripts.run_inference \\
        --backend checkpoint --matcher lightglue \\
        --extractor_ckpt outputs/training/superpoint_slam_run/checkpoint_best.tar \\
        --matcher_ckpt   outputs/training/lightglue_slam_run/checkpoint_best.tar \\
        --input data/output/slam/images/rgb \\
        --output all --output_dir data/output/slam/visualizations/match_inference_rgb_light

    # SuperPoint alone, from an exported .pt, static PNGs only
    MPLBACKEND=Agg python -m gluefactory.scripts.run_inference \\
        --backend exported --matcher none \\
        --extractor_pt superpoint.pt \\
        --input data/output/slam/images/rgb \\
        --output png --output_dir outputs/sp_detections
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from gluefactory.models.extractors.superpoint_open import (
    batched_nms,
    sample_descriptors,
    select_top_k_keypoints,
)
from gluefactory.slam.io import list_images, load_image
from gluefactory.slam.matcher import SLAMMatcher, run_match_directory
from gluefactory.visualization import dashboard, viz2d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_inference")


# ─────────────────────────────────────────────────────────────────────────────
# Exported-model (.pt) backend
# ─────────────────────────────────────────────────────────────────────────────

class ExportedMatcher:
    """Checkpoint-free inference backend: loads .pt TorchScript graphs from
    gluefactory.scripts.export_model and decodes SuperPoint's dense heatmap
    with the real training-time NMS/threshold/top-k/descriptor-sampling
    logic, rather than a crude np.where + top-k re-implementation.

    Exposes the same `match_pair`/`match_directory` interface as
    `gluefactory.slam.matcher.SLAMMatcher` so both backends can be driven
    identically from `main()` below.
    """

    def __init__(self, extractor_pt, matcher_pt=None, device=None, conf=None,
                 matcher="superglue"):
        matcher = matcher or "none"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.has_matcher = matcher != "none"
        conf = conf or {}
        self.nms_radius = conf.get("nms_radius", 3)
        self.max_num_keypoints = conf.get("max_num_keypoints", 512)
        self.detection_threshold = conf.get("detection_threshold", 0.005)
        self.remove_borders = conf.get("remove_borders", 4)
        self.filter_threshold = conf.get("filter_threshold", 0.01)

        self._sp = torch.jit.load(str(extractor_pt), map_location=self.device).eval()
        self._matcher = None
        if self.has_matcher:
            if matcher_pt is None:
                raise ValueError("matcher_pt is required unless matcher='none'")
            self._matcher = torch.jit.load(str(matcher_pt), map_location=self.device).eval()
        logger.info(f"ExportedMatcher ({matcher}) loaded on {self.device}")

    def _extract(self, img_gray):
        """Decode one image: keypoints [N,2], scores [N], descriptors [1,256,N],
        and the [1,1,H,W] input tensor (needed by the matcher's image0/1 input).
        """
        t = torch.from_numpy(img_gray).float().to(self.device).unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            out = self._sp({"image": t})
        scores = batched_nms(out["scores"], self.nms_radius)
        if self.remove_borders:
            pad = self.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1
        scores = scores.squeeze(0)
        idxs = torch.where(scores > self.detection_threshold)
        kpts = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        kscores = scores[idxs]
        kpts, kscores = select_top_k_keypoints(kpts, kscores, self.max_num_keypoints)
        with torch.no_grad():
            desc = sample_descriptors(kpts.unsqueeze(0), out["descriptors"], s=8)
        return kpts, kscores, desc, t

    def extract(self, img_gray):
        """Run the extractor alone on a single image (no matcher forward)."""
        kpts, kscores, _, _ = self._extract(img_gray)
        return {"keypoints": kpts.cpu().numpy(), "scores": kscores.cpu().numpy()}

    def match_pair(self, img0_gray, img1_gray):
        kpts0, sc0, desc0, t0 = self._extract(img0_gray)
        kpts1, sc1, desc1, t1 = self._extract(img1_gray)
        n0 = len(kpts0)

        if not self.has_matcher:
            matches0 = np.full(n0, -1, dtype=np.int64)
            mscores0 = np.zeros(n0, dtype=np.float32)
        else:
            with torch.no_grad():
                out = self._matcher({
                    "keypoints0": kpts0.unsqueeze(0),
                    "keypoints1": kpts1.unsqueeze(0),
                    "descriptors0": desc0,
                    "descriptors1": desc1,
                    "scores0": sc0.unsqueeze(0),
                    "scores1": sc1.unsqueeze(0),
                    "image0": t0,
                    "image1": t1,
                })
            matches0 = out["matches0"][0].cpu().numpy()
            mscores0 = out["matching_scores0"][0].cpu().numpy()

        return {
            "keypoints0": kpts0.cpu().numpy(),
            "keypoints1": kpts1.cpu().numpy(),
            "scores0": sc0.cpu().numpy(),
            "scores1": sc1.cpu().numpy(),
            "matches0": matches0,
            "mscores0": mscores0,
        }

    def match_directory(self, input_dir, max_pairs=None, resize=0):
        return run_match_directory(self, self.filter_threshold, input_dir, max_pairs, resize)


# ─────────────────────────────────────────────────────────────────────────────
# Extractor-only (--matcher none) driver: one pass per image, not per pair.
# ─────────────────────────────────────────────────────────────────────────────

def _run_extractor_only(backend, input_path, output_dir, outputs, resize, max_pairs):
    image_paths = list_images(input_path)
    if not image_paths:
        logger.error(f"No images found in: {input_path}")
        return
    if max_pairs:
        image_paths = image_paths[:max_pairs]

    png_dir = output_dir / "images"
    if "png" in outputs or "html" in outputs:
        png_dir.mkdir(parents=True, exist_ok=True)

    records = []
    png_paths = []
    for idx, p in enumerate(tqdm(image_paths, desc="Extracting")):
        img_bgr, img_gray = load_image(p, resize)
        out = backend.extract(img_gray)
        kpts, scores = out["keypoints"], out["scores"]
        logger.info(f"[{idx}] {p.name}: {len(kpts)} keypoints")

        records.append({
            "idx": idx, "name": p.name,
            "keypoints": kpts.tolist(), "scores": scores.tolist(),
            "num_keypoints": int(len(kpts)),
        })

        if "png" in outputs or "html" in outputs:
            png_path = png_dir / f"{p.stem}_keypoints.png"
            viz2d.plot_keypoints_overlay(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), kpts, scores, png_path,
                title=f"{p.name} ({len(kpts)} kpts)", colorbar=True,
            )
            png_paths.append(png_path)

    if "data" in outputs:
        data_path = output_dir / "keypoints_data.json"
        data_path.write_text(json.dumps(records, indent=2))
        logger.info(f"Saved {len(records)} record(s) → {data_path}")

    if "html" in outputs:
        dashboard.render_image_grid(png_dir, png_paths, title="Keypoint Visualizer")

    print(f"\nTotal processed images: {len(records)}")
    print(f"Results: {output_dir.resolve()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Pair (--matcher superglue|lightglue) driver
# ─────────────────────────────────────────────────────────────────────────────

def _run_pairs(backend, input_path, output_dir, outputs, resize, max_pairs, matcher_name):
    results = backend.match_directory(input_path, max_pairs=max_pairs, resize=resize)
    if not results:
        logger.error("No pairs were matched.")
        return

    images_dir = output_dir / "images"
    plots_dir = output_dir / "plots"
    # "data" mode still needs the raw view images on disk: matches_data.js
    # references them by URL for any external HTML viewer to load.
    need_images = "html" in outputs or "data" in outputs
    need_plots = "png" in outputs
    if need_images:
        images_dir.mkdir(parents=True, exist_ok=True)
    if need_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for r in tqdm(results, desc="Saving"):
        idx = r["idx"]
        record = {
            "idx": idx,
            "name0": r["name0"],
            "name1": r["name1"],
            "keypoints0": r["keypoints0"].tolist(),
            "keypoints1": r["keypoints1"].tolist(),
            "scores0": r["scores0"].tolist(),
            "scores1": r["scores1"].tolist(),
            "matches": [
                [int(i0), int(i1), float(r["mscores0"][i0])]
                for i0, i1 in enumerate(r["matches0"]) if i1 != -1
            ],
            "metrics": r["metrics"],
            "image_width": int(r["img0_bgr"].shape[1]),
            "image_height": int(r["img0_bgr"].shape[0]),
        }

        if need_images:
            n0, n1 = f"pair_{idx}_view0.png", f"pair_{idx}_view1.png"
            cv2.imwrite(str(images_dir / n0), r["img0_bgr"])
            cv2.imwrite(str(images_dir / n1), r["img1_bgr"])
            record["image0_url"] = f"images/{n0}"
            record["image1_url"] = f"images/{n1}"

        if need_plots:
            plot_name = f"pair_{idx}_matches.png"
            img0_rgb = cv2.cvtColor(r["img0_bgr"], cv2.COLOR_BGR2RGB)
            img1_rgb = cv2.cvtColor(r["img1_bgr"], cv2.COLOR_BGR2RGB)
            viz2d.plot_pair_matches(
                img0_rgb, img1_rgb, r["keypoints0"], r["keypoints1"],
                r["matches0"], r["mscores0"], plots_dir / plot_name,
                title=f"{r['name0']} ↔ {r['name1']}",
            )
            record["plot_url"] = f"plots/{plot_name}"

        m = record["metrics"]
        logger.info(
            f"Pair {idx}: {r['name0']} ↔ {r['name1']}  "
            f"kpts {m['total_kpts0']}/{m['total_kpts1']}  "
            f"matches {m['num_matches']} ({m['match_ratio']:.1%})  avg {m['avg_mscore']:.2f}"
        )
        records.append(record)

    if "html" in outputs:
        # render_dashboard writes matches_data.js itself; avoid writing it twice.
        title = f"SuperPoint + {matcher_name.capitalize()} Visualizer"
        html_path = dashboard.render_dashboard(output_dir, records, title=title)
        print(f"Dashboard: file://{html_path.resolve()}")
    elif "data" in outputs:
        dashboard.write_matches_data(output_dir, records)

    print(f"\nTotal processed pairs: {len(records)}")
    print(f"Results: {output_dir.resolve()}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_OUTPUT_ALIASES = {"all": ("html", "data", "png")}


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", choices=["checkpoint", "exported"], default="checkpoint")
    parser.add_argument("--matcher", choices=["none", "superglue", "lightglue"],
                         default="superglue")

    parser.add_argument("--extractor_ckpt", type=str, default=None,
                         help="SuperPoint training checkpoint (.tar); required for --backend checkpoint")
    parser.add_argument("--matcher_ckpt", type=str, default=None,
                         help="Matcher training checkpoint (.tar); required for --backend checkpoint "
                              "unless --matcher none")
    parser.add_argument("--extractor_pt", type=str, default=None,
                         help="Exported SuperPoint model (.pt); required for --backend exported")
    parser.add_argument("--matcher_pt", type=str, default=None,
                         help="Exported matcher model (.pt); required for --backend exported "
                              "unless --matcher none")

    parser.add_argument("--input", type=str, required=True,
                         help="Image directory, single image, or 'img0.png,img1.png'")
    parser.add_argument("--output", choices=["html", "data", "png", "all"], default="all")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resize", type=int, default=640)
    parser.add_argument("--max_pairs", type=int, default=0, help="Cap on pairs/images (0 = all)")

    parser.add_argument("--nms_radius", type=int, default=3)
    parser.add_argument("--max_num_keypoints", type=int, default=512)
    parser.add_argument("--detection_threshold", type=float, default=0.005)
    parser.add_argument("--filter_threshold", type=float, default=0.01)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    outputs = _OUTPUT_ALIASES.get(args.output, (args.output,))

    conf = {
        "nms_radius": args.nms_radius,
        "max_num_keypoints": args.max_num_keypoints,
        "detection_threshold": args.detection_threshold,
        "filter_threshold": args.filter_threshold,
    }

    if args.backend == "checkpoint":
        if not args.extractor_ckpt:
            raise SystemExit("--extractor_ckpt is required for --backend checkpoint")
        if args.matcher != "none" and not args.matcher_ckpt:
            raise SystemExit("--matcher_ckpt is required unless --matcher none")
        backend = SLAMMatcher(
            args.matcher_ckpt, args.extractor_ckpt, device=args.device, conf=conf,
            matcher=args.matcher,
        )
    else:
        if not args.extractor_pt:
            raise SystemExit("--extractor_pt is required for --backend exported")
        if args.matcher != "none" and not args.matcher_pt:
            raise SystemExit("--matcher_pt is required unless --matcher none")
        backend = ExportedMatcher(
            args.extractor_pt, args.matcher_pt, device=args.device, conf=conf,
            matcher=args.matcher,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_pairs = args.max_pairs if args.max_pairs > 0 else None

    if args.matcher == "none":
        _run_extractor_only(backend, args.input, output_dir, outputs, args.resize, max_pairs)
    else:
        _run_pairs(backend, args.input, output_dir, outputs, args.resize, max_pairs, args.matcher)


if __name__ == "__main__":
    main()
