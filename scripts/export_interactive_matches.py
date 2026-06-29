"""
Export raw match data from SuperPoint + SuperGlue inference.

Writes matches_data.js (loadable by any match visualizer HTML) and, optionally,
copies images for use in the browser.

Supports two modes:
  1. Checkpoint inference (--input + --checkpoint_superglue + --checkpoint_superpoint)
  2. Legacy HomographyDataset (--dataset + --conf)
"""

import json
import logging
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export_matches")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def write_data(out_dir, records):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    js_path = out_dir / "matches_data.js"
    with open(js_path, "w") as f:
        f.write("const MATCHES_DATA = ")
        json.dump(records, f, indent=2)
        f.write(";\n")

    logger.info(f"Saved {len(records)} pairs → {js_path}")
    print(f"\nOutput: {out_dir.resolve()}")
    print(f"Data:   {js_path.resolve()}\n")


# ---------------------------------------------------------------------------
# Mode A: checkpoint-based inference
# ---------------------------------------------------------------------------

def run_checkpoint_mode(args):
    from gluefactory.slam.matcher import SLAMMatcher

    sg_ckpt = Path(args.checkpoint_superglue)
    sp_ckpt = Path(args.checkpoint_superpoint)

    if not sg_ckpt.exists():
        logger.error(f"SuperGlue checkpoint not found: {sg_ckpt}"); return
    if not sp_ckpt.exists():
        logger.error(f"SuperPoint checkpoint not found: {sp_ckpt}"); return

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
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for r in tqdm(results, desc="Saving"):
        idx = r["idx"]
        n0 = f"pair_{idx}_view0.png"
        n1 = f"pair_{idx}_view1.png"
        cv2.imwrite(str(img_dir / n0), r["img0_bgr"])
        cv2.imwrite(str(img_dir / n1), r["img1_bgr"])

        matches = [
            [int(i0), int(i1), float(r["mscores0"][i0])]
            for i0, i1 in enumerate(r["matches0"]) if i1 != -1
        ]
        m = r["metrics"]
        logger.info(
            f"[{idx}] {r['name0']} ↔ {r['name1']}  "
            f"kpts {m['total_kpts0']}/{m['total_kpts1']}  "
            f"matches {m['num_matches']} ({m['match_ratio']:.1%})  avg {m['avg_mscore']:.3f}"
        )
        records.append({
            "idx":        idx,
            "name0":      r["name0"],
            "name1":      r["name1"],
            "image0_url": f"images/{n0}",
            "image1_url": f"images/{n1}",
            "keypoints0": r["keypoints0"].tolist(),
            "keypoints1": r["keypoints1"].tolist(),
            "scores0":    r["scores0"].tolist(),
            "scores1":    r["scores1"].tolist(),
            "matches":    matches,
            "metrics":    m,
            "image_width":  int(r["img0_bgr"].shape[1]),
            "image_height": int(r["img0_bgr"].shape[0]),
        })

    write_data(out_dir, records)


# ---------------------------------------------------------------------------
# Mode B: HomographyDataset (legacy)
# ---------------------------------------------------------------------------

def run_dataset_mode(args):
    from omegaconf import OmegaConf
    from gluefactory.datasets.homographies import HomographyDataset
    from gluefactory.geometry.gt_generation import gt_matches_from_homography
    from gluefactory.settings import DATA_PATH
    import torch

    dataset_dir = DATA_PATH / args.dataset
    out_dir     = dataset_dir / "visualizations/matches_interactive"
    img_dir     = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    conf = OmegaConf.load(args.conf)
    conf.data.data_dir = args.dataset
    dataset = HomographyDataset(conf.data)

    records = []
    for split in ["train", "val"]:
        sub = dataset.get_dataset(split)
        logger.info(f"Split '{split}': {len(sub)} samples")

        for idx in range(len(sub)):
            item = sub[idx]
            name = item["name"]

            img0_np = item["view0"]["image"].permute(1, 2, 0).numpy()
            img1_np = item["view1"]["image"].permute(1, 2, 0).numpy()
            img0_bgr = (img0_np * 255).astype(np.uint8)[:, :, ::-1]
            img1_bgr = (img1_np * 255).astype(np.uint8)[:, :, ::-1]

            n0 = f"{split}_{idx}_view0.jpg"
            n1 = f"{split}_{idx}_view1.jpg"
            cv2.imwrite(str(img_dir / n0), img0_bgr)
            cv2.imwrite(str(img_dir / n1), img1_bgr)

            kp0     = item["view0"]["cache"]["keypoints"]
            kp1     = item["view1"]["cache"]["keypoints"]
            scores0 = item["view0"]["cache"]["keypoint_scores"]
            scores1 = item["view1"]["cache"]["keypoint_scores"]
            H       = torch.from_numpy(item["H_0to1"]).unsqueeze(0)

            md = gt_matches_from_homography(
                kp0.unsqueeze(0), kp1.unsqueeze(0), H,
                pos_th=conf.model.ground_truth.th_positive,
                neg_th=conf.model.ground_truth.th_negative,
            )
            m0 = md["matches0"][0].numpy()
            matches = [[int(i0), int(i1), 1.0] for i0, i1 in enumerate(m0) if i1 >= 0]
            n_match = len(matches)

            logger.info(
                f"[{split} {idx}] {name}  "
                f"kpts {len(kp0)}/{len(kp1)}  matches {n_match}"
            )

            records.append({
                "idx":         idx,
                "name0":       f"{split}/{name}",
                "name1":       f"{split}/{name} (warped)",
                "image0_url":  f"images/{n0}",
                "image1_url":  f"images/{n1}",
                "keypoints0":  kp0.tolist(),
                "keypoints1":  kp1.tolist(),
                "scores0":     scores0.tolist(),
                "scores1":     scores1.tolist(),
                "matches":     matches,
                "H_0to1":      item["H_0to1"].tolist(),
                "metrics": {
                    "total_kpts0": int(len(kp0)),
                    "total_kpts1": int(len(kp1)),
                    "num_matches": n_match,
                    "match_ratio": float(n_match / max(1, len(kp0))),
                    "avg_mscore":  1.0,
                },
                "image_width":  int(img0_bgr.shape[1]),
                "image_height": int(img0_bgr.shape[0]),
            })

    write_data(out_dir, records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ck = parser.add_argument_group("Checkpoint inference mode")
    ck.add_argument("--input", type=str, default=None,
                    help="Image directory or 'img0.png,img1.png'")
    ck.add_argument("--checkpoint_superglue", type=str, default=None,
                    help="SuperGlue checkpoint (.tar)")
    ck.add_argument("--checkpoint_superpoint", type=str, default=None,
                    help="SuperPoint checkpoint (.tar)")
    ck.add_argument("--output_dir", type=str, default=None,
                    help="Output directory (default: <input_parent>/visualizations/matches_interactive)")
    ck.add_argument("--resize", type=int, default=512,
                    help="Resize longest side (0 = no resize)")
    ck.add_argument("--max_pairs", type=int, default=0,
                    help="Max pairs to process (0 = all)")
    ck.add_argument("--nms_radius", type=int, default=3)
    ck.add_argument("--max_num_keypoints", type=int, default=512)
    ck.add_argument("--detection_threshold", type=float, default=0.005)
    ck.add_argument("--filter_threshold", type=float, default=0.01)

    lg = parser.add_argument_group("Legacy HomographyDataset mode")
    lg.add_argument("--dataset", type=str, default="output/sample_data",
                    help="Dataset dir under data/")
    lg.add_argument("--conf", type=str,
                    default="gluefactory/configs/superpoint_custom+superglue_homography_custom_dataset.yaml")

    args = parser.parse_args()

    if args.checkpoint_superglue and args.checkpoint_superpoint:
        if args.input is None:
            parser.error("--input is required in checkpoint mode")
        if args.output_dir is None:
            args.output_dir = str(
                Path(args.input).parent / "visualizations/matches_interactive"
            )
        run_checkpoint_mode(args)
    else:
        run_dataset_mode(args)


if __name__ == "__main__":
    main()
