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
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export_matches")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_image(path, resize_max=0):
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape[:2]
    if resize_max > 0:
        scale = resize_max / max(h, w)
        w_new = int(round(w * scale / 8) * 8)
        h_new = int(round(h * scale / 8) * 8)
        img_gray = cv2.resize(img_gray, (w_new, h_new), interpolation=cv2.INTER_AREA)
        img_bgr  = cv2.resize(img_bgr,  (w_new, h_new), interpolation=cv2.INTER_AREA)
    else:
        w_new = (w // 8) * 8
        h_new = (h // 8) * 8
        if w_new != w or h_new != h:
            img_gray = cv2.resize(img_gray, (w_new, h_new), interpolation=cv2.INTER_AREA)
            img_bgr  = cv2.resize(img_bgr,  (w_new, h_new), interpolation=cv2.INTER_AREA)
    return img_bgr, img_gray


def get_image_pairs(input_path):
    path = Path(input_path)
    if "," in input_path:
        parts = [Path(p.strip()) for p in input_path.split(",")]
        if len(parts) == 2:
            return [(parts[0], parts[1])]

    if not path.is_dir():
        logger.error(f"Not a directory: {input_path}")
        return []

    exts = {"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff",
            "*.PNG", "*.JPG", "*.JPEG"}
    files = sorted({f for ext in exts for f in path.glob(ext)})
    return [(files[i], files[i + 1]) for i in range(len(files) - 1)]


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
    from gluefactory.utils.experiments import load_experiment

    sg_ckpt = Path(args.checkpoint_superglue)
    sp_ckpt = Path(args.checkpoint_superpoint)

    if not sg_ckpt.exists():
        logger.error(f"SuperGlue checkpoint not found: {sg_ckpt}"); return
    if not sp_ckpt.exists():
        logger.error(f"SuperPoint checkpoint not found: {sp_ckpt}"); return

    pairs = get_image_pairs(args.input)
    if not pairs:
        logger.error(f"No image pairs found in: {args.input}"); return
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    logger.info(f"Processing {len(pairs)} pairs on "
                f"{'cuda' if torch.cuda.is_available() else 'cpu'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_experiment(str(sg_ckpt), {
        "extractor": {
            "name": "superpoint_open",
            "weights": str(sp_ckpt),
            "nms_radius": args.nms_radius,
            "max_num_keypoints": args.max_num_keypoints,
            "detection_threshold": args.detection_threshold,
            "remove_borders": 4,
            "trainable": False,
        },
        "matcher": {
            "name": "gluefactory_nonfree.superglue",
            "filter_threshold": args.filter_threshold,
        },
    }).to(device).eval()

    out_dir   = Path(args.output_dir)
    img_dir   = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, (p0, p1) in enumerate(tqdm(pairs, desc="Matching")):
        try:
            img0_bgr, img0_gray = load_image(p0, args.resize)
            img1_bgr, img1_gray = load_image(p1, args.resize)

            n0 = f"pair_{idx}_view0.png"
            n1 = f"pair_{idx}_view1.png"
            cv2.imwrite(str(img_dir / n0), img0_bgr)
            cv2.imwrite(str(img_dir / n1), img1_bgr)

            t0 = torch.from_numpy(img0_gray).float().div(255).unsqueeze(0).unsqueeze(0).to(device)
            t1 = torch.from_numpy(img1_gray).float().div(255).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model({"view0": {"image": t0}, "view1": {"image": t1}})

            kpts0    = pred["keypoints0"][0].cpu().numpy()
            kpts1    = pred["keypoints1"][0].cpu().numpy()
            scores0  = pred["keypoint_scores0"][0].cpu().numpy()
            scores1  = pred["keypoint_scores1"][0].cpu().numpy()
            matches0 = pred["matches0"][0].cpu().numpy()
            mscores0 = pred["matching_scores0"][0].cpu().numpy()

            matches = [
                [int(i0), int(i1), float(mscores0[i0])]
                for i0, i1 in enumerate(matches0) if i1 != -1
            ]
            n_match = len(matches)
            ratio   = n_match / max(1, min(len(kpts0), len(kpts1)))
            avg_s   = float(np.mean([m[2] for m in matches])) if matches else 0.0

            logger.info(
                f"[{idx}] {p0.name} ↔ {p1.name}  "
                f"kpts {len(kpts0)}/{len(kpts1)}  "
                f"matches {n_match} ({ratio:.1%})  avg {avg_s:.3f}"
            )

            records.append({
                "idx":         idx,
                "name0":       p0.name,
                "name1":       p1.name,
                "image0_url":  f"images/{n0}",
                "image1_url":  f"images/{n1}",
                "keypoints0":  kpts0.tolist(),
                "keypoints1":  kpts1.tolist(),
                "scores0":     scores0.tolist(),
                "scores1":     scores1.tolist(),
                "matches":     matches,
                "metrics": {
                    "total_kpts0": int(len(kpts0)),
                    "total_kpts1": int(len(kpts1)),
                    "num_matches": n_match,
                    "match_ratio": float(ratio),
                    "avg_mscore":  float(avg_s),
                },
                "image_width":  int(img0_bgr.shape[1]),
                "image_height": int(img0_bgr.shape[0]),
            })

        except Exception as e:
            logger.error(f"Error on pair {idx}: {e}", exc_info=True)

    write_data(out_dir, records)


# ---------------------------------------------------------------------------
# Mode B: HomographyDataset (legacy)
# ---------------------------------------------------------------------------

def run_dataset_mode(args):
    from omegaconf import OmegaConf
    from gluefactory.datasets.homographies import HomographyDataset
    from gluefactory.geometry.gt_generation import gt_matches_from_homography
    from gluefactory.settings import DATA_PATH

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
