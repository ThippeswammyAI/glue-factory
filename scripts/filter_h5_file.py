"""
Filter keypoints in an HDF5 feature file by a minimum score threshold.

Reads an H5 file produced by SuperPoint export scripts, drops all keypoints
(and their descriptors) below --min_score, and writes a new filtered file.
Creates a .bak backup of the input file before overwriting it in place, or
writes to a separate output path if --output is given.

Usage:
    python scripts/filter_h5_file.py \\
        --input  data/output/slam/exports/sp_features_slam.h5 \\
        --min_score 0.02

    # Write to a new file instead of overwriting:
    python scripts/filter_h5_file.py \\
        --input  data/output/slam/exports/sp_features_slam.h5 \\
        --output data/output/slam/exports/sp_features_filtered.h5 \\
        --min_score 0.05
"""

import argparse
import h5py
import numpy as np
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Filter H5 keypoints by score threshold.")
    parser.add_argument("--input", type=str, required=True,
                        help="Input H5 file path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output H5 file path (default: overwrite input after backup)")
    parser.add_argument("--min_score", type=float, default=0.02,
                        help="Minimum keypoint score to keep (default: 0.02)")
    args = parser.parse_args()

    h5_path  = Path(args.input)
    out_path = Path(args.output) if args.output else h5_path
    tmp_path = h5_path.with_suffix(".h5.tmp")

    if not h5_path.exists():
        print(f"Error: input file not found: {h5_path}")
        return

    # Backup only when overwriting in place
    if out_path == h5_path:
        bak_path = h5_path.with_suffix(".h5.bak")
        print(f"Backing up {h5_path} → {bak_path}")
        shutil.copyfile(h5_path, bak_path)

    print(f"Filtering {h5_path}  (min_score={args.min_score}) → {out_path}")

    with h5py.File(h5_path, "r") as f_in, h5py.File(tmp_path, "w") as f_out:
        for group_name in f_in.keys():
            g_in  = f_in[group_name]
            g_out = f_out.create_group(group_name)

            for img_name in g_in.keys():
                ig_in  = g_in[img_name]
                ig_out = g_out.create_group(img_name)

                scores = ig_in["keypoint_scores"][:]
                mask   = scores >= args.min_score

                ig_out.create_dataset("keypoint_scores", data=scores[mask],                    dtype=np.float16)
                ig_out.create_dataset("keypoints",       data=ig_in["keypoints"][:][mask],     dtype=np.float16)
                ig_out.create_dataset("descriptors",     data=ig_in["descriptors"][:][mask],   dtype=np.float16)

                print(f"  {img_name}: {len(scores)} → {int(mask.sum())} keypoints")

    tmp_path.replace(out_path)
    print("Done.")


if __name__ == "__main__":
    main()
