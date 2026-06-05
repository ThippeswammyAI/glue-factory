import h5py
import numpy as np
import shutil
from pathlib import Path

def main():
    h5_path = Path("data/exports/custom_dataset_custom_SP-k2048-nms4.h5")
    bak_path = h5_path.with_suffix(".h5.bak")
    tmp_path = h5_path.with_suffix(".h5.tmp")
    
    # 1. Create a backup
    print(f"Creating backup of {h5_path} at {bak_path}...")
    shutil.copyfile(h5_path, bak_path)
    
    # 2. Read from original and write filtered to tmp
    print("Filtering keypoints with score >= 0.02...")
    with h5py.File(h5_path, "r") as f_in, h5py.File(tmp_path, "w") as f_out:
        for group_name in f_in.keys():
            g_in = f_in[group_name]
            g_out = f_out.create_group(group_name)
            
            for img_name in g_in.keys():
                img_grp_in = g_in[img_name]
                img_grp_out = g_out.create_group(img_name)
                
                scores = img_grp_in["keypoint_scores"].__array__()
                mask = scores >= 0.02
                
                filtered_scores = scores[mask]
                filtered_kpts = img_grp_in["keypoints"].__array__()[mask]
                filtered_desc = img_grp_in["descriptors"].__array__()[mask]
                
                img_grp_out.create_dataset("keypoint_scores", data=filtered_scores, dtype=np.float16)
                img_grp_out.create_dataset("keypoints", data=filtered_kpts, dtype=np.float16)
                img_grp_out.create_dataset("descriptors", data=filtered_desc, dtype=np.float16)
                
                print(f"  {img_name}: {len(scores)} -> {len(filtered_scores)}")
                
    # 3. Replace the original file with the tmp file
    print(f"Replacing {h5_path} with the filtered file...")
    tmp_path.replace(h5_path)
    print("Filtering complete!")

if __name__ == "__main__":
    main()
