import os
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from omegaconf import OmegaConf
from gluefactory.datasets.homographies import HomographyDataset
from gluefactory.geometry.gt_generation import gt_matches_from_homography

def main():
    # Define directories
    output_dir = Path("data/custom_dataset/visualizations/matches_interactive")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    conf_path = "gluefactory/configs/superpoint_custom+superglue_homography_custom_dataset.yaml"
    conf = OmegaConf.load(conf_path)
    data_conf = conf.data
    
    # Initialize dataset
    print("Initializing dataset...")
    dataset = HomographyDataset(data_conf)
    
    all_data = []
    
    # Process both train and val splits
    for split in ["train", "val"]:
        sub_dataset = dataset.get_dataset(split)
        print(f"Processing split '{split}' containing {len(sub_dataset)} samples...")
        
        for idx in range(len(sub_dataset)):
            item = sub_dataset[idx]
            name = item["name"]
            
            # Extract images and save them
            # Images are float tensors [3, H, W] in [0, 1]
            img0_np = item["view0"]["image"].permute(1, 2, 0).numpy()
            img1_np = item["view1"]["image"].permute(1, 2, 0).numpy()
            
            # Convert float [0, 1] RGB to uint8 BGR for cv2
            img0_bgr = (img0_np * 255.0).astype(np.uint8)[:, :, ::-1]
            img1_bgr = (img1_np * 255.0).astype(np.uint8)[:, :, ::-1]
            
            img0_filename = f"{split}_{idx}_view0.jpg"
            img1_filename = f"{split}_{idx}_view1.jpg"
            
            cv2.imwrite(str(images_dir / img0_filename), img0_bgr)
            cv2.imwrite(str(images_dir / img1_filename), img1_bgr)
            
            # Extract keypoints and compute matches
            kp0 = item["view0"]["cache"]["keypoints"]
            kp1 = item["view1"]["cache"]["keypoints"]
            scores0 = item["view0"]["cache"]["keypoint_scores"]
            scores1 = item["view1"]["cache"]["keypoint_scores"]
            H = torch.from_numpy(item["H_0to1"]).unsqueeze(0)
            
            # Compute matches
            matches_dict = gt_matches_from_homography(
                kp0.unsqueeze(0),
                kp1.unsqueeze(0),
                H,
                pos_th=conf.model.ground_truth.th_positive,
                neg_th=conf.model.ground_truth.th_negative
            )
            
            matches0 = matches_dict["matches0"][0].numpy()
            
            # Convert tensors/numpy to native python lists for JSON serialization
            kp0_list = kp0.tolist()
            kp1_list = kp1.tolist()
            scores0_list = scores0.tolist()
            scores1_list = scores1.tolist()
            
            # Create list of matches: [[idx0, idx1], ...]
            matches_list = []
            for idx0, idx1 in enumerate(matches0):
                if idx1 >= 0:
                    matches_list.append([idx0, int(idx1)])
                    
            H_list = item["H_0to1"].tolist()
            
            sample_info = {
                "idx": idx,
                "name": name,
                "split": split,
                "image0_url": f"images/{img0_filename}",
                "image1_url": f"images/{img1_filename}",
                "keypoints0": kp0_list,
                "keypoints1": kp1_list,
                "scores0": scores0_list,
                "scores1": scores1_list,
                "matches": matches_list,
                "H_0to1": H_list,
                "image_width": img0_bgr.shape[1],
                "image_height": img0_bgr.shape[0]
            }
            
            all_data.append(sample_info)
            print(f"  Sample {idx}: {len(kp0_list)} / {len(kp1_list)} kpts, {len(matches_list)} matches.")
            
    # Save matches data as a js file
    output_js_path = output_dir / "matches_data.js"
    with open(output_js_path, "w") as f:
        f.write("const MATCHES_DATA = ")
        json.dump(all_data, f, indent=2)
        f.write(";\n")
        
    print(f"Data saved to {output_js_path}")

if __name__ == "__main__":
    main()
