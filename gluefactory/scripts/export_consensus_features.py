"""
Extract SuperPoint descriptors at consensus keypoint locations and save to H5.

Takes a pseudo-labels H5 file (containing consensus keypoints) and a trained
SuperPoint checkpoint, re-runs the dense descriptor backbone on each image, then
samples descriptors at the consensus keypoint coordinates. The result is an H5
file with keypoints, keypoint_scores, and descriptors ready for SuperGlue
training or evaluation.

Usage:
    python -m gluefactory.scripts.export_consensus_features \\
        --dataset         output/slam \\
        --pseudo_labels_h5 data/output/slam/exports/pseudo_labels_slam.h5 \\
        --weights         outputs/training/superpoint_slam_run/checkpoint_best.tar \\
        --output_h5       data/output/slam/exports/sp_features_slam.h5 \\
        --modality        rgb
"""

import argparse
import h5py
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm

from gluefactory.models import get_model
from gluefactory.settings import DATA_PATH
from gluefactory.models.extractors.superpoint_open import sample_descriptors

def main():
    parser = argparse.ArgumentParser(description="Extract descriptors for consensus keypoints")
    parser.add_argument("--dataset", type=str, default="custom_dataset", help="Dataset name under data/")
    parser.add_argument("--pseudo_labels_h5", type=str, required=True, help="Path to pseudo labels H5 file")
    parser.add_argument("--weights", type=str, required=True, help="Path to custom SuperPoint weights")
    parser.add_argument("--output_h5", type=str, required=True, help="Path to save final H5 file")
    parser.add_argument("--modality", type=str, default="reflectivity", help="Modality prefix")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load custom model
    print(f"Loading custom SuperPoint model from {args.weights}...")
    model_conf = {
        "name": "superpoint_open",
        "nms_radius": 4,
        "max_num_keypoints": 2048,
        "detection_threshold": 0.0,
        "trainable": False,
        "weights": args.weights
    }
    model = get_model("superpoint_open")(model_conf).to(device).eval()
    
    pseudo_labels_path = Path(args.pseudo_labels_h5)
    if not pseudo_labels_path.exists():
        print(f"Error: pseudo labels file not found at {pseudo_labels_path}")
        return
        
    output_path = Path(args.output_h5)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    dataset_dir = DATA_PATH / args.dataset
    images_dir = dataset_dir / "images" / args.modality
    
    with h5py.File(pseudo_labels_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        # Create root group for the modality
        root_grp = f_out.create_group(args.modality)
        
        image_names = list(f_in.keys())
        print(f"Processing {len(image_names)} images...")
        
        for img_name in tqdm(image_names, desc="Extracting descriptors"):
            # Load keypoints and scores
            kpts = f_in[img_name]["keypoints"][...]
            scores = f_in[img_name]["keypoint_scores"][...]
            
            # Read image
            img_path = images_dir / img_name
            if not img_path.exists():
                print(f"Warning: Image {img_path} not found, skipping.")
                continue
                
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"Warning: Could not read image {img_path}, skipping.")
                continue
                
            h, w = img_gray.shape
            
            # Prepare tensor
            img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            
            with torch.no_grad():
                # Get dense descriptors from backbone + descriptor head
                features = model.backbone(img_tensor)
                descriptors_dense = torch.nn.functional.normalize(
                    model.descriptor(features), p=2, dim=1
                )
                
                if len(kpts) > 0:
                    kpts_tensor = torch.from_numpy(kpts).float().unsqueeze(0).to(device)
                    # Sample descriptors at keypoints
                    desc = sample_descriptors(kpts_tensor, descriptors_dense, model.stride)
                    desc = desc.squeeze(0).transpose(0, 1).cpu().numpy() # (N, 256)
                else:
                    desc = np.zeros((0, 256), dtype=np.float32)
            
            # Shift keypoints by 0.5 to match SuperPoint output standard
            kpts_shifted = kpts + 0.5
            
            # Save to output H5 file
            grp = root_grp.create_group(img_name)
            grp.create_dataset("keypoints", data=kpts_shifted)
            grp.create_dataset("keypoint_scores", data=scores)
            grp.create_dataset("descriptors", data=desc)
            
    print(f"Successfully saved features to {output_path}")

if __name__ == "__main__":
    main()
