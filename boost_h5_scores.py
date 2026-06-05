import h5py
import numpy as np
import argparse
from pathlib import Path

def boost_scores(input_h5, output_h5, scale_factor=5.0):
    input_path = Path(input_h5)
    output_path = Path(output_h5)
    
    if not input_path.exists():
        print(f"Error: Input file {input_h5} does not exist.")
        return
        
    print(f"Reading from: {input_path}")
    print(f"Writing to:   {output_path}")
    print(f"Applying scale factor: x{scale_factor}")
    
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        def visitor(name, obj):
            if isinstance(obj, h5py.Group):
                if name not in f_out:
                    f_out.create_group(name)
            elif isinstance(obj, h5py.Dataset):
                data = obj[...]
                if name.split('/')[-1] == "keypoint_scores":
                    boosted_data = np.clip(data * scale_factor, 0.0, 1.0)
                    f_out.create_dataset(name, data=boosted_data)
                    original_above_02 = np.sum(data >= 0.2)
                    boosted_above_02 = np.sum(boosted_data >= 0.2)
                    print(f"  {name}: scores >= 0.2 went from {original_above_02} to {boosted_above_02} (out of {len(data)})")
                else:
                    f_out.create_dataset(name, data=data)
                    
        f_in.visititems(visitor)
        
    print(f"Done! Saved boosted features to {output_h5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boost H5 feature scores by a scale factor.")
    parser.add_argument("--input", type=str, required=True, help="Input H5 file path")
    parser.add_argument("--output", type=str, required=True, help="Output H5 file path")
    parser.add_argument("--scale", type=float, default=5.0, help="Scale factor to multiply scores by")
    args = parser.parse_args()
    
    boost_scores(args.input, args.output, args.scale)
