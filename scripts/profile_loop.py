"""Benchmark the data-loading + homography-warp loop without model inference."""

import time
import numpy as np

def profile_loop():
    h, w = 207, 512
    num_warps = 15
    modalities = 4
    kpts_per_warp = 2048
    
    joint_accumulator = np.zeros((h, w), dtype=np.float32)
    valid_mask = np.random.choice([True, False], size=(h, w))
    warped_coord_map = np.random.uniform(0, 511, size=(h, w, 2)).astype(np.float32)
    
    # Generate mock keypoints for 60 warps
    mock_warps_kpts = [np.random.uniform(0, 511, size=(kpts_per_warp, 2)).astype(np.float32) for _ in range(num_warps * modalities)]
    mock_warps_scores = [np.random.uniform(0, 1, size=(kpts_per_warp,)).astype(np.float32) for _ in range(num_warps * modalities)]
    
    t0 = time.time()
    for warp_idx in range(num_warps * modalities):
        kpts_warped = mock_warps_kpts[warp_idx]
        scores_warped = mock_warps_scores[warp_idx]
        
        for (x_prime, y_prime), score in zip(kpts_warped, scores_warped):
            ix_prime = int(round(x_prime))
            iy_prime = int(round(y_prime))
            if 0 <= ix_prime < w and 0 <= iy_prime < h:
                if valid_mask[iy_prime, ix_prime]:
                    x_orig = warped_coord_map[iy_prime, ix_prime, 0]
                    y_orig = warped_coord_map[iy_prime, ix_prime, 1]
                    ix_orig, iy_orig = int(round(x_orig)), int(round(y_orig))
                    if 0 <= ix_orig < w and 0 <= iy_orig < h:
                        joint_accumulator[iy_orig, ix_orig] += score
    t1 = time.time()
    print(f"Time for splatting loop: {t1-t0:.4f}s")

if __name__ == '__main__':
    profile_loop()
