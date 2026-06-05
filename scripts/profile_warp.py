import time
import numpy as np
import cv2

def profile_warp():
    H, W = 207, 512 # based on camera.info height: 207, width: 512
    depth_map = np.random.uniform(0.5, 10.0, size=(H, W)).astype(np.float32)
    img = np.random.randint(0, 255, size=(H, W), dtype=np.uint8)
    K = np.array([[256., 0., 256.], [0., 256., 103.9], [0., 0., 1.]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.array([0.1, 0.02, 0.05], dtype=np.float32)
    
    t0 = time.time()
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid_depth_mask = (depth_map > 0)
    u_valid = u[valid_depth_mask]
    v_valid = v[valid_depth_mask]
    d_valid = depth_map[valid_depth_mask]
    
    K_inv = np.linalg.inv(K)
    pixels_homo = np.stack([u_valid, v_valid, np.ones_like(u_valid)], axis=0)
    t1 = time.time()
    
    P_3D = d_valid * (K_inv @ pixels_homo)
    P_prime_3D = R @ P_3D + t[:, None]
    z_prime = P_prime_3D[2, :]
    t2 = time.time()
    
    valid_z_mask = (z_prime > 1e-3)
    projected = K @ P_prime_3D[:, valid_z_mask]
    u_prime = projected[0, :] / z_prime[valid_z_mask]
    v_prime = projected[1, :] / z_prime[valid_z_mask]
    z_prime = z_prime[valid_z_mask]
    u_orig = u_valid[valid_z_mask]
    v_orig = v_valid[valid_z_mask]
    t3 = time.time()
    
    u_idx = np.round(u_prime).astype(np.int32)
    v_idx = np.round(v_prime).astype(np.int32)
    in_bounds = (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)
    
    u_idx = u_idx[in_bounds]
    v_idx = v_idx[in_bounds]
    z_prime = z_prime[in_bounds]
    u_orig = u_orig[in_bounds]
    v_orig = v_orig[in_bounds]
    t4 = time.time()
    
    sort_idx = np.argsort(-z_prime)
    u_idx_sorted = u_idx[sort_idx]
    v_idx_sorted = v_idx[sort_idx]
    u_orig_sorted = u_orig[sort_idx]
    v_orig_sorted = v_orig[sort_idx]
    t5 = time.time()
    
    warped_img = np.zeros((H, W), dtype=img.dtype)
    colors = img[v_orig_sorted, u_orig_sorted]
    warped_img[v_idx_sorted, u_idx_sorted] = colors
    t6 = time.time()
    
    print(f"Meshgrid & Setup: {t1-t0:.4f}s")
    print(f"3D projection: {t2-t1:.4f}s")
    print(f"2D reprojection: {t3-t2:.4f}s")
    print(f"In-bounds masking: {t4-t3:.4f}s")
    print(f"Sorting (argsort): {t5-t4:.4f}s")
    print(f"Image warping: {t6-t5:.4f}s")
    print(f"Total: {t6-t0:.4f}s")

if __name__ == '__main__':
    profile_warp()
