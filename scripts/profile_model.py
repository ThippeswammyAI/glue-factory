import time
import torch
from gluefactory.models import get_model

def profile_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    model = get_model("superpoint_open")({
        "nms_radius": 4,
        "max_num_keypoints": 2048,
        "detection_threshold": 0.0,
        "trainable": False
    }).to(device).eval()
    
    # Warmup
    img = torch.rand(1, 1, 207, 512).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model({"image": img})
            
    # Benchmark
    t0 = time.time()
    for _ in range(60):
        with torch.no_grad():
            pred = model({"image": img})
            kpts = pred["keypoints"][0].cpu().numpy()
            scores = pred["keypoint_scores"][0].cpu().numpy()
    t1 = time.time()
    
    print(f"Time for 60 forward passes: {t1-t0:.4f}s (Average: {(t1-t0)/60:.4f}s per pass)")

if __name__ == '__main__':
    profile_model()
