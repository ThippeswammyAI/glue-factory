"""
Dataset class for Pose-Based SLAM Images
"""
import logging
from pathlib import Path
from collections.abc import Iterable
import cv2
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from ..geometry.wrappers import Camera, Pose
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor
from ..utils.tools import fork_rng
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class SlamPosedDataset(BaseDataset):
    default_conf = {
        "data_dir": "output/slam",
        "scene": "slam_scene",
        "train_size": "???",
        "val_size": "???",
        "pose_file": "poses_odom_RGBD_slam.txt",
        "calib_dir": "images/calib",
        "image_dir": "images/rgb",
        "depth_dir": "images/depth",
        "depth_format": "png", # 16-bit PNG mm -> meters
        
        "pairs_train": "pairs_train.txt",
        "pairs_val": "pairs_val.txt",
        
        "read_depth": True,
        "read_image": True,
        "grayscale": True,
        "preprocessing": ImagePreprocessor.default_conf,
        "p_rotate": 0.0,
        "reseed": False,
        "seed": 0,
        
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
        },
    }

    def _init(self, conf):
        pass
        
    def get_dataset(self, split):
        return _SlamPairDataset(self.conf, split)

class _SlamPairDataset(torch.utils.data.Dataset):
    def __init__(self, conf, split, load_sample=True):
        self.root = DATA_PATH / conf.data_dir
        assert self.root.exists(), f"Data root not found: {self.root}"
        self.split = split
        self.conf = conf
        
        self.scene = conf.scene
        
        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)
            
        self.preprocessor = ImagePreprocessor(conf.preprocessing)
        
        # Parse Poses
        self.poses_w2c = {} # timestamp -> Pose wrapper
        poses_path = self.root / conf.pose_file
        with open(poses_path, "r") as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.strip().split()
                if len(parts) < 8: continue
                ts = parts[0]
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                
                rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
                t = np.array([tx, ty, tz])
                
                # R_cw = R_wc^T, t_cw = -R_wc^T * t_wc
                R_cw = rot.T
                t_cw = -R_cw @ t
                T_cw = np.eye(4, dtype=np.float32)
                T_cw[:3, :3] = R_cw
                T_cw[:3, 3] = t_cw
                
                self.poses_w2c[ts] = Pose.from_4x4mat(T_cw).float()
                
        # Parse camera calibration
        self.K = None
        calib_dir = self.root / conf.calib_dir
        calib_files = list(calib_dir.glob("*.yaml"))
        if len(calib_files) > 0:
            import yaml
            with open(calib_files[0], "r") as f:
                lines = f.readlines()
                if lines and lines[0].startswith("%YAML"):
                    lines = lines[1:]
                data = yaml.unsafe_load("".join(lines))
            k_flat = data["camera_matrix"]["data"]
            K = np.array(k_flat, dtype=np.float32).reshape(3, 3)
            self.K = K
            self.camera = Camera.from_calibration_matrix(self.K).float()
        else:
            raise FileNotFoundError(f"No calib yaml found in {calib_dir}")
            
        # Parse pairs
        self.items = []
        pairs_file = conf[f"pairs_{split}"]
        if isinstance(pairs_file, str) and pairs_file:
            pairs_path = self.root / pairs_file
            if pairs_path.exists():
                with open(pairs_path, "r") as f:
                    for line in f:
                        if not line.strip(): continue
                        im0, im1 = line.strip().split()
                        # e.g., rgb/1775717079.106858.png
                        self.items.append((im0, im1))
            else:
                # If pairs file doesn't exist, we might be training SP and need an image list
                logger.warning(f"Pairs file {pairs_path} not found. Using single images if list exists.")
                list_path = self.root / f"image_list_{split}.txt"
                if list_path.exists():
                    with open(list_path, "r") as f:
                        for line in f:
                            if not line.strip(): continue
                            self.items.append((line.strip(), None))
                            
    def _read_view(self, image_rel_path):
        # image_rel_path is e.g. "rgb/1775717079.106858.png"
        path = self.root / self.conf.image_dir / Path(image_rel_path).name
        name = path.name
        ts = path.stem
        
        # read image
        if self.conf.read_image:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if self.conf.grayscale else cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read image {path}")
            if self.conf.grayscale:
                img = np.expand_dims(img, axis=-1)
            img = img.transpose(2, 0, 1) # HWC -> CHW
            img = torch.from_numpy(img).float()
        else:
            img = torch.zeros([1, 100, 100]).float() # Dummy
            
        # read depth
        depth = None
        if self.conf.read_depth:
            depth_path = self.root / self.conf.depth_dir / name
            if depth_path.exists():
                depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth_img is not None:
                    depth_m = depth_img.astype(np.float32) / 1000.0
                    depth = torch.from_numpy(depth_m).unsqueeze(0)
                
        # pose
        T_w2cam = self.poses_w2c.get(ts, Pose.from_4x4mat(np.eye(4, dtype=np.float32)).float())
        
        data = self.preprocessor(img)
        if depth is not None:
            data["depth"] = self.preprocessor(depth, interpolation="nearest")["image"][0]
            
        data = {
            "name": name,
            "scene": self.scene,
            "T_w2cam": T_w2cam,
            "depth": depth,
            "camera": self.camera.scale(data["scales"]),
            **data,
        }
        
        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            data = {"cache": features, **data}
            
        return data

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def getitem(self, idx):
        item = self.items[idx]
        if item[1] is not None:
            # 2 views
            data0 = self._read_view(item[0])
            data1 = self._read_view(item[1])
            data = {
                "view0": data0,
                "view1": data1,
            }
            data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
            data["T_1to0"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()
            data["name"] = f"{self.scene}/{data0['name']}_{data1['name']}"
        else:
            # 1 view
            data = self._read_view(item[0])
            data["scene"] = self.scene
            data["name"] = f"{self.scene}/{data['name']}"
            
        data["idx"] = idx
        return data

    def __len__(self):
        return len(self.items)
