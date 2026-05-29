"""PyTorch implementation of the SuperPoint model,
derived from the TensorFlow re-implementation (2018).
Authors: Rémi Pautrat, Paul-Edouard Sarlin
https://github.com/rpautrat/SuperPoint
The implementation of this model and its trained weights are made
available under the MIT license.
"""

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..utils.misc import pad_and_stack
from ...geometry.homography import warp_points_torch


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


def batched_nms(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def select_top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


class VGGBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding
        )
        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        bn = nn.BatchNorm2d(c_out, eps=0.001)
        super().__init__(
            OrderedDict(
                [
                    ("conv", conv),
                    ("activation", activation),
                    ("bn", bn),
                ]
            )
        )


class SuperPoint(BaseModel):
    default_conf = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": None,
        "force_num_keypoints": False,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "channels": [64, 64, 128, 128, 256],
        "dense_outputs": True,
        "weights": None,  # local path of pretrained weights
        "lambda_d": 1.0,
    }

    checkpoint_url = "https://github.com/rpautrat/SuperPoint/raw/master/weights/superpoint_v6_from_tf.pth"  # noqa: E501

    def _init(self, conf):
        self.conf = SimpleNamespace(**conf)
        self.stride = 2 ** (len(self.conf.channels) - 2)
        channels = [1, *self.conf.channels[:-1]]

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.conf.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.conf.descriptor_dim, 1, relu=False),
        )

        if conf.weights is not None and Path(conf.weights).exists():
            state_dict = torch.load(conf.weights, map_location="cpu")
        else:
            state_dict = torch.hub.load_state_dict_from_url(self.checkpoint_url)
        self.load_state_dict(state_dict)

    def _forward(self, data):
        image = data["image"]
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        features = self.backbone(image)
        descriptors_dense = torch.nn.functional.normalize(
            self.descriptor(features), p=2, dim=1
        )

        # Decode the detection scores
        logits_raw = self.detector(features)
        scores = torch.nn.functional.softmax(logits_raw, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(
            b, h * self.stride, w * self.stride
        )
        scores = batched_nms(scores, self.conf.nms_radius)

        # Discard keypoints near the image borders
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints
        if b > 1:
            idxs = torch.where(scores > self.conf.detection_threshold)
            mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
        else:  # Faster shortcut
            scores = scores.squeeze(0)
            idxs = torch.where(scores > self.conf.detection_threshold)

        # Convert (i, j) to (x, y)
        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        scores_all = scores[idxs]

        keypoints = []
        scores = []
        for i in range(b):
            if b > 1:
                k = keypoints_all[mask[i]]
                s = scores_all[mask[i]]
            else:
                k = keypoints_all
                s = scores_all
            if self.conf.max_num_keypoints is not None:
                k, s = select_top_k_keypoints(k, s, self.conf.max_num_keypoints)

            keypoints.append(k)
            scores.append(s)

        if self.conf.force_num_keypoints:
            keypoints = pad_and_stack(
                keypoints,
                self.conf.max_num_keypoints,
                -2,
                mode="random_c",
                bounds=(
                    0,
                    data.get("image_size", torch.tensor(image.shape[-2:])).min().item(),
                ),
            )
            scores = pad_and_stack(
                scores, self.conf.max_num_keypoints, -1, mode="zeros"
            )
        else:
            keypoints = torch.stack(keypoints, 0)
            scores = torch.stack(scores, 0)

        if len(keypoints) == 1 or self.conf.force_num_keypoints:
            # Batch sampling of the descriptors
            desc = sample_descriptors(keypoints, descriptors_dense, self.stride)
        else:
            desc = [
                sample_descriptors(k[None], d[None], self.stride)[0]
                for k, d in zip(keypoints, descriptors_dense)
            ]

        pred = {
            "keypoints": keypoints + 0.5,
            "keypoint_scores": scores,
            "descriptors": desc.transpose(-1, -2),
            "logits": logits_raw,
        }
        if self.conf.dense_outputs:
            pred["dense_descriptors"] = descriptors_dense

        return pred

    def loss(self, pred, data):
        # We need both views for training
        assert "logits0" in pred and "logits1" in pred
        assert "view0" in data and "view1" in data

        def keypoints_to_grid(keypoints, image_shape, device):
            """Convert keypoints to a 65-channel classification target."""
            b = len(keypoints)
            h, w = image_shape
            hc, wc = h // 8, w // 8
            targets = torch.full((b, hc, wc), 64, dtype=torch.long, device=device)
            for i in range(b):
                kpts = keypoints[i]
                if len(kpts) == 0:
                    continue
                valid = (kpts[:, 0] >= 0) & (kpts[:, 0] < w) & (kpts[:, 1] >= 0) & (kpts[:, 1] < h)
                kpts = kpts[valid]
                if len(kpts) == 0:
                    continue
                xi = torch.clamp(torch.floor(kpts[:, 0]).long(), 0, w - 1)
                yi = torch.clamp(torch.floor(kpts[:, 1]).long(), 0, h - 1)
                cx = xi // 8
                cy = yi // 8
                ix = xi % 8
                iy = yi % 8
                label = iy * 8 + ix
                targets[i, cy, cx] = label
            return targets

        # 1. Detector Loss (Cross Entropy)
        loss_det = 0.0
        for i in ["0", "1"]:
            logits = pred[f"logits{i}"]
            img = data[f"view{i}"]["image"]
            h, w = img.shape[-2:]
            
            cache = data[f"view{i}"].get("cache", None)
            if cache is None or "keypoints" not in cache:
                raise ValueError(
                    f"Ground-truth keypoints cache not found in view{i}. "
                    "Ensure load_features.do is enabled and populated."
                )
            gt_kpts = cache["keypoints"]
            targets = keypoints_to_grid(gt_kpts, (h, w), logits.device)
            loss_det += torch.nn.functional.cross_entropy(logits, targets)
        loss_det = loss_det / 2.0

        # 2. Descriptor Loss (Grid-based contrastive loss)
        assert "dense_descriptors0" in pred and "dense_descriptors1" in pred
        desc0 = pred["dense_descriptors0"]
        desc1 = pred["dense_descriptors1"]
        b, c, hc, wc = desc0.shape
        h, w = hc * 8, wc * 8
        device = desc0.device

        # Create a grid of cell centers
        grid_y, grid_x = torch.meshgrid(
            torch.arange(hc, device=device),
            torch.arange(wc, device=device),
            indexing="ij"
        )
        centers0 = torch.stack([grid_x, grid_y], dim=-1).float() * 8.0 + 4.0
        centers0 = centers0.reshape(-1, 2)
        centers1 = centers0.clone()

        H_0to1 = data["H_0to1"]
        loss_desc = 0.0
        zeros = torch.tensor(0.0, device=device)

        for i in range(b):
            warped_centers0 = warp_points_torch(centers0[None], H_0to1[i], inverse=False)[0]
            dist = torch.norm(warped_centers0[:, None, :] - centers1[None, :, :], dim=-1)
            Y = (dist < 8.0).float()
            
            # Filter points warping outside image bounds
            valid_warp = (warped_centers0[:, 0] >= 0) & (warped_centers0[:, 0] < w) & \
                         (warped_centers0[:, 1] >= 0) & (warped_centers0[:, 1] < h)
            Y = Y * valid_warp[:, None]

            # Cosine similarity between dense descriptors
            d0 = desc0[i].reshape(c, -1)
            d1 = desc1[i].reshape(c, -1)
            S = torch.matmul(d0.t(), d1)

            loss_pos = Y * torch.max(zeros, 1.0 - S)
            loss_neg = (1.0 - Y) * torch.max(zeros, S - 0.2)
            loss_desc += (loss_pos + loss_neg).mean()

        loss_desc = loss_desc / b

        lambda_d = self.conf.lambda_d if hasattr(self.conf, "lambda_d") else 1.0
        total_loss = loss_det + lambda_d * loss_desc

        losses = {
            "total": total_loss,
            "detector": loss_det,
            "descriptor": loss_desc,
        }
        metrics = {
            "loss/total": total_loss.detach(),
            "loss/detector": loss_det.detach(),
            "loss/descriptor": loss_desc.detach(),
        }
        return losses, metrics
