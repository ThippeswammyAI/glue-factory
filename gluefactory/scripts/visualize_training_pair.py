import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

from gluefactory.datasets import get_dataset
from gluefactory.models import get_model


CONFIG = "gluefactory/configs/superpoint+lightglue_homography.yaml"


def draw_matches(img0, img1, kp0, kp1, matches):

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    H = max(h0, h1)

    canvas = np.zeros((H, w0 + w1, 3), dtype=np.uint8)

    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0+w1] = img1

    for i, j in matches:

        x0, y0 = kp0[i]
        x1, y1 = kp1[j]

        x0 = int(round(x0))
        y0 = int(round(y0))

        x1 = int(round(x1)) + w0
        y1 = int(round(y1))

        cv2.circle(canvas, (x0, y0), 3, (0,255,0), -1)
        cv2.circle(canvas, (x1, y1), 3, (0,255,0), -1)

        cv2.line(
            canvas,
            (x0,y0),
            (x1,y1),
            (255,0,0),
            1,
            cv2.LINE_AA
        )

    return canvas


def main():

    conf = OmegaConf.load(CONFIG)

    ############################################
    # Dataset
    ############################################

    dataset = get_dataset("homographies")(conf.data)
    train_ds = dataset.get_dataset("train")

    sample = train_ds[0]

    print("Sample name:", sample["name"])

    ############################################
    # SuperPoint
    ############################################

    sp = get_model(
        conf.model.extractor.name
    )(conf.model.extractor)

    sp.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = sp.to(device)

    img0 = sample["view0"]["image"].unsqueeze(0).to(device)
    img1 = sample["view1"]["image"].unsqueeze(0).to(device)

    with torch.no_grad():

        pred0 = sp({"image": img0})
        pred1 = sp({"image": img1})

    kp0 = pred0["keypoints"][0]
    kp1 = pred1["keypoints"][0]

    print("KP0:", kp0.shape)
    print("KP1:", kp1.shape)

    ############################################
    # Homography GT matcher
    ############################################

    gt_matcher = get_model(
        conf.model.ground_truth.name
    )(conf.model.ground_truth)

    data = {
        **sample,
        "keypoints0": kp0.unsqueeze(0),
        "keypoints1": kp1.unsqueeze(0),
    }

    if not isinstance(data["H_0to1"], torch.Tensor):
        data["H_0to1"] = torch.from_numpy(
            data["H_0to1"]
        ).float()

    # DEBUG
    print(type(data["H_0to1"]))
    print(data["H_0to1"].shape)
    print(type(data["keypoints0"]))
    print(data["keypoints0"].shape)

    # DEBUG
    print(type(data["H_0to1"]))
    print(data["H_0to1"].shape)
    print(type(data["keypoints0"]))
    print(data["keypoints0"].shape)

    ####################################################
    # Generate GT matches
    ####################################################

    gt = gt_matcher(data)

    print("\nGT Keys:")
    print(gt.keys())

    print("matches0:", gt["matches0"].shape)
    print("matches1:", gt["matches1"].shape)

    ####################################################
    # Extract matches
    ####################################################

    matches0 = gt["matches0"][0].cpu().numpy()

    valid = np.where(matches0 >= 0)[0]

    matches = []

    for i in valid:
        matches.append((i, matches0[i]))

    print("\nGT Matches:", len(matches))
    gt = gt_matcher(data)

    print("\nGT Keys:")
    print(gt.keys())

    ####################################################
    # Extract matches
    ####################################################

    # matches0 = gt["gt_matches0"][0].cpu().numpy()
    matches0 = gt["matches0"][0].cpu().numpy()

    valid = np.where(matches0 >= 0)[0]

    matches = []

    for i in valid:
        matches.append((i, matches0[i]))

    print("\nGT Matches:", len(matches))

    ####################################################
    # Convert images
    ####################################################

    img0_np = (
        sample["view0"]["image"]
        .permute(1,2,0)
        .numpy()
        * 255
    ).astype(np.uint8)

    img1_np = (
        sample["view1"]["image"]
        .permute(1,2,0)
        .numpy()
        * 255
    ).astype(np.uint8)

    kp0_np = kp0.cpu().numpy()
    kp1_np = kp1.cpu().numpy()

    ####################################################
    # Draw
    ####################################################

    vis = draw_matches(
        img0_np,
        img1_np,
        kp0_np,
        kp1_np,
        matches[:300]
    )

    plt.figure(figsize=(18,8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(
        f"GT Homography Matches ({len(matches)})"
    )

    plt.tight_layout()

    out_file = "training_pair_visualization.png"

    plt.savefig(
        out_file,
        dpi=200,
        bbox_inches="tight"
    )

    print("\nSaved:", out_file)

    plt.show()


if __name__ == "__main__":
    main()