import os
import csv
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

from gluefactory.datasets import get_dataset
from gluefactory.models import get_model


CONFIG = "gluefactory/configs/superpoint+lightglue_homography.yaml"

NUM_STEPS = 500

SAVE_DIR = "training_dataset_visualization"

VIS_STEPS = {
    0,
    50,
    100,
    200,
    300,
    400,
    499,
}


def draw_matches(img0, img1, kp0, kp1, matches):

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    H = max(h0, h1)

    canvas = np.zeros(
        (H, w0 + w1, 3),
        dtype=np.uint8
    )

    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0+w1] = img1

    for i, j in matches:

        x0, y0 = kp0[i]
        x1, y1 = kp1[j]

        x0 = int(round(x0))
        y0 = int(round(y0))

        x1 = int(round(x1)) + w0
        y1 = int(round(y1))

        cv2.circle(
            canvas,
            (x0, y0),
            3,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            canvas,
            (x1, y1),
            3,
            (0, 255, 0),
            -1
        )

        cv2.line(
            canvas,
            (x0, y0),
            (x1, y1),
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

    return canvas


def tensor_to_uint8(img_tensor):

    img = (
        img_tensor
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    img = np.clip(img * 255, 0, 255)

    return img.astype(np.uint8)


def save_visualization(
    out_file,
    img0,
    img1,
    kp0,
    kp1,
    matches,
    title,
):

    vis = draw_matches(
        img0,
        img1,
        kp0,
        kp1,
        matches[:300]
    )

    plt.figure(figsize=(18, 8))

    plt.imshow(
        cv2.cvtColor(
            vis,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.axis("off")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        out_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def main():

    os.makedirs(
        SAVE_DIR,
        exist_ok=True
    )

    csv_file = os.path.join(
        SAVE_DIR,
        "training_statistics.csv"
    )

    conf = OmegaConf.load(CONFIG)

    print("Loading dataset...")

    dataset = get_dataset(
        "homographies"
    )(conf.data)

    train_ds = dataset.get_dataset(
        "train"
    )

    print(
        "Training images:",
        len(train_ds.image_names)
    )

    print("Loading SuperPoint...")

    sp = get_model(
        conf.model.extractor.name
    )(conf.model.extractor)

    sp.eval()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    sp = sp.to(device)

    print(
        "Running on:",
        device
    )

    gt_matcher = get_model(
        conf.model.ground_truth.name
    )(conf.model.ground_truth)

    stats = []

    for step in range(NUM_STEPS):

        idx = step % len(train_ds)

        sample = train_ds[idx]

        img0 = (
            sample["view0"]["image"]
            .unsqueeze(0)
            .to(device)
        )

        img1 = (
            sample["view1"]["image"]
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():

            pred0 = sp(
                {"image": img0}
            )

            pred1 = sp(
                {"image": img1}
            )

        kp0 = pred0["keypoints"][0]
        kp1 = pred1["keypoints"][0]

        data = {
            **sample,
            "keypoints0": kp0.unsqueeze(0),
            "keypoints1": kp1.unsqueeze(0),
        }

        if not isinstance(
            data["H_0to1"],
            torch.Tensor,
        ):
            data["H_0to1"] = (
                torch.from_numpy(
                    data["H_0to1"]
                ).float()
            )

        gt = gt_matcher(data)

        matches0 = (
            gt["matches0"][0]
            .cpu()
            .numpy()
        )

        valid = np.where(
            matches0 >= 0
        )[0]

        matches = []

        for i in valid:
            matches.append(
                (
                    int(i),
                    int(matches0[i])
                )
            )

        num_matches = len(matches)

        stats.append(
            [
                step,
                idx,
                sample["name"],
                kp0.shape[0],
                kp1.shape[0],
                num_matches,
            ]
        )

        print(
            f"[{step+1}/{NUM_STEPS}] "
            f"{sample['name']} "
            f"KP0={kp0.shape[0]} "
            f"KP1={kp1.shape[0]} "
            f"GT={num_matches}"
        )

        if step in VIS_STEPS:

            img0_np = tensor_to_uint8(
                sample["view0"]["image"]
            )

            img1_np = tensor_to_uint8(
                sample["view1"]["image"]
            )

            kp0_np = (
                kp0.cpu().numpy()
            )

            kp1_np = (
                kp1.cpu().numpy()
            )

            out_file = os.path.join(
                SAVE_DIR,
                f"iter_{step:05d}.png"
            )

            save_visualization(
                out_file,
                img0_np,
                img1_np,
                kp0_np,
                kp1_np,
                matches,
                (
                    f"Step={step} "
                    f"GT Matches={num_matches}"
                ),
            )

    with open(
        csv_file,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "step",
                "dataset_idx",
                "image_name",
                "kp0",
                "kp1",
                "gt_matches",
            ]
        )

        writer.writerows(stats)

    match_counts = [
        s[5]
        for s in stats
    ]

    print("\n===================")
    print("SUMMARY")
    print("===================")

    print(
        "Min GT Matches:",
        np.min(match_counts)
    )

    print(
        "Max GT Matches:",
        np.max(match_counts)
    )

    print(
        "Mean GT Matches:",
        np.mean(match_counts)
    )

    print(
        "Median GT Matches:",
        np.median(match_counts)
    )

    print(
        "\nResults saved to:"
    )

    print(SAVE_DIR)

    print(
        "\nVisualizations:"
    )

    for s in sorted(VIS_STEPS):
        print(
            f"iter_{s:05d}.png"
        )

    print(
        "\nStatistics CSV:"
    )

    print(csv_file)


if __name__ == "__main__":
    main()
