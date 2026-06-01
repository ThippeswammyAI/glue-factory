# Configuration Parameter Guide: `superpoint_custom_homography.yaml`

This document details what each configuration section and parameter does inside the `superpoint_custom_homography.yaml` training configuration file.

---

## 1. `data` (Dataset Settings)
This block configures the training and validation data loader using a **Homography Dataset**.

```yaml
data:
    name: homographies
    data_dir: custom_dataset1
    image_dir: images
    image_list: custom_image_list.txt
    train_size: 16
    val_size: 4
    batch_size: 1
    num_workers: 2
```
* **`name`**: `homographies` — Specifies that the dataset loader should instantiate the `HomographyDataset` class (defined in `gluefactory/datasets/homographies.py`).
* **`data_dir`**: `custom_dataset1` — The top-level dataset folder under `data/` (resolves to `data/custom_dataset1/`).
* **`image_dir`**: `images` — The directory inside `data_dir` where the images are located.
* **`image_list`**: `custom_image_list.txt` — A text file listing all raw images relative to the `image_dir`.
* **`train_size`**: `16` — The first `16` images in the `custom_image_list.txt` will be used for training.
* **`val_size`**: `4` — The next `4` images following the training split will be used for validation.
* **`batch_size`**: `1` — The number of images processed per training step.
* **`num_workers`**: `2` — The number of parallel CPU worker threads/subprocesses dedicated to loading images and performing augmentations.

---

### `data.homography` (Homographic Transformations)
These options control how the target image pairs (`view0` and `view1`) are synthesized via random homographies to train the keypoint extractor.

```yaml
    homography:
        difficulty: 0.7
        max_angle: 45
        patch_shape: [512, 512]
```
* **`difficulty`**: `0.7` — A scaling factor between `0.0` and `1.0` controlling the severity of the perspective warp (translations, scale changes, and shearing forces). Higher values produce more extreme angle distortions.
* **`max_angle`**: `45` — The maximum out-of-plane rotation (in degrees) that can be applied to simulate viewpoint changes.
* **`patch_shape`**: `[512, 512]` — The target output image resolution `[Height, Width]` of the warped images after applying homographic transformation.

---

### `data.photometric` (Visual Augmentations)
```yaml
    photometric:
        name: lg
```
* **`name`**: `lg` — Specifies the photometric augmentation technique block to apply (e.g., random brightness, contrast adjustments, noise, shadows, motion blur) matching the LightGlue or custom augmentation preset.

---

### `data.load_features` (Pseudo-Label Cache)
Precomputed feature keys are highly useful when training SuperPoint self-supervised or using knowledge distillation.

```yaml
    load_features:
        do: True
        path: "custom_dataset1/exports/pseudo_labels.h5"
        collate: False
        max_num_keypoints: 512
        force_num_keypoints: True
```
* **`do`**: `True` — Enables loading precomputed features (pseudo-labels) instead of computing ground truth matches/keypoints dynamically.
* **`path`**: `"custom_dataset1/exports/pseudo_labels.h5"` — Path to the cached labels HDF5 file containing the pre-extracted keypoints/descriptors.
* **`collate`**: `False` — Disables batch collation on the local features level when loading.
* **`max_num_keypoints`**: `512` — Caps the number of loaded keypoints per image to the top 512 features based on their confidence scores.
* **`force_num_keypoints`**: `True` — Forces the output tensor size to be exactly `512` keypoints. If less than 512 keypoints are cached, the system pads the remaining space with dummy keypoints.

---

## 2. `model` (Architecture Configuration)
Defines the deep learning architecture used for the training run.

```yaml
model:
    name: two_view_pipeline
    extractor:
        name: extractors.superpoint_open
        max_num_keypoints: 512
        force_num_keypoints: True
        detection_threshold: -1
        nms_radius: 3
        trainable: True
        lambda_d: 1.0
        dense_outputs: True
    ground_truth:
        name: null
    matcher:
        name: null
```
* **`name`**: `two_view_pipeline` — A wrapper model pipeline that handles keypoint extraction, matching, and ground truth calculations across a pair of views.
* **`extractor`**: Configures the keypoint extraction neural network.
  * **`name`**: `extractors.superpoint_open` — Specifies the open-source PyTorch implementation of the **SuperPoint** neural network.
  * **`max_num_keypoints`**: `512` — Retains only the top `512` highest-scoring keypoints returned by the network.
  * **`force_num_keypoints`**: `True` — Pads keypoint vectors to exactly 512 length to ensure static tensor shapes for faster GPU compilation.
  * **`detection_threshold`**: `-1` — Keypoint score detection threshold. Setting this to `-1` disables the threshold entirely, forcing the network to always return the top 512 points regardless of their absolute confidence.
  * **`nms_radius`**: `3` — Non-Maximum Suppression radius in pixels. Prevents duplicate keypoints from being detected within a `3x3` pixel neighborhood.
  * **`trainable`**: `True` — Enables gradient updates for the SuperPoint feature extractor weights during training.
  * **`lambda_d`**: `1.0` — Coefficient scaling factor for the detector loss term relative to descriptor loss.
  * **`dense_outputs`**: `True` — Forces the extractor to produce dense detector and descriptor heatmaps before point sampling.
* **`ground_truth`**: `null` — Disables ground-truth correspondence filtering since we train self-supervised on Homographies using direct waping.
* **`matcher`**: `null` — Disables matchers (like LightGlue/SuperGlue) since the objective of this configuration is to train the **Feature Extractor** (`SuperPoint`) alone.

---

## 3. `train` (Optimization and Training Loop)
Controls learning rate, scheduling, epochs, logging, and evaluation intervals.

```yaml
train:
    seed: 0
    epochs: 100
    log_every_iter: 1
    eval_every_iter: 10
    lr: 1e-4
    lr_schedule:
        start: 20
        type: exp
        on_epoch: true
        exp_div_10: 10
    plot: [1, 'gluefactory.visualization.visualize_batch.make_keypoint_figures']
```
* **`seed`**: `0` — Seed for the random number generator to ensure training reproducibility.
* **`epochs`**: `100` — The model will train for 100 complete cycles over the dataset.
* **`log_every_iter`**: `1` — Write loss and performance statistics to TensorBoard/logs after every single training iteration (batch).
* **`eval_every_iter`**: `10` — Perform a validation pass every 10 iterations to inspect performance and loss curves on the validation set.
* **`lr`**: `1e-4` — Initial training learning rate ($0.0001$) for the Adam optimizer.
* **`lr_schedule`**: Configures the dynamic adjustment of the learning rate.
  * **`start`**: `20` — Keep the learning rate constant at `1e-4` until epoch 20, then start decaying it.
  * **`type`**: `exp` — Uses exponential learning rate decay.
  * **`on_epoch`**: `true` — Decay step updates are computed at the end of each epoch rather than per-iteration.
  * **`exp_div_10`**: `10` — Controls the speed of exponential decay (decays by a factor of 10 over the course of `10` epochs).
* **`plot`**: `[1, 'gluefactory.visualization.visualize_batch.make_keypoint_figures']` — Renders a keypoint visualization plot showing predicted keypoints and waped target keypoint matches every `1` epoch, using the specified plotting function helper.
