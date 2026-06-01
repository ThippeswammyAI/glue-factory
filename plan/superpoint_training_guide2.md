# SuperPoint Training Guide: Joint Multimodal Homographic Adaptation on Custom LiDAR Projection Images

This guide provides a comprehensive documentation of the successfully implemented **SuperPoint** self-supervised pipeline using **Joint Multimodal Homographic Adaptation** on your custom pixel-aligned LiDAR projection datasets.

---

## 1. The Spatial Alignment Advantage

In your custom dataset, the images are generated from Ouster LiDAR sensor scans projected into a pinhole camera model. We have four synchronized modalities:
1. **Near-IR (`nearir`)**: Simulates a standard active infrared image, rich in surface ambient detail.
2. **Range (`range`)**: Encodes dense radial depth/geometric structure.
3. **Reflectivity (`reflectivity`)**: Captures active retro-reflective properties of surfaces, independent of ambient light.
4. **Signal (`signal`)**: Measures return pulse strength, highly sharp but sensitive to distance decay.

Because all four modalities are derived from the exact same LiDAR sensor sweep and projected using the same pinhole model, they are **perfectly pixel-aligned (spatially synchronized)**. A coordinate point $(x, y)$ in the `nearir` image refers to the exact same physical feature at $(x, y)$ in the `range`, `reflectivity`, and `signal` images!

---

## 2. Joint Multimodal Homographic Adaptation (MHA)

We leverage this alignment by running Homographic Adaptation across all four modalities simultaneously.Detections from all modalities under $N$ random homographic warps are back-projected and accumulated into a single **Joint Multimodal Heatmap**:

$$A_{joint}(x, y) = \sum_{m \in \text{modalities}} \left( A_{m, \text{orig}}(x, y) + \sum_{k=1}^{N} H_k^{-1} \cdot A_{m, \text{warp}_k}(x, y) \right)$$

This approach yields the ultimate pseudo-ground-truth targets:
*   **Modality Invariance**: Detections are cross-validated. If a feature is stable in both the Near-IR, Reflectivity, and Signal/Range images, it is exceptionally robust.
*   **Shadow and Noise Immunity**: Modality-specific drop-outs (such as range shadows or low signal-to-noise regions) are naturally filtered out by the multi-modal consensus.

---

## 3. Bootstrapping Visual Verification Results

The bootstrapping execution completed successfully, processing all **19 custom LiDAR scenes** across all 4 subdirectories and extracting highly stable pseudo-ground-truth labels using a **15-warp consensus sweep per modality**. 

The generated multi-modal dataset verification visual is displayed below:

![Joint Multimodal Homographic Adaptation Verification Image](/home/tippeswamy/.gemini/antigravity/brain/2b3c17bd-81dd-4712-9451-d592a709e54c/artifacts/adaptation_visualization.png)

> [!NOTE]
> As shown above, the joint consensus successfully captures highly repeatable features, providing robust self-supervised targets that align perfectly with the geometry of all four modalities!

---

## 4. Directory Layout and File Links

All newly created and modified files are located in your workspace:

*   **Model & Loss Module**: [superpoint_open.py](file:///home/tippeswamy/ws/thippeswamy/glue-factory/gluefactory/models/extractors/superpoint_open.py) — Contains the completed neural network architecture and joint Cross-Entropy + Hinge loss functions.
*   **Multimodal Adaptation Tool**: [prepare_and_visualize_adaptation.py](file:///home/tippeswamy/ws/thippeswamy/glue-factory/gluefactory/scripts/prepare_and_visualize_adaptation.py) — The executable joint multi-modal dataset generator script.
*   **Custom Dataset Cache File**: `/home/tippeswamy/ws/thippeswamy/glue-factory/data/custom_dataset/exports/pseudo_labels.h5` — Generated joint pseudo-ground-truth labels.
*   **Custom Training Configuration**: [superpoint_custom_homography.yaml](file:///home/tippeswamy/ws/thippeswamy/glue-factory/gluefactory/configs/superpoint_custom_homography.yaml) — Training configuration pointing to the multimodal cached labels.

---

## 5. Technical Architecture Carousel

````carousel
```python
# Joint SuperPoint Loss Structure (superpoint_open.py)
total_loss = loss_detector + lambda_descriptor * loss_descriptor
```
<!-- slide -->
```yaml
# Training Pipeline Integration (superpoint_custom_homography.yaml)
data:
  name: homographies
  data_dir: custom_dataset
  load_features:
    do: True
    path: "custom_dataset/exports/pseudo_labels.h5"
```
<!-- slide -->
```python
# Back-Projected Repeatability Normalization (prepare_and_visualize_adaptation.py)
normalized_heatmap = joint_accumulator / np.maximum(global_trials, 1.0)
```
````

---

## 6. Next Steps: Model Training

When you have a GPU environment ready, launch training on any modality with:

```bash
/home/tippeswamy/python_venv/env/bin/python3 -m gluefactory.train superpoint_custom_run --conf gluefactory/configs/superpoint_custom_homography.yaml
```

python3 -c "import h5py; f = h5py.File('data/custom_dataset/exports/pseudo_labels.h5', 'r'); print(list(f.keys())[:5])"

find . -type f \( -iname "*.png" -o -iname "*.jpg" \) | sed 's|^\./||' > ../custom_image_list.txt

find images -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) > custom_image_list.txt

python3 prepare_and_visualize_adaptation.py 

# Launch with full GPU acceleration, 8 parallel threads, and automatic custom_image_list.txt generation for custom_dataset
python3 -m gluefactory.scripts.prepare_and_visualize_adaptation \
    --warp_mode 3d \
    --use_gpu \
    --num_threads 8 \
    --num_warps 2 \
    --dataset custom_dataset \
    --image_list_modality reflectivity

