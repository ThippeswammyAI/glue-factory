# Walkthrough - Custom SuperPoint Model Evaluation

We successfully evaluated the custom SuperPoint checkpoint (`outputs/training/superpoint_custom_run/checkpoint_best.tar`) against the official pretrained SuperPoint model.

## 1. Standard Dataset (HPatches) Evaluation

We ran HPatches evaluation with the nearest neighbor matcher using both your custom checkpoint and the official pretrained checkpoint. We also ran an evaluation using your custom SuperPoint coupled with the official SuperGlue matcher (with increased NMS=4 and detection threshold=0.005).

### Metric Comparison on HPatches

| Metric | Official SuperPoint + NN | Custom SuperPoint + NN | Custom SuperPoint + SuperGlue (NMS=4, th=0.005) |
| :--- | :--- | :--- | :--- |
| **H_error_ransac@1px** | 0.3181 | **0.3325** | 0.2686 |
| **H_error_ransac@3px** | 0.5088 | **0.5117** | 0.3887 |
| **H_error_ransac@5px** | **0.6222** | 0.6126 | 0.4649 |
| **H_error_ransac_mAA** | 0.4830 | **0.4856** | 0.3741 |
| **mprec@1px** | 0.267 | 0.305 | **0.320** |
| **mprec@3px** | **0.749** | 0.738 | 0.743 |
| **mnum_matches** | **576.5** | 458.0 | 163.0 |

> [!NOTE]
> Your custom SuperPoint model slightly outperforms the official pretrained Magic Leap SuperPoint model on standard RGB HPatches benchmark across several metrics, including **RANSAC@1px**, **RANSAC@3px**, and **precision@1px**.

> [!IMPORTANT]
> **SuperGlue Mismatch Analysis**: 
> When using your custom SuperPoint model with the **official SuperGlue**, RANSAC accuracy drops significantly (mAA from `0.4856` to `0.3741`), and the average number of matches drops to `163.0`.
> 
> This is a classic feature/matcher mismatch: the official SuperGlue weights are trained to match descriptors from the *official* SuperPoint descriptor space. Because your custom-trained SuperPoint model has a different descriptor space, SuperGlue fails to associate matches correctly. To use SuperGlue/LightGlue effectively with your custom model, you must train/fine-tune the matcher (SuperGlue/LightGlue) on your custom descriptor space.

---

## 2. Custom Dataset Evaluation

We evaluated your custom model on the custom validation split (`custom_dataset1` used for training) using the newly implemented [homographies.py](file:///home/thippe/workspaces/AiMl/glue-factory/gluefactory/eval/homographies.py) pipeline.

### Metrics on Custom Dataset
```json
{
  "H_error_dlt@1px": 0.0,
  "H_error_dlt@3px": 0.0,
  "H_error_dlt@5px": 0.0,
  "H_error_ransac@1px": 0.0,
  "H_error_ransac@3px": 0.0454,
  "H_error_ransac@5px": 0.117,
  "H_error_ransac_mAA": 0.0541,
  "mH_error_dlt": 399.947,
  "mH_error_ransac": 13.386,
  "mnum_keypoints": 2048.0,
  "mnum_matches": 501.5,
  "mprec@1px": 0.058,
  "mprec@3px": 0.29,
  "mransac_inl": 38.0,
  "mransac_inl%": 0.072
}
```
