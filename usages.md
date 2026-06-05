1. Command to Train SuperGlue
To start training SuperGlue using your custom dataset configuration:

bash
python3 -m gluefactory.train superpoint_custom+superglue_homography_custom_dataset \
  --conf gluefactory/configs/superpoint_custom+superglue_homography_custom_dataset.yaml
Multi-GPU (Distributed) Training: If you have multiple GPUs on your system, you can train in distributed mode by appending the --distributed flag:
bash
python3 -m gluefactory.train superpoint_custom+superglue_homography_custom_dataset \
  --conf gluefactory/configs/superpoint_custom+superglue_homography_custom_dataset.yaml \
  --distributed
2. Command to Run TensorBoard
All training logs and summaries are saved in the outputs/training/ directory. Run the following command from the project root to launch TensorBoard:

bash
tensorboard --logdir outputs/training/