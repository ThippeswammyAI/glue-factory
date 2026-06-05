ippe@thippe:~/workspaces/AiMl/glue-factory$ python3 -m gluefactory.eval.hpatches \
  --checkpoint outputs/training/superpoint_custom_run/checkpoint_best.tar \
  --conf gluefactory/configs/superpoint+lsd+gluestick.yaml \
  --overwrite
Running benchmark: hpatches
Experiment tag: superpoint+lsd+gluestick
Config:
{'checkpoint': 'outputs/training/superpoint_custom_run/checkpoint_best.tar',
 'data': {'batch_size': 1,
          'name': 'hpatches',
          'num_workers': 16,
          'preprocessing': {'resize': 480, 'side': 'short'}},
 'eval': {'estimator': 'opencv', 'ransac_th': -1},
 'model': {'extractor': {'line_extractor': {'force_num_lines': False,
                                            'max_num_lines': 512,
                                            'min_length': 15,
                                            'name': 'gluefactory.models.lines.lsd',
                                            'trainable': False},
                         'name': 'gluefactory.models.lines.wireframe',
                         'point_extractor': {'dense_outputs': True,
                                             'detection_threshold': 0.005,
                                             'force_num_keypoints': False,
                                             'max_num_keypoints': 2048,
                                             'name': 'extractors.superpoint_open',
                                             'nms_radius': 4,
                                             'trainable': False},
                         'wireframe_params': {'merge_line_endpoints': True,
                                              'merge_points': True,
                                              'nms_radius': 3}},
           'ground_truth': {'name': None},
           'matcher': {'name': 'gluefactory.models.matchers.gluestick',
                       'weights': 'checkpoint_GlueStick_MD'},
           'name': 'gluefactory.models.two_view_pipeline'}}
[06/05/2026 16:57:28 gluefactory.eval.eval_pipeline INFO] Running eval pipeline HPatchesPipeline.
[06/05/2026 16:57:28 gluefactory.eval.eval_pipeline INFO] Loop 1: Exporting predictions to "/home/thippe/workspaces/AiMl/glue-factory/outputs/results/hpatches/superpoint+lsd+gluestick".
[06/05/2026 16:57:28 gluefactory.utils.experiments INFO] Loading checkpoint checkpoint_best.tar
[06/05/2026 16:57:28 gluefactory.utils.experiments WARNING] Missing 396 parameters in 
[06/05/2026 16:57:28 gluefactory.datasets.base_dataset INFO] Creating dataset HPatches
100%|█████████████████████████████████████████| 540/540 [03:58<00:00,  2.27it/s]
[06/05/2026 17:01:29 gluefactory.eval.eval_pipeline INFO] Loop 1 finished. Predictions saved to /home/thippe/workspaces/AiMl/glue-factory/outputs/results/hpatches/superpoint+lsd+gluestick/predictions.h5.
[06/05/2026 17:01:29 gluefactory.eval.eval_pipeline INFO] Loop 2: Evaluating predictions in /home/thippe/workspaces/AiMl/glue-factory/outputs/results/hpatches/superpoint+lsd+gluestick/predictions.h5.
[06/05/2026 17:01:29 gluefactory.datasets.base_dataset INFO] Creating dataset HPatches
100%|█████████████████████████████████████████| 540/540 [01:05<00:00,  8.20it/s]
Tested ransac setup with following results:
AUC {0.5: [0.2227, 0.336, 0.416], 1.0: [0.1384, 0.3187, 0.4137], 1.5: [0.1151, 0.3031, 0.4104], 2.0: [0.1111, 0.3019, 0.4106], 2.5: [0.1071, 0.3044, 0.414], 3.0: [0.1078, 0.3034, 0.4128]}
mAA {0.5: 0.32489999999999997, 1.0: 0.2902666666666667, 1.5: 0.2762, 2.0: 0.27453333333333335, 2.5: 0.2751666666666666, 3.0: 0.27466666666666667}
best threshold = 0.5
[06/05/2026 17:02:35 gluefactory.eval.eval_pipeline INFO] Loop 2 finished. Results saved to /home/thippe/workspaces/AiMl/glue-factory/outputs/results/hpatches/superpoint+lsd+gluestick.
{'H_error_dlt@1px': 0.0191,
 'H_error_dlt@3px': 0.1497,
 'H_error_dlt@5px': 0.2531,
 'H_error_ransac@1px': 0.2227,
 'H_error_ransac@3px': 0.336,
 'H_error_ransac@5px': 0.416,
 'H_error_ransac_mAA': 0.32489999999999997,
 'mH_error_dlt': nan,
 'mH_error_ransac': 3.376,
 'mnum_keypoints': 1824.25,
 'mnum_matches': 242.0,
 'mprec@1px': 0.263,
 'mprec@3px': 0.62,
 'mransac_inl': 31.5,
 'mransac_inl%': 0.165}
thippe@thippe:~/workspaces/AiMl/glue-factory$ 
