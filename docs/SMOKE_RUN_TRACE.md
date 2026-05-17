# Smoke Run Trace: `inference_w_noise.py` with `v2x-reg++`

Date: 2026-04-22

## Command

```bash
PYTHONPATH=/home/qqxluca/HEAL_benchmark_20260422 \
/home/qqxluca/miniconda3/envs/heal/bin/python \
/home/qqxluca/HEAL_benchmark_20260422/opencood/tools/inference_w_noise.py \
  --model_dir /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/HeterBaseline_DAIR_lidar_pastat_noise1_2026_01_14_19_55_25 \
  --fusion_method intermediate \
  --save_vis_interval 1 \
  --num-workers 0 \
  --max-eval-samples 1 \
  --pos-std-list 0 \
  --rot-std-list 0 \
  --note smoke_v2xregpp \
  --pose-correction v2xregpp_initfree \
  --stage1-result /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_stage1/merged_stage1_val.json
```

## Result

- Smoke run succeeded.
- Model, dataset, pose correction, inference, evaluation, and visualization paths were all reached.
- AP:
  - IoU 0.3: 0.04
  - IoU 0.5: 0.00
  - IoU 0.7: 0.00

## Runtime Notes

- `PYTHONPATH=/home/qqxluca/HEAL_benchmark_20260422` is required when calling the script directly.
- `--save_vis_interval 0` currently crashes with `ZeroDivisionError` in `opencood/tools/inference_w_noise.py` because the code does `i % opt.save_vis_interval`.
- For this smoke run, `--save_vis_interval 1` was used as a workaround.

## Trace Artifact

- Full trace JSON: `.trace_smoke_v2xregpp.json`
- Recorded project-internal files: 142
- Recorded call edges: 531

## Core Runtime Call Chain

```text
opencood/tools/inference_w_noise.py::main
  -> opencood/hypes_yaml/yaml_utils.py::load_yaml
  -> opencood/tools/train_utils.py::create_model
      -> opencood/models/heter_model_baseline.py::__init__
          -> opencood/models/heter_encoders.py::__init__
          -> opencood/models/fuse_modules/pastat_fusion.py::__init__
          -> opencood/models/sub_modules/base_bev_backbone.py::__init__
  -> opencood/tools/train_utils.py::load_saved_model
  -> opencood/data_utils/datasets/__init__.py::build_dataset
      -> opencood/data_utils/datasets/intermediate_heter_fusion_dataset.py::getIntermediateheterFusionDataset
      -> opencood/data_utils/datasets/intermediate_heter_fusion_dataset.py::__init__
          -> opencood/data_utils/post_processor/voxel_postprocessor.py::generate_anchor_box
  -> opencood/extrinsics/pose_correction/pose_solver.py::build_pose_corrector
      -> opencood/extrinsics/pose_correction/stage1_v2xregpp.py::Stage1V2XRegPPPoseCorrector
  -> opencood/extrinsics/pose_correction/pose_solver.py::run_pose_solver
      -> opencood/data_utils/datasets/basedataset/dairv2x_basedataset.py::retrieve_base_data
      -> opencood/extrinsics/pose_correction/stage1_v2xregpp.py::apply
          -> opencood/extrinsics/pose_correction/stage1_v2xregpp.py::_extract_boxes
              -> opencood/extrinsics/bbox_utils.py::corners_to_bbox3d_list
          -> opencood/extrinsics/pose_correction/stage1_v2xregpp.py::_estimate_rel_T
              -> calib/filters/pipeline.py::apply
              -> calib/matching/engine.py::compute
                  -> legacy/v2x_calib/corresponding/BoxesMatch.py::get_matches_with_score
                  -> legacy/v2x_calib/corresponding/BoxesMatch.py::get_stability
          -> opencood/extrinsics/pose_correction/stage1_v2xregpp.py::_quality
              -> legacy/v2x_calib/search/Matches2Extrinsics.py::get_combined_extrinsic
  -> opencood/tools/train_utils.py::to_device
  -> opencood/tools/train_utils.py::maybe_apply_pose_provider
  -> opencood/tools/inference_utils.py::inference_intermediate_fusion
      -> opencood/tools/inference_utils.py::inference_early_fusion
          -> model(...)
              -> opencood/models/heter_model_baseline.py::forward
          -> opencood/data_utils/datasets/intermediate_heter_fusion_dataset.py::post_process
              -> opencood/data_utils/post_processor/voxel_postprocessor.py::post_process
              -> opencood/data_utils/post_processor/base_postprocessor.py::generate_gt_bbx_by_iou
  -> opencood/utils/eval_utils.py::caluclate_tp_fp
  -> opencood/utils/eval_utils.py::eval_final_results
  -> opencood/visualization/simple_vis.py::visualize
      -> opencood/visualization/simple_plot3d/canvas_bev.py::draw_boxes
```

## Files That Were Actually Loaded

The full list is in `.trace_smoke_v2xregpp.json`. For this successful path, the important files fall into these groups:

- Entrypoint and orchestration:
  - `opencood/tools/inference_w_noise.py`
  - `opencood/tools/inference_utils.py`
  - `opencood/tools/train_utils.py`
- Config and model setup:
  - `opencood/hypes_yaml/yaml_utils.py`
  - `opencood/models/heter_model_baseline.py`
  - `opencood/models/heter_encoders.py`
  - `opencood/models/fuse_modules/pastat_fusion.py`
  - several files under `opencood/models/sub_modules/`
- Dataset and labels:
  - `opencood/data_utils/datasets/__init__.py`
  - `opencood/data_utils/datasets/intermediate_heter_fusion_dataset.py`
  - `opencood/data_utils/datasets/basedataset/dairv2x_basedataset.py`
  - `opencood/data_utils/post_processor/base_postprocessor.py`
  - `opencood/data_utils/post_processor/voxel_postprocessor.py`
- Pose correction (`v2x-reg++` path):
  - `opencood/extrinsics/pose_correction/pose_solver.py`
  - `opencood/extrinsics/pose_correction/stage1_v2xregpp.py`
  - `opencood/extrinsics/bbox_utils.py`
  - `calib/config.py`
  - `calib/filters/pipeline.py`
  - `calib/matching/engine.py`
  - multiple `legacy/v2x_calib/*` files for box matching and extrinsic solving
- Shared utilities used heavily:
  - `opencood/utils/common_utils.py`
  - `opencood/utils/box_utils.py`
  - `opencood/utils/transformation_utils.py`
  - `opencood/utils/pose_utils.py`
  - `opencood/utils/eval_utils.py`
- Visualization:
  - `opencood/visualization/simple_vis.py`
  - `opencood/visualization/simple_plot3d/canvas_bev.py`

## Important Distinction

The trace includes both:

- files on the core runtime path above, and
- files loaded only because some package `__init__.py` or registry-style import pulled in alternative datasets, post-processors, or pose-correction methods.

For example, the trace records many files under `opencood/data_utils/datasets/`, `opencood/data_utils/post_processor/`, and `opencood/extrinsics/pose_correction/`, but the smoke run's actual main path is much narrower than the raw 142-file list.

Within `opencood/tools/`, only these files were loaded in this smoke run:

- `opencood/tools/__init__.py`
- `opencood/tools/inference_w_noise.py`
- `opencood/tools/inference_utils.py`
- `opencood/tools/train_utils.py`
