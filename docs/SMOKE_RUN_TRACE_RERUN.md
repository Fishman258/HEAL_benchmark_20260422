# Smoke Run Trace Rerun: `inference_w_noise.py` with `v2x-reg++`

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
  --note smoke_v2xregpp_rerun \
  --pose-correction v2xregpp_initfree \
  --stage1-result /home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/freealign_repro_dair_stage1/merged_stage1_val.json \
  --log-interval 1
```

## Result

- Exit code: `0`
- Elapsed: `61.013s`
- Pose correction: `v2xregpp_initfree`
- Noise: `pos=0`, `rot=0`
- Evaluated samples: `1`
- AP@0.3: `0.04`
- AP@0.5: `0.00`
- AP@0.7: `0.00`

## Artifacts

- Trace JSON: `.trace_smoke_v2xregpp_rerun.json`
- Trace meta: `.trace_smoke_v2xregpp_rerun.meta.json`
- AP YAML: `/home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/HeterBaseline_DAIR_lidar_pastat_noise1_2026_01_14_19_55_25/AP030507_v2xregpp_initfreesmoke_v2xregpp_rerun.yaml`
- Eval YAML: `/home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/HeterBaseline_DAIR_lidar_pastat_noise1_2026_01_14_19_55_25/eval_0.0_0.0_0.0_0.0_intermediatesmoke_v2xregpp_rerun_v2xregpp_initfree.yaml`
- Visualization dir: `/home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/HeterBaseline_DAIR_lidar_pastat_noise1_2026_01_14_19_55_25/vis_0.0_0.0_0.0_0.0_intermediatesmoke_v2xregpp_rerun_v2xregpp_initfree`

## Trace Counts

- Total traced file identifiers: `158`
- Real project-internal Python files executed: `142`
- Recorded Python call edges: `1408`

The `158` total includes runtime pseudo-files such as importlib internals. The complete `142`-file project-internal list is stored in `.trace_smoke_v2xregpp_rerun.json` under `files`.

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
  -> opencood/registration/runtime/pose_solver.py::build_pose_corrector
      -> opencood/registration/runtime/stage1_v2xregpp.py::Stage1V2XRegPPPoseCorrector
  -> opencood/registration/runtime/pose_solver.py::run_pose_solver
      -> opencood/data_utils/datasets/basedataset/dairv2x_basedataset.py::retrieve_base_data
      -> opencood/utils/pose_utils.py::add_noise_data_dict
      -> opencood/utils/pose_utils.py::attach_pose_confidence
      -> opencood/registration/runtime/stage1_v2xregpp.py::apply
          -> opencood/registration/runtime/stage1_v2xregpp.py::_extract_boxes
              -> opencood/registration/utils/bbox.py::corners_to_bbox3d_list
          -> opencood/registration/runtime/stage1_v2xregpp.py::_estimate_rel_T
              -> opencood/registration/estimators/v2xregpp_runtime/filters/pipeline.py::apply
              -> opencood/registration/estimators/v2xregpp_runtime/matching/engine.py::compute
                  -> opencood/registration/estimators/box_matching/boxes_match.py::get_matches_with_score
                  -> opencood/registration/estimators/box_matching/boxes_match.py::get_stability
          -> opencood/registration/runtime/stage1_v2xregpp.py::_quality
              -> opencood/registration/estimators/box_matching/corresponding_detector.py::get_distance_corresponding_precision
              -> opencood/registration/estimators/v2xregpp_runtime/matches_to_extrinsics.py::get_combined_extrinsic
  -> opencood/tools/train_utils.py::to_device
  -> opencood/tools/train_utils.py::maybe_apply_pose_provider
      -> opencood/registration/runtime/pose_provider_runtime.py::from_hypes
  -> opencood/tools/inference_utils.py::inference_intermediate_fusion
      -> opencood/tools/inference_utils.py::inference_early_fusion
          -> model(...)
              -> opencood/models/heter_model_baseline.py::forward
                  -> opencood/utils/transformation_utils.py::normalize_pairwise_tfm
          -> opencood/data_utils/datasets/intermediate_heter_fusion_dataset.py::post_process
              -> opencood/data_utils/post_processor/voxel_postprocessor.py::post_process
              -> opencood/data_utils/post_processor/base_postprocessor.py::generate_gt_bbx_by_iou
  -> opencood/utils/eval_utils.py::eval_final_results
      -> opencood/utils/eval_utils.py::calculate_ap
      -> opencood/hypes_yaml/yaml_utils.py::save_yaml
  -> opencood/visualization/simple_vis.py::visualize
      -> opencood/visualization/simple_plot3d/canvas_bev.py::draw_boxes
```

## Notes

- This rerun matches the earlier smoke result: the full path from model load to dataset, `v2x-reg++` pose correction, inference, evaluation, and visualization is reachable.
- Within `opencood/tools/`, only these files were actually executed in this rerun:
  - `opencood/tools/__init__.py`
  - `opencood/tools/inference_utils.py`
  - `opencood/tools/inference_w_noise.py`
  - `opencood/tools/train_utils.py`
