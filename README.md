# HEAL_benchmark_20260422

Benchmark workspace for staged cooperative perception experiments.

The current project is organized around a three-step benchmark view:

1. detection produces or records local 3D boxes as `stage1` cache.
2. registration estimates corrected relative poses from detection outputs.
3. fusion runs cooperative perception with the selected pose source and reports AP / runtime / payload proxy metrics.

## Main Layout

- `benchmarks/`: benchmark configs, staged pipeline orchestration, plotting, profiling, schemas, validation, and data-preparation scripts.
- `opencood/detection/`: single-agent detection models, pre/post-processing, and stage1 export scripts.
- `opencood/registration/`: matching, pose estimation, pose override generation, and runtime pose-provider code.
- `opencood/fusion/`: cooperative perception datasets, models, fusion modules, and inference helpers.
- `opencood/tools/inference_w_noise.py`: low-level benchmark executor still used by the staged pipeline.
- `outputs/`: local benchmark outputs. This directory is intentionally ignored by git.

Legacy root-level compatibility folders such as `scripts/`, `tools/`, `calib/`,
`v2x_calib/`, `configs/`, and `legacy/` have been removed. Use the canonical
paths under `benchmarks/` and `opencood/`.

## Benchmark Entry

Use the staged pipeline entry point:

```bash
/home/qqxluca/miniconda3/envs/heal/bin/python \
  benchmarks/pipelines/run_pipeline.py \
  benchmarks/configs/opv2v_camera_pipeline_smoke.yaml
```

Dry-run without launching inference jobs:

```bash
/home/qqxluca/miniconda3/envs/heal/bin/python \
  benchmarks/pipelines/run_pipeline.py \
  benchmarks/configs/opv2v_camera_pipeline_smoke.yaml \
  --dry-run --print-run-dir
```

Current benchmark configs:

- `benchmarks/configs/opv2v_camera_pipeline_smoke.yaml`
- `benchmarks/configs/opv2v_camera_benchmark_a_profile_smoke.yaml`
- `benchmarks/configs/opv2v_camera_benchmark_a_profile_full2170.yaml`
- `benchmarks/configs/pubmap_opv2v_lidar_benchmark_ab.yaml`

For full Benchmark A/AB sweeps, the top-level `registration` and `fusion`
stages are usually disabled in YAML because the `benchmark` stage internally
creates registration solver jobs and downstream fusion jobs for every method
and noise point.

## Checkpoints

Current benchmark configs no longer depend on external checkpoint directories.
Local checkpoint copies are stored here:

- `opencood/detection/checkpoints/`
- `opencood/fusion/checkpoints/`

Current local checkpoint directories:

- `opv2v_camera_v2xvit_full_prope`
- `pubmap_full_heal_pointpillar_2026_05_08_16_08_40`

Each directory contains the small `config.yaml` plus the required local
`net_epoch*.pth` file. The `.pth` files are intentionally ignored by git, but
they exist on this machine so current benchmark commands can run without
reading checkpoint weights from `/home/qqxluca/projects/...` or `/data2/...`.

The source paths and roles are recorded in:

- `benchmarks/manifests/paths.yaml`

## Data And Cache Inputs

Large datasets are still external or symlinked. Current important inputs are:

- OPV2V test split: `/data2/OPV2V/test`
- OPV2V depth maps: `/data2/opv2v_depth/test`
- PubMap paired OPV2V inputs: `/data2/pubmap_full_training/paired_benchmark_inputs/latest_pubmap_paired_opv2v/datasets/heal_pointpillar_opv2v_paired/test`
- OPV2V camera-depth stage1 cache: `outputs/image_depth_stage1_cache/run_20260513_191711_opv2v_image_depth_camera_model_fullprope_full2170/test/stage1_boxes_image_depth_camera_model.json`
- PubMap lidar stage1 cache: `/data2/pubmap_full_training/stage1_cache/pubmap_pointpillar_bestval51_paired_local_20260509_041950/test/stage1_boxes.json`

Large outputs, datasets, logs, and model weights are ignored by git.

## Outputs

Pipeline runs write timestamped folders under:

```text
outputs/pipelines/<benchmark_name>/run_<timestamp>_<note>/
```

Typical files include:

- `config_resolved.yaml`
- `manifest.json`
- `commands.sh`
- `benchmark_summary.json`
- `benchmarkA_points.csv`
- `profile_rows.csv`
- `profile_summary.json`
- generated plots and AP YAML files

## Git

This repository uses a dedicated local git identity:

- Name: `David Liu`
- Email: `LIUZ0112@e.ntu.edu.sg`

