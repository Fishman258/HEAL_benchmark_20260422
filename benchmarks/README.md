# Benchmark Layout

This directory is the benchmark layer. `opencood/tools/` remains the low-level
OpenCOOD execution layer.

## Directory Roles

- `configs/`: human-readable benchmark run configs.
- `data_prep/`: benchmark-specific input and stage1 cache preparation.
- `launchers/`: legacy launchers kept for old command reproduction only.
- `manifests/`: dataset, checkpoint, and stage1 cache path records.
- `pipelines/`: staged benchmark orchestration (`detection -> benchmark -> registration -> fusion -> evaluation`).
- `plotting/`: plotting wrappers and future centralized plot code.
- `profiling/`: profiling notes and future centralized profile code.
- `schemas/`: JSON contracts for stage records and major stage outputs.
- `validators/`: stdlib-only checks for stage1 cache, pose override, and stage records.

## Recommended Entry

Run every benchmark through the staged pipeline:

```bash
/home/qqxluca/miniconda3/envs/heal/bin/python \
  benchmarks/pipelines/run_pipeline.py \
  benchmarks/configs/opv2v_camera_pipeline_smoke.yaml
```

Dry-run the same config before a real run:

```bash
/home/qqxluca/miniconda3/envs/heal/bin/python \
  benchmarks/pipelines/run_pipeline.py \
  benchmarks/configs/opv2v_camera_pipeline_smoke.yaml \
  --dry-run --print-run-dir
```

Full OPV2V camera-depth Benchmark A:

```bash
/home/qqxluca/miniconda3/envs/heal/bin/python \
  benchmarks/pipelines/run_pipeline.py \
  benchmarks/configs/opv2v_camera_benchmark_a_profile_full2170.yaml
```

PubMap paired OPV2V lidar Benchmark A/B:

```bash
/home/qqxluca/miniconda3/envs/heal/bin/python \
  benchmarks/pipelines/run_pipeline.py \
  benchmarks/configs/pubmap_opv2v_lidar_benchmark_ab.yaml
```

The staged pipeline records separate stage directories and delegates low-level
model execution to `opencood/tools/inference_w_noise.py`. Benchmark-level job
graphs, summaries, and plots live under `benchmarks/pipelines/`.

Each staged run writes an immutable timestamped directory under
`outputs/pipelines/<name>/run_<timestamp>_<note>/`. Legacy executor output
directories printed by `inference_w_noise.py` are captured in `manifest.json`
and in each stage `result/summary.json`.

Validate common artifacts:

```bash
/home/qqxluca/miniconda3/envs/heal/bin/python \
  -m benchmarks.validators.validate_stage1_cache \
  /path/to/stage1_boxes.json --require-contiguous-keys

/home/qqxluca/miniconda3/envs/heal/bin/python \
  -m benchmarks.validators.validate_pose_override \
  /path/to/pose_override.json --require-contiguous-keys
```

## Entry Points

- `opencood/tools/inference_w_noise.py`
- `benchmarks/pipelines/run_pipeline.py`
- `benchmarks/data_prep/build_pubmap_paired_opv2v_inputs.py`
- `benchmarks/data_prep/run_pubmap_paired_stage1_export.py`

Legacy launcher entry points remain in `benchmarks/launchers/` only to reproduce
old runs. New benchmark configs should not use `launcher:`.
