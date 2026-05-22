# Benchmark Layout

This directory is the benchmark layer. `opencood/tools/` remains the low-level
OpenCOOD execution layer.

## Directory Roles

- `configs/`: human-readable benchmark run configs.
- `data_prep/`: benchmark-specific input and stage1 cache preparation.
- `launchers/`: config-driven benchmark launchers and job orchestration.
- `manifests/`: dataset, checkpoint, and stage1 cache path records.
- `plotting/`: plotting wrappers and future centralized plot code.
- `profiling/`: profiling notes and future centralized profile code.

## Recommended Entry

Run a benchmark from a config:

```bash
python benchmarks/launchers/run_benchmark_config.py \
  benchmarks/configs/opv2v_camera_benchmark_a_profile_smoke.yaml
```

To print the command without executing:

```bash
python benchmarks/launchers/run_benchmark_config.py \
  benchmarks/configs/opv2v_camera_benchmark_a_profile_smoke.yaml \
  --print-only
```

To force the underlying launcher into dry-run mode:

```bash
python benchmarks/launchers/run_benchmark_config.py \
  benchmarks/configs/opv2v_camera_benchmark_a_profile_full2170.yaml \
  --dry-run
```

## Entry Points

- `opencood/tools/inference_w_noise.py`
- `benchmarks/data_prep/build_pubmap_paired_opv2v_inputs.py`
- `benchmarks/data_prep/run_pubmap_paired_stage1_export.py`
- `benchmarks/launchers/run_benchmark_ab_local.py`
- `benchmarks/launchers/run_opv2v_benchmark_a_profile.py`
- `benchmarks/launchers/run_pubmap_opv2v_benchmark_ab.py`
