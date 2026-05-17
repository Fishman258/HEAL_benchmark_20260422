# Benchmark Layout

This directory is the new, low-risk benchmark layer. It does not replace the
legacy entry points yet. Existing scripts under `scripts/` and
`opencood/tools/` remain valid.

## Directory Roles

- `configs/`: human-readable benchmark run configs.
- `launchers/`: config-driven launchers and compatibility wrappers.
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

## Compatibility Rule

During refactor phase 1, do not delete or move these existing entry points:

- `opencood/tools/inference_w_noise.py`
- `scripts/run_opv2v_benchmark_a_profile.py`
- `scripts/run_pubmap_opv2v_benchmark_ab.py`
- `scripts/run_benchmark_ab_local.py`
