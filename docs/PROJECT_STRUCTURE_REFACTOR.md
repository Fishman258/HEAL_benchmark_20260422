# Project Structure Refactor

This document describes the current structure and the phase-1 target layout.
Phase 1 is compatibility-first: old paths remain valid.

## Current Important Paths

- `opencood/tools/inference_w_noise.py`: low-level benchmark executor.
- `scripts/run_opv2v_benchmark_a_profile.py`: OPV2V Benchmark A orchestration and plotting.
- `scripts/run_pubmap_opv2v_benchmark_ab.py`: PubMap/OPV2V Benchmark A/B orchestration.
- `scripts/run_benchmark_ab_local.py`: DAIR local Benchmark A/B orchestration.
- `opencood/extrinsics/pose_correction/`: runtime pose correction methods.
- `opencood/extrinsics/late_fusion/`: estimator implementations and legacy late-fusion adapters.
- `opencood/models/`: fusion and detection model code.
- `opencood/visualization/`: runtime visualization helpers.
- `opencood/tools/plot_noise_sweep.py`: legacy AP sweep plotting script.
- `outputs/`: timestamped experiment outputs, ignored by git.
- `opencood/logs/`: checkpoints, configs, caches, and historical run outputs, ignored by git.
- `data/` and `dataset/`: local data/cache entries, ignored by git.

## Phase-1 Additions

- `benchmarks/configs/`: readable YAML configs for benchmark runs.
- `benchmarks/launchers/run_benchmark_config.py`: converts config files into existing launcher commands.
- `benchmarks/manifests/paths.yaml`: records dataset/checkpoint/cache paths and whether they are inside or outside this project.
- `benchmarks/plotting/`: compatibility wrappers for plotting scripts.
- `benchmarks/profiling/`: profiling notes and future home for profile table/plot generation.

## Target Direction

Use this target structure for future cleanup:

```text
HEAL_benchmark_20260422/
  opencood/                 # library/runtime/model code only
  benchmarks/
    configs/                # benchmark configs
    launchers/              # benchmark launchers
    plotting/               # result plotting
    profiling/              # runtime/communication summaries
    manifests/              # path and artifact manifests
  checkpoints/              # optional small symlinks or manifests, no large binary files
  stage1_cache/             # optional small symlinks or manifests, no large generated caches
  datasets/                 # symlinks/manifests only
  outputs/                  # timestamped run results, ignored by git
  docs/                     # run notes and structure documentation
```

## Rules During Refactor

- Do not move `inference_w_noise.py` until all launchers use the new config layer.
- Do not delete `scripts/*.py` until compatibility wrappers and smoke tests pass.
- Do not commit `outputs/`, `opencood/logs/`, `data/`, or `dataset/`.
- Every real benchmark output must remain in a timestamped run directory.
- Large datasets, checkpoints, and stage1 caches should be referenced by manifest or symlink, not copied into git.
