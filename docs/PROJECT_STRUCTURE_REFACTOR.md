# Project Structure Refactor

This document describes the current structure after the benchmark cleanup.

## Current Important Paths

- `opencood/tools/inference_w_noise.py`: low-level benchmark executor.
- `benchmarks/launchers/run_opv2v_benchmark_a_profile.py`: OPV2V Benchmark A orchestration and plotting.
- `benchmarks/launchers/run_pubmap_opv2v_benchmark_ab.py`: PubMap/OPV2V Benchmark A/B orchestration.
- `benchmarks/launchers/run_benchmark_ab_local.py`: DAIR local Benchmark A/B orchestration.
- `benchmarks/data_prep/build_pubmap_paired_opv2v_inputs.py`: PubMap single-CAV to paired OPV2V input preparation.
- `benchmarks/data_prep/run_pubmap_paired_stage1_export.py`: paired-local stage1 cache export and shard merge.
- `opencood/registration/runtime/`: runtime pose-correction interface layer used by benchmark execution.
- `opencood/registration/estimators/`: concrete extrinsic-estimator algorithm implementations.
- `opencood/registration/utils/`: shared bbox, transform, and type helpers for registration code.
- `opencood/utils/path_utils.py`: repo-relative path resolution shared across datasets, benchmark code, and estimator code.
- `opencood/models/`: fusion and detection model code.
- `opencood/visualization/`: runtime visualization helpers.
- `benchmarks/plotting/plot_noise_sweep.py`: AP sweep plotting from `AP030507_*.yaml`.
- `opencood/tools/plot_noise_sweep.py`: compatibility wrapper for the benchmark plotting script.
- `outputs/`: timestamped experiment outputs, ignored by git.
- `opencood/logs/`: checkpoints, configs, caches, and historical run outputs, ignored by git.
- `data/` and `dataset/`: local data/cache entries, ignored by git.

## Phase-1 Additions

- `benchmarks/configs/`: readable YAML configs for benchmark runs.
- `benchmarks/data_prep/`: benchmark-specific dataset views and stage1 cache preparation.
- `benchmarks/launchers/run_benchmark_config.py`: converts config files into benchmark launcher commands.
- `benchmarks/manifests/paths.yaml`: records dataset/checkpoint/cache paths and whether they are inside or outside this project.
- `benchmarks/plotting/`: plotting wrappers and OPV2V Benchmark A plot generation.
- `benchmarks/profiling/`: OPV2V Benchmark A profile table and summary generation.

## Phase-2 Changes

- `benchmarks/launchers/run_opv2v_benchmark_a_profile.py` now focuses on job construction,
  process execution, command logging, and manifest writing.
- `benchmarks/profiling/opv2v_benchmark_a.py` owns OPV2V Benchmark A CSV tables,
  profile summary JSON, communication payload proxy summaries, and timing/memory
  summaries.
- `benchmarks/plotting/opv2v_benchmark_a.py` owns OPV2V Benchmark A AP,
  payload, runtime, and CUDA memory plots.

## Phase-3 Registration Runtime/Estimator Integration

- `opencood/registration/estimators/` is now the canonical home for old
  pose-estimation, calibration, and concrete estimator algorithm code.
- `opencood/registration/estimators/v2xregpp_runtime/` contains the V2X-Reg++
  runtime config, filters, matching engine, and SVD transform solving.
- `opencood/registration/estimators/v2xregpp_runtime/configs/` contains
  estimator-runtime YAML configs used by `--v2xregpp-config`.
- `opencood/registration/estimators/box_matching/` contains box correspondence
  utilities shared by V2X-Reg++, CBM-style matching, and related methods.
- `opencood/registration/utils/` contains reusable bbox geometry, IoU, rigid
  transform, and dataclass/type helpers used by registration code.
- `opencood/registration/runtime/pose_provider_runtime.py` contains the
  runtime bridge that applies pose correction to model batches before fusion.
- `opencood/registration/pose_estimation/` has been retired; runtime benchmark
  code should import concrete algorithms from `estimators/` and shared
  helpers from `utils/`.
- Root-level compatibility folders `calib/`, `v2x_calib/`, `configs/`,
  `legacy/`, `scripts/`, and `tools/` have been removed.

## Target Direction

Use this target structure for future cleanup:

```text
HEAL_benchmark_20260422/
  opencood/                 # library/runtime/model code only
    registration/
      runtime/              # runtime interface layer
      estimators/           # concrete extrinsic estimator algorithms
      utils/                # shared helpers
    utils/
      path_utils.py         # repo-relative path resolution
  benchmarks/
    configs/                # benchmark configs
    data_prep/              # benchmark-specific input/cache preparation
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
- Do not commit `outputs/`, `opencood/logs/`, `data/`, or `dataset/`.
- Every real benchmark output must remain in a timestamped run directory.
- Large datasets, checkpoints, and stage1 caches should be referenced by manifest or symlink, not copied into git.
