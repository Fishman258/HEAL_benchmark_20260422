# HEAL_benchmark_20260422

Benchmark workspace initialized on 2026-04-22.

## Included

- `opencood/tools/inference_w_noise.py` as the benchmark entrypoint
- `opencood/` codebase required by `opencood/tools/inference_w_noise.py`
- minimal `opencood/tools/` runtime subset:
  - `__init__.py`
  - `inference_w_noise.py`
  - `inference_utils.py`
  - `train_utils.py`
- `opencood/logs/heter_modality_assign/opv2v_4modality.json`
- local checkpoint run dir:
  - `opencood/logs/HeterBaseline_DAIR_lidar_pastat_noise1_2026_01_14_19_55_25/`
- local stage1 cache:
  - `opencood/logs/freealign_repro_dair_stage1/merged_stage1_val.json`
- `dataset/` copied with symlinks preserved
- `calib/`, `v2x_calib/`, `legacy/`, `configs/`
- `requirements.txt`, `setup.py`
- backup archive for removed tool scripts:
  - `_backup/opencood_tools_before_cleanup_20260422.tar.gz`
- benchmark output root:
  - `outputs/`
- project notes and smoke-run records:
  - `docs/`

## Not Included

- large detection caches under the original top-level `data/`

This workspace now contains the specific checkpoint directory and stage1 cache needed for the local `inference_w_noise.py` smoke path, but it still does not include the broader original training-log / cache inventory.

## Git

This repository uses a dedicated local git identity:

- Name: `David Liu`
- Email: `LIUZ0112@e.ntu.edu.sg`
