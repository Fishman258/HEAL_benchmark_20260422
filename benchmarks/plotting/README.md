# Plotting

Plotting code is being centralized here. During refactor phase 1, this folder
contains wrappers around existing plotting scripts rather than moved code.

## Existing Plot Sources

- `opencood/tools/plot_noise_sweep.py`: plots AP curves/heatmaps from
  `AP030507_*.yaml` outputs.
- `scripts/run_opv2v_benchmark_a_profile.py`: writes Benchmark A AP/profile
  plots inside each run directory.
- `scripts/run_pubmap_opv2v_benchmark_ab.py`: writes PubMap A/B plots inside
  each run directory.

## Compatibility Wrapper

Use:

```bash
/home/qqxluca/miniconda3/envs/heal/bin/python benchmarks/plotting/plot_noise_sweep.py --help
```

This forwards to `opencood/tools/plot_noise_sweep.py`.
