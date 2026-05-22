# Plotting

Plotting code is being centralized here.

## Existing Plot Sources

- `benchmarks/plotting/plot_noise_sweep.py`: plots AP curves/heatmaps from
  `AP030507_*.yaml` outputs.
- `benchmarks/plotting/opv2v_benchmark_a.py`: writes OPV2V Benchmark A AP and
  profiling plots inside each run directory.
- `benchmarks/launchers/run_pubmap_opv2v_benchmark_ab.py`: writes PubMap A/B
  plots inside each run directory.

## Compatibility Wrapper

Use:

```bash
/home/qqxluca/miniconda3/envs/heal/bin/python benchmarks/plotting/plot_noise_sweep.py --help
```

The old `opencood/tools/plot_noise_sweep.py` path remains as a compatibility
wrapper.
