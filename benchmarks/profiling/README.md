# Profiling

Runtime, communication, and memory summaries for OPV2V Benchmark A are now
centralized in:

- `benchmarks/profiling/opv2v_benchmark_a.py`

The pipeline stage `benchmarks/pipelines/benchmark_stage.py` owns job
construction and process execution, then calls this module to write:

- `profile_rows.csv`
- `benchmarkA_points.csv`
- `profile_overhead_vs_baseline.csv`
- `profile_summary_full2170.json`
- `profile_summary.json`
