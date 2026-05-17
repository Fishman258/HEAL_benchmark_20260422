# Profiling

Runtime, communication, and memory summaries are currently produced by the
benchmark launcher itself:

- `scripts/run_opv2v_benchmark_a_profile.py`

The next refactor step should move profile table and plot generation from that
launcher into this directory, after compatibility smoke tests pass.

