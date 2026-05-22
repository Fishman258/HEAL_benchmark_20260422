# Benchmark Data Preparation

This directory contains benchmark-specific data preparation entry points.

- `build_pubmap_paired_opv2v_inputs.py`: builds paired OPV2V-style test inputs
  from PubMap single-CAV stage1 cache and contract files.
- `run_pubmap_paired_stage1_export.py`: exports paired-local stage1 boxes in
  shards and merges them into one `stage1_boxes.json`.
