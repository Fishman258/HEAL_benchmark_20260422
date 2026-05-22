#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path("/home/qqxluca/miniconda3/envs/heal/bin/python")
DEFAULT_BASE_HYPES = Path("/data2/pubmap_full_training/configs/full_heal_pointpillar_framelevel_stage1.yaml")
DEFAULT_CHECKPOINT = Path("/data2/pubmap_full_training/logs/pubmap_full_heal_pointpillar_2026_05_08_16_08_40/net_epoch_bestval_at51.pth")
DEFAULT_PAIRED_DATASET = Path(
    "/data2/pubmap_full_training/paired_benchmark_inputs/latest_pubmap_paired_opv2v/datasets/heal_pointpillar_opv2v_paired/test"
)
DEFAULT_OUTPUT_ROOT = Path("/data2/pubmap_full_training/stage1_cache")


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _build_hypes(base_hypes: Path, paired_dataset: Path, out_path: Path) -> None:
    with base_hypes.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["root_dir"] = str(paired_dataset)
    cfg["validate_dir"] = str(paired_dataset)
    cfg["test_dir"] = str(paired_dataset)
    cfg.setdefault("train_params", {})["max_cav"] = 2
    _write_yaml(out_path, cfg)


def _merge_shards(stage1_dir: Path, num_shards: int, expected_samples: int) -> Path:
    merged: Dict[str, Any] = {}
    for shard_idx in range(int(num_shards)):
        shard_path = stage1_dir / f"stage1_boxes_shard{shard_idx:02d}of{int(num_shards):02d}.json"
        if not shard_path.exists():
            raise FileNotFoundError(f"missing shard output: {shard_path}")
        with shard_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"shard output is not a dict: {shard_path}")
        overlap = sorted(set(merged) & set(payload))
        if overlap:
            raise ValueError(f"duplicate sample keys in shard {shard_idx}: {overlap[:5]}")
        merged.update(payload)

    if expected_samples > 0 and len(merged) != int(expected_samples):
        raise ValueError(f"merged sample count mismatch: expected={expected_samples} actual={len(merged)}")
    int_keys = sorted(int(k) for k in merged.keys())
    if int_keys != list(range(len(int_keys))):
        raise ValueError("merged stage1 keys are not contiguous from 0")

    out_path = stage1_dir / "stage1_boxes.json"
    ordered = {str(k): merged[str(k)] for k in int_keys}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, sort_keys=True)
        f.write("\n")
    return out_path


def _launch_shards(opt: argparse.Namespace, run_dir: Path, hypes_yaml: Path) -> None:
    gpus = _safe_list(opt.gpus)
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    num_shards = int(opt.num_shards or len(gpus))
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for shard_idx in range(num_shards):
        gpu = gpus[shard_idx % len(gpus)]
        log_path = logs_dir / f"export_shard{shard_idx:02d}of{num_shards:02d}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = str(ROOT)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        cmd = [
            str(opt.python_bin),
            str(ROOT / "opencood" / "tools" / "export_stage1_boxes_per_cav.py"),
            "--hypes_yaml",
            str(hypes_yaml),
            "--checkpoint_path",
            str(opt.checkpoint_path),
            "--output_dir",
            str(run_dir),
            "--split",
            "test",
            "--num_shards",
            str(num_shards),
            "--shard_index",
            str(shard_idx),
            "--nms_pre_topk",
            str(int(opt.nms_pre_topk)),
            "--log_interval",
            str(int(opt.log_interval)),
        ]
        if int(opt.max_samples or 0) > 0:
            cmd.extend(["--max_samples", str(int(opt.max_samples))])
        with log_path.open("w", encoding="utf-8") as log_f:
            log_f.write(" ".join(cmd) + "\n")
        log_f = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((proc, log_f, log_path, shard_idx, gpu))

    failed = []
    try:
        for proc, log_f, log_path, shard_idx, gpu in procs:
            ret = proc.wait()
            log_f.close()
            if ret != 0:
                failed.append((shard_idx, gpu, ret, str(log_path)))
    finally:
        for proc, log_f, _, _, _ in procs:
            if proc.poll() is None:
                proc.terminate()
            try:
                log_f.close()
            except Exception:
                pass
    if failed:
        raise RuntimeError(f"stage1 export shard failures: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel export paired-local PubMap/OPV2V stage1 cache.")
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--base-hypes", type=Path, default=DEFAULT_BASE_HYPES)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--paired-dataset", type=Path, default=DEFAULT_PAIRED_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--expected-samples", type=int, default=5985)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--nms-pre-topk", type=int, default=300)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    opt = parse_args()
    run_name = str(opt.run_name or f"pubmap_pointpillar_bestval51_paired_local_{_timestamp()}").strip()
    run_dir = opt.output_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    hypes_yaml = run_dir / "config.yaml"
    _build_hypes(opt.base_hypes, opt.paired_dataset, hypes_yaml)

    manifest = {
        "schema_version": "pubmap_paired_stage1_export_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workspace_root": str(ROOT),
        "base_hypes": str(opt.base_hypes),
        "hypes_yaml": str(hypes_yaml),
        "checkpoint_path": str(opt.checkpoint_path),
        "paired_dataset": str(opt.paired_dataset),
        "gpus": _safe_list(opt.gpus),
        "num_shards": int(opt.num_shards),
        "expected_samples": int(opt.expected_samples),
        "max_samples": int(opt.max_samples or 0),
        "nms_pre_topk": int(opt.nms_pre_topk),
    }
    _write_json(run_dir / "manifest.json", manifest)

    _launch_shards(opt, run_dir, hypes_yaml)
    expected = int(opt.max_samples) if int(opt.max_samples or 0) > 0 else int(opt.expected_samples)
    stage1_path = _merge_shards(run_dir / "test", int(opt.num_shards), expected)
    manifest["stage1_output"] = str(stage1_path)
    manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(run_dir / "manifest.json", manifest)
    print(stage1_path)


if __name__ == "__main__":
    main()
