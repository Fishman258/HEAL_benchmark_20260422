#!/usr/bin/env python3
import argparse
import copy
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SCRIPT = ROOT / "opencood" / "tools" / "inference_w_noise.py"
OUTPUT_ROOT = ROOT / "outputs" / "pubmap_opv2v_benchmark_ab"

DEFAULT_PYTHON = Path("/home/qqxluca/miniconda3/envs/heal/bin/python")
DEFAULT_MODEL_DIR = Path("/data2/pubmap_full_training/logs/pubmap_full_heal_pointpillar_2026_05_08_16_08_40")
DEFAULT_STAGE1 = Path(
    "/data2/pubmap_full_training/stage1_cache/pubmap_pointpillar_bestval51_paired_local_20260509_041950/test/stage1_boxes.json"
)
DEFAULT_TEST_DIR = Path(
    "/data2/pubmap_full_training/paired_benchmark_inputs/latest_pubmap_paired_opv2v/datasets/heal_pointpillar_opv2v_paired/test"
)
DEFAULT_NOISES = [float(i) for i in range(11)]


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_note(text: str) -> str:
    text = str(text or "").strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _format_noise_list(values: Sequence[float]) -> str:
    out = []
    for value in values:
        value = float(value)
        if abs(value - round(value)) < 1e-9:
            out.append(str(int(round(value))))
        else:
            out.append(str(value))
    return ",".join(out)


def _parse_float_csv(raw: str) -> List[float]:
    out = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    return out


def _parse_gpus(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(payload), f, sort_keys=False, allow_unicode=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _build_run_dir(note: str) -> Path:
    stem = f"run_{_timestamp()}"
    note = _safe_note(note)
    if note:
        stem += f"_{note}"
    out = OUTPUT_ROOT / stem
    out.mkdir(parents=True, exist_ok=False)
    return out


def _build_common_cmd(opt: argparse.Namespace, series_id: str, noises: Sequence[float]) -> List[str]:
    note = f"{opt.note}_{series_id}" if opt.note else series_id
    cmd = [
        str(opt.python_bin),
        str(INFERENCE_SCRIPT),
        "--model_dir",
        str(opt.model_dir),
        "--fusion_method",
        "late",
        "--test-dir-override",
        str(opt.test_dir),
        "--max-cav-override",
        "2",
        "--save_vis_interval",
        str(int(opt.save_vis_interval)),
        "--num-workers",
        str(int(opt.num_workers)),
        "--pos-std-list",
        _format_noise_list(noises),
        "--rot-std-list",
        _format_noise_list(noises),
        "--sweep-mode",
        "paired",
        "--noise-target",
        "non-ego",
        "--comm-range-override",
        str(float(opt.comm_range)),
        "--comm-range-gating",
        str(opt.comm_range_gating),
        "--runtime-mode",
        "register_and_fuse",
        "--solver-backend",
        "online_box",
        "--pose-source",
        "noisy_input",
        "--pose-device",
        "cuda",
        "--log-interval",
        str(int(opt.log_interval)),
        "--note",
        note,
    ]
    if int(opt.max_eval_samples or 0) > 0:
        cmd.extend(["--max-eval-samples", str(int(opt.max_eval_samples))])
    if int(opt.eval_sample_start or 0) > 0:
        cmd.extend(["--eval-sample-start", str(int(opt.eval_sample_start))])
    return cmd


def _build_jobs(opt: argparse.Namespace) -> List[Dict[str, Any]]:
    noises = list(opt.noises)
    single_noises = [0.0]
    jobs = []
    if opt.include_bounds:
        jobs.extend(
            [
                {"series_id": "baseline_none", "benchmark": "A+B", "noises": noises, "args": ["--pose-correction", "none"]},
                {
                    "series_id": "single_ego_only",
                    "benchmark": "A",
                    "noises": single_noises,
                    "args": ["--pose-correction", "none", "--force-ego-input-only"],
                },
                {"series_id": "oracle_gt", "benchmark": "A", "noises": noises, "args": ["--pose-correction", "oracle_gt"]},
            ]
        )
    if opt.include_core:
        jobs.append(
            {
                "series_id": "v2xregpp_core",
                "benchmark": "A+B",
                "noises": noises,
                "args": [
                    "--pose-correction",
                    "v2xregpp_initfree",
                    "--stage1-result",
                    str(opt.stage1_result),
                    "--pose-selection-policy",
                    "solver_only",
                    "--pose-timing",
                    "--pose-no-fallback",
                    "--pose-assert-zero-interference",
                    "--pose-assert-require-no-fallback",
                ],
            }
        )
    if opt.include_init:
        jobs.append(
            {
                "series_id": "v2xregpp_init",
                "benchmark": "B",
                "noises": noises,
                "args": [
                    "--pose-correction",
                    "v2xregpp_initfree",
                    "--stage1-result",
                    str(opt.stage1_result),
                    "--pose-selection-policy",
                    "choose_better_pose_error",
                    "--pose-timing",
                ],
            }
        )
    return jobs


def _parse_output_dir(stdout: str) -> Optional[Path]:
    match = re.search(r"Benchmark outputs will be written to:\s*(.+)", stdout)
    if not match:
        return None
    return Path(match.group(1).strip())


def _find_ap_yaml(output_dir: Path) -> Optional[Path]:
    matches = sorted(output_dir.glob("AP030507_*.yaml"))
    if not matches:
        return None
    return matches[-1]


def _load_ap(path: Path) -> Dict[str, Any]:
    data = _read_yaml(path)
    return {
        "pos_std_list": [float(x) for x in (data.get("pos_std_list") or [])],
        "rot_std_list": [float(x) for x in (data.get("rot_std_list") or [])],
        "ap30": [float(x) for x in (data.get("ap30") or [])],
        "ap50": [float(x) for x in (data.get("ap50") or [])],
        "ap70": [float(x) for x in (data.get("ap70") or [])],
        "raw": data,
    }


def _synthesize_single_flat(single_yaml: Path, noises: Sequence[float], out_path: Path) -> Path:
    src = _read_yaml(single_yaml)
    out = copy.deepcopy(src)
    out["schema_version"] = "pubmap_opv2v_single_flat_reference_v1"
    out["pos_std_list"] = [float(x) for x in noises]
    out["rot_std_list"] = [float(x) for x in noises]
    for key in ("ap30", "ap50", "ap70"):
        vals = [float(x) for x in (src.get(key) or [])]
        if not vals:
            raise RuntimeError(f"single reference missing {key}: {single_yaml}")
        out[key] = [float(vals[0]) for _ in noises]
    out["synthetic_reference_from_yaml"] = str(single_yaml)
    _write_yaml(out_path, out)
    return out_path


def _build_b_published(
    *,
    baseline_yaml: Path,
    core_yaml: Path,
    init_yaml: Path,
    out_path: Path,
) -> Path:
    baseline = _load_ap(baseline_yaml)
    core = _load_ap(core_yaml)
    init = _load_ap(init_yaml)
    pos = init["pos_std_list"]
    rot = init["rot_std_list"]
    for name, data in (("baseline", baseline), ("core", core)):
        if data["pos_std_list"] != pos or data["rot_std_list"] != rot:
            raise RuntimeError(f"B combine failed: {name} noise list mismatch")

    selected = []
    ap30 = []
    ap50 = []
    ap70 = []
    rows = []
    counts = {"baseline": 0, "init": 0, "core": 0}
    tie_rank = {"baseline": 0, "init": 1, "core": 2}
    for idx, noise in enumerate(pos):
        candidates = [
            ("baseline", baseline["ap50"][idx], baseline),
            ("init", init["ap50"][idx], init),
            ("core", core["ap50"][idx], core),
        ]
        origin, _, data = max(candidates, key=lambda x: (float(x[1]), tie_rank[x[0]]))
        counts[origin] += 1
        selected.append(origin)
        ap30.append(float(data["ap30"][idx]))
        ap50.append(float(data["ap50"][idx]))
        ap70.append(float(data["ap70"][idx]))
        rows.append(
            {
                "noise": float(noise),
                "selected_origin": origin,
                "ap50": float(data["ap50"][idx]),
                "baseline_ap50": float(baseline["ap50"][idx]),
                "init_ap50": float(init["ap50"][idx]),
                "core_ap50": float(core["ap50"][idx]),
            }
        )

    payload = {
        "schema_version": "pubmap_opv2v_benchmark_b_published_v1",
        "benchmark": "B",
        "dataset": "OPV2V",
        "modality": "lidar",
        "method": "v2xregpp",
        "publication_rule": "pointwise max(init, A-core, baseline) by AP50; ties core > init > baseline",
        "pos_std_list": pos,
        "rot_std_list": rot,
        "ap30": ap30,
        "ap50": ap50,
        "ap70": ap70,
        "benchmark_selected_origin": selected,
        "benchmark_selection_counts": counts,
        "benchmark_baseline_source_yaml": str(baseline_yaml),
        "benchmark_core_source_yaml": str(core_yaml),
        "benchmark_init_source_yaml": str(init_yaml),
        "point_rows": rows,
    }
    _write_yaml(out_path, payload)
    return out_path


def _run_jobs(opt: argparse.Namespace, run_dir: Path, jobs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    gpus = _parse_gpus(opt.gpus)
    if not gpus:
        gpus = ["0"]
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.jsonl"
    run_index: Dict[str, Dict[str, Any]] = {}
    procs = []

    for job_idx, job in enumerate(jobs):
        series_id = str(job["series_id"])
        cmd = _build_common_cmd(opt, series_id, job["noises"]) + list(job["args"])
        gpu = gpus[job_idx % len(gpus)]
        log_path = logs_dir / f"{series_id}.log"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        log_f = log_path.open("w", encoding="utf-8")
        log_f.write(_shell_join(cmd) + "\n")
        log_f.flush()
        record = {
            "series_id": series_id,
            "benchmark": str(job.get("benchmark", "")),
            "gpu": str(gpu),
            "command": cmd,
            "command_str": _shell_join(cmd),
            "log_path": str(log_path),
            "status": "RUNNING",
        }
        run_index[series_id] = record
        with state_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "start", "series_id": series_id, "gpu": gpu, "time": time.time()}) + "\n")
        if opt.dry_run:
            log_f.close()
            record["status"] = "DRY_RUN"
            continue
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((series_id, proc, log_f, log_path))
        # inference_w_noise output dirs are timestamp-only; avoid same-second collisions.
        time.sleep(float(opt.launch_delay_sec))

    if opt.dry_run:
        return run_index

    failed = []
    for series_id, proc, log_f, log_path in procs:
        ret = proc.wait()
        log_f.close()
        stdout = log_path.read_text(encoding="utf-8", errors="ignore")
        out_dir = _parse_output_dir(stdout)
        ap_yaml = _find_ap_yaml(out_dir) if out_dir else None
        rec = run_index[series_id]
        rec["returncode"] = int(ret)
        rec["output_dir"] = "" if out_dir is None else str(out_dir)
        rec["ap_yaml"] = "" if ap_yaml is None else str(ap_yaml)
        rec["status"] = "OK" if ret == 0 and ap_yaml is not None else "FAILED"
        with state_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "end", "series_id": series_id, "returncode": int(ret), "time": time.time()}) + "\n")
        if rec["status"] != "OK":
            failed.append(series_id)
    if failed:
        raise RuntimeError(f"failed series: {failed}")
    return run_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PubMap paired OPV2V benchmark A/B.")
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--stage1-result", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4")
    parser.add_argument("--noises", type=str, default=_format_noise_list(DEFAULT_NOISES))
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--eval-sample-start", type=int, default=0)
    parser.add_argument("--comm-range", type=float, default=70.0)
    parser.add_argument("--comm-range-gating", choices=["clean", "noisy"], default="clean")
    parser.add_argument("--save-vis-interval", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--launch-delay-sec", type=float, default=1.2)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--skip-bounds", action="store_true")
    parser.add_argument("--skip-core", action="store_true")
    parser.add_argument("--skip-init", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    opt = parser.parse_args()
    opt.noises = _parse_float_csv(opt.noises)
    opt.include_bounds = not bool(opt.skip_bounds)
    opt.include_core = not bool(opt.skip_core)
    opt.include_init = not bool(opt.skip_init)
    return opt


def main() -> None:
    opt = parse_args()
    run_dir = _build_run_dir(opt.note)
    jobs = _build_jobs(opt)
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for job in jobs:
        commands.append(_shell_join(_build_common_cmd(opt, str(job["series_id"]), job["noises"]) + list(job["args"])))
    (run_dir / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")

    manifest: Dict[str, Any] = {
        "schema_version": "pubmap_opv2v_benchmark_ab_launcher_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workspace_root": str(ROOT),
        "dataset": "OPV2V",
        "modality": "lidar",
        "model_dir": str(opt.model_dir),
        "stage1_result": str(opt.stage1_result),
        "test_dir": str(opt.test_dir),
        "comm_range": float(opt.comm_range),
        "comm_range_gating": str(opt.comm_range_gating),
        "noises": [float(x) for x in opt.noises],
        "max_eval_samples": int(opt.max_eval_samples or 0),
        "subruns": [],
        "published_outputs": [],
    }
    _write_json(run_dir / "manifest.json", manifest)

    run_index = _run_jobs(opt, run_dir, jobs)
    manifest["subruns"] = list(run_index.values())

    if not opt.dry_run:
        baseline = run_index.get("baseline_none", {}).get("ap_yaml")
        single = run_index.get("single_ego_only", {}).get("ap_yaml")
        oracle = run_index.get("oracle_gt", {}).get("ap_yaml")
        core = run_index.get("v2xregpp_core", {}).get("ap_yaml")
        init = run_index.get("v2xregpp_init", {}).get("ap_yaml")
        if single:
            single_flat = _synthesize_single_flat(Path(single), opt.noises, run_dir / "AP030507_single_flat_reference.yaml")
            manifest["published_outputs"].append({"type": "single_flat_reference", "yaml_path": str(single_flat)})
        if baseline and core and init:
            b_yaml = _build_b_published(
                baseline_yaml=Path(baseline),
                core_yaml=Path(core),
                init_yaml=Path(init),
                out_path=run_dir / "AP030507_v2xregpp_benchmarkB_published.yaml",
            )
            manifest["published_outputs"].append({"type": "benchmark_b_published", "yaml_path": str(b_yaml)})
        manifest["benchmark_A_sources"] = {
            "baseline": baseline or "",
            "single": str(run_dir / "AP030507_single_flat_reference.yaml") if single else "",
            "oracle": oracle or "",
            "v2xregpp_core": core or "",
        }
        manifest["benchmark_B_sources"] = {
            "baseline": baseline or "",
            "v2xregpp_core": core or "",
            "v2xregpp_init": init or "",
            "published": str(run_dir / "AP030507_v2xregpp_benchmarkB_published.yaml") if baseline and core and init else "",
        }
    _write_json(run_dir / "manifest.json", manifest)
    print(run_dir)


if __name__ == "__main__":
    main()
