#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.profiling.opv2v_benchmark_a import build_tables_and_summary as build_profile_summary

INFERENCE_SCRIPT = ROOT / "opencood" / "tools" / "inference_w_noise.py"
OUTPUT_ROOT = ROOT / "outputs" / "opv2v_benchmark_a_profile"

DEFAULT_PYTHON = Path("/home/qqxluca/miniconda3/envs/heal/bin/python")
DEFAULT_MODEL_DIR = Path(
    "/home/qqxluca/projects/v2xreg_private/HEAL/opencood/logs/opv2v_camera_v2xvit_full_prope"
)
DEFAULT_STAGE1 = Path(
    "/home/qqxluca/HEAL_benchmark_20260422/outputs/image_depth_stage1_cache/run_20260513_191711_opv2v_image_depth_camera_model_fullprope_full2170/test/stage1_boxes_image_depth_camera_model.json"
)
DEFAULT_TEST_DIR = Path("/data2/OPV2V/test")
DEFAULT_DEPTH_ROOT = Path("/data2/opv2v_depth/test")
DEFAULT_NOISES = [float(i) for i in range(11)]
DEFAULT_CORE_NOISES = [0.0, 7.0, 10.0]
DEFAULT_ORACLE_ANCHOR = 9.0

CORE_POSE_CORR = {
    "cbm": "cbm_initfree",
    "freealign": "freealign_paper",
    "v2xregpp": "v2xregpp_initfree",
    "vips": "vips_initfree",
}


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_note(raw: str) -> str:
    text = str(raw or "").strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _shell_join(parts: Sequence[Any]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _parse_float_csv(raw: str) -> List[float]:
    return [float(x) for x in _parse_csv(raw)]


def _format_noise(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value)


def _noise_tag(value: float) -> str:
    return _format_noise(value).replace(".", "p")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _build_run_dir(note: str) -> Path:
    stem = f"run_{_timestamp()}_opv2v_benchmarkA_profile"
    suffix = _safe_note(note)
    if suffix:
        stem += f"_{suffix}"
    out = OUTPUT_ROOT / stem
    out.mkdir(parents=True, exist_ok=False)
    return out


def _common_infer_cmd(
    opt: argparse.Namespace,
    *,
    note: str,
    noise: float,
    include_runtime_pose_provider: bool,
) -> List[str]:
    cmd = [
        str(opt.python_bin),
        str(INFERENCE_SCRIPT),
        "--model_dir",
        str(opt.model_dir),
        "--fusion_method",
        "intermediate",
        "--test-dir-override",
        str(opt.test_dir),
        "--depth-root-override",
        str(opt.depth_root),
        "--save_vis_interval",
        str(int(opt.save_vis_interval)),
        "--num-workers",
        str(int(opt.num_workers)),
        "--pos-std-list",
        _format_noise(noise),
        "--rot-std-list",
        _format_noise(noise),
        "--sweep-mode",
        "paired",
        "--noise-target",
        "non-ego",
        "--comm-range-override",
        str(float(opt.comm_range)),
        "--comm-range-gating",
        "clean",
        "--pose-device",
        "cuda",
        "--log-interval",
        str(int(opt.log_interval)),
        "--note",
        note,
    ]
    if include_runtime_pose_provider:
        cmd.extend(
            [
                "--runtime-mode",
                "register_and_fuse",
                "--solver-backend",
                "online_box",
                "--pose-source",
                "noisy_input",
                "--pose-timing",
            ]
        )
    if opt.force_pose_confidence is not None:
        cmd.extend(["--force-pose-confidence", str(float(opt.force_pose_confidence))])
    if int(opt.max_eval_samples or 0) > 0:
        cmd.extend(["--max-eval-samples", str(int(opt.max_eval_samples))])
    if int(opt.eval_sample_start or 0) > 0:
        cmd.extend(["--eval-sample-start", str(int(opt.eval_sample_start))])
    return cmd


def _build_jobs(opt: argparse.Namespace, run_dir: Path) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    note_prefix = _safe_note(opt.note) or "opv2v_benchmarkA_profile"

    bounds = set(opt.bounds)
    if "baseline" in bounds:
        for noise in opt.noises:
            sid = f"baseline_n{_noise_tag(noise)}"
            cmd = _common_infer_cmd(
                opt,
                note=f"{note_prefix}_{sid}",
                noise=noise,
                include_runtime_pose_provider=True,
            )
            cmd.extend(["--pose-correction", "none"])
            jobs.append(
                {
                    "series_id": sid,
                    "line_id": "baseline",
                    "role": "reference",
                    "job_type": "bound",
                    "noise": float(noise),
                    "cmd": cmd,
                    "deps": [],
                }
            )

    if "single" in bounds:
        noise = 0.0
        sid = "single_n0"
        cmd = _common_infer_cmd(
            opt,
            note=f"{note_prefix}_{sid}",
            noise=noise,
            include_runtime_pose_provider=True,
        )
        cmd.extend(["--pose-correction", "none", "--force-ego-input-only"])
        jobs.append(
            {
                "series_id": sid,
                "line_id": "single",
                "role": "reference",
                "job_type": "bound",
                "noise": float(noise),
                "cmd": cmd,
                "deps": [],
            }
        )

    if "oracle" in bounds:
        noise = float(opt.oracle_anchor_noise)
        sid = f"oracle_n{_noise_tag(noise)}"
        cmd = _common_infer_cmd(
            opt,
            note=f"{note_prefix}_{sid}",
            noise=noise,
            include_runtime_pose_provider=True,
        )
        cmd.extend(["--pose-correction", "oracle_gt"])
        jobs.append(
            {
                "series_id": sid,
                "line_id": "oracle",
                "role": "reference",
                "job_type": "bound",
                "noise": float(noise),
                "cmd": cmd,
                "deps": [],
            }
        )

    for method in opt.methods:
        pose_corr = CORE_POSE_CORR[method]
        for noise in opt.core_noises:
            tag = _noise_tag(noise)
            solver_sid = f"{method}_solver_n{tag}"
            down_sid = f"{method}_n{tag}"
            override_path = run_dir / "overrides" / "opv2v" / method / f"{method}_n{tag}.json"
            solver_cmd = [
                str(opt.python_bin),
                str(INFERENCE_SCRIPT),
                "--model_dir",
                str(opt.model_dir),
                "--fusion_method",
                "intermediate",
                "--test-dir-override",
                str(opt.test_dir),
                "--depth-root-override",
                str(opt.depth_root),
                "--pose-correction",
                pose_corr,
                "--solver-backend",
                "offline_map",
                "--pose-solver-only",
                "--pose-selection-policy",
                "solver_only",
                "--pose-source",
                "identity",
                "--comm-range-override",
                str(float(opt.comm_range)),
                "--noise-target",
                "non-ego",
                "--sweep-mode",
                "paired",
                "--pos-std-list",
                _format_noise(noise),
                "--rot-std-list",
                _format_noise(noise),
                "--num-workers",
                str(int(opt.num_workers)),
                "--save_vis_interval",
                str(int(opt.save_vis_interval)),
                "--pose-device",
                "cuda",
                "--stage1-result",
                str(opt.stage1_result),
                "--pose-override-export-path",
                str(override_path),
                "--log-interval",
                str(int(opt.log_interval)),
                "--note",
                f"{note_prefix}_{solver_sid}",
            ]
            if int(opt.max_eval_samples or 0) > 0:
                solver_cmd.extend(["--max-eval-samples", str(int(opt.max_eval_samples))])
            if int(opt.eval_sample_start or 0) > 0:
                solver_cmd.extend(["--eval-sample-start", str(int(opt.eval_sample_start))])
            jobs.append(
                {
                    "series_id": solver_sid,
                    "line_id": method,
                    "role": "method",
                    "job_type": "solver",
                    "noise": float(noise),
                    "cmd": solver_cmd,
                    "deps": [],
                    "override_path": str(override_path),
                }
            )

            down_cmd = _common_infer_cmd(
                opt,
                note=f"{note_prefix}_{down_sid}",
                noise=noise,
                include_runtime_pose_provider=False,
            )
            down_cmd.extend(["--pose-correction", "none", "--pose-override-path", str(override_path)])
            jobs.append(
                {
                    "series_id": down_sid,
                    "line_id": method,
                    "role": "method",
                    "job_type": "downstream",
                    "noise": float(noise),
                    "cmd": down_cmd,
                    "deps": [solver_sid],
                    "override_path": str(override_path),
                    "solver_series_id": solver_sid,
                }
            )
    return jobs


def _parse_output_dir(stdout: str) -> Optional[Path]:
    match = re.search(r"Benchmark outputs will be written to:\s*(.+)", stdout)
    if not match:
        return None
    return Path(match.group(1).strip())


def _find_yaml(output_dir: Optional[Path], job_type: str) -> Optional[Path]:
    if output_dir is None or not output_dir.exists():
        return None
    pattern = "POSE_SOLVER_ONLY_*.yaml" if job_type == "solver" else "AP030507_*.yaml"
    matches = sorted(output_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return matches[-1]


def _run_jobs(opt: argparse.Namespace, run_dir: Path, jobs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.jsonl"
    gpus = [x.strip() for x in str(opt.gpus or "").split(",") if x.strip()]
    if not gpus:
        gpus = ["0"]
    max_parallel = min(int(opt.max_parallel or len(gpus)), len(gpus))

    records: Dict[str, Dict[str, Any]] = {}
    pending: Dict[str, Dict[str, Any]] = {str(job["series_id"]): dict(job) for job in jobs}
    running: List[Tuple[str, subprocess.Popen, Any, Path, float, str]] = []
    completed: Dict[str, str] = {}
    gpu_cursor = 0

    for job in jobs:
        sid = str(job["series_id"])
        rec = {
            "series_id": sid,
            "line_id": str(job.get("line_id", "")),
            "role": str(job.get("role", "")),
            "job_type": str(job.get("job_type", "")),
            "noise": float(job.get("noise", 0.0)),
            "deps": list(job.get("deps") or []),
            "command": list(job.get("cmd") or []),
            "command_str": _shell_join(job.get("cmd") or []),
            "override_path": str(job.get("override_path") or ""),
            "status": "PENDING",
        }
        records[sid] = rec

    if opt.dry_run:
        for rec in records.values():
            rec["status"] = "DRY_RUN"
        return records

    def _append_state(payload: Mapping[str, Any]) -> None:
        with state_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")

    while pending or running:
        started = True
        while started and len(running) < max_parallel:
            started = False
            for sid, job in list(pending.items()):
                deps = [str(x) for x in (job.get("deps") or [])]
                if any(completed.get(dep) != "OK" for dep in deps):
                    if any(dep in completed and completed.get(dep) != "OK" for dep in deps):
                        records[sid]["status"] = "SKIPPED_DEP_FAILED"
                        completed[sid] = "SKIPPED_DEP_FAILED"
                        pending.pop(sid)
                        _append_state({"event": "skip", "series_id": sid, "time": time.time(), "reason": "dep_failed"})
                    continue

                gpu = gpus[gpu_cursor % len(gpus)]
                gpu_cursor += 1
                log_path = logs_dir / f"{sid}.log"
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("OMP_NUM_THREADS", "1")
                env.setdefault("MKL_NUM_THREADS", "1")
                log_f = log_path.open("w", encoding="utf-8")
                log_f.write(_shell_join(job["cmd"]) + "\n")
                log_f.flush()
                proc = subprocess.Popen(
                    list(job["cmd"]),
                    cwd=str(ROOT),
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                )
                start_time = time.time()
                records[sid].update(
                    {
                        "status": "RUNNING",
                        "gpu": str(gpu),
                        "log_path": str(log_path),
                        "start_time": start_time,
                    }
                )
                _append_state(
                    {
                        "event": "start",
                        "series_id": sid,
                        "job_type": str(job.get("job_type", "")),
                        "line_id": str(job.get("line_id", "")),
                        "noise": float(job.get("noise", 0.0)),
                        "gpu": str(gpu),
                        "time": start_time,
                    }
                )
                running.append((sid, proc, log_f, log_path, start_time, str(gpu)))
                pending.pop(sid)
                started = True
                time.sleep(float(opt.launch_delay_sec))
                break

        time.sleep(float(opt.poll_sec))
        still_running: List[Tuple[str, subprocess.Popen, Any, Path, float, str]] = []
        for sid, proc, log_f, log_path, start_time, gpu in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((sid, proc, log_f, log_path, start_time, gpu))
                continue
            log_f.close()
            end_time = time.time()
            stdout = log_path.read_text(encoding="utf-8", errors="ignore")
            output_dir = _parse_output_dir(stdout)
            yaml_path = _find_yaml(output_dir, records[sid]["job_type"])
            status = "OK" if int(ret) == 0 and yaml_path is not None else "FAILED"
            records[sid].update(
                {
                    "status": status,
                    "returncode": int(ret),
                    "end_time": end_time,
                    "wall_sec": float(end_time - start_time),
                    "output_dir": "" if output_dir is None else str(output_dir),
                    "yaml_path": "" if yaml_path is None else str(yaml_path),
                }
            )
            completed[sid] = status
            _append_state(
                {
                    "event": "end",
                    "series_id": sid,
                    "status": status,
                    "returncode": int(ret),
                    "time": end_time,
                    "wall_sec": float(end_time - start_time),
                    "yaml_path": records[sid]["yaml_path"],
                }
            )
        running = still_running

        failed = [sid for sid, status in completed.items() if status == "FAILED"]
        if failed and bool(opt.abort_on_failure):
            for sid, proc, log_f, log_path, start_time, gpu in running:
                proc.terminate()
                log_f.close()
                records[sid]["status"] = "TERMINATED_AFTER_FAILURE"
                completed[sid] = "TERMINATED_AFTER_FAILURE"
            raise RuntimeError(f"failed jobs: {failed}")

    failed = [sid for sid, rec in records.items() if rec.get("status") not in {"OK", "DRY_RUN"}]
    if failed:
        raise RuntimeError(f"benchmark jobs did not all finish successfully: {failed}")
    return records



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OPV2V full-test Benchmark A with profiling summaries.")
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--stage1-result", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--depth-root", type=Path, default=DEFAULT_DEPTH_ROOT)
    parser.add_argument("--gpus", type=str, default="1,5,6,7,8,9")
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--noises", type=str, default=",".join(_format_noise(x) for x in DEFAULT_NOISES))
    parser.add_argument("--core-noises", type=str, default=",".join(_format_noise(x) for x in DEFAULT_CORE_NOISES))
    parser.add_argument("--oracle-anchor-noise", type=float, default=DEFAULT_ORACLE_ANCHOR)
    parser.add_argument("--methods", type=str, default="cbm,freealign,v2xregpp,vips")
    parser.add_argument("--bounds", type=str, default="baseline,single,oracle")
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--eval-sample-start", type=int, default=0)
    parser.add_argument("--comm-range", type=float, default=70.0)
    parser.add_argument("--force-pose-confidence", type=float, default=1.0)
    parser.add_argument("--save-vis-interval", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--launch-delay-sec", type=float, default=1.2)
    parser.add_argument("--poll-sec", type=float, default=10.0)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--abort-on-failure", action="store_true", default=True)
    parser.add_argument("--no-abort-on-failure", dest="abort_on_failure", action="store_false")
    opt = parser.parse_args()
    opt.noises = _parse_float_csv(opt.noises)
    opt.core_noises = _parse_float_csv(opt.core_noises)
    opt.methods = [m for m in _parse_csv(opt.methods) if m]
    opt.bounds = [b for b in _parse_csv(opt.bounds) if b]
    for method in opt.methods:
        if method not in CORE_POSE_CORR:
            raise ValueError(f"unsupported method: {method}")
    for bound in opt.bounds:
        if bound not in {"baseline", "single", "oracle"}:
            raise ValueError(f"unsupported bound: {bound}")
    if int(opt.max_parallel or 0) <= 0:
        opt.max_parallel = len([x for x in str(opt.gpus).split(",") if x.strip()]) or 1
    for label, path in (
        ("python_bin", opt.python_bin),
        ("model_dir", opt.model_dir),
        ("stage1_result", opt.stage1_result),
        ("test_dir", opt.test_dir),
        ("depth_root", opt.depth_root),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    return opt


def main() -> None:
    opt = parse_args()
    note = opt.note or ("full2170" if int(opt.max_eval_samples or 0) <= 0 else f"smoke{int(opt.max_eval_samples)}")
    run_dir = _build_run_dir(note)
    jobs = _build_jobs(opt, run_dir)

    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for job in jobs:
        commands.append(f"# {job['series_id']}")
        commands.append(_shell_join(job["cmd"]))
        commands.append("")
    (run_dir / "commands.sh").write_text("\n".join(commands), encoding="utf-8")

    manifest = {
        "schema_version": "opv2v_benchmark_a_profile_manifest_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workspace_root": str(ROOT),
        "run_dir": str(run_dir),
        "dataset": "OPV2V",
        "dataset_frames_expected": 2170,
        "benchmark": "A",
        "model_dir": str(opt.model_dir),
        "stage1_result": str(opt.stage1_result),
        "test_dir": str(opt.test_dir),
        "depth_root": str(opt.depth_root),
        "gpus": str(opt.gpus),
        "max_parallel": int(opt.max_parallel),
        "max_eval_samples": int(opt.max_eval_samples or 0),
        "noises": [float(x) for x in opt.noises],
        "core_noises": [float(x) for x in opt.core_noises],
        "oracle_anchor_noise": float(opt.oracle_anchor_noise),
        "methods": list(opt.methods),
        "bounds": list(opt.bounds),
        "jobs": [
            {
                "series_id": str(job["series_id"]),
                "line_id": str(job.get("line_id", "")),
                "job_type": str(job.get("job_type", "")),
                "noise": float(job.get("noise", 0.0)),
                "deps": list(job.get("deps") or []),
                "override_path": str(job.get("override_path") or ""),
                "command": list(job.get("cmd") or []),
            }
            for job in jobs
        ],
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(f"Run dir: {run_dir}")
    print(f"Jobs: {len(jobs)}")

    records = _run_jobs(opt, run_dir, jobs)
    manifest["records"] = list(records.values())
    _write_json(run_dir / "manifest.json", manifest)

    if not opt.dry_run:
        summary = build_profile_summary(opt, run_dir, records)
        print("Summary:", summary["artifacts"]["profile_rows_csv"])
        print("Profile summary:", run_dir / "profile_summary_full2170.json")


if __name__ == "__main__":
    main()
