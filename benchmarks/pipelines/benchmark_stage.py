#!/usr/bin/env python3
"""Benchmark sweep orchestration inside the staged pipeline.

This module replaces the old benchmark launcher path for config-driven runs.
It still delegates low-level model execution to `opencood/tools/inference_w_noise.py`,
but the experiment graph, records, summaries, and plots are owned by the
pipeline layer.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - project runtime depends on PyYAML.
    yaml = None

from benchmarks.pipelines.artifacts import (
    ROOT,
    make_stage_dirs,
    resolve_path,
    shell_join,
    write_json,
)
from benchmarks.profiling.opv2v_benchmark_a import build_tables_and_summary as build_opv2v_profile_summary


INFERENCE_SCRIPT = ROOT / "opencood" / "tools" / "inference_w_noise.py"
OUTPUT_DIR_RE = re.compile(r"Benchmark outputs will be written to:\s*(.+)")

OPV2V_A_CORE_POSE_CORR = {
    "cbm": "cbm_initfree",
    "freealign": "freealign_paper",
    "v2xregpp": "v2xregpp_initfree",
    "vips": "vips_initfree",
}


def _safe_note(raw: Any) -> str:
    text = str(raw or "").strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _as_mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [value]


def _float_list(value: Any) -> List[float]:
    return [float(x) for x in _as_list(value)]


def _str_list(value: Any) -> List[str]:
    return [str(x).strip() for x in _as_list(value) if str(x).strip()]


def _format_noise(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value)


def _format_noise_list(values: Sequence[float]) -> str:
    return ",".join(_format_noise(float(x)) for x in values)


def _noise_tag(value: float) -> str:
    return _format_noise(value).replace(".", "p")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write benchmark YAML outputs")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(payload), f, sort_keys=False, allow_unicode=True)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read benchmark YAML outputs")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def _config_path(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] not in {None, ""}:
            return config[key]
    return default


def _stage1_from_detection(config: Mapping[str, Any], detection_record: Mapping[str, Any]) -> str:
    inputs = _as_mapping(config.get("inputs"))
    value = _config_path(inputs, "stage1_result", "stage1_path")
    if value:
        return str(resolve_path(value))
    value = _config_path(config, "stage1_result", "stage1_path")
    if value:
        return str(resolve_path(value))
    output = detection_record.get("output") if isinstance(detection_record, Mapping) else {}
    if isinstance(output, Mapping) and output.get("stage1_result"):
        return str(resolve_path(output["stage1_result"]))
    raise ValueError("benchmark stage requires stage1_result or a detection stage output")


def _parse_output_dir(text: str) -> Optional[Path]:
    matches = list(OUTPUT_DIR_RE.finditer(text or ""))
    if not matches:
        return None
    return Path(matches[-1].group(1).strip())


def _find_yaml(output_dir: Optional[Path], job_type: str) -> Optional[Path]:
    if output_dir is None or not output_dir.exists():
        return None
    pattern = "POSE_SOLVER_ONLY_*.yaml" if job_type == "solver" else "AP030507_*.yaml"
    matches = sorted(output_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _commands_for_jobs(jobs: Sequence[Mapping[str, Any]]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"export PYTHONPATH={shell_join([str(ROOT)])}:${{PYTHONPATH:-}}", ""]
    for job in jobs:
        lines.append(f"# {job['series_id']}")
        lines.append(shell_join(job.get("cmd") or []))
        lines.append("")
    return "\n".join(lines)


def _run_job_graph(
    *,
    jobs: Sequence[Mapping[str, Any]],
    run_dir: Path,
    logs_dir: Path,
    dry_run: bool,
    gpus: Sequence[str],
    max_parallel: int,
    launch_delay_sec: float,
    poll_sec: float,
    abort_on_failure: bool,
) -> Dict[str, Dict[str, Any]]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.jsonl"
    gpu_list = [str(x).strip() for x in gpus if str(x).strip()] or ["0"]
    max_parallel = max(1, int(max_parallel or len(gpu_list)))
    max_parallel = min(max_parallel, len(gpu_list))

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
            "benchmark": str(job.get("benchmark", "")),
            "noise": float(job.get("noise", 0.0)),
            "noises": [float(x) for x in (job.get("noises") or [])],
            "deps": list(job.get("deps") or []),
            "command": list(job.get("cmd") or []),
            "command_str": shell_join(job.get("cmd") or []),
            "override_path": str(job.get("override_path") or ""),
            "status": "PENDING",
        }
        records[sid] = rec

    if dry_run:
        for rec in records.values():
            rec["status"] = "DRY_RUN"
        return records

    def append_state(payload: Mapping[str, Any]) -> None:
        with state_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")

    while pending or running:
        started_any = True
        while started_any and len(running) < max_parallel:
            started_any = False
            for sid, job in list(pending.items()):
                deps = [str(x) for x in (job.get("deps") or [])]
                if any(completed.get(dep) != "OK" for dep in deps):
                    if any(dep in completed and completed.get(dep) != "OK" for dep in deps):
                        records[sid]["status"] = "SKIPPED_DEP_FAILED"
                        completed[sid] = "SKIPPED_DEP_FAILED"
                        pending.pop(sid)
                        append_state({"event": "skip", "series_id": sid, "time": time.time(), "reason": "dep_failed"})
                    continue

                gpu = gpu_list[gpu_cursor % len(gpu_list)]
                gpu_cursor += 1
                log_path = logs_dir / f"{sid}.log"
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("OMP_NUM_THREADS", "1")
                env.setdefault("MKL_NUM_THREADS", "1")
                log_f = log_path.open("w", encoding="utf-8")
                log_f.write(shell_join(job["cmd"]) + "\n")
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
                append_state(
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
                started_any = True
                time.sleep(float(launch_delay_sec))
                break

        time.sleep(float(poll_sec))
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
                    "ap_yaml": "" if yaml_path is None else str(yaml_path),
                }
            )
            completed[sid] = status
            append_state(
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
        if failed and abort_on_failure:
            for sid, proc, log_f, _log_path, _start_time, _gpu in running:
                proc.terminate()
                log_f.close()
                records[sid]["status"] = "TERMINATED_AFTER_FAILURE"
                completed[sid] = "TERMINATED_AFTER_FAILURE"
            raise RuntimeError(f"failed benchmark jobs: {failed}")

    failed = [sid for sid, rec in records.items() if rec.get("status") not in {"OK", "DRY_RUN"}]
    if failed:
        raise RuntimeError(f"benchmark jobs did not all finish successfully: {failed}")
    return records


def _opv2v_common_cmd(opt: argparse.Namespace, *, note: str, noise: float, runtime_pose_provider: bool) -> List[str]:
    cmd = [
        str(opt.python_bin),
        str(INFERENCE_SCRIPT),
        "--model_dir",
        str(opt.model_dir),
        "--fusion_method",
        str(opt.fusion_method),
        "--test-dir-override",
        str(opt.test_dir),
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
        str(opt.comm_range_gating),
        "--pose-device",
        str(opt.pose_device),
        "--log-interval",
        str(int(opt.log_interval)),
        "--note",
        note,
    ]
    if opt.depth_root:
        cmd.extend(["--depth-root-override", str(opt.depth_root)])
    if runtime_pose_provider:
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


def _build_opv2v_a_jobs(opt: argparse.Namespace, run_dir: Path) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    note_prefix = _safe_note(opt.note) or "opv2v_benchmarkA_profile"
    bounds = set(opt.bounds)

    if "baseline" in bounds:
        for noise in opt.noises:
            sid = f"baseline_n{_noise_tag(noise)}"
            cmd = _opv2v_common_cmd(opt, note=f"{note_prefix}_{sid}", noise=noise, runtime_pose_provider=True)
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
        sid = "single_n0"
        cmd = _opv2v_common_cmd(opt, note=f"{note_prefix}_{sid}", noise=0.0, runtime_pose_provider=True)
        cmd.extend(["--pose-correction", "none", "--force-ego-input-only"])
        jobs.append(
            {
                "series_id": sid,
                "line_id": "single",
                "role": "reference",
                "job_type": "bound",
                "noise": 0.0,
                "cmd": cmd,
                "deps": [],
            }
        )

    if "oracle" in bounds:
        noise = float(opt.oracle_anchor_noise)
        sid = f"oracle_n{_noise_tag(noise)}"
        cmd = _opv2v_common_cmd(opt, note=f"{note_prefix}_{sid}", noise=noise, runtime_pose_provider=True)
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
        if method not in OPV2V_A_CORE_POSE_CORR:
            raise ValueError(f"unsupported OPV2V Benchmark A method: {method}")
        pose_corr = OPV2V_A_CORE_POSE_CORR[method]
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
                str(opt.fusion_method),
                "--test-dir-override",
                str(opt.test_dir),
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
                str(opt.pose_device),
                "--stage1-result",
                str(opt.stage1_result),
                "--pose-override-export-path",
                str(override_path),
                "--log-interval",
                str(int(opt.log_interval)),
                "--note",
                f"{note_prefix}_{solver_sid}",
            ]
            if opt.depth_root:
                solver_cmd.extend(["--depth-root-override", str(opt.depth_root)])
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

            down_cmd = _opv2v_common_cmd(opt, note=f"{note_prefix}_{down_sid}", noise=noise, runtime_pose_provider=False)
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


def _opv2v_a_options(config: Mapping[str, Any], detection_record: Mapping[str, Any], run_dir: Path) -> argparse.Namespace:
    inputs = _as_mapping(config.get("inputs"))
    execution = _as_mapping(config.get("execution"))
    sweep = _as_mapping(config.get("sweep"))
    inference = _as_mapping(config.get("inference"))
    stage1_result = _stage1_from_detection(config, detection_record)
    methods = _str_list(sweep.get("methods", ["cbm", "freealign", "v2xregpp", "vips"]))
    bounds = _str_list(sweep.get("bounds", ["baseline", "single", "oracle"]))
    for method in methods:
        if method not in OPV2V_A_CORE_POSE_CORR:
            raise ValueError(f"unsupported OPV2V Benchmark A method: {method}")
    for bound in bounds:
        if bound not in {"baseline", "single", "oracle"}:
            raise ValueError(f"unsupported OPV2V Benchmark A bound: {bound}")
    gpus = _str_list(execution.get("gpus", "0"))
    max_parallel = int(execution.get("max_parallel") or len(gpus) or 1)
    name_note = config.get("note") or run_dir.name
    return argparse.Namespace(
        python_bin=resolve_path(config.get("python_bin") or config.get("python-bin") or execution.get("python_bin") or execution.get("python-bin") or os.sys.executable),
        model_dir=resolve_path(inputs.get("model_dir") or config.get("model_dir")),
        stage1_result=Path(stage1_result),
        test_dir=resolve_path(inputs.get("test_dir") or inputs.get("test_dir_override") or config.get("test_dir")),
        depth_root=resolve_path(inputs.get("depth_root") or inputs.get("depth_root_override")) if (inputs.get("depth_root") or inputs.get("depth_root_override")) else None,
        gpus=gpus,
        max_parallel=max_parallel,
        launch_delay_sec=float(execution.get("launch_delay_sec", 1.2)),
        poll_sec=float(execution.get("poll_sec", 10.0)),
        abort_on_failure=bool(execution.get("abort_on_failure", True)),
        noises=_float_list(sweep.get("noises", list(range(11)))),
        core_noises=_float_list(sweep.get("core_noises", [0, 7, 10])),
        oracle_anchor_noise=float(sweep.get("oracle_anchor_noise", 9)),
        methods=methods,
        bounds=bounds,
        fusion_method=str(inference.get("fusion_method", "intermediate")),
        save_vis_interval=int(inference.get("save_vis_interval", 0)),
        num_workers=int(inference.get("num_workers", 0)),
        max_eval_samples=int(inference.get("max_eval_samples", 0) or 0),
        eval_sample_start=int(inference.get("eval_sample_start", 0) or 0),
        comm_range=float(inference.get("comm_range", inference.get("comm_range_override", 70.0))),
        comm_range_gating=str(inference.get("comm_range_gating", "clean")),
        force_pose_confidence=inference.get("force_pose_confidence"),
        pose_device=str(inference.get("pose_device", "cuda")),
        log_interval=int(inference.get("log_interval", 500)),
        note=str(name_note),
    )


def _run_opv2v_benchmark_a_profile(
    config: Mapping[str, Any],
    *,
    run_dir: Path,
    stage_dirs: Mapping[str, Path],
    detection_record: Mapping[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    started = time.time()
    opt = _opv2v_a_options(config, detection_record, run_dir)
    jobs = _build_opv2v_a_jobs(opt, run_dir)
    commands_text = _commands_for_jobs(jobs)
    (stage_dirs["stage"] / "commands.sh").write_text(commands_text, encoding="utf-8")
    (run_dir / "commands.sh").write_text(commands_text, encoding="utf-8")
    records = _run_job_graph(
        jobs=jobs,
        run_dir=run_dir,
        logs_dir=stage_dirs["logs"],
        dry_run=dry_run,
        gpus=opt.gpus,
        max_parallel=opt.max_parallel,
        launch_delay_sec=opt.launch_delay_sec,
        poll_sec=opt.poll_sec,
        abort_on_failure=opt.abort_on_failure,
    )
    artifacts = [str(stage_dirs["stage"] / "commands.sh"), str(run_dir / "commands.sh")]
    summary: Dict[str, Any] = {}
    if not dry_run:
        summary = build_opv2v_profile_summary(opt, run_dir, records)
        artifacts.extend(
            str(path)
            for path in (
                run_dir / "profile_rows.csv",
                run_dir / "benchmarkA_points.csv",
                run_dir / "profile_overhead_vs_baseline.csv",
                run_dir / "profile_summary.json",
                run_dir / "profile_summary_full2170.json",
            )
            if path.exists()
        )
        plots = ((summary.get("artifacts") or {}).get("plots") or {}) if isinstance(summary, dict) else {}
        artifacts.extend(str(x) for x in plots.values() if x)
    legacy_dirs = sorted({str(rec.get("output_dir")) for rec in records.values() if rec.get("output_dir")})
    yaml_paths = sorted({str(rec.get("yaml_path")) for rec in records.values() if rec.get("yaml_path")})
    artifacts.extend(yaml_paths)
    return {
        "schema_version": "benchmark_stage_record_v1",
        "stage": "benchmark",
        "mode": "opv2v_benchmark_a_profile",
        "status": "dry_run" if dry_run else "completed",
        "inputs": dict(config),
        "output": {
            "stage1_result": str(opt.stage1_result),
            "commands_path": str(stage_dirs["stage"] / "commands.sh"),
            "legacy_output_dirs": legacy_dirs,
            "artifacts": sorted(set(artifacts)),
        },
        "result": {
            "job_count": len(jobs),
            "records": list(records.values()),
            "profile_summary": summary,
        },
        "elapsed_sec": float(time.time() - started),
    }


def _pubmap_common_cmd(opt: argparse.Namespace, series_id: str, noises: Sequence[float]) -> List[str]:
    note = f"{opt.note}_{series_id}" if opt.note else series_id
    cmd = [
        str(opt.python_bin),
        str(INFERENCE_SCRIPT),
        "--model_dir",
        str(opt.model_dir),
        "--fusion_method",
        str(opt.fusion_method),
        "--test-dir-override",
        str(opt.test_dir),
        "--max-cav-override",
        str(int(opt.max_cav_override)),
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
        str(opt.pose_device),
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


def _build_pubmap_ab_jobs(opt: argparse.Namespace) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    noises = list(opt.noises)
    if opt.include_bounds:
        specs = [
            ("baseline_none", "baseline", "A+B", noises, ["--pose-correction", "none"]),
            ("single_ego_only", "single", "A", [0.0], ["--pose-correction", "none", "--force-ego-input-only"]),
            ("oracle_gt", "oracle", "A", noises, ["--pose-correction", "oracle_gt"]),
        ]
        for series_id, line_id, benchmark, job_noises, extra in specs:
            jobs.append(
                {
                    "series_id": series_id,
                    "line_id": line_id,
                    "role": "reference",
                    "job_type": "bound",
                    "benchmark": benchmark,
                    "noise": float(job_noises[0] if job_noises else 0.0),
                    "noises": [float(x) for x in job_noises],
                    "cmd": _pubmap_common_cmd(opt, series_id, job_noises) + extra,
                    "deps": [],
                }
            )
    if opt.include_core:
        extra = [
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
        ]
        jobs.append(
            {
                "series_id": "v2xregpp_core",
                "line_id": "v2xregpp_core",
                "role": "method",
                "job_type": "core",
                "benchmark": "A+B",
                "noise": float(noises[0] if noises else 0.0),
                "noises": [float(x) for x in noises],
                "cmd": _pubmap_common_cmd(opt, "v2xregpp_core", noises) + extra,
                "deps": [],
            }
        )
    if opt.include_init:
        extra = [
            "--pose-correction",
            "v2xregpp_initfree",
            "--stage1-result",
            str(opt.stage1_result),
            "--pose-selection-policy",
            "choose_better_pose_error",
            "--pose-timing",
        ]
        jobs.append(
            {
                "series_id": "v2xregpp_init",
                "line_id": "v2xregpp_init",
                "role": "method",
                "job_type": "init",
                "benchmark": "B",
                "noise": float(noises[0] if noises else 0.0),
                "noises": [float(x) for x in noises],
                "cmd": _pubmap_common_cmd(opt, "v2xregpp_init", noises) + extra,
                "deps": [],
            }
        )
    return jobs


def _pubmap_ab_options(config: Mapping[str, Any], detection_record: Mapping[str, Any], run_dir: Path) -> argparse.Namespace:
    inputs = _as_mapping(config.get("inputs"))
    execution = _as_mapping(config.get("execution"))
    sweep = _as_mapping(config.get("sweep"))
    inference = _as_mapping(config.get("inference"))
    benchmark = _as_mapping(config.get("benchmark"))
    stage1_result = _stage1_from_detection(config, detection_record)
    gpus = _str_list(execution.get("gpus", "0"))
    return argparse.Namespace(
        python_bin=resolve_path(config.get("python_bin") or config.get("python-bin") or execution.get("python_bin") or execution.get("python-bin") or os.sys.executable),
        model_dir=resolve_path(inputs.get("model_dir") or config.get("model_dir")),
        stage1_result=Path(stage1_result),
        test_dir=resolve_path(inputs.get("test_dir") or inputs.get("test_dir_override") or config.get("test_dir")),
        gpus=gpus,
        max_parallel=int(execution.get("max_parallel") or len(gpus) or 1),
        launch_delay_sec=float(execution.get("launch_delay_sec", 1.2)),
        poll_sec=float(execution.get("poll_sec", 10.0)),
        abort_on_failure=bool(execution.get("abort_on_failure", True)),
        noises=_float_list(sweep.get("noises", list(range(11)))),
        include_bounds=bool(benchmark.get("include_bounds", True)),
        include_core=bool(benchmark.get("include_core", True)),
        include_init=bool(benchmark.get("include_init", True)),
        fusion_method=str(inference.get("fusion_method", "late")),
        max_cav_override=int(inference.get("max_cav_override", 2)),
        save_vis_interval=int(inference.get("save_vis_interval", 0)),
        num_workers=int(inference.get("num_workers", 0)),
        max_eval_samples=int(inference.get("max_eval_samples", 0) or 0),
        eval_sample_start=int(inference.get("eval_sample_start", 0) or 0),
        comm_range=float(inference.get("comm_range", inference.get("comm_range_override", 70.0))),
        comm_range_gating=str(inference.get("comm_range_gating", "clean")),
        pose_device=str(inference.get("pose_device", "cuda")),
        log_interval=int(inference.get("log_interval", 200)),
        note=str(config.get("note") or run_dir.name),
    )


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


def _build_b_published(*, baseline_yaml: Path, core_yaml: Path, init_yaml: Path, out_path: Path) -> Path:
    baseline = _load_ap(baseline_yaml)
    core = _load_ap(core_yaml)
    init = _load_ap(init_yaml)
    pos = init["pos_std_list"]
    rot = init["rot_std_list"]
    for name, data in (("baseline", baseline), ("core", core)):
        if data["pos_std_list"] != pos or data["rot_std_list"] != rot:
            raise RuntimeError(f"B combine failed: {name} noise list mismatch")

    selected: List[str] = []
    ap30: List[float] = []
    ap50: List[float] = []
    ap70: List[float] = []
    rows: List[Dict[str, Any]] = []
    counts = {"baseline": 0, "init": 0, "core": 0}
    tie_rank = {"baseline": 0, "init": 1, "core": 2}
    for idx, noise in enumerate(pos):
        candidates = [
            ("baseline", baseline["ap50"][idx], baseline),
            ("init", init["ap50"][idx], init),
            ("core", core["ap50"][idx], core),
        ]
        origin, _score, data = max(candidates, key=lambda x: (float(x[1]), tie_rank[x[0]]))
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


def _plot_yaml_sources(run_dir: Path, name: str, sources: Mapping[str, str]) -> Dict[str, str]:
    paths = {label: Path(path) for label, path in sources.items() if path and Path(path).exists()}
    if not paths:
        return {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, str] = {}
    for metric in ("ap30", "ap50", "ap70"):
        plt.figure(figsize=(8, 4.6))
        for label, path in paths.items():
            data = _read_yaml(path)
            xs = [float(x) for x in (data.get("pos_std_list") or [])]
            ys = [float(x) for x in (data.get(metric) or [])]
            points = [
                (float(x), float(y))
                for x, y in zip(xs, ys)
                if math.isfinite(float(x)) and math.isfinite(float(y))
            ]
            points.sort(key=lambda item: item[0])
            if not points:
                continue
            plt.plot([x for x, _ in points], [y for _, y in points], marker="o", linewidth=1.8, markersize=3.5, label=label)
        plt.xlabel("pose noise std (m / deg, paired)")
        plt.ylabel(metric.upper())
        plt.title(f"{name} {metric.upper()}")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.xlim(0, 10)
        plt.ylim(0, 1.0)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        path = plots_dir / f"{name}_{metric}.png"
        plt.savefig(path, dpi=160)
        plt.close()
        out[f"{name}_{metric}"] = str(path)
    return out


def _run_pubmap_opv2v_benchmark_ab(
    config: Mapping[str, Any],
    *,
    run_dir: Path,
    stage_dirs: Mapping[str, Path],
    detection_record: Mapping[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    started = time.time()
    opt = _pubmap_ab_options(config, detection_record, run_dir)
    jobs = _build_pubmap_ab_jobs(opt)
    commands_text = _commands_for_jobs(jobs)
    (stage_dirs["stage"] / "commands.sh").write_text(commands_text, encoding="utf-8")
    (run_dir / "commands.sh").write_text(commands_text, encoding="utf-8")
    records = _run_job_graph(
        jobs=jobs,
        run_dir=run_dir,
        logs_dir=stage_dirs["logs"],
        dry_run=dry_run,
        gpus=opt.gpus,
        max_parallel=opt.max_parallel,
        launch_delay_sec=opt.launch_delay_sec,
        poll_sec=opt.poll_sec,
        abort_on_failure=opt.abort_on_failure,
    )

    published_outputs: List[Dict[str, Any]] = []
    benchmark_a_sources: Dict[str, str] = {}
    benchmark_b_sources: Dict[str, str] = {}
    plot_paths: Dict[str, str] = {}
    artifacts = [str(stage_dirs["stage"] / "commands.sh"), str(run_dir / "commands.sh")]

    if not dry_run:
        baseline = records.get("baseline_none", {}).get("yaml_path")
        single = records.get("single_ego_only", {}).get("yaml_path")
        oracle = records.get("oracle_gt", {}).get("yaml_path")
        core = records.get("v2xregpp_core", {}).get("yaml_path")
        init = records.get("v2xregpp_init", {}).get("yaml_path")
        single_flat = ""
        b_yaml = ""
        if single:
            single_flat_path = _synthesize_single_flat(Path(str(single)), opt.noises, run_dir / "AP030507_single_flat_reference.yaml")
            single_flat = str(single_flat_path)
            published_outputs.append({"type": "single_flat_reference", "yaml_path": single_flat})
        if baseline and core and init:
            b_yaml_path = _build_b_published(
                baseline_yaml=Path(str(baseline)),
                core_yaml=Path(str(core)),
                init_yaml=Path(str(init)),
                out_path=run_dir / "AP030507_v2xregpp_benchmarkB_published.yaml",
            )
            b_yaml = str(b_yaml_path)
            published_outputs.append({"type": "benchmark_b_published", "yaml_path": b_yaml})
        benchmark_a_sources = {
            "baseline": str(baseline or ""),
            "single": single_flat,
            "oracle": str(oracle or ""),
            "v2xregpp_core": str(core or ""),
        }
        benchmark_b_sources = {
            "baseline": str(baseline or ""),
            "v2xregpp_core": str(core or ""),
            "v2xregpp_init": str(init or ""),
            "published": b_yaml,
        }
        plot_paths.update(_plot_yaml_sources(run_dir, "benchmarkA", benchmark_a_sources))
        plot_paths.update(_plot_yaml_sources(run_dir, "benchmarkB", benchmark_b_sources))
        artifacts.extend([x["yaml_path"] for x in published_outputs if x.get("yaml_path")])
        artifacts.extend(plot_paths.values())

    legacy_dirs = sorted({str(rec.get("output_dir")) for rec in records.values() if rec.get("output_dir")})
    yaml_paths = sorted({str(rec.get("yaml_path")) for rec in records.values() if rec.get("yaml_path")})
    artifacts.extend(yaml_paths)
    summary = {
        "schema_version": "pubmap_opv2v_benchmark_ab_pipeline_summary_v1",
        "dataset": "OPV2V",
        "modality": "lidar",
        "model_dir": str(opt.model_dir),
        "stage1_result": str(opt.stage1_result),
        "test_dir": str(opt.test_dir),
        "noises": [float(x) for x in opt.noises],
        "max_eval_samples": int(opt.max_eval_samples or 0),
        "records": list(records.values()),
        "published_outputs": published_outputs,
        "benchmark_A_sources": benchmark_a_sources,
        "benchmark_B_sources": benchmark_b_sources,
        "plots": plot_paths,
    }
    write_json(run_dir / "benchmark_summary.json", summary)
    return {
        "schema_version": "benchmark_stage_record_v1",
        "stage": "benchmark",
        "mode": "pubmap_opv2v_benchmark_ab",
        "status": "dry_run" if dry_run else "completed",
        "inputs": dict(config),
        "output": {
            "stage1_result": str(opt.stage1_result),
            "commands_path": str(stage_dirs["stage"] / "commands.sh"),
            "legacy_output_dirs": legacy_dirs,
            "artifacts": sorted(set(artifacts + [str(run_dir / "benchmark_summary.json")])),
        },
        "result": summary,
        "elapsed_sec": float(time.time() - started),
    }


def run_benchmark_stage(
    config: Mapping[str, Any],
    *,
    run_dir: Path,
    detection_record: Mapping[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    stage_dirs = make_stage_dirs(run_dir, "benchmark")
    mode = str(config.get("mode") or "opv2v_benchmark_a_profile").strip()
    if mode == "skip":
        record = {
            "schema_version": "benchmark_stage_record_v1",
            "stage": "benchmark",
            "mode": "skip",
            "status": "skipped",
            "output": {},
            "result": {},
            "inputs": dict(config),
        }
    elif mode == "opv2v_benchmark_a_profile":
        record = _run_opv2v_benchmark_a_profile(
            config,
            run_dir=run_dir,
            stage_dirs=stage_dirs,
            detection_record=detection_record,
            dry_run=dry_run,
        )
    elif mode == "pubmap_opv2v_benchmark_ab":
        record = _run_pubmap_opv2v_benchmark_ab(
            config,
            run_dir=run_dir,
            stage_dirs=stage_dirs,
            detection_record=detection_record,
            dry_run=dry_run,
        )
    else:
        raise ValueError(f"unsupported benchmark stage mode: {mode}")
    write_json(stage_dirs["result"] / "summary.json", record)
    return record
