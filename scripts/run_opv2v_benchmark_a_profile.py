#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]
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


def _load_model_config(model_dir: Path) -> Dict[str, Any]:
    cfg_path = model_dir / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.Loader) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected model config mapping: {cfg_path}")
    return data


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


def _extract_single_yaml_point(path: Path) -> Dict[str, Any]:
    data = _read_yaml(path)
    out: Dict[str, Any] = {
        "source_yaml": str(path),
        "ap30": None,
        "ap50": None,
        "ap70": None,
        "samples": None,
        "infer_sec": None,
        "infer_fps": None,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
        "pose_sec": None,
        "pose_solver_avg_time_sec": None,
        "pose_solver_applied": None,
        "pose_solver_samples": None,
    }
    for key in ("ap30", "ap50", "ap70"):
        vals = data.get(key) or []
        if vals:
            out[key] = float(vals[0])
    timing = data.get("timing_stats") or []
    if isinstance(timing, list) and timing:
        t0 = timing[0] or {}
        out["samples"] = int(t0.get("samples") or 0)
        out["infer_sec"] = None if t0.get("infer_sec") is None else float(t0.get("infer_sec"))
        out["infer_fps"] = None if t0.get("infer_fps") is None else float(t0.get("infer_fps"))
        out["pose_sec"] = None if t0.get("pose_sec") is None else float(t0.get("pose_sec"))
        mem = t0.get("cuda_memory") or {}
        if isinstance(mem, dict):
            out["cuda_peak_allocated_bytes"] = mem.get("peak_allocated_bytes")
            out["cuda_peak_reserved_bytes"] = mem.get("peak_reserved_bytes")
    solver_stats = data.get("pose_solver_only_stats") or []
    if isinstance(solver_stats, list) and solver_stats:
        s0 = solver_stats[0] or {}
        out["samples"] = int(s0.get("samples") or 0)
        out["pose_solver_avg_time_sec"] = None if s0.get("avg_time_sec") is None else float(s0.get("avg_time_sec"))
        out["pose_solver_applied"] = None if s0.get("applied") is None else int(s0.get("applied"))
        out["pose_solver_samples"] = int(s0.get("samples") or 0)
        mem = s0.get("cuda_memory") or {}
        if isinstance(mem, dict):
            out["cuda_peak_allocated_bytes"] = mem.get("peak_allocated_bytes")
            out["cuda_peak_reserved_bytes"] = mem.get("peak_reserved_bytes")
    rel_stats = data.get("rel_error_stats") or []
    if isinstance(rel_stats, list) and rel_stats:
        r0 = rel_stats[0] or {}
        rel_trans = r0.get("rel_trans_m") or {}
        rel_yaw = r0.get("rel_yaw_deg") or {}
        rel_succ = r0.get("rel_success_at_m") or {}
        for stat in ("mean", "median", "p90"):
            out[f"rel_trans_{stat}_m"] = rel_trans.get(stat)
            out[f"rel_yaw_{stat}_deg"] = rel_yaw.get(stat)
        for thr in ("1", "2", "3", "5", "10"):
            out[f"success_at_{thr}m"] = rel_succ.get(thr)
    return out


def _canonical_core_anchor(noise: float, measured: Sequence[float]) -> float:
    n = float(noise)
    measured_set = {float(x) for x in measured}
    if not {0.0, 7.0, 10.0}.issubset(measured_set):
        return min(measured_set, key=lambda x: abs(x - n))
    if n <= 3.0:
        return 0.0
    if n <= 8.0:
        return 7.0
    return 10.0


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return {"count": 0}
    def pct(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        idx = (len(vals) - 1) * p / 100.0
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)
    return {
        "count": len(vals),
        "min": vals[0],
        "max": vals[-1],
        "mean": sum(vals) / len(vals),
        "median": pct(50.0),
        "p90": pct(90.0),
        "p95": pct(95.0),
        "sum": sum(vals),
    }


def _array_nbytes(raw: Any) -> int:
    if raw is None:
        return 0
    try:
        import numpy as np
        arr = np.asarray(raw, dtype=np.float32)
        return int(arr.size * 4)
    except Exception:
        return 0


def _stage1_payload_stats(stage1_path: Path) -> Dict[str, Any]:
    with stage1_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    non_ego_payloads: List[float] = []
    all_agent_payloads: List[float] = []
    non_ego_counts: List[float] = []
    box_counts_non_ego: List[float] = []
    for _, entry in sorted(data.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0])):
        if not isinstance(entry, dict):
            continue
        cav_ids = entry.get("cav_id_list") or []
        cav_count = len(cav_ids) if isinstance(cav_ids, list) else 0
        non_ego_counts.append(float(max(0, cav_count - 1)))
        pred = entry.get("pred_corner3d_np_list") or []
        unc = entry.get("uncertainty_np_list") or []
        score = entry.get("pred_score_np_list") or []
        non_ego_bytes = 0
        all_bytes = 0
        non_ego_boxes = 0
        if isinstance(pred, list):
            for idx, boxes in enumerate(pred):
                b = _array_nbytes(boxes)
                if isinstance(unc, list) and idx < len(unc):
                    b += _array_nbytes(unc[idx])
                if isinstance(score, list) and idx < len(score):
                    b += _array_nbytes(score[idx])
                all_bytes += b
                if idx > 0:
                    non_ego_bytes += b
                    try:
                        non_ego_boxes += len(boxes)
                    except Exception:
                        pass
        non_ego_payloads.append(float(non_ego_bytes))
        all_agent_payloads.append(float(all_bytes))
        box_counts_non_ego.append(float(non_ego_boxes))
    return {
        "stage1_cache_file_bytes": int(stage1_path.stat().st_size),
        "entries": len(data) if isinstance(data, dict) else None,
        "non_ego_cav_count": _stats(non_ego_counts),
        "non_ego_box_count": _stats(box_counts_non_ego),
        "non_ego_box_uncertainty_score_payload_bytes": _stats(non_ego_payloads),
        "all_agent_box_uncertainty_score_payload_bytes": _stats(all_agent_payloads),
    }


def _feature_payload_proxy(model_dir: Path, stage1_payload: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = _load_model_config(model_dir)
    model_args = ((cfg.get("model") or {}).get("args") or {})
    heter = cfg.get("heter") or {}
    modality_setting = heter.get("modality_setting") or {}
    if modality_setting:
        ego_modality = str((model_args.get("ego_modality") or heter.get("ego_modality") or "")).split("&")[0]
        modality_name = ego_modality if ego_modality in modality_setting else sorted(modality_setting.keys())[0]
        modal = modality_setting.get(modality_name) or {}
        sensor_type = str(modal.get("sensor_type") or "").lower()
        if sensor_type == "camera":
            encoder_args = modal.get("encoder_args") or {}
            grid_conf = encoder_args.get("grid_conf") or {}
            xbound = grid_conf.get("xbound") or [-51.2, 51.2, 0.4]
            ybound = grid_conf.get("ybound") or [-51.2, 51.2, 0.4]
            try:
                width = int(round((float(xbound[1]) - float(xbound[0])) / float(xbound[2])))
                height = int(round((float(ybound[1]) - float(ybound[0])) / float(ybound[2])))
            except Exception:
                width, height = 256, 256
            channels = int(encoder_args.get("img_features") or 64)
            shrink = modal.get("shrink_header") or {}
            shrink_stride = 1
            try:
                stride_raw = shrink.get("stride", 1)
                if isinstance(stride_raw, (list, tuple)):
                    shrink_stride = int(stride_raw[0])
                else:
                    shrink_stride = int(stride_raw)
            except Exception:
                shrink_stride = 1
            if shrink.get("dim"):
                try:
                    dim_raw = shrink.get("dim")
                    channels = int(dim_raw[0] if isinstance(dim_raw, (list, tuple)) else dim_raw)
                except Exception:
                    channels = int(model_args.get("in_head") or channels)
            backbone = modal.get("backbone_args") or {}
            upsample = backbone.get("upsample_strides") or []
            feature_stride = 1
            if upsample:
                try:
                    min_upsample = min(float(x) for x in upsample)
                    if min_upsample > 0:
                        feature_stride = max(1, int(round(1.0 / min_upsample)))
                except Exception:
                    feature_stride = 1
            feature_stride *= max(1, shrink_stride)
            h = int(math.ceil(float(height) / float(feature_stride)))
            w = int(math.ceil(float(width) / float(feature_stride)))
            per_non_ego = int(channels * h * w * 4)
            non_ego_mean = float(((stage1_payload.get("non_ego_cav_count") or {}).get("mean") or 0.0))
            return {
                "interpretation": "camera/LSS BEV feature payload proxy for intermediate fusion; not packet capture",
                "feature_dtype": "float32",
                "sensor_type": "camera",
                "modality_name": modality_name,
                "bev_grid_w_h": [int(width), int(height)],
                "feature_map": {
                    "channels": int(channels),
                    "height": int(h),
                    "width": int(w),
                    "effective_stride": int(feature_stride),
                    "fp32_bytes": int(per_non_ego),
                },
                "per_non_ego_cav_bytes": int(per_non_ego),
                "mean_non_ego_cavs_from_stage1_cache": non_ego_mean,
                "mean_per_sample_bytes": float(per_non_ego * non_ego_mean),
            }

    bb = model_args.get("base_bev_backbone") or {}
    scatter = model_args.get("point_pillar_scatter") or {}
    grid = scatter.get("grid_size")
    try:
        grid_list = [int(x) for x in list(grid)]
        width, height = int(grid_list[0]), int(grid_list[1])
    except Exception:
        width, height = 704, 200
    layer_strides = [int(x) for x in (bb.get("layer_strides") or [2, 2, 2])]
    num_filters = [int(x) for x in (bb.get("num_filters") or [64, 128, 256])]
    h, w = int(height), int(width)
    multiscale = []
    for channels, stride in zip(num_filters, layer_strides):
        h = int(math.ceil(h / float(stride)))
        w = int(math.ceil(w / float(stride)))
        bytes_fp32 = int(channels * h * w * 4)
        multiscale.append({"channels": channels, "height": h, "width": w, "fp32_bytes": bytes_fp32})
    per_non_ego = int(sum(x["fp32_bytes"] for x in multiscale))
    non_ego_mean = float(((stage1_payload.get("non_ego_cav_count") or {}).get("mean") or 0.0))
    return {
        "interpretation": "dense multiscale BEV feature payload proxy for intermediate fusion; not packet capture",
        "feature_dtype": "float32",
        "grid_size_w_h": [int(width), int(height)],
        "multiscale_feature_maps": multiscale,
        "per_non_ego_cav_bytes": per_non_ego,
        "mean_non_ego_cavs_from_stage1_cache": non_ego_mean,
        "mean_per_sample_bytes": float(per_non_ego * non_ego_mean),
    }


def _camera_input_payload_stats(test_dir: Path, depth_root: Path, stage1_path: Path) -> Dict[str, Any]:
    with stage1_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    image_bytes_all: List[float] = []
    depth_bytes_all: List[float] = []
    image_depth_bytes_all: List[float] = []
    image_bytes_non_ego: List[float] = []
    depth_bytes_non_ego: List[float] = []
    image_depth_bytes_non_ego: List[float] = []
    missing: List[str] = []
    test_dir = Path(test_dir)
    depth_root = Path(depth_root)
    for _, entry in sorted(data.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0])):
        if not isinstance(entry, dict):
            continue
        scenario = str(entry.get("scenario_name") or "")
        timestamp = str(entry.get("timestamp") or entry.get("veh_frame_id") or "")
        cav_ids = entry.get("cav_id_list") or []
        sample_img_all = 0
        sample_depth_all = 0
        sample_img_non_ego = 0
        sample_depth_non_ego = 0
        for cav_idx, cav_id in enumerate(cav_ids):
            cav_img = 0
            cav_depth = 0
            for cam_id in range(4):
                image_path = test_dir / scenario / str(cav_id) / f"{timestamp}_camera{cam_id}.png"
                depth_path = depth_root / scenario / str(cav_id) / f"{timestamp}_camera{cam_id}_depth.npy"
                if image_path.exists():
                    cav_img += int(image_path.stat().st_size)
                else:
                    missing.append(str(image_path))
                if depth_path.exists():
                    cav_depth += int(depth_path.stat().st_size)
                else:
                    missing.append(str(depth_path))
            sample_img_all += cav_img
            sample_depth_all += cav_depth
            if cav_idx > 0:
                sample_img_non_ego += cav_img
                sample_depth_non_ego += cav_depth
        image_bytes_all.append(float(sample_img_all))
        depth_bytes_all.append(float(sample_depth_all))
        image_depth_bytes_all.append(float(sample_img_all + sample_depth_all))
        image_bytes_non_ego.append(float(sample_img_non_ego))
        depth_bytes_non_ego.append(float(sample_depth_non_ego))
        image_depth_bytes_non_ego.append(float(sample_img_non_ego + sample_depth_non_ego))
    return {
        "interpretation": "raw camera PNG + depth NPY file-size proxy. Non-ego values approximate cooperative upstream payload if raw sensors are transmitted.",
        "entries": len(data) if isinstance(data, dict) else None,
        "missing_count": int(len(missing)),
        "missing_examples": missing[:10],
        "all_cav_image_png_bytes": _stats(image_bytes_all),
        "all_cav_depth_npy_bytes": _stats(depth_bytes_all),
        "all_cav_image_plus_depth_bytes": _stats(image_depth_bytes_all),
        "non_ego_image_png_bytes": _stats(image_bytes_non_ego),
        "non_ego_depth_npy_bytes": _stats(depth_bytes_non_ego),
        "non_ego_image_plus_depth_bytes": _stats(image_depth_bytes_non_ego),
    }


def _build_tables_and_summary(opt: argparse.Namespace, run_dir: Path, records: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    measured: Dict[Tuple[str, float], Dict[str, Any]] = {}
    solver_by_downstream: Dict[str, Mapping[str, Any]] = {}
    profile_rows: List[Dict[str, Any]] = []

    for sid, rec in records.items():
        yaml_path_raw = rec.get("yaml_path")
        if not yaml_path_raw:
            continue
        yaml_path = Path(str(yaml_path_raw))
        point = _extract_single_yaml_point(yaml_path)
        line_id = str(rec.get("line_id", ""))
        noise = float(rec.get("noise", 0.0))
        row = {
            "series_id": sid,
            "line_id": line_id,
            "role": str(rec.get("role", "")),
            "job_type": str(rec.get("job_type", "")),
            "noise": noise,
            "status": str(rec.get("status", "")),
            "gpu": str(rec.get("gpu", "")),
            "wall_sec": rec.get("wall_sec"),
            "yaml_path": str(yaml_path),
            "output_dir": str(rec.get("output_dir", "")),
            "override_path": str(rec.get("override_path", "")),
            **point,
        }
        profile_rows.append(row)
        if rec.get("job_type") == "downstream" or rec.get("job_type") == "bound":
            measured[(line_id, noise)] = row
        elif rec.get("job_type") == "solver":
            solver_by_downstream[f"{line_id}_n{_noise_tag(noise)}"] = row

    for row in profile_rows:
        if row.get("job_type") == "downstream":
            key = f"{row['line_id']}_n{_noise_tag(float(row['noise']))}"
            solver = solver_by_downstream.get(key)
            if solver:
                row["paired_solver_series_id"] = solver.get("series_id")
                row["paired_solver_wall_sec"] = solver.get("wall_sec")
                row["paired_solver_cuda_peak_allocated_bytes"] = solver.get("cuda_peak_allocated_bytes")
                row["paired_solver_cuda_peak_reserved_bytes"] = solver.get("cuda_peak_reserved_bytes")

    stage1_payload = _stage1_payload_stats(Path(opt.stage1_result))
    feature_payload = _feature_payload_proxy(Path(opt.model_dir), stage1_payload)
    camera_input_payload = _camera_input_payload_stats(Path(opt.test_dir), Path(opt.depth_root), Path(opt.stage1_result))
    registration_mean_bytes = float(
        (stage1_payload.get("non_ego_box_uncertainty_score_payload_bytes") or {}).get("mean") or 0.0
    )
    feature_mean_bytes = float(feature_payload.get("mean_per_sample_bytes") or 0.0)

    point_rows: List[Dict[str, Any]] = []
    all_noises = [float(x) for x in opt.noises]
    line_order = []
    for line in ("baseline", "single", "oracle", *opt.methods):
        if line in {"baseline", "single", "oracle"}:
            if line in opt.bounds:
                line_order.append(line)
        else:
            line_order.append(line)

    for line in line_order:
        measured_noises = sorted([noise for (line_id, noise) in measured if line_id == line])
        for noise in all_noises:
            if line == "single":
                anchor = 0.0
            elif line == "oracle":
                anchor = float(opt.oracle_anchor_noise)
            elif line in CORE_POSE_CORR:
                anchor = _canonical_core_anchor(noise, measured_noises)
            else:
                anchor = float(noise)
            src = measured.get((line, float(anchor)))
            if not src:
                continue
            is_synth = abs(float(noise) - float(anchor)) > 1e-9
            role = "method" if line in CORE_POSE_CORR else "reference"
            extra_registration = registration_mean_bytes if line in CORE_POSE_CORR else 0.0
            feature_proxy = 0.0 if line == "single" else feature_mean_bytes
            row = {
                "dataset": "OPV2V",
                "noise": float(noise),
                "series_id": line,
                "label": line,
                "role": role,
                "ap30": src.get("ap30"),
                "ap50": src.get("ap50"),
                "ap70": src.get("ap70"),
                "samples": src.get("samples"),
                "is_synthetic": bool(is_synth),
                "measured_noise": float(anchor),
                "source_yaml": src.get("yaml_path"),
                "override_path": src.get("override_path", ""),
                "infer_sec": src.get("infer_sec"),
                "infer_fps": src.get("infer_fps"),
                "wall_sec": src.get("wall_sec"),
                "cuda_peak_allocated_bytes": src.get("cuda_peak_allocated_bytes"),
                "cuda_peak_reserved_bytes": src.get("cuda_peak_reserved_bytes"),
                "registration_extra_payload_bytes_per_sample_proxy": extra_registration,
                "intermediate_feature_payload_bytes_per_sample_proxy": feature_proxy,
                "total_payload_bytes_per_sample_proxy": extra_registration + feature_proxy,
            }
            if line in CORE_POSE_CORR:
                key = f"{line}_n{_noise_tag(float(anchor))}"
                solver = solver_by_downstream.get(key) or {}
                row["source_solver_yaml"] = solver.get("yaml_path", "")
                row["solver_wall_sec"] = solver.get("wall_sec")
                row["solver_cuda_peak_allocated_bytes"] = solver.get("cuda_peak_allocated_bytes")
                row["solver_cuda_peak_reserved_bytes"] = solver.get("cuda_peak_reserved_bytes")
            point_rows.append(row)

    profile_csv = run_dir / "profile_rows.csv"
    _write_csv(profile_csv, profile_rows)
    points_csv = run_dir / "benchmarkA_points.csv"
    _write_csv(points_csv, point_rows)

    overhead_rows = _build_overhead_rows(profile_rows)
    overhead_csv = run_dir / "profile_overhead_vs_baseline.csv"
    _write_csv(overhead_csv, overhead_rows)

    plot_paths = _write_plots(run_dir, point_rows, profile_rows)
    summary = {
        "schema_version": "opv2v_benchmark_a_profile_summary_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "OPV2V",
        "dataset_frames_expected": 2170,
        "run_dir": str(run_dir),
        "model_dir": str(opt.model_dir),
        "stage1_result": str(opt.stage1_result),
        "test_dir": str(opt.test_dir),
        "depth_root": str(opt.depth_root),
        "max_eval_samples": int(opt.max_eval_samples or 0),
        "noises": [float(x) for x in opt.noises],
        "core_measured_noises": [float(x) for x in opt.core_noises],
        "oracle_anchor_noise": float(opt.oracle_anchor_noise),
        "communication_interpretation": {
            "registration_extra_payload": "non-ego stage1 boxes + uncertainty + scores, counted as a payload proxy for registration methods only",
            "intermediate_feature_payload_proxy": "dense multiscale BEV FP32 feature payload proxy for cooperative intermediate fusion; not measured network traffic",
            "single": "force-ego-input-only, so feature payload proxy is reported as 0",
        },
        "stage1_payload_stats": stage1_payload,
        "camera_input_payload_stats": camera_input_payload,
        "intermediate_feature_payload_proxy": feature_payload,
        "artifacts": {
            "profile_rows_csv": str(profile_csv),
            "benchmarkA_points_csv": str(points_csv),
            "profile_overhead_vs_baseline_csv": str(overhead_csv),
            "plots": plot_paths,
        },
        "records": list(records.values()),
    }
    _write_json(run_dir / "profile_summary_full2170.json", summary)
    _write_json(run_dir / "profile_summary.json", summary)
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _build_overhead_rows(profile_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    baseline_by_noise = {
        float(r["noise"]): r
        for r in profile_rows
        if r.get("line_id") == "baseline" and r.get("job_type") == "bound"
    }
    solver_by_key = {
        (str(r.get("line_id")), float(r.get("noise", 0.0))): r
        for r in profile_rows
        if r.get("job_type") == "solver"
    }
    out: List[Dict[str, Any]] = []
    for row in profile_rows:
        if row.get("job_type") not in {"bound", "downstream"}:
            continue
        samples = float(row.get("samples") or 0.0)
        if samples <= 0:
            continue
        noise = float(row.get("noise", 0.0))
        line = str(row.get("line_id", ""))
        solver = solver_by_key.get((line, noise)) if row.get("job_type") == "downstream" else None
        solver_wall = float((solver or {}).get("wall_sec") or 0.0)
        downstream_sec = float(row.get("infer_sec") or 0.0)
        total_sec = solver_wall + downstream_sec
        baseline = baseline_by_noise.get(noise)
        baseline_sec = float((baseline or {}).get("infer_sec") or 0.0)
        out.append(
            {
                "line_id": line,
                "noise": noise,
                "samples": int(samples),
                "solver_wall_sec": solver_wall,
                "downstream_infer_sec": downstream_sec,
                "pipeline_total_sec": total_sec,
                "pipeline_sec_per_sample": total_sec / samples,
                "baseline_infer_sec_same_noise": baseline_sec,
                "baseline_sec_per_sample_same_noise": (baseline_sec / samples) if baseline_sec > 0 else "",
                "delta_sec_per_sample_vs_baseline": ((total_sec - baseline_sec) / samples) if baseline_sec > 0 else "",
                "ratio_vs_baseline": (total_sec / baseline_sec) if baseline_sec > 0 else "",
            }
        )
    return out


def _write_plots(run_dir: Path, point_rows: Sequence[Mapping[str, Any]], profile_rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"error": f"matplotlib unavailable: {type(exc).__name__}: {exc}"}

    line_order = ["baseline", "single", "oracle", "cbm", "freealign", "v2xregpp", "vips"]
    colors = {
        "baseline": "#4C566A",
        "single": "#A3BE8C",
        "oracle": "#D08770",
        "cbm": "#5E81AC",
        "freealign": "#B48EAD",
        "v2xregpp": "#BF616A",
        "vips": "#EBCB8B",
    }
    paths: Dict[str, str] = {}
    for metric in ("ap30", "ap50", "ap70"):
        plt.figure(figsize=(8, 4.6))
        for line in line_order:
            rows = [r for r in point_rows if r.get("series_id") == line and r.get(metric) not in {None, ""}]
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: float(r["noise"]))
            plt.plot(
                [float(r["noise"]) for r in rows],
                [float(r[metric]) for r in rows],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=line,
                color=colors.get(line),
            )
        plt.xlabel("pose noise std (m / deg, paired)")
        plt.ylabel(metric.upper())
        plt.title(f"OPV2V Benchmark A {metric.upper()}")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.xlim(0, 10)
        plt.ylim(0, 1.0)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        out = plots_dir / f"benchmarkA_{metric}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        paths[f"benchmarkA_{metric}"] = str(out)

    def _mean_for_line(line: str, key: str, *, include_solver: bool = False) -> float:
        vals = []
        for r in profile_rows:
            if r.get("line_id") != line:
                continue
            if r.get("job_type") == "solver" and not include_solver:
                continue
            val = r.get(key)
            if val not in {None, ""}:
                vals.append(float(val))
        return sum(vals) / len(vals) if vals else 0.0

    def _bar_plot(name: str, ylabel: str, values: Dict[str, float]) -> None:
        labels = [x for x in line_order if x in values]
        plt.figure(figsize=(8, 4.2))
        plt.bar(labels, [values[x] for x in labels], color=[colors.get(x, "#888888") for x in labels])
        plt.ylabel(ylabel)
        plt.xticks(rotation=25, ha="right")
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        out = plots_dir / f"{name}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        paths[name] = str(out)

    payload_values = {}
    for line in line_order:
        rows = [r for r in point_rows if r.get("series_id") == line]
        vals = [float(r.get("total_payload_bytes_per_sample_proxy") or 0.0) for r in rows]
        if vals:
            payload_values[line] = sum(vals) / len(vals)
    _bar_plot("payload_proxy_bytes_per_sample", "payload proxy bytes / sample", payload_values)

    time_values = {}
    solver_by_key = {
        (str(r.get("line_id")), float(r.get("noise", 0.0))): r
        for r in profile_rows
        if r.get("job_type") == "solver"
    }
    for line in line_order:
        vals = []
        for r in profile_rows:
            if r.get("line_id") != line or r.get("job_type") not in {"bound", "downstream"}:
                continue
            samples = float(r.get("samples") or 0.0)
            if samples <= 0:
                continue
            total = float(r.get("infer_sec") or 0.0)
            solver = solver_by_key.get((line, float(r.get("noise", 0.0))))
            if solver:
                total += float(solver.get("wall_sec") or 0.0)
            vals.append(total / samples)
        if vals:
            time_values[line] = sum(vals) / len(vals)
    _bar_plot("pipeline_sec_per_sample", "pipeline seconds / sample", time_values)

    mem_values = {
        line: _mean_for_line(line, "cuda_peak_allocated_bytes", include_solver=True) / (1024.0 * 1024.0)
        for line in line_order
    }
    mem_values = {k: v for k, v in mem_values.items() if v > 0}
    _bar_plot("cuda_peak_allocated_mib", "CUDA peak allocated MiB", mem_values)
    return paths


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
        summary = _build_tables_and_summary(opt, run_dir, records)
        print("Summary:", summary["artifacts"]["profile_rows_csv"])
        print("Profile summary:", run_dir / "profile_summary_full2170.json")


if __name__ == "__main__":
    main()
