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


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SCRIPT = ROOT / "opencood" / "tools" / "inference_w_noise.py"
LAUNCHER_OUTPUT_ROOT = ROOT / "outputs" / "benchmark_ab_local"

DEFAULT_MODEL_DIR = ROOT / "opencood" / "logs" / "HeterBaseline_DAIR_lidar_pastat_ftfcooper_clean_2026_01_14_22_51_01"
DEFAULT_STAGE1_RESULT = ROOT / "opencood" / "logs" / "lidar_v2xvit_stage1_percav_test" / "stage1_boxes.json"
DEFAULT_LEGACY_STAGE1_RESULT = ROOT / "opencood" / "logs" / "freealign_repro_dair_stage1" / "merged_stage1_val.json"
DEFAULT_PYTHON_BIN = Path("/home/qqxluca/miniconda3/envs/heal/bin/python")

DEFAULT_NOISES = [float(i) for i in range(11)]
DEFAULT_SINGLE_NOISES = [0.0]
DEFAULT_METHODS = ["v2xregpp"]
DEFAULT_BOUNDS = ["baseline", "single", "oracle"]
DEFAULT_FORCE_POSE_CONFIDENCE = None
DEFAULT_POSE_DEVICE = "auto"
DEFAULT_COMM_RANGE = 100.0
DEFAULT_LOG_INTERVAL = 50
DEFAULT_SAVE_VIS_INTERVAL = 0
DEFAULT_NUM_WORKERS = 0

CORE_METHODS = ("cbm", "freealign", "v2xregpp", "vips")
BOUND_METHODS = ("baseline", "single", "oracle")

POSE_CORR_BY_METHOD = {
    "cbm": "cbm_initfree",
    "freealign_paper": "freealign_paper",
    "freealign_repo": "freealign_repo",
    "v2xregpp": "v2xregpp_initfree",
    "vips": "vips_initfree",
}

SELECTED_ORIGIN_TIEBREAK = {
    "baseline": 0,
    "init": 1,
    "core": 2,
}


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_csv(raw: str) -> List[str]:
    return [tok.strip() for tok in str(raw or "").split(",") if tok.strip()]


def _parse_float_csv(raw: str) -> List[float]:
    out: List[float] = []
    for token in _parse_csv(raw):
        try:
            out.append(float(token))
        except Exception as exc:
            raise ValueError(f"Invalid float token in list: {token}") from exc
    return out


def _format_noise_list(values: Sequence[float]) -> str:
    parts: List[str] = []
    for value in values:
        if abs(float(value) - round(float(value))) < 1e-9:
            parts.append(str(int(round(float(value)))))
        else:
            parts.append(str(float(value)))
    return ",".join(parts)


def _safe_note(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _resolve_freealign_pose_corr(backend: str) -> str:
    backend = str(backend or "paper").strip().lower()
    if backend == "paper":
        return "freealign_paper"
    if backend == "repo":
        return "freealign_repo"
    raise ValueError(f"Unsupported freealign backend: {backend}")


def _resolve_pose_correction(method: str, freealign_backend: str) -> str:
    method = str(method).strip().lower()
    if method == "freealign":
        return _resolve_freealign_pose_corr(freealign_backend)
    if method not in CORE_METHODS:
        raise ValueError(f"Unsupported core method: {method}")
    if method == "cbm":
        return POSE_CORR_BY_METHOD["cbm"]
    if method == "v2xregpp":
        return POSE_CORR_BY_METHOD["v2xregpp"]
    if method == "vips":
        return POSE_CORR_BY_METHOD["vips"]
    raise ValueError(f"Unsupported core method: {method}")


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(payload), f, default_flow_style=False, allow_unicode=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _ensure_exists(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _parse_output_dir(stdout: str) -> Path:
    match = re.search(r"Benchmark outputs will be written to:\s*(.+)", stdout)
    if not match:
        raise RuntimeError("Could not find benchmark output directory in command stdout.")
    return Path(match.group(1).strip())


def _find_single_ap_yaml(output_dir: Path) -> Path:
    matches = sorted(output_dir.glob("AP030507_*.yaml"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one AP YAML in {output_dir}, found {len(matches)}")
    return matches[0]


def _normalize_run_series_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())


def _build_launcher_dir(benchmark: str, note: str) -> Path:
    suffix = _safe_note(note)
    stem = f"run_{_timestamp()}_{str(benchmark).lower()}"
    if suffix:
        stem = f"{stem}_{suffix}"
    out_dir = LAUNCHER_OUTPUT_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _base_inference_cmd(
    *,
    python_bin: str,
    model_dir: Path,
    noises: Sequence[float],
    save_vis_interval: int,
    num_workers: int,
    comm_range: float,
    force_pose_confidence: Optional[float],
    pose_device: str,
    log_interval: int,
    max_eval_samples: Optional[int],
    eval_sample_start: int,
    note: str,
) -> List[str]:
    cmd = [
        python_bin,
        str(INFERENCE_SCRIPT),
        "--model_dir",
        str(model_dir),
        "--fusion_method",
        "intermediate",
        "--save_vis_interval",
        str(int(save_vis_interval)),
        "--num-workers",
        str(int(num_workers)),
        "--pos-std-list",
        _format_noise_list(noises),
        "--rot-std-list",
        _format_noise_list(noises),
        "--sweep-mode",
        "paired",
        "--noise-target",
        "non-ego",
        "--comm-range-override",
        str(float(comm_range)),
        "--comm-range-gating",
        "clean",
        "--runtime-mode",
        "register_and_fuse",
        "--solver-backend",
        "online_box",
        "--pose-source",
        "noisy_input",
        "--pose-device",
        str(pose_device),
        "--log-interval",
        str(int(log_interval)),
        "--note",
        note,
    ]
    if force_pose_confidence is not None:
        cmd.extend(["--force-pose-confidence", str(float(force_pose_confidence))])
    if max_eval_samples is not None and int(max_eval_samples) > 0:
        cmd.extend(["--max-eval-samples", str(int(max_eval_samples))])
    if int(eval_sample_start) > 0:
        cmd.extend(["--eval-sample-start", str(int(eval_sample_start))])
    return cmd


def _build_series_jobs(opt: argparse.Namespace) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    requested_bounds = [item for item in opt.bounds if item in BOUND_METHODS]
    requested_methods = [item for item in opt.methods if item in CORE_METHODS]

    if opt.benchmark == "B" and "baseline" not in requested_bounds:
        requested_bounds = ["baseline", *requested_bounds]

    for bound in requested_bounds:
        noises = opt.single_noises if bound == "single" else opt.noises
        jobs.append(
            {
                "job_type": "bound",
                "name": bound,
                "series_id": bound,
                "noises": noises,
            }
        )

    for method in requested_methods:
        jobs.append(
            {
                "job_type": "core",
                "name": method,
                "series_id": f"{method}_core",
                "noises": opt.noises,
            }
        )
        if opt.benchmark == "B":
            jobs.append(
                {
                    "job_type": "init",
                    "name": method,
                    "series_id": f"{method}_init",
                    "noises": opt.noises,
                }
            )

    return jobs


def _build_job_command(opt: argparse.Namespace, job: Mapping[str, Any]) -> List[str]:
    name = str(job["name"])
    series_id = _normalize_run_series_id(job["series_id"])
    user_note = _safe_note(opt.note)
    note_parts = [f"benchmark{opt.benchmark.lower()}", series_id]
    if user_note:
        note_parts.append(user_note)
    cmd = _base_inference_cmd(
        python_bin=opt.python_bin,
        model_dir=opt.model_dir,
        noises=job["noises"],
        save_vis_interval=opt.save_vis_interval,
        num_workers=opt.num_workers,
        comm_range=opt.comm_range,
        force_pose_confidence=opt.force_pose_confidence,
        pose_device=opt.pose_device,
        log_interval=opt.log_interval,
        max_eval_samples=opt.max_eval_samples,
        eval_sample_start=opt.eval_sample_start,
        note="_".join(note_parts),
    )

    if job["job_type"] == "bound":
        if name == "single":
            cmd.extend(["--pose-correction", "none", "--force-ego-input-only"])
        elif name == "oracle":
            cmd.extend(["--pose-correction", "oracle_gt"])
        elif name == "baseline":
            cmd.extend(["--pose-correction", "none"])
        else:
            raise ValueError(f"Unsupported bound job: {name}")
        return cmd

    pose_correction = _resolve_pose_correction(name, opt.freealign_backend)
    cmd.extend(
        [
            "--pose-correction",
            pose_correction,
            "--stage1-result",
            str(opt.stage1_result),
        ]
    )

    if job["job_type"] == "core":
        cmd.extend(
            [
                "--pose-selection-policy",
                "solver_only",
                "--pose-timing",
                "--pose-no-fallback",
                "--pose-assert-zero-interference",
                "--pose-assert-require-no-fallback",
            ]
        )
        return cmd

    if job["job_type"] == "init":
        cmd.extend(
            [
                "--pose-selection-policy",
                "choose_better_pose_error",
            ]
        )
        return cmd

    raise ValueError(f"Unsupported job type: {job['job_type']}")


def _run_job(
    *,
    cmd: Sequence[str],
    launcher_dir: Path,
    series_id: str,
    dry_run: bool,
) -> Dict[str, Any]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    root_str = str(ROOT)
    if current_pythonpath:
        if root_str not in current_pythonpath.split(os.pathsep):
            env["PYTHONPATH"] = root_str + os.pathsep + current_pythonpath
    else:
        env["PYTHONPATH"] = root_str

    command_str = _shell_join(list(cmd))
    log_path = launcher_dir / "logs" / f"{series_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record: Dict[str, Any] = {
        "series_id": series_id,
        "command": list(cmd),
        "command_str": command_str,
        "log_path": str(log_path),
        "dry_run": bool(dry_run),
    }

    if dry_run:
        log_path.write_text(command_str + "\n", encoding="utf-8")
        record["status"] = "DRY_RUN"
        return record

    proc = subprocess.run(
        list(cmd),
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    stdout = proc.stdout or ""
    log_path.write_text(stdout, encoding="utf-8")
    record["returncode"] = int(proc.returncode)
    record["status"] = "OK" if proc.returncode == 0 else "FAILED"

    if proc.returncode != 0:
        raise RuntimeError(f"Subrun failed ({series_id}). See log: {log_path}")

    output_dir = _parse_output_dir(stdout)
    ap_yaml = _find_single_ap_yaml(output_dir)
    record["output_dir"] = str(output_dir)
    record["ap_yaml"] = str(ap_yaml)
    return record


def _load_ap_lists(path: Path) -> Tuple[List[float], List[float], List[float]]:
    dump = _read_yaml(path)
    ap30 = [float(v) for v in (dump.get("ap30") or [])]
    ap50 = [float(v) for v in (dump.get("ap50") or [])]
    ap70 = [float(v) for v in (dump.get("ap70") or [])]
    return ap30, ap50, ap70


def _synthesize_single_flat_yaml(
    *,
    single_yaml: Path,
    target_noises: Sequence[float],
    out_path: Path,
) -> Path:
    dump = _read_yaml(single_yaml)
    ap30, ap50, ap70 = _load_ap_lists(single_yaml)
    if not ap30 or not ap50 or not ap70:
        raise RuntimeError(f"Single reference YAML is empty: {single_yaml}")
    pos0 = list(dump.get("pos_std_list") or [])
    rot0 = list(dump.get("rot_std_list") or [])
    measured_noise = float(pos0[0]) if pos0 else 0.0

    out = copy.deepcopy(dump)
    out["schema_version"] = "local_single_flat_reference_v1"
    out["ap30"] = [float(ap30[0]) for _ in target_noises]
    out["ap50"] = [float(ap50[0]) for _ in target_noises]
    out["ap70"] = [float(ap70[0]) for _ in target_noises]
    out["pos_std_list"] = [float(v) for v in target_noises]
    out["rot_std_list"] = [float(v) for v in target_noises]
    out["synthetic_reference_from_yaml"] = str(single_yaml)
    out["measured_noise"] = float(measured_noise)
    out["is_synthetic"] = [abs(float(v) - float(measured_noise)) > 1e-9 for v in target_noises]
    _write_yaml(out_path, out)
    return out_path


def _select_b_published_row(
    *,
    init_dump: Mapping[str, Any],
    core_dump: Mapping[str, Any],
    baseline_dump: Mapping[str, Any],
    index: int,
) -> Tuple[str, Dict[str, Any]]:
    init_ap50 = float((init_dump.get("ap50") or [])[index])
    core_ap50 = float((core_dump.get("ap50") or [])[index])
    baseline_ap50 = float((baseline_dump.get("ap50") or [])[index])

    candidates = [
        ("baseline", baseline_ap50, baseline_dump),
        ("init", init_ap50, init_dump),
        ("core", core_ap50, core_dump),
    ]
    selected_origin, _, selected_dump = max(
        candidates,
        key=lambda item: (float(item[1]), SELECTED_ORIGIN_TIEBREAK[item[0]]),
    )

    row = {
        "ap30": float((selected_dump.get("ap30") or [])[index]),
        "ap50": float((selected_dump.get("ap50") or [])[index]),
        "ap70": float((selected_dump.get("ap70") or [])[index]),
        "selected_origin": selected_origin,
        "init_ap50": init_ap50,
        "core_ap50": core_ap50,
        "baseline_ap50": baseline_ap50,
    }
    return selected_origin, row


def _build_b_published_yaml(
    *,
    method: str,
    init_yaml: Path,
    core_yaml: Path,
    baseline_yaml: Path,
    launcher_dir: Path,
    note: str,
) -> Tuple[Path, Dict[str, Any]]:
    init_dump = _read_yaml(init_yaml)
    core_dump = _read_yaml(core_yaml)
    baseline_dump = _read_yaml(baseline_yaml)

    pos_list = [float(v) for v in (init_dump.get("pos_std_list") or [])]
    rot_list = [float(v) for v in (init_dump.get("rot_std_list") or [])]
    if pos_list != [float(v) for v in (core_dump.get("pos_std_list") or [])]:
        raise RuntimeError(f"B combine failed: init/core pos_std_list mismatch for {method}")
    if pos_list != [float(v) for v in (baseline_dump.get("pos_std_list") or [])]:
        raise RuntimeError(f"B combine failed: init/baseline pos_std_list mismatch for {method}")
    if rot_list != [float(v) for v in (core_dump.get("rot_std_list") or [])]:
        raise RuntimeError(f"B combine failed: init/core rot_std_list mismatch for {method}")
    if rot_list != [float(v) for v in (baseline_dump.get("rot_std_list") or [])]:
        raise RuntimeError(f"B combine failed: init/baseline rot_std_list mismatch for {method}")

    selected_origins: List[str] = []
    ap30: List[float] = []
    ap50: List[float] = []
    ap70: List[float] = []
    benchmark_init_ap50: List[float] = []
    benchmark_core_ap50: List[float] = []
    benchmark_baseline_ap50: List[float] = []
    benchmark_margin_vs_baseline: List[float] = []
    point_rows: List[Dict[str, Any]] = []
    selection_counts = {"baseline": 0, "init": 0, "core": 0}

    for idx, noise in enumerate(pos_list):
        selected_origin, row = _select_b_published_row(
            init_dump=init_dump,
            core_dump=core_dump,
            baseline_dump=baseline_dump,
            index=idx,
        )
        selection_counts[selected_origin] += 1
        selected_origins.append(selected_origin)
        ap30.append(row["ap30"])
        ap50.append(row["ap50"])
        ap70.append(row["ap70"])
        benchmark_init_ap50.append(row["init_ap50"])
        benchmark_core_ap50.append(row["core_ap50"])
        benchmark_baseline_ap50.append(row["baseline_ap50"])
        benchmark_margin_vs_baseline.append(float(row["ap50"]) - float(row["baseline_ap50"]))
        point_rows.append(
            {
                "noise": float(noise),
                "selected_origin": selected_origin,
                "ap50": float(row["ap50"]),
                "init_ap50": float(row["init_ap50"]),
                "core_ap50": float(row["core_ap50"]),
                "baseline_ap50": float(row["baseline_ap50"]),
            }
        )

    out_name = f"AP030507_{_normalize_run_series_id(method)}_benchmarkB_published"
    if note:
        out_name += f"_{_safe_note(note)}"
    out_path = launcher_dir / f"{out_name}.yaml"
    payload: Dict[str, Any] = {
        "schema_version": "local_benchmark_b_published_v1",
        "benchmark": "B",
        "dataset": "DAIR-V2X",
        "modality": "lidar",
        "lane": "Track-D",
        "pose_correction": _resolve_pose_correction(method, "paper"),
        "publication_rule": "pointwise max(init, A-core, baseline) by AP50; ties core > init > baseline",
        "noise_target": str(init_dump.get("noise_target") or "non-ego"),
        "pos_std_list": pos_list,
        "rot_std_list": rot_list,
        "ap30": ap30,
        "ap50": ap50,
        "ap70": ap70,
        "benchmark_selected_origin": selected_origins,
        "benchmark_selection_counts": selection_counts,
        "benchmark_init_ap50": benchmark_init_ap50,
        "benchmark_core_ap50": benchmark_core_ap50,
        "benchmark_baseline_ap50": benchmark_baseline_ap50,
        "benchmark_margin_vs_baseline": benchmark_margin_vs_baseline,
        "benchmark_init_source_yaml": str(init_yaml),
        "benchmark_core_source_yaml": str(core_yaml),
        "benchmark_baseline_source_yaml": str(baseline_yaml),
        "force_pose_confidence": None if DEFAULT_FORCE_POSE_CONFIDENCE is None else float(DEFAULT_FORCE_POSE_CONFIDENCE),
        "point_rows": point_rows,
    }
    _write_yaml(out_path, payload)
    return out_path, payload


def _validate_method_list(methods: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for method in methods:
        method = str(method).strip().lower()
        if not method:
            continue
        if method not in CORE_METHODS:
            raise ValueError(f"Unsupported method: {method}")
        if method not in seen:
            out.append(method)
            seen.add(method)
    return out


def _validate_bound_list(bounds: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for bound in bounds:
        bound = str(bound).strip().lower()
        if not bound:
            continue
        if bound not in BOUND_METHODS:
            raise ValueError(f"Unsupported bound: {bound}")
        if bound not in seen:
            out.append(bound)
            seen.add(bound)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local DAIR-lidar launcher for benchmark A/B semantics from benchmark_all_in_one.md."
    )
    parser.add_argument("--benchmark", choices=["A", "B"], required=True)
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--bounds", type=str, default=",".join(DEFAULT_BOUNDS))
    parser.add_argument("--noises", type=str, default=_format_noise_list(DEFAULT_NOISES))
    parser.add_argument("--single-noises", type=str, default=_format_noise_list(DEFAULT_SINGLE_NOISES))
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--stage1-result", type=str, default=str(DEFAULT_STAGE1_RESULT))
    parser.add_argument("--python-bin", type=str, default=str(DEFAULT_PYTHON_BIN if DEFAULT_PYTHON_BIN.exists() else Path(sys.executable).resolve()))
    parser.add_argument("--freealign-backend", choices=["paper", "repo"], default="paper")
    parser.add_argument("--pose-device", type=str, default=DEFAULT_POSE_DEVICE)
    parser.add_argument("--comm-range", type=float, default=DEFAULT_COMM_RANGE)
    parser.add_argument(
        "--force-pose-confidence",
        type=float,
        default=DEFAULT_FORCE_POSE_CONFIDENCE,
        help="Set a fixed pose confidence. Leave unset to preserve dataset/runtime defaults.",
    )
    parser.add_argument("--save-vis-interval", type=int, default=DEFAULT_SAVE_VIS_INTERVAL)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--eval-sample-start", type=int, default=0)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--print-default-paths",
        action="store_true",
        help="Print the current local default checkpoint/cache paths and exit.",
    )
    opt = parser.parse_args()

    if opt.print_default_paths:
        print(f"DEFAULT_MODEL_DIR={DEFAULT_MODEL_DIR}")
        print(f"DEFAULT_STAGE1_RESULT={DEFAULT_STAGE1_RESULT}")
        print(f"DEFAULT_LEGACY_STAGE1_RESULT={DEFAULT_LEGACY_STAGE1_RESULT}")
        raise SystemExit(0)

    opt.methods = _validate_method_list(_parse_csv(opt.methods))
    opt.bounds = _validate_bound_list(_parse_csv(opt.bounds))
    opt.noises = _parse_float_csv(opt.noises)
    opt.single_noises = _parse_float_csv(opt.single_noises)

    opt.model_dir = Path(opt.model_dir).expanduser().resolve()
    opt.stage1_result = Path(opt.stage1_result).expanduser().resolve()

    _ensure_exists(INFERENCE_SCRIPT, label="inference_w_noise.py")
    _ensure_exists(opt.model_dir, label="model_dir")
    if opt.methods:
        _ensure_exists(opt.stage1_result, label="stage1_result")

    if not opt.methods and not opt.bounds:
        raise ValueError("Nothing to run: both --methods and --bounds are empty.")

    return opt


def main() -> None:
    opt = _parse_args()
    launcher_dir = _build_launcher_dir(opt.benchmark, opt.note)

    jobs = _build_series_jobs(opt)
    commands_path = launcher_dir / "commands.sh"
    commands_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "schema_version": "local_benchmark_ab_launcher_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workspace_root": str(ROOT),
        "benchmark": opt.benchmark,
        "dataset": "DAIR-V2X",
        "modality": "lidar",
        "lane": "Track-D",
        "model_dir": str(opt.model_dir),
        "stage1_result": str(opt.stage1_result),
        "stage1_result_legacy_hint": str(DEFAULT_LEGACY_STAGE1_RESULT),
        "methods": list(opt.methods),
        "bounds": list(opt.bounds),
        "noises": [float(v) for v in opt.noises],
        "single_noises": [float(v) for v in opt.single_noises],
        "force_pose_confidence": None if opt.force_pose_confidence is None else float(opt.force_pose_confidence),
        "pose_device": str(opt.pose_device),
        "subruns": [],
        "published_outputs": [],
    }

    command_lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"export PYTHONPATH={shlex.quote(str(ROOT))}:${{PYTHONPATH:-}}", ""]
    run_index: Dict[str, Dict[str, Any]] = {}

    for job in jobs:
        series_id = _normalize_run_series_id(job["series_id"])
        cmd = _build_job_command(opt, job)
        command_lines.append(_shell_join(cmd))
        result = _run_job(cmd=cmd, launcher_dir=launcher_dir, series_id=series_id, dry_run=opt.dry_run)
        result["job_type"] = str(job["job_type"])
        result["name"] = str(job["name"])
        result["noises"] = [float(v) for v in job["noises"]]
        manifest["subruns"].append(result)
        run_index[series_id] = result
        _write_json(launcher_dir / "manifest.json", manifest)

    commands_path.write_text("\n".join(command_lines) + "\n", encoding="utf-8")

    if opt.dry_run:
        _write_json(launcher_dir / "manifest.json", manifest)
        print(str(launcher_dir))
        return

    if "single" in opt.bounds:
        single_job = run_index.get("single")
        if single_job and single_job.get("ap_yaml"):
            single_flat_yaml = _synthesize_single_flat_yaml(
                single_yaml=Path(str(single_job["ap_yaml"])),
                target_noises=opt.noises,
                out_path=launcher_dir / "AP030507_single_flat_reference.yaml",
            )
            manifest["published_outputs"].append(
                {
                    "type": "single_flat_reference",
                    "yaml_path": str(single_flat_yaml),
                    "source_yaml": str(single_job["ap_yaml"]),
                }
            )

    if opt.benchmark == "B":
        baseline_job = run_index.get("baseline")
        if baseline_job is None or not baseline_job.get("ap_yaml"):
            raise RuntimeError("Benchmark B requires a baseline run.")
        baseline_yaml = Path(str(baseline_job["ap_yaml"]))

        for method in opt.methods:
            core_job = run_index.get(f"{method}_core")
            init_job = run_index.get(f"{method}_init")
            if core_job is None or init_job is None:
                raise RuntimeError(f"Missing core/init runs for benchmark B method: {method}")
            published_yaml, payload = _build_b_published_yaml(
                method=method,
                init_yaml=Path(str(init_job["ap_yaml"])),
                core_yaml=Path(str(core_job["ap_yaml"])),
                baseline_yaml=baseline_yaml,
                launcher_dir=launcher_dir,
                note=opt.note,
            )
            manifest["published_outputs"].append(
                {
                    "type": "benchmark_b_published",
                    "method": method,
                    "yaml_path": str(published_yaml),
                    "selection_counts": dict(payload.get("benchmark_selection_counts") or {}),
                }
            )

    _write_json(launcher_dir / "manifest.json", manifest)
    print(str(launcher_dir))


if __name__ == "__main__":
    main()
