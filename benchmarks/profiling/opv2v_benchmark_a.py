import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import yaml

from benchmarks.plotting.opv2v_benchmark_a import write_plots


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


def _load_model_config(model_dir: Path) -> Dict[str, Any]:
    cfg_path = model_dir / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.Loader) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected model config mapping: {cfg_path}")
    return data


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
    if not measured_set:
        return n
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


def build_tables_and_summary(
    opt: argparse.Namespace,
    run_dir: Path,
    records: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
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
    core_methods = set(str(x) for x in (opt.methods or []))
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
            elif line in core_methods:
                anchor = _canonical_core_anchor(noise, measured_noises)
            else:
                anchor = float(noise)
            src = measured.get((line, float(anchor)))
            if not src:
                continue
            is_synth = abs(float(noise) - float(anchor)) > 1e-9
            role = "method" if line in core_methods else "reference"
            extra_registration = registration_mean_bytes if line in core_methods else 0.0
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
            if line in core_methods:
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

    plot_paths = write_plots(run_dir, point_rows, profile_rows)
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
