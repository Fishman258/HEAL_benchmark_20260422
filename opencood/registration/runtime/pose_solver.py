from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import os
from time import perf_counter
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from opencood.utils.pose_utils import add_noise_data_dict, attach_pose_confidence, override_lidar_poses
from opencood.utils.transformation_utils import pose_to_tfm


@dataclass
class GTPoseCorrector:
    """
    Override lidar poses with clean (GT) poses from the dataset.
    """

    freeze_ego: bool = True

    def apply(
        self,
        *,
        sample_idx: int,
        cav_id_list: Sequence[Any],
        base_data_dict: MutableMapping[Any, Dict[str, Any]],
        **kwargs,
    ) -> bool:
        updated_any = False
        for cav_id in cav_id_list:
            if cav_id not in base_data_dict:
                continue
            if self.freeze_ego and bool(base_data_dict[cav_id].get("ego", False)):
                continue
            params = base_data_dict[cav_id].get("params") or {}
            pose_clean = params.get("lidar_pose_clean")
            if pose_clean is None:
                continue
            params["lidar_pose"] = list(pose_clean)
            base_data_dict[cav_id]["params"] = params
            updated_any = True
        return updated_any


def _filter_kwargs(cls, payload: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        from dataclasses import fields

        allowed = {f.name for f in fields(cls)}
    except Exception:
        allowed = set()
    return {k: v for k, v in (payload or {}).items() if k in allowed}


def build_pose_corrector(method: str, *, args: Optional[Mapping[str, Any]] = None):
    """
    Build a Stage-1 pose corrector from a method tag and argument dict.
    """
    args = dict(args or {})
    name = str(method or "").lower().strip()

    if name == "v2xregpp":
        from opencood.registration.runtime import Stage1V2XRegPPPoseCorrector

        return Stage1V2XRegPPPoseCorrector(**args)

    if name == "freealign":
        backend = str(args.get("backend") or args.get("mode") or "paper").lower().strip()
        from opencood.registration.runtime import Stage1FreeAlignPoseCorrector, Stage1FreeAlignRepoPoseCorrector
        from opencood.registration.estimators.freealign_paper import FreeAlignPaperConfig
        from opencood.registration.estimators.freealign_repo import FreeAlignRepoConfig

        if backend in {"repo", "released", "match_v7", "match_v7_with_detection"}:
            cfg_kwargs = _filter_kwargs(FreeAlignRepoConfig, args)
            corrector_kwargs = _filter_kwargs(Stage1FreeAlignRepoPoseCorrector, args)
            corrector_kwargs.pop("cfg", None)
            return Stage1FreeAlignRepoPoseCorrector(cfg=FreeAlignRepoConfig(**cfg_kwargs), **corrector_kwargs)

        cfg_kwargs = _filter_kwargs(FreeAlignPaperConfig, args)
        corrector_kwargs = _filter_kwargs(Stage1FreeAlignPoseCorrector, args)
        corrector_kwargs.pop("cfg", None)
        return Stage1FreeAlignPoseCorrector(cfg=FreeAlignPaperConfig(**cfg_kwargs), **corrector_kwargs)

    if name == "vips":
        from opencood.registration.runtime import Stage1VIPSPoseCorrector

        return Stage1VIPSPoseCorrector(**args)

    if name == "cbm":
        from opencood.registration.runtime import Stage1CBMPoseCorrector

        return Stage1CBMPoseCorrector(**args)

    if name == "image_match":
        from dataclasses import fields
        from opencood.registration.estimators.image_matching import ImageMatchingConfig
        from opencood.registration.runtime import Stage1ImageMatchPoseCorrector

        allowed_cfg = {f.name for f in fields(ImageMatchingConfig)}
        cfg_payload = args.get("cfg") or {}
        cfg_kwargs = {k: v for k, v in (cfg_payload or {}).items() if k in allowed_cfg}
        cfg_kwargs.update({k: v for k, v in args.items() if k in allowed_cfg})
        img_cfg = ImageMatchingConfig(**cfg_kwargs)
        corrector_kwargs = {k: v for k, v in args.items() if k not in allowed_cfg}
        corrector_kwargs.pop("cfg", None)
        return Stage1ImageMatchPoseCorrector(cfg=img_cfg, **corrector_kwargs)

    if name == "lidar_reg":
        from dataclasses import fields
        from opencood.registration.estimators.lidar_registration import LidarRegistrationConfig
        from opencood.registration.runtime import Stage1LidarRegPoseCorrector

        allowed_cfg = {f.name for f in fields(LidarRegistrationConfig)}
        cfg_payload = args.get("cfg") or {}
        cfg_kwargs = {k: v for k, v in (cfg_payload or {}).items() if k in allowed_cfg}
        cfg_kwargs.update({k: v for k, v in args.items() if k in allowed_cfg})
        lidar_cfg = LidarRegistrationConfig(**cfg_kwargs)
        corrector_kwargs = {k: v for k, v in args.items() if k not in allowed_cfg}
        corrector_kwargs.pop("cfg", None)
        return Stage1LidarRegPoseCorrector(cfg=lidar_cfg, **corrector_kwargs)

    if name == "pgc":
        from opencood.registration.runtime import Stage1PGCPoseCorrector

        return Stage1PGCPoseCorrector(**args)

    if name == "gt":
        return GTPoseCorrector(freeze_ego=bool(args.get("freeze_ego", True)))

    raise ValueError(f"Unknown pose corrector method: {method}")

def _summ(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "p95": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def _compute_rel_errors(base_data_dict) -> Tuple[list, list]:
    try:
        cav_ids = list(base_data_dict.keys())
        if not cav_ids:
            return [], []
        poses = [base_data_dict[cav_id]["params"]["lidar_pose"] for cav_id in cav_ids]
        poses_clean = [base_data_dict[cav_id]["params"]["lidar_pose_clean"] for cav_id in cav_ids]
        T_world = pose_to_tfm(np.asarray(poses, dtype=np.float64))
        T_world_clean = pose_to_tfm(np.asarray(poses_clean, dtype=np.float64))
        ego_T_world = T_world[0]
        ego_T_world_clean = T_world_clean[0]
    except Exception:
        return [], []
    rel_trans_errors = []
    rel_yaw_errors = []
    for cav_idx in range(1, min(T_world.shape[0], T_world_clean.shape[0])):
        rel = np.linalg.inv(ego_T_world) @ T_world[cav_idx]
        rel_clean = np.linalg.inv(ego_T_world_clean) @ T_world_clean[cav_idx]
        err = np.linalg.inv(rel_clean) @ rel
        rel_trans_errors.append(float(np.linalg.norm(err[:2, 3])))
        yaw = float(np.degrees(np.arctan2(err[1, 0], err[0, 0])))
        rel_yaw_errors.append(abs(yaw))
    return rel_trans_errors, rel_yaw_errors


@dataclass
class PoseOverrideResult:
    overrides: Dict[str, Any]
    metrics: Dict[str, Any]


def run_pose_solver(
    dataset,
    *,
    corrector,
    stage1_result: Optional[Mapping[str, Any]] = None,
    pose_result: Optional[Mapping[str, Any]] = None,
    noise_setting: Optional[Mapping[str, Any]] = None,
    max_samples: Optional[int] = None,
    seed: Optional[int] = 303,
    simple_override_cfg: Optional[Mapping[str, Any]] = None,
    telemetry_path: Optional[str] = None,
    telemetry_context: Optional[Mapping[str, Any]] = None,
    force_ego_only_when_incomplete: bool = False,
) -> PoseOverrideResult:
    """
    Run a pose corrector offline and return a pose override map + metrics.

    This keeps pose correction out of the dataset: corrected poses are cached
    and later injected via `apply_pose_overrides`.
    """
    if seed is not None:
        np.random.seed(int(seed))

    overrides: Dict[str, Any] = {}
    telemetry_f = None
    if telemetry_path:
        tpath = str(telemetry_path)
        parent = os.path.dirname(tpath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if tpath.endswith('.gz'):
            telemetry_f = gzip.open(tpath, 'wt', encoding='utf-8')
        else:
            telemetry_f = open(tpath, 'w', encoding='utf-8')

    rel_trans_errors: list = []
    rel_yaw_errors: list = []
    applied_count = 0
    total_time = 0.0

    max_len = len(dataset)
    if max_samples is not None:
        max_len = min(max_len, int(max_samples))

    for idx in range(max_len):
        base_data_dict = dataset.retrieve_base_data(idx)
        if noise_setting is not None:
            base_data_dict = add_noise_data_dict(base_data_dict, noise_setting)

        if isinstance(simple_override_cfg, Mapping):
            override_lidar_poses(
                base_data_dict,
                mode=str(simple_override_cfg.get("mode", "ego") or "ego"),
                apply_to=str(simple_override_cfg.get("apply_to", "non-ego") or "non-ego"),
                ego_id=next(iter(base_data_dict.keys()), None),
                set_confidence=simple_override_cfg.get("set_confidence"),
            )

        start = perf_counter()
        applied = False
        if corrector is not None:
            if stage1_result is not None:
                applied = bool(
                    corrector.apply(
                        sample_idx=idx,
                        cav_id_list=list(base_data_dict.keys()),
                        base_data_dict=base_data_dict,
                        stage1_result=stage1_result,
                    )
                )
            elif pose_result is not None:
                applied = bool(
                    corrector.apply(
                        sample_idx=idx,
                        cav_id_list=list(base_data_dict.keys()),
                        base_data_dict=base_data_dict,
                        pose_result=pose_result,
                    )
                )
            else:
                applied = bool(
                    corrector.apply(
                        sample_idx=idx,
                        cav_id_list=list(base_data_dict.keys()),
                        base_data_dict=base_data_dict,
                    )
                )
        total_time += float(perf_counter() - start)
        if applied:
            applied_count += 1

        force_ego_only = False
        pair_total = None
        pair_applied = None
        if bool(force_ego_only_when_incomplete):
            try:
                stats = getattr(corrector, "last_stats", None)
                if isinstance(stats, Mapping):
                    pair_total = stats.get("pose_corr_pair_total_count")
                    pair_applied = stats.get("pose_corr_applied_pair_count")
                    pair_total_i = int(round(float(pair_total)))
                    pair_applied_i = int(round(float(pair_applied)))
                    force_ego_only = bool(pair_total_i > 0 and pair_applied_i < pair_total_i)
            except Exception:
                force_ego_only = False

        if telemetry_f is not None:
            rows = getattr(corrector, 'last_telemetry', None)
            if isinstance(rows, list) and rows:
                noise_args = {}
                if isinstance(noise_setting, Mapping):
                    args = noise_setting.get('args')
                    if isinstance(args, Mapping):
                        noise_args = dict(args)
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    rec = dict(row)
                    if telemetry_context:
                        try:
                            rec.update(dict(telemetry_context))
                        except Exception:
                            pass
                    if noise_args:
                        rec['noise_pos_std'] = float(noise_args.get('pos_std', 0.0) or 0.0)
                        rec['noise_rot_std'] = float(noise_args.get('rot_std', 0.0) or 0.0)
                        rec['noise_pos_mean'] = float(noise_args.get('pos_mean', 0.0) or 0.0)
                        rec['noise_rot_mean'] = float(noise_args.get('rot_mean', 0.0) or 0.0)
                        rec['noise_target'] = str(noise_args.get('target', ''))
                    telemetry_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        attach_pose_confidence(base_data_dict)
        cav_id_list = list(base_data_dict.keys())
        poses = [base_data_dict[cav_id]["params"]["lidar_pose"] for cav_id in cav_id_list]
        confs = [base_data_dict[cav_id]["params"].get("pose_confidence", 1.0) for cav_id in cav_id_list]
        overrides[str(idx)] = {
            "cav_id_list": cav_id_list,
            "lidar_pose_pred_np": poses,
            "pose_confidence_np": confs,
        }
        if bool(force_ego_only_when_incomplete):
            overrides[str(idx)]["force_ego_only"] = bool(force_ego_only)
            if pair_total is not None:
                overrides[str(idx)]["pose_corr_pair_total_count"] = pair_total
            if pair_applied is not None:
                overrides[str(idx)]["pose_corr_applied_pair_count"] = pair_applied

        rel_t, rel_yaw = _compute_rel_errors(base_data_dict)
        rel_trans_errors.extend(rel_t)
        rel_yaw_errors.extend(rel_yaw)

    if telemetry_f is not None:
        telemetry_f.close()

    metrics = {
        "samples": int(max_len),
        "applied": int(applied_count),
        "avg_time_sec": float(total_time / max(1, max_len)),
        "rel_trans_m": _summ(rel_trans_errors),
        "rel_yaw_deg": _summ(rel_yaw_errors),
    }
    return PoseOverrideResult(overrides=overrides, metrics=metrics)
