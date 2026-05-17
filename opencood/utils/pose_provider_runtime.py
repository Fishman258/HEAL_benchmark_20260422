from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from opencood.utils.common_utils import read_json
from opencood.utils.pose_utils import (
    _confidence_from_value,
    _pose6_from_value,
    _resolve_pose_override_entry,
    load_pose_override_map,
)
from opencood.utils.transformation_utils import (
    get_pairwise_transformation_torch,
    pose_to_tfm,
    tfm_to_pose_torch,
)


try:
    from opencood.extrinsics.pose_correction.online_box_solver import (
        solve_relative_pose_from_stage1_entry,
    )
except Exception:  # pragma: no cover
    solve_relative_pose_from_stage1_entry = None

try:
    from opencood.extrinsics.pose_correction.online_feature_refiner import (
        refine_relative_pose_from_stage1_entry,
    )
except Exception:  # pragma: no cover
    refine_relative_pose_from_stage1_entry = None


_ALLOWED_RUNTIME_MODES = {
    "single_only",
    "fusion_only",
    "register_only",
    "register_and_fuse",
}

_ALLOWED_POSE_SOURCES = {
    "gt",
    "identity",
    "noisy_input",
    "pred",
    "raw",
}

_GPU_STAGE1_METHODS = {"v2xregpp", "freealign", "vips", "cbm"}


# NOTE: Camera datasets normalize images with ImageNet mean/std before model ingestion.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _find_camera_inputs(
    batch: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[Mapping[str, Any]], Optional[str]]:
    """
    Best-effort discovery of camera inputs from dataset batches.

    Returns (key, payload, modality_tag):
    - key: the batch key that holds the payload ("image_inputs" or "inputs_{tag}").
    - payload: mapping with at least imgs/intrins/extrinsics.
    - modality_tag: parsed from inputs_{tag} (e.g. "m2") to align with agent_modality_list.
    """
    cand = batch.get("image_inputs")
    if isinstance(cand, Mapping) and all(k in cand for k in ("imgs", "intrins", "extrinsics")):
        return "image_inputs", cand, None

    for key, val in batch.items():
        if not isinstance(key, str) or not key.startswith("inputs_"):
            continue
        if not isinstance(val, Mapping):
            continue
        if not all(k in val for k in ("imgs", "intrins", "extrinsics")):
            continue
        tag = key[len("inputs_") :] or None
        return key, val, tag

    return None, None, None


def _tensor_to_uint8_rgb(image_chw: torch.Tensor) -> Optional[np.ndarray]:
    """Convert a normalized CHW float tensor into an HWC uint8 RGB NumPy image."""
    if not isinstance(image_chw, torch.Tensor) or image_chw.ndim != 3:
        return None
    if int(image_chw.shape[0]) <= 0:
        return None

    img = image_chw.detach()
    if int(img.shape[0]) >= 3:
        img = img[:3]
    else:
        img = img.repeat(3, 1, 1)

    img = img.to(device="cpu", dtype=torch.float32)
    # Undo standard normalization when possible. If tensors are already unnormalized,
    # this will produce nonsense; downstream matching will simply fail and no pose
    # update will be applied.
    try:
        img = img * _IMAGENET_STD + _IMAGENET_MEAN
    except Exception:
        pass
    img = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return img.permute(1, 2, 0).contiguous().numpy()


def _augment_intrinsics_from_post(
    K: torch.Tensor,
    post_rot: Optional[torch.Tensor],
    post_tran: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Adjust intrinsics to match augmented images (resize/crop/flip/rotate).

    lift-splat-shoot aug keeps original K and stores post_rot/post_tran that map:
      p_aug = post_rot * p_orig + post_tran
    in homogeneous pixel coords, H = post_rot with translation injected into [:2,2], so:
      K_aug = H @ K
    """
    if not isinstance(K, torch.Tensor):
        return K
    if post_rot is None or post_tran is None:
        return K
    if not isinstance(post_rot, torch.Tensor) or not isinstance(post_tran, torch.Tensor):
        return K
    if post_rot.shape != (3, 3) or post_tran.numel() < 2:
        return K
    H = post_rot.clone()
    try:
        flat = post_tran.view(-1)
        H[0, 2] = flat[0]
        H[1, 2] = flat[1]
    except Exception:
        return K
    try:
        return H.to(dtype=K.dtype, device=K.device) @ K
    except Exception:
        return K


def _maybe_attach_online_camera_payload(
    *,
    batch: Mapping[str, Any],
    base_data_dict: Dict[Any, Dict[str, Any]],
    cav_ids: Sequence[Any],
    agent_modalities: Optional[Sequence[Any]],
    require_payload: bool,
) -> bool:
    """
    Populate base_data_dict[*]['camera_data'] and params.camera* from runtime batch tensors.

    This is required for `image_match` pose correction in the online backend.
    """
    key, payload, modality_tag = _find_camera_inputs(batch)
    if payload is None:
        if require_payload:
            raise RuntimeError(
                "online image_match requires camera tensors in batch (expected image_inputs or inputs_* with imgs/intrins/extrinsics)"
            )
        return False

    imgs = payload.get("imgs")
    intrins = payload.get("intrins")
    extrinsics = payload.get("extrinsics")
    post_rots = payload.get("post_rots")
    post_trans = payload.get("post_trans")

    if not isinstance(imgs, torch.Tensor) or not isinstance(intrins, torch.Tensor) or not isinstance(extrinsics, torch.Tensor):
        if require_payload:
            raise RuntimeError(f"{key} payload missing required tensors (imgs/intrins/extrinsics)")
        return False

    # Expected shapes:
    # - imgs: [N_cam_cav, N_cam, C, H, W]
    # - intrins: [N_cam_cav, N_cam, 3, 3]
    # - extrinsics: [N_cam_cav, N_cam, 4, 4]  (camera->lidar from dataset)
    if imgs.ndim != 5:
        if require_payload:
            raise RuntimeError(f"Unsupported {key}.imgs shape for image_match: {tuple(imgs.shape)}")
        return False
    n_cam_cav = int(imgs.shape[0])
    n_cam = int(imgs.shape[1])

    cav_is_camera = [True] * len(cav_ids)
    if modality_tag and agent_modalities and len(agent_modalities) == len(cav_ids):
        cav_is_camera = [str(m) == str(modality_tag) for m in agent_modalities]
    expected = sum(1 for v in cav_is_camera if v)
    if require_payload and expected > n_cam_cav:
        raise RuntimeError(
            f"{key} camera payload has {n_cam_cav} CAV entries, but batch expects {expected} camera CAVs (tag={modality_tag})"
        )

    cam_cursor = 0
    attached_any = False

    # Only convert as many camera CAVs as needed for this sample.
    for i, cav_id in enumerate(cav_ids):
        if cav_id not in base_data_dict:
            continue
        if not cav_is_camera[i]:
            continue
        if cam_cursor >= n_cam_cav:
            break

        imgs_cav = imgs[cam_cursor]  # [N_cam, C, H, W]
        intrins_cav = intrins[cam_cursor] if intrins.ndim >= 4 else None
        extr_cav = extrinsics[cam_cursor] if extrinsics.ndim >= 4 else None
        post_rot_cav = post_rots[cam_cursor] if isinstance(post_rots, torch.Tensor) and post_rots.ndim >= 4 else None
        post_tran_cav = post_trans[cam_cursor] if isinstance(post_trans, torch.Tensor) and post_trans.ndim >= 3 else None

        camera_data: List[np.ndarray] = []
        params = base_data_dict[cav_id].get("params") or {}

        for cam_idx in range(n_cam):
            img_np = _tensor_to_uint8_rgb(imgs_cav[cam_idx])
            if img_np is None:
                continue
            camera_data.append(img_np)

            K_list = None
            if intrins_cav is not None and cam_idx < int(intrins_cav.shape[0]):
                K = intrins_cav[cam_idx].detach().cpu()
                pr = None
                pt = None
                if post_rot_cav is not None and cam_idx < int(post_rot_cav.shape[0]):
                    pr = post_rot_cav[cam_idx].detach().cpu()
                if post_tran_cav is not None and cam_idx < int(post_tran_cav.shape[0]):
                    pt = post_tran_cav[cam_idx].detach().cpu()
                K_aug = _augment_intrinsics_from_post(K, pr, pt)
                K_list = K_aug.detach().cpu().numpy().astype(np.float64).tolist()

            T_cam_lidar_list = None
            if extr_cav is not None and cam_idx < int(extr_cav.shape[0]):
                # Dataset provides camera->lidar; image_match expects lidar->camera (T_cam_lidar).
                ext = extr_cav[cam_idx].detach().cpu().to(torch.float64)
                try:
                    T_cam_lidar = torch.inverse(ext).numpy()
                    T_cam_lidar_list = np.asarray(T_cam_lidar, dtype=np.float64).tolist()
                except Exception:
                    T_cam_lidar_list = None

            cam_params: Dict[str, Any] = {}
            if K_list is not None:
                cam_params["intrinsic"] = K_list
            if T_cam_lidar_list is not None:
                cam_params["extrinsic"] = T_cam_lidar_list
            if cam_params:
                params[f"camera{int(cam_idx)}"] = cam_params

        if camera_data:
            base_data_dict[cav_id]["camera_data"] = camera_data
            base_data_dict[cav_id]["params"] = params
            attached_any = True

        cam_cursor += 1

    return bool(attached_any)


def _maybe_attach_online_lidar_payload_from_vsa(
    *,
    batch: Mapping[str, Any],
    base_data_dict: Dict[Any, Dict[str, Any]],
    cav_ids: Sequence[Any],
    global_idx_by_cav: Mapping[Any, int],
    require_payload: bool,
) -> bool:
    """
    Attach per-CAV LiDAR point clouds from `origin_lidar_for_vsa` when available.

    Some dataset variants export `origin_lidar_for_vsa` as [N, 1+F] where the first
    column is a global cav index aligned with concatenated lidar_pose order.
    """
    # Preferred lightweight payload: a flat list aligned with concatenated cav order.
    raw_list = batch.get("lidar_np_by_cav")
    if isinstance(raw_list, (list, tuple)) and raw_list:
        attached_any = False
        for cav_id in cav_ids:
            if cav_id not in base_data_dict:
                continue
            gidx = global_idx_by_cav.get(cav_id)
            if gidx is None:
                continue
            gi = int(gidx)
            if gi < 0 or gi >= len(raw_list):
                continue
            cloud = raw_list[gi]
            if cloud is None:
                continue
            try:
                base_data_dict[cav_id]["lidar_np"] = np.asarray(cloud)
                attached_any = True
            except Exception:
                continue
        if attached_any:
            return True

    pts = batch.get("origin_lidar_for_vsa")
    if isinstance(pts, torch.Tensor):
        if pts.ndim != 2 or int(pts.shape[1]) < 4:
            if require_payload:
                raise RuntimeError(
                    "online lidar_reg requires per-CAV raw points; expected batch['origin_lidar_for_vsa'] with shape [N, >=4]"
                )
            return False
        idx_col = pts[:, 0].long()
        attached_any = False
        for cav_id in cav_ids:
            if cav_id not in base_data_dict:
                continue
            gidx = global_idx_by_cav.get(cav_id)
            if gidx is None:
                continue
            mask = idx_col == int(gidx)
            if not bool(mask.any().item()):
                continue
            cloud = pts[mask][:, 1:].detach().cpu().numpy()
            base_data_dict[cav_id]["lidar_np"] = cloud
            attached_any = True
        return bool(attached_any)

    if isinstance(pts, np.ndarray):
        arr = np.asarray(pts)
        if arr.ndim != 2 or arr.shape[1] < 4:
            if require_payload:
                raise RuntimeError(
                    "online lidar_reg requires per-CAV raw points; expected batch['origin_lidar_for_vsa'] with shape [N, >=4]"
                )
            return False
        idx_col = arr[:, 0].astype(np.int64)
        attached_any = False
        for cav_id in cav_ids:
            if cav_id not in base_data_dict:
                continue
            gidx = global_idx_by_cav.get(cav_id)
            if gidx is None:
                continue
            mask = idx_col == int(gidx)
            if not bool(mask.any()):
                continue
            cloud = arr[mask][:, 1:]
            base_data_dict[cav_id]["lidar_np"] = cloud
            attached_any = True
        return bool(attached_any)

    if require_payload:
        raise RuntimeError(
            "online lidar_reg requires per-CAV raw points; expected batch['lidar_np_by_cav'] or batch['origin_lidar_for_vsa']"
        )
    return False


@dataclass
class PoseProviderConfig:
    enabled: bool = False
    # Legacy mode kept for backward compatibility.
    mode: str = "legacy"
    # New mode interface used by the unified benchmark/runtime path.
    runtime_mode: Optional[str] = None
    # offline_map (legacy override map), online_box, online_box_feat_refine
    solver_backend: str = "offline_map"
    # Used when runtime_mode=fusion_only.
    pose_source: str = "noisy_input"

    apply_to: str = "non-ego"
    freeze_ego: bool = True
    recompute_pairwise: bool = False
    override_map: Optional[Dict[str, Any]] = None
    override_path: Optional[str] = None
    pose_field: str = "lidar_pose_pred_np"
    confidence_field: str = "pose_confidence_np"
    proj_first: Optional[bool] = None
    max_cav: Optional[int] = None

    # Online solver payload.
    online_method: Optional[str] = None
    online_args: Dict[str, Any] = field(default_factory=dict)
    online_stage1_result: Optional[Dict[str, Any]] = None
    online_stage1_result_path: Optional[str] = None
    online_pose_result: Optional[Dict[str, Any]] = None
    online_pose_result_path: Optional[str] = None

    # Runtime caches.
    _online_corrector: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_hypes(cls, hypes: Mapping[str, Any]) -> "PoseProviderConfig":
        payload = dict(hypes.get("pose_provider") or {})

        fusion_cfg = hypes.get("fusion") if isinstance(hypes, Mapping) else None
        fusion_args = {}
        if isinstance(fusion_cfg, Mapping):
            maybe_args = fusion_cfg.get("args")
            if isinstance(maybe_args, Mapping):
                fusion_args = dict(maybe_args)

        train_params = hypes.get("train_params") if isinstance(hypes, Mapping) else None

        enabled = bool(payload.get("enabled", False))
        mode = str(payload.get("mode") or "legacy").lower().strip()

        runtime_mode_raw = payload.get("runtime_mode")
        runtime_mode = (
            str(runtime_mode_raw).lower().strip() if runtime_mode_raw is not None else None
        )
        if runtime_mode and runtime_mode not in _ALLOWED_RUNTIME_MODES:
            runtime_mode = None

        pose_source = str(payload.get("pose_source") or "noisy_input").lower().strip()
        if pose_source not in _ALLOWED_POSE_SOURCES:
            pose_source = "noisy_input"

        # Keep legacy mode semantics while allowing runtime_mode as primary selector.
        if runtime_mode is not None:
            if runtime_mode == "single_only":
                mode = "single_only"
            elif runtime_mode == "fusion_only" and pose_source == "gt":
                mode = "gt_only"
            elif runtime_mode in {"fusion_only", "register_only", "register_and_fuse"}:
                mode = runtime_mode

        solver_backend = str(payload.get("solver_backend") or "offline_map").lower().strip()
        if not solver_backend:
            solver_backend = "offline_map"

        apply_to = str(payload.get("apply_to") or "non-ego")
        freeze_ego = bool(payload.get("freeze_ego", True))
        recompute_pairwise = bool(payload.get("recompute_pairwise", enabled))

        pose_field = str(payload.get("pose_field") or "lidar_pose_pred_np")
        confidence_field = str(payload.get("confidence_field") or "pose_confidence_np")

        proj_first = payload.get("proj_first", None)
        max_cav = payload.get("max_cav", None)
        if proj_first is None and "proj_first" in fusion_args:
            proj_first = fusion_args.get("proj_first")
        if max_cav is None and isinstance(train_params, Mapping):
            max_cav = train_params.get("max_cav", None)

        override_map = payload.get("pose_override_map") or payload.get("override_map")
        override_path = payload.get("pose_override_path") or payload.get("override_path")

        # Online payload can be passed under pose_provider.online or flat keys.
        online_payload = payload.get("online")
        if not isinstance(online_payload, Mapping):
            online_payload = {}
        online_method = (
            online_payload.get("method")
            or payload.get("online_method")
            or payload.get("method")
            or payload.get("pose_method")
        )
        online_method = str(online_method).lower().strip() if online_method else None

        online_args = online_payload.get("args")
        if not isinstance(online_args, Mapping):
            online_args = payload.get("online_args") or payload.get("solver_args") or {}
        online_args = dict(online_args) if isinstance(online_args, Mapping) else {}

        online_stage1_result = (
            online_payload.get("stage1_result")
            if isinstance(online_payload.get("stage1_result"), Mapping)
            else None
        )
        online_pose_result = (
            online_payload.get("pose_result")
            if isinstance(online_payload.get("pose_result"), Mapping)
            else None
        )

        online_stage1_result_path = (
            online_payload.get("stage1_result_path")
            or payload.get("online_stage1_result_path")
            or payload.get("stage1_result")
        )
        if isinstance(online_stage1_result_path, Mapping):
            online_stage1_result_path = None

        online_pose_result_path = (
            online_payload.get("pose_result_path")
            or payload.get("online_pose_result_path")
            or payload.get("pose_result")
        )
        if isinstance(online_pose_result_path, Mapping):
            online_pose_result_path = None

        # Legacy pose_override compatibility.
        legacy_override = payload.get("pose_override") or hypes.get("pose_override") or {}
        if isinstance(legacy_override, Mapping):
            if override_map is None:
                override_map = legacy_override.get("pose_map")
            if override_path is None:
                override_path = (
                    legacy_override.get("path")
                    or legacy_override.get("pose_path")
                    or legacy_override.get("pose_result")
                )
            pose_field = str(legacy_override.get("pose_field") or pose_field)
            confidence_field = str(legacy_override.get("confidence_field") or confidence_field)
            apply_to = str(legacy_override.get("apply_to") or apply_to)
            if "freeze_ego" in legacy_override:
                freeze_ego = bool(legacy_override.get("freeze_ego"))

        return cls(
            enabled=enabled,
            mode=mode,
            runtime_mode=runtime_mode,
            solver_backend=solver_backend,
            pose_source=pose_source,
            apply_to=apply_to,
            freeze_ego=freeze_ego,
            recompute_pairwise=recompute_pairwise,
            override_map=override_map if isinstance(override_map, dict) else None,
            override_path=str(override_path) if override_path else None,
            pose_field=pose_field,
            confidence_field=confidence_field,
            proj_first=proj_first,
            max_cav=int(max_cav) if max_cav is not None else None,
            online_method=online_method,
            online_args=online_args,
            online_stage1_result=online_stage1_result if isinstance(online_stage1_result, dict) else None,
            online_stage1_result_path=str(online_stage1_result_path) if online_stage1_result_path else None,
            online_pose_result=online_pose_result if isinstance(online_pose_result, dict) else None,
            online_pose_result_path=str(online_pose_result_path) if online_pose_result_path else None,
        )


def _canonical_runtime_mode(cfg: PoseProviderConfig) -> str:
    if cfg.runtime_mode in _ALLOWED_RUNTIME_MODES:
        return str(cfg.runtime_mode)

    mode = str(cfg.mode or "legacy").lower().strip()
    if mode == "single_only":
        return "single_only"
    if mode == "gt_only":
        return "fusion_only"
    if mode == "register_only":
        return "register_only"
    if mode == "register_and_fuse":
        return "register_and_fuse"
    if mode == "fusion_only":
        return "fusion_only"
    # Legacy/default behavior: register + fuse when provider is enabled.
    return "register_and_fuse"


def _resolve_fusion_pose_source(cfg: PoseProviderConfig) -> str:
    src = str(cfg.pose_source or "noisy_input").lower().strip()
    if src not in _ALLOWED_POSE_SOURCES:
        src = "noisy_input"
    if str(cfg.mode or "").lower().strip() == "gt_only":
        src = "gt"
    return src


def _ensure_override_map(cfg: PoseProviderConfig) -> Optional[Dict[str, Any]]:
    if isinstance(cfg.override_map, dict) and cfg.override_map:
        return cfg.override_map
    if cfg.override_path:
        cfg.override_map = load_pose_override_map(cfg.override_path)
    return cfg.override_map


def _split_cav_id_list(cav_id_list: Any, record_len: torch.Tensor) -> Optional[list]:
    if cav_id_list is None:
        return None
    if (
        isinstance(cav_id_list, (list, tuple))
        and cav_id_list
        and isinstance(cav_id_list[0], (list, tuple))
    ):
        return [list(x) for x in cav_id_list]
    if isinstance(cav_id_list, (list, tuple)):
        if len(record_len) == 1:
            return [list(cav_id_list)]
    return None


def _split_sample_idx(sample_idx: Any, batch_size: int) -> list:
    if isinstance(sample_idx, torch.Tensor):
        if sample_idx.ndim == 0:
            return [int(sample_idx.item()) for _ in range(batch_size)]
        vals = [int(x) for x in sample_idx.detach().cpu().view(-1).tolist()]
        return vals if vals else [0 for _ in range(batch_size)]
    if isinstance(sample_idx, (list, tuple)):
        vals = [int(x) for x in sample_idx] if sample_idx else [0]
        return vals
    if sample_idx is None:
        return [0 for _ in range(batch_size)]
    return [int(sample_idx) for _ in range(batch_size)]


def _build_pose_by_id(entry: Mapping[str, Any], pose_field: str) -> Optional[Dict[str, Any]]:
    cav_ids = entry.get("cav_id_list") or entry.get("cav_ids") or entry.get("agent_ids")
    poses = entry.get(pose_field)
    if isinstance(poses, dict):
        return poses
    if isinstance(cav_ids, list) and isinstance(poses, list):
        return {str(cid): poses[i] for i, cid in enumerate(cav_ids) if i < len(poses)}
    return None


def _build_conf_by_id(entry: Mapping[str, Any], confidence_field: str) -> Optional[Dict[str, Any]]:
    if not confidence_field:
        return None
    cav_ids = entry.get("cav_id_list") or entry.get("cav_ids") or entry.get("agent_ids")
    confs = entry.get(confidence_field)
    if isinstance(confs, dict):
        return confs
    if isinstance(cav_ids, list) and isinstance(confs, list):
        return {str(cid): confs[i] for i, cid in enumerate(cav_ids) if i < len(confs)}
    return None


def apply_overrides_to_batch(batch: Dict[str, Any], cfg: PoseProviderConfig) -> bool:
    override_map = _ensure_override_map(cfg)
    if not isinstance(override_map, dict) or not override_map:
        return False

    lidar_pose = batch.get("lidar_pose")
    record_len = batch.get("record_len")
    sample_idx = batch.get("sample_idx")
    cav_id_list = batch.get("cav_id_list")

    if lidar_pose is None or record_len is None or sample_idx is None:
        return False

    cav_id_by_batch = _split_cav_id_list(cav_id_list, record_len)
    if cav_id_by_batch is None:
        return False

    sample_idx_list = _split_sample_idx(sample_idx, len(record_len))

    apply_to = str(cfg.apply_to or "non-ego").lower().strip()
    freeze_ego = bool(cfg.freeze_ego)

    pose_conf = batch.get("pose_confidence")
    if pose_conf is None:
        pose_conf = torch.ones(
            (lidar_pose.shape[0],), device=lidar_pose.device, dtype=lidar_pose.dtype
        )
        batch["pose_confidence"] = pose_conf

    applied = False
    offset = 0
    for b, cav_ids in enumerate(cav_id_by_batch):
        if b >= len(record_len):
            break
        num = int(record_len[b])
        if num <= 0:
            continue
        idx = sample_idx_list[b] if b < len(sample_idx_list) else sample_idx_list[0]
        entry = _resolve_pose_override_entry(override_map, idx)
        if entry is None or not isinstance(entry, dict):
            offset += num
            continue

        pose_by_id = _build_pose_by_id(entry, cfg.pose_field)
        if not isinstance(pose_by_id, dict):
            offset += num
            continue

        conf_by_id = _build_conf_by_id(entry, cfg.confidence_field)

        for i, cav_id in enumerate(cav_ids[:num]):
            is_ego = i == 0
            if apply_to in {"non-ego", "non_ego", "cav", "other", "others"} and is_ego:
                continue
            if freeze_ego and is_ego:
                continue

            pose_val = pose_by_id.get(str(cav_id))
            pose6 = _pose6_from_value(pose_val)
            if pose6 is None:
                continue

            lidar_pose[offset + i] = torch.tensor(
                pose6, device=lidar_pose.device, dtype=lidar_pose.dtype
            )
            if conf_by_id is not None:
                conf_val = _confidence_from_value(conf_by_id.get(str(cav_id)))
                if conf_val is not None:
                    pose_conf[offset + i] = torch.tensor(
                        conf_val, device=lidar_pose.device, dtype=lidar_pose.dtype
                    )
            applied = True

        offset += num

    return applied


def _ensure_online_runtime(cfg: PoseProviderConfig) -> Tuple[Optional[Any], Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    stage1_result = cfg.online_stage1_result
    pose_result = cfg.online_pose_result

    if stage1_result is None and cfg.online_stage1_result_path:
        try:
            stage1_result = read_json(cfg.online_stage1_result_path)
        except Exception:
            stage1_result = None
        if isinstance(stage1_result, dict):
            cfg.online_stage1_result = stage1_result

    if pose_result is None and cfg.online_pose_result_path:
        try:
            pose_result = read_json(cfg.online_pose_result_path)
        except Exception:
            pose_result = None
        if isinstance(pose_result, dict):
            cfg.online_pose_result = pose_result

    if cfg._online_corrector is None and cfg.online_method:
        try:
            from opencood.extrinsics.pose_correction.pose_solver import build_pose_corrector

            # `online_args` may include runtime-only switches (e.g., gpu_stage1_solver)
            # that are not constructor fields for CPU correctors. Strip them so we
            # can still build the corrector for methods like image_match/lidar_reg.
            build_args = dict(cfg.online_args or {})
            for drop_key in ("gpu_stage1_solver", "skip_pairwise_rebuild", "require_payload"):
                build_args.pop(drop_key, None)
            cfg._online_corrector = build_pose_corrector(
                cfg.online_method, args=build_args
            )
        except Exception:
            cfg._online_corrector = None

    return cfg._online_corrector, stage1_result, pose_result




def _apply_online_gpu_stage1_solver_to_batch(
    batch: Dict[str, Any],
    cfg: PoseProviderConfig,
    stage1_result: Optional[Mapping[str, Any]],
) -> Tuple[bool, bool, float, float, int, int]:
    if solve_relative_pose_from_stage1_entry is None:
        return False, False, 0.0, 0.0, 0, 0
    if not isinstance(stage1_result, Mapping):
        return False, False, 0.0, 0.0, 0, 0

    method = str(cfg.online_method or '').lower().strip()
    if method not in _GPU_STAGE1_METHODS:
        return False, False, 0.0, 0.0, 0, 0

    lidar_pose = batch.get('lidar_pose')
    record_len = batch.get('record_len')
    if lidar_pose is None or record_len is None:
        return False, False, 0.0, 0.0, 0, 0

    cav_id_by_batch = _split_cav_id_list(batch.get('cav_id_list'), record_len)
    if cav_id_by_batch is None:
        cav_id_by_batch = [list(range(int(n))) for n in record_len.detach().cpu().tolist()]

    sample_idx_list = _split_sample_idx(batch.get('sample_idx'), len(record_len))

    pose_conf = batch.get('pose_confidence')
    if pose_conf is None:
        pose_conf = torch.ones((lidar_pose.shape[0],), device=lidar_pose.device, dtype=lidar_pose.dtype)
        batch['pose_confidence'] = pose_conf

    args = dict(cfg.online_args or {})
    field = str(args.get('stage1_field') or 'pred_corner3d_np_list')
    topk = int(args.get('max_boxes', 60) or 60)
    use_score_weight = bool(args.get('use_score_weight', True))
    min_matches = max(2, int(args.get('min_matches', 3) or 3))
    max_match_distance_m = float(args.get('max_match_distance_m', 6.0) or 6.0)
    if method == "freealign":
        min_matches = max(2, int(args.get('min_nodes', min_matches) or min_matches))
        max_match_distance_m = float(args.get('box_error', max_match_distance_m) or max_match_distance_m)
        use_score_weight = bool(args.get('use_score_weight', False))
    elif method == "vips":
        max_match_distance_m = float(
            args.get('match_distance_thr_m', max_match_distance_m) or max_match_distance_m
        )
    elif method == "cbm":
        max_match_distance_m = float(
            args.get('absolute_dis_lim_m', max_match_distance_m) or max_match_distance_m
        )

    solver_backend = str(cfg.solver_backend or "").lower().strip()
    use_feat_refine = solver_backend == "online_box_feat_refine"
    refine_steps = int(args.get("refine_steps", 4) or 4)
    refine_init_step_xy_m = float(args.get("refine_init_step_xy_m", 0.4) or 0.4)
    refine_init_step_yaw_deg = float(args.get("refine_init_step_yaw_deg", 1.5) or 1.5)
    refine_decay = float(args.get("refine_decay", 0.6) or 0.6)
    refine_min_step_xy_m = float(args.get("refine_min_step_xy_m", 0.03) or 0.03)
    refine_min_step_yaw_deg = float(args.get("refine_min_step_yaw_deg", 0.15) or 0.15)
    refine_min_improvement_m = float(args.get("refine_min_improvement_m", 1e-4) or 1e-4)

    applied_any = False
    t0 = time.perf_counter()
    refine_sec = 0.0
    refine_attempted_count = 0
    refine_applied_count = 0
    offset = 0

    for b, num_raw in enumerate(record_len.detach().cpu().tolist()):
        num = int(num_raw)
        if num <= 1:
            offset += max(num, 0)
            continue

        cav_ids = cav_id_by_batch[b] if b < len(cav_id_by_batch) else list(range(num))
        if len(cav_ids) < num:
            cav_ids = list(cav_ids) + list(range(len(cav_ids), num))

        sample_idx = sample_idx_list[b] if b < len(sample_idx_list) else sample_idx_list[0]
        ego_id = cav_ids[0]

        # Map batch cav positions -> stage1 cav_id strings when DAIR caches store
        # role strings (e.g., ["infrastructure","vehicle"]). This avoids accidentally
        # treating numeric cav ids as indices into a role-ordered stage1 list.
        stage1_ids_by_pos: Optional[List[Any]] = None
        try:
            entry = None
            for key in (sample_idx, str(sample_idx), f"{int(sample_idx):06d}", f"{int(sample_idx):04d}"):
                if key in stage1_result:
                    cand = stage1_result.get(key)
                    if isinstance(cand, Mapping):
                        entry = cand
                        break
            if isinstance(entry, Mapping):
                stage1_cav_ids = entry.get("cav_id_list")
                stage1_clean = entry.get("lidar_pose_clean_np")
                batch_clean = batch.get("lidar_pose_clean")
                if (
                    isinstance(stage1_cav_ids, Sequence)
                    and isinstance(stage1_clean, Sequence)
                    and isinstance(batch_clean, torch.Tensor)
                ):
                    stage1_pose = torch.as_tensor(stage1_clean, device=lidar_pose.device, dtype=lidar_pose.dtype).view(-1, 6)
                    if int(stage1_pose.shape[0]) >= int(num) and int(batch_clean.shape[0]) >= int(offset + num):
                        batch_pose = batch_clean[offset : offset + num].to(device=lidar_pose.device, dtype=lidar_pose.dtype)
                        # Use (x,y,yaw) to disambiguate vehicle vs infrastructure.
                        dims = [0, 1, 4] if int(batch_pose.shape[1]) >= 5 else [0, 1]
                        cost = torch.cdist(batch_pose[:, dims], stage1_pose[: len(stage1_cav_ids), dims], p=2)
                        if int(num) == 2 and cost.shape == (2, 2):
                            a0 = float(cost[0, 0] + cost[1, 1])
                            a1 = float(cost[0, 1] + cost[1, 0])
                            mapping = [1, 0] if a1 < a0 else [0, 1]
                        else:
                            mapping = torch.argmin(cost, dim=1).detach().cpu().tolist()
                        stage1_ids_by_pos = []
                        for j in mapping:
                            try:
                                stage1_ids_by_pos.append(stage1_cav_ids[int(j)])
                            except Exception:
                                stage1_ids_by_pos.append(int(j))
        except Exception:
            stage1_ids_by_pos = None

        stage1_ego_id = stage1_ids_by_pos[0] if stage1_ids_by_pos and len(stage1_ids_by_pos) >= 1 else ego_id
        ego_world_raw = pose_to_tfm(lidar_pose[offset : offset + 1])
        if isinstance(ego_world_raw, torch.Tensor):
            ego_world = ego_world_raw.view(4, 4).to(device=lidar_pose.device, dtype=lidar_pose.dtype)
        else:
            ego_world = torch.as_tensor(
                ego_world_raw, device=lidar_pose.device, dtype=lidar_pose.dtype
            ).view(4, 4)

        for i in range(1, num):
            cav_id = cav_ids[i]
            stage1_cav_id = (
                stage1_ids_by_pos[i] if stage1_ids_by_pos and i < len(stage1_ids_by_pos) else cav_id
            )
            sol = solve_relative_pose_from_stage1_entry(
                stage1_result=stage1_result,
                sample_idx=int(sample_idx),
                ego_cav_id=stage1_ego_id,
                cav_id=stage1_cav_id,
                field=field,
                min_matches=min_matches,
                max_match_distance_m=max_match_distance_m,
                topk=topk,
                use_score_weight=use_score_weight,
                device=lidar_pose.device,
            )
            if not isinstance(sol, Mapping):
                continue
            T_rel = sol.get('T_rel')
            if not isinstance(T_rel, torch.Tensor) or T_rel.shape != (4, 4):
                continue
            T_rel = T_rel.to(device=lidar_pose.device, dtype=lidar_pose.dtype)
            sol_residual = float(sol.get('mean_residual_m', 1.0) or 1.0)

            if use_feat_refine and refine_relative_pose_from_stage1_entry is not None:
                refine_attempted_count += 1
                t_refine = time.perf_counter()
                ref = refine_relative_pose_from_stage1_entry(
                    stage1_result=stage1_result,
                    sample_idx=int(sample_idx),
                    ego_cav_id=stage1_ego_id,
                    cav_id=stage1_cav_id,
                    T_init=T_rel,
                    field=field,
                    topk=topk,
                    use_score_weight=use_score_weight,
                    refine_steps=refine_steps,
                    init_step_xy_m=refine_init_step_xy_m,
                    init_step_yaw_deg=refine_init_step_yaw_deg,
                    decay=refine_decay,
                    min_step_xy_m=refine_min_step_xy_m,
                    min_step_yaw_deg=refine_min_step_yaw_deg,
                    min_improvement_m=refine_min_improvement_m,
                    device=lidar_pose.device,
                )
                refine_sec += float(time.perf_counter() - t_refine)
                if isinstance(ref, Mapping):
                    T_ref = ref.get("T_rel")
                    if isinstance(T_ref, torch.Tensor) and T_ref.shape == (4, 4):
                        T_rel = T_ref.to(device=lidar_pose.device, dtype=lidar_pose.dtype)
                        sol_residual = float(ref.get("mean_residual_m", sol_residual) or sol_residual)
                        if bool(ref.get("refined", False)):
                            refine_applied_count += 1

            T_world = torch.matmul(ego_world, T_rel)
            pose6 = tfm_to_pose_torch(T_world.view(1, 4, 4), dof=6).view(-1)
            lidar_pose[offset + i] = pose6.to(device=lidar_pose.device, dtype=lidar_pose.dtype)

            conf_val = 1.0 / (1.0 + max(0.0, sol_residual))
            pose_conf[offset + i] = torch.tensor(conf_val, device=pose_conf.device, dtype=pose_conf.dtype)
            applied_any = True

        offset += num

    return (
        True,
        applied_any,
        float(time.perf_counter() - t0),
        float(refine_sec),
        int(refine_attempted_count),
        int(refine_applied_count),
    )


def _apply_online_solver_to_batch(
    batch: Dict[str, Any], cfg: PoseProviderConfig
) -> Tuple[bool, int, float, float, int, int, Dict[str, int]]:
    corrector, stage1_result, pose_result = _ensure_online_runtime(cfg)
    method = str(cfg.online_method or '').lower().strip()
    online_args = dict(cfg.online_args or {})
    require_payload = bool(online_args.get("require_payload", True))

    # GPU stage1 solver is an *optional* fast path. Keep it opt-in so the default
    # online backend uses the method's actual corrector implementation.
    #
    # NOTE: inference_w_noise.py exposes this as `--online-gpu-stage1-solver`.
    use_gpu_stage1_solver = bool(online_args.get("gpu_stage1_solver", False))
    if use_gpu_stage1_solver:
        gpu_handled, gpu_applied, gpu_sec, gpu_refine_sec, gpu_refine_attempted, gpu_refine_applied = _apply_online_gpu_stage1_solver_to_batch(
            batch, cfg, stage1_result
        )
        if gpu_handled:
            return (
                bool(gpu_applied),
                0,
                float(gpu_sec),
                float(gpu_refine_sec),
                int(gpu_refine_attempted),
                int(gpu_refine_applied),
                {},
            )
        if method in _GPU_STAGE1_METHODS:
            # Keep strict GPU residency for supported online methods.
            return False, 0, float(gpu_sec), float(gpu_refine_sec), 0, 0, {}

    # Fast-path oracle (GT) pose override without building CPU payloads.
    if method == "gt":
        lidar_pose = batch.get("lidar_pose")
        record_len = batch.get("record_len")
        lidar_pose_clean = batch.get("lidar_pose_clean")
        if (
            isinstance(lidar_pose, torch.Tensor)
            and isinstance(record_len, torch.Tensor)
            and isinstance(lidar_pose_clean, torch.Tensor)
            and lidar_pose_clean.shape == lidar_pose.shape
        ):
            freeze_ego = bool((cfg.online_args or {}).get("freeze_ego", True))
            pose_conf = batch.get("pose_confidence")
            if pose_conf is None:
                pose_conf = torch.ones((lidar_pose.shape[0],), device=lidar_pose.device, dtype=lidar_pose.dtype)
                batch["pose_confidence"] = pose_conf
            t0 = time.perf_counter()
            applied_any = False
            offset = 0
            for num_raw in record_len.detach().cpu().tolist():
                num = int(num_raw)
                if num <= 0:
                    continue
                for i in range(num):
                    if freeze_ego and i == 0:
                        continue
                    global_idx = offset + i
                    lidar_pose[global_idx] = lidar_pose_clean[global_idx].to(
                        device=lidar_pose.device, dtype=lidar_pose.dtype
                    )
                    pose_conf[global_idx] = torch.tensor(
                        1.0, device=pose_conf.device, dtype=pose_conf.dtype
                    )
                    applied_any = True
                offset += num
            return bool(applied_any), 0, float(time.perf_counter() - t0), 0.0, 0, 0, {}

    if corrector is None:
        return False, 0, 0.0, 0.0, 0, 0, {}

    lidar_pose = batch.get("lidar_pose")
    record_len = batch.get("record_len")
    if lidar_pose is None or record_len is None:
        return False, 0, 0.0, 0.0, 0, 0, {}

    cav_id_by_batch = _split_cav_id_list(batch.get("cav_id_list"), record_len)
    if cav_id_by_batch is None:
        cav_id_by_batch = [list(range(int(n))) for n in record_len.detach().cpu().tolist()]

    sample_idx_list = _split_sample_idx(batch.get("sample_idx"), len(record_len))

    pose_conf = batch.get("pose_confidence")
    if pose_conf is None:
        pose_conf = torch.ones(
            (lidar_pose.shape[0],), device=lidar_pose.device, dtype=lidar_pose.dtype
        )
        batch["pose_confidence"] = pose_conf

    lidar_pose_clean = batch.get("lidar_pose_clean")

    cpu_fallback_count = 0
    applied_any = False
    pose_corr_stats_total: Dict[str, int] = {}
    start = time.perf_counter()

    offset = 0
    for b, num_raw in enumerate(record_len.detach().cpu().tolist()):
        num = int(num_raw)
        if num <= 0:
            continue
        cav_ids = cav_id_by_batch[b] if b < len(cav_id_by_batch) else list(range(num))
        sample_idx = sample_idx_list[b] if b < len(sample_idx_list) else sample_idx_list[0]

        # Some dataloaders may omit stable cav ids (falling back to [0,1,...]),
        # while stage1 caches store the real cav_id_list. Remap cav ids by
        # matching clean poses so CPU correctors can index stage1 content.
        cav_ids_eff = cav_ids
        if isinstance(stage1_result, Mapping):
            try:
                stage1_ids_by_pos: Optional[List[Any]] = None
                entry = None
                for key in (sample_idx, str(sample_idx), f"{int(sample_idx):06d}", f"{int(sample_idx):04d}"):
                    if key in stage1_result:
                        cand = stage1_result.get(key)
                        if isinstance(cand, Mapping):
                            entry = cand
                            break
                if isinstance(entry, Mapping):
                    stage1_cav_ids = entry.get("cav_id_list")
                    stage1_clean = entry.get("lidar_pose_clean_np")
                    batch_clean = lidar_pose_clean
                    if (
                        isinstance(stage1_cav_ids, Sequence)
                        and isinstance(stage1_clean, Sequence)
                        and isinstance(batch_clean, torch.Tensor)
                    ):
                        stage1_pose = torch.as_tensor(
                            stage1_clean, device=lidar_pose.device, dtype=lidar_pose.dtype
                        ).view(-1, 6)
                        if int(stage1_pose.shape[0]) >= int(num) and int(batch_clean.shape[0]) >= int(offset + num):
                            batch_pose = batch_clean[offset : offset + num].to(
                                device=lidar_pose.device, dtype=lidar_pose.dtype
                            )
                            dims = [0, 1, 4] if int(batch_pose.shape[1]) >= 5 else [0, 1]
                            cost = torch.cdist(batch_pose[:, dims], stage1_pose[: len(stage1_cav_ids), dims], p=2)
                            if int(num) == 2 and cost.shape == (2, 2):
                                a0 = float(cost[0, 0] + cost[1, 1])
                                a1 = float(cost[0, 1] + cost[1, 0])
                                mapping = [1, 0] if a1 < a0 else [0, 1]
                            else:
                                mapping = torch.argmin(cost, dim=1).detach().cpu().tolist()
                            stage1_ids_by_pos = []
                            for j in mapping:
                                try:
                                    stage1_ids_by_pos.append(stage1_cav_ids[int(j)])
                                except Exception:
                                    stage1_ids_by_pos.append(int(j))
                if stage1_ids_by_pos and len(stage1_ids_by_pos) >= num:
                    # Only use the remap when it's a bijection (avoid duplicate dict keys).
                    if len(set(stage1_ids_by_pos[:num])) == num:
                        cav_ids_eff = list(stage1_ids_by_pos[:num])
            except Exception:
                cav_ids_eff = cav_ids
        agent_modalities = None
        raw_modalities = batch.get("agent_modality_list")
        if isinstance(raw_modalities, (list, tuple)):
            if raw_modalities and all(isinstance(x, (list, tuple)) for x in raw_modalities):
                # Nested list: one entry per batch.
                if b < len(raw_modalities):
                    agent_modalities = list(raw_modalities[b])
            else:
                # Flat list aligned with concatenated cav order.
                try:
                    agent_modalities = list(raw_modalities[offset : offset + num])
                except Exception:
                    agent_modalities = None

        base_data_dict: Dict[Any, Dict[str, Any]] = {}
        global_idx_by_cav: Dict[Any, int] = {}
        for i in range(num):
            global_idx = offset + i
            cav_id = cav_ids_eff[i] if i < len(cav_ids_eff) else i
            global_idx_by_cav[cav_id] = int(global_idx)

            params: Dict[str, Any] = {
                "lidar_pose": lidar_pose[global_idx].detach().cpu().numpy().reshape(-1).tolist(),
                "pose_confidence": float(pose_conf[global_idx].detach().cpu().item()),
            }
            if isinstance(lidar_pose_clean, torch.Tensor) and global_idx < int(lidar_pose_clean.shape[0]):
                params["lidar_pose_clean"] = (
                    lidar_pose_clean[global_idx].detach().cpu().numpy().reshape(-1).tolist()
                )

            base_data_dict[cav_id] = {
                "ego": i == 0,
                "params": params,
            }

        solver_kwargs: Dict[str, Any] = {}
        if isinstance(stage1_result, Mapping):
            solver_kwargs["stage1_result"] = stage1_result
        if isinstance(pose_result, Mapping):
            solver_kwargs["pose_result"] = pose_result

        # Attach any additional runtime payload required by CPU correctors.
        if method == "image_match":
            _maybe_attach_online_camera_payload(
                batch=batch,
                base_data_dict=base_data_dict,
                cav_ids=list(cav_ids_eff[:num]),
                agent_modalities=agent_modalities,
                require_payload=require_payload,
            )
        elif method == "lidar_reg":
            _maybe_attach_online_lidar_payload_from_vsa(
                batch=batch,
                base_data_dict=base_data_dict,
                cav_ids=list(cav_ids_eff[:num]),
                global_idx_by_cav=global_idx_by_cav,
                require_payload=require_payload,
            )

        applied = False
        apply_exc = False
        try:
            # Legacy corrector implementations still expect CPU/NumPy payloads.
            cpu_fallback_count += 1
            applied = bool(
                corrector.apply(
                    sample_idx=int(sample_idx),
                    cav_id_list=list(cav_ids_eff[:num]),
                    base_data_dict=base_data_dict,
                    **solver_kwargs,
                )
            )
        except Exception:
            applied = False
            apply_exc = True
            # Best-effort failure accounting so benchmarks can distinguish "no-op due to method"
            # vs. "no-op due to runtime exception".
            pair_total = max(0, int(num) - 1)
            pose_corr_stats_total["pose_corr_pair_total_count"] = pose_corr_stats_total.get(
                "pose_corr_pair_total_count", 0
            ) + int(pair_total)
            pose_corr_stats_total["pose_corr_skip_exception_count"] = pose_corr_stats_total.get(
                "pose_corr_skip_exception_count", 0
            ) + int(pair_total)

        # Stage-1 pose correctors (freealign/vips/cbm) expose per-sample reason breakdown via
        # `last_stats`. Ingest it so downstream YAML timing_stats can summarize skip/applied
        # reasons (keys end with *_count, so inference_w_noise will aggregate them).
        if not apply_exc:
            last_stats = getattr(corrector, "last_stats", None)
            if isinstance(last_stats, dict) and last_stats:
                for k, v in last_stats.items():
                    key = str(k)
                    if not key.endswith("_count"):
                        continue
                    try:
                        iv = int(v)
                    except Exception:
                        continue
                    pose_corr_stats_total[key] = pose_corr_stats_total.get(key, 0) + iv

        if applied:
            for i in range(num):
                global_idx = offset + i
                cav_id = cav_ids_eff[i] if i < len(cav_ids_eff) else i
                entry = base_data_dict.get(cav_id, {})
                params = entry.get("params", {}) if isinstance(entry, Mapping) else {}

                pose6 = _pose6_from_value(params.get("lidar_pose"))
                if pose6 is not None:
                    lidar_pose[global_idx] = torch.tensor(
                        pose6, device=lidar_pose.device, dtype=lidar_pose.dtype
                    )

                conf_val = _confidence_from_value(params.get("pose_confidence"))
                if conf_val is not None:
                    pose_conf[global_idx] = torch.tensor(
                        conf_val, device=pose_conf.device, dtype=pose_conf.dtype
                    )
            applied_any = True

        offset += num

    return (
        applied_any,
        int(cpu_fallback_count),
        float(time.perf_counter() - start),
        0.0,
        0,
        0,
        pose_corr_stats_total,
    )


def _infer_max_cav(batch: Dict[str, Any], record_len: torch.Tensor, cfg: PoseProviderConfig) -> int:
    if cfg.max_cav is not None:
        return int(cfg.max_cav)
    existing = batch.get("pairwise_t_matrix")
    if isinstance(existing, torch.Tensor) and existing.ndim >= 2:
        return int(existing.shape[1])
    return int(record_len.max().item()) if record_len.numel() > 0 else 1


def _infer_proj_first(batch: Dict[str, Any], cfg: PoseProviderConfig) -> bool:
    if cfg.proj_first is not None:
        return bool(cfg.proj_first)
    raw = batch.get("proj_first")
    return bool(raw) if raw is not None else False


def _set_pairwise(batch: Dict[str, Any], pairwise: torch.Tensor) -> None:
    batch["pairwise_t_matrix"] = pairwise
    label_dict = batch.get("label_dict")
    if isinstance(label_dict, dict):
        label_dict["pairwise_t_matrix"] = pairwise


def _identity_pairwise(
    record_len: torch.Tensor, max_cav: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    batch_size = int(record_len.shape[0])
    eye = torch.eye(4, device=device, dtype=dtype).view(1, 1, 1, 4, 4)
    return eye.repeat(batch_size, max_cav, max_cav, 1, 1)


def apply_pose_provider(batch_data: Dict[str, Any], cfg: PoseProviderConfig) -> Dict[str, Any]:
    if not cfg.enabled:
        return batch_data

    target = batch_data.get("ego") if isinstance(batch_data.get("ego"), dict) else batch_data
    if not isinstance(target, dict):
        return batch_data

    lidar_pose = target.get("lidar_pose")
    record_len = target.get("record_len")
    if lidar_pose is None or record_len is None:
        return batch_data

    runtime_mode = _canonical_runtime_mode(cfg)
    solver_backend = str(cfg.solver_backend or "offline_map").lower().strip()

    override_sec = 0.0
    online_solver_sec = 0.0
    refine_sec = 0.0
    refine_attempted_count = 0
    refine_applied_count = 0
    pairwise_sec = 0.0
    cpu_fallback_count = 0
    applied = False
    pose_corr_stats_total: Dict[str, int] = {}
    total_start = time.perf_counter()

    if runtime_mode in {"register_only", "register_and_fuse"}:
        if solver_backend in {"online_box", "online_box_feat_refine"}:
            (
                applied,
                cpu_fallback_count,
                online_solver_sec,
                refine_sec,
                refine_attempted_count,
                refine_applied_count,
                pose_corr_stats_total,
            ) = _apply_online_solver_to_batch(target, cfg)
            override_sec = online_solver_sec
        else:
            t0 = time.perf_counter()
            applied = apply_overrides_to_batch(target, cfg)
            override_sec = time.perf_counter() - t0

    # Pose source selection for pairwise rebuild.
    pose_source_tensor = lidar_pose
    identity_pairwise = False
    fusion_pose_source = _resolve_fusion_pose_source(cfg)

    if runtime_mode == "fusion_only":
        if fusion_pose_source == "gt":
            pose_source_tensor = target.get("lidar_pose_clean")
            if pose_source_tensor is None:
                pose_source_tensor = lidar_pose
        elif fusion_pose_source == "identity":
            pose_source_tensor = lidar_pose
            identity_pairwise = True
        else:
            pose_source_tensor = lidar_pose
    elif str(cfg.mode or "").lower().strip() == "gt_only":
        pose_source_tensor = target.get("lidar_pose_clean")
        if pose_source_tensor is None:
            pose_source_tensor = lidar_pose

    # Rebuild pairwise for fusion/register modes, or when explicitly requested.
    should_rebuild = bool(cfg.recompute_pairwise)
    if runtime_mode in {"fusion_only", "register_only", "register_and_fuse"}:
        should_rebuild = True

    # Oracle runtime can optionally reuse dataset pairwise to tighten strict parity
    # against offline-map (which also consumes dataset-built pairwise).
    if (
        solver_backend in {"online_box", "online_box_feat_refine"}
        and str(cfg.online_method or "").lower().strip() == "gt"
        and bool((cfg.online_args or {}).get("skip_pairwise_rebuild", False))
    ):
        should_rebuild = False

    if should_rebuild:
        t1 = time.perf_counter()
        max_cav = _infer_max_cav(target, record_len, cfg)
        proj_first = _infer_proj_first(target, cfg)
        if proj_first or identity_pairwise:
            pairwise = _identity_pairwise(
                record_len,
                max_cav,
                pose_source_tensor.dtype,
                pose_source_tensor.device,
            )
        else:
            dof = int(pose_source_tensor.shape[1]) if pose_source_tensor.ndim > 1 else 6
            pairwise = get_pairwise_transformation_torch(
                pose_source_tensor, max_cav, record_len, dof
            )
        _set_pairwise(target, pairwise)
        pairwise_sec = time.perf_counter() - t1

    total_sec = time.perf_counter() - total_start
    pose_timing_payload: Dict[str, Any] = {
        "pose_provider_total_sec": float(total_sec),
        "pose_override_sec": float(override_sec),
        "online_solver_sec": float(online_solver_sec),
        "match_sec": float(online_solver_sec),
        "solver_sec": float(online_solver_sec),
        "refine_sec": float(refine_sec),
        "refine_attempted_count": int(refine_attempted_count),
        "refine_applied_count": int(refine_applied_count),
        "pairwise_rebuild_sec": float(pairwise_sec),
        "cpu_fallback_count": int(cpu_fallback_count),
        # Numeric version so downstream benchmark ingest (keys ending with *_count) can summarize it.
        "pose_provider_applied_count": int(bool(applied)),
        "pose_provider_applied": bool(applied),
        "solver_backend": solver_backend,
        "runtime_mode": runtime_mode,
        "pose_source": fusion_pose_source if runtime_mode == "fusion_only" else "n/a",
    }
    if pose_corr_stats_total:
        # Add stage-1 correction reason breakdown (counts) when available.
        for k, v in pose_corr_stats_total.items():
            try:
                pose_timing_payload[str(k)] = int(v)
            except Exception:
                continue
    target["pose_timing"] = pose_timing_payload

    return batch_data
