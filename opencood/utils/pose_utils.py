# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import json
import numpy as np
import torch
import torch.distributions as dist


def pose_xy_error(pose: np.ndarray, pose_clean: np.ndarray) -> float:
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    pose_clean = np.asarray(pose_clean, dtype=np.float32).reshape(-1)
    if pose.shape[0] < 2 or pose_clean.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(pose[:2] - pose_clean[:2], ord=2))


def pose_confidence_from_epsilon(epsilon: float) -> float:
    eps = float(epsilon)
    return float(1.0 / (1.0 + eps * eps))


def attach_pose_confidence(data_dict, *, key: str = "pose_confidence") -> None:
    """
    Attach a scalar pose confidence to each CAV if missing.

    Following V2VLoc Eq.(6): sigma = 1 / (1 + epsilon^2).
    Here epsilon is approximated by XY translation error between lidar_pose and lidar_pose_clean.
    """
    for _, cav_content in data_dict.items():
        params = cav_content.get("params") or {}
        if key in params:
            continue
        pose = params.get("lidar_pose")
        pose_clean = params.get("lidar_pose_clean")
        if pose is None or pose_clean is None:
            params[key] = 1.0
            continue
        eps = pose_xy_error(pose, pose_clean)
        params[key] = pose_confidence_from_epsilon(eps)
        cav_content["params"] = params


def add_noise_data_dict(data_dict, noise_setting):
    """ Update the base data dict. 
        We retrieve lidar_pose and add_noise to it.
        And set a clean pose.
    """
    noise_args = noise_setting.get('args', {}) if isinstance(noise_setting, dict) else {}
    target = noise_args.get('target', 'all') if isinstance(noise_args, dict) else 'all'
    # Optional: simulate localization outage by reusing the last pose for a fraction of frames.
    # This keeps the downstream pipeline intact while pausing fresh noise injection.
    dropout_prob = 0.0
    if isinstance(noise_args, dict):
        try:
            dropout_prob = float(noise_args.get('dropout_prob', 0.0) or 0.0)
        except Exception:
            dropout_prob = 0.0
    if dropout_prob < 0.0:
        dropout_prob = 0.0
    if dropout_prob > 1.0:
        dropout_prob = 1.0

    def _should_apply_noise(cav_id, cav_content) -> bool:
        if target is None or target == 'all':
            return True
        if target in {'ego', 'self'}:
            return bool(cav_content.get('ego', False))
        if target in {'non-ego', 'cav', 'other', 'others'}:
            return not bool(cav_content.get('ego', False))
        if isinstance(target, (list, tuple, set)):
            return cav_id in target
        try:
            return int(cav_id) == int(target)
        except Exception:
            # Backward compatible fallback: if unknown target, apply to all.
            return True

    state = getattr(add_noise_data_dict, "_dropout_state", None)
    if state is None:
        state = {"last_pose": {}}
        setattr(add_noise_data_dict, "_dropout_state", state)
    last_pose = state["last_pose"]

    if noise_setting['add_noise']:
        # Draw once per frame for deterministic dropout across agents.
        skip_noise = False
        if dropout_prob > 0.0:
            skip_noise = bool(np.random.rand() < dropout_prob)
        for cav_id, cav_content in data_dict.items():
            cav_content['params']['lidar_pose_clean'] = cav_content['params']['lidar_pose'] # 6 dof pose

            if skip_noise:
                if cav_id in last_pose:
                    cav_content['params']['lidar_pose'] = np.asarray(last_pose[cav_id], dtype=np.float32)
                cav_content['params']['pose_confidence'] = 0.0
                continue

            cav_content['params'].pop('pose_confidence', None)

            if _should_apply_noise(cav_id, cav_content):
                if "laplace" in noise_setting['args'].keys() and noise_setting['args']['laplace'] is True:
                    cav_content['params']['lidar_pose'] = cav_content['params']['lidar_pose'] + \
                                                            generate_noise_laplace( # we just use the same key name
                                                                noise_setting['args']['pos_std'],
                                                                noise_setting['args']['rot_std'],
                                                                noise_setting['args']['pos_mean'],
                                                                noise_setting['args']['rot_mean']
                                                            )
                else:
                    cav_content['params']['lidar_pose'] = cav_content['params']['lidar_pose'] + \
                                                                generate_noise(
                                                                    noise_setting['args']['pos_std'],
                                                                    noise_setting['args']['rot_std'],
                                                                    noise_setting['args']['pos_mean'],
                                                                    noise_setting['args']['rot_mean']
                                                                )

            last_pose[cav_id] = np.asarray(cav_content['params']['lidar_pose'], dtype=np.float32)

    else:
        for cav_id, cav_content in data_dict.items():
            cav_content['params']['lidar_pose_clean'] = cav_content['params']['lidar_pose'] # 6 dof pose
            cav_content['params'].pop('pose_confidence', None)
            last_pose[cav_id] = np.asarray(cav_content['params']['lidar_pose'], dtype=np.float32)

            
    return data_dict


def override_lidar_poses(
    data_dict,
    *,
    mode: str = "ego",
    apply_to: str = "non-ego",
    ego_id=None,
    reference_pose=None,
    set_confidence=None,
) -> None:
    """
    Override per-agent lidar poses in-place.

    This is primarily for "no extrinsics / no localization" experiments where we
    intentionally hide all relative transforms from the model and rely on a
    calibration-free estimator (e.g. FreeAlign/V2XReg++) to recover them.

    Args:
        mode: "ego" (copy ego pose) or "zero"/"identity" (set to [0,0,0,0,0,0]).
        apply_to: "all" or "non-ego".
        ego_id: Optional ego key in data_dict (used to locate the ego pose).
        reference_pose: Optional 6DoF pose to use when mode == "ego".
        set_confidence: Optional float to assign to params["pose_confidence"] for overridden agents.
    """
    if not isinstance(data_dict, dict):
        return

    mode = str(mode or "ego").lower().strip()
    apply_to = str(apply_to or "non-ego").lower().strip()

    ref = None
    if mode in {"zero", "zeros", "identity", "none"}:
        ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        if reference_pose is not None:
            ref = list(reference_pose)
        elif ego_id is not None and ego_id in data_dict:
            ref = list((data_dict.get(ego_id) or {}).get("params", {}).get("lidar_pose", []))
        else:
            for cav_id, cav_content in data_dict.items():
                if isinstance(cav_content, dict) and cav_content.get("ego", False):
                    ref = list((cav_content.get("params") or {}).get("lidar_pose", []))
                    break
        if not ref or len(ref) < 6:
            ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    ref6 = [float(x) for x in (ref[:6] if len(ref) >= 6 else (list(ref) + [0.0] * 6)[:6])]

    for cav_id, cav_content in data_dict.items():
        if not isinstance(cav_content, dict):
            continue
        if apply_to in {"non-ego", "non_ego", "cav", "other", "others"}:
            if cav_id == ego_id or bool(cav_content.get("ego", False)):
                continue
        params = cav_content.get("params") or {}
        if not isinstance(params, dict):
            continue
        params["lidar_pose"] = list(ref6)
        if set_confidence is not None:
            try:
                params["pose_confidence"] = float(set_confidence)
            except Exception:
                params["pose_confidence"] = 0.0
        else:
            params.pop("pose_confidence", None)
        cav_content["params"] = params

def _resolve_pose_override_entry(override_map, sample_idx):
    if not isinstance(override_map, dict):
        return None
    if sample_idx in override_map:
        return override_map.get(sample_idx)
    key = str(sample_idx)
    if key in override_map:
        return override_map.get(key)
    try:
        idx_int = int(sample_idx)
    except Exception:
        idx_int = None
    if idx_int is not None and idx_int in override_map:
        return override_map.get(idx_int)
    if idx_int is not None and str(idx_int) in override_map:
        return override_map.get(str(idx_int))
    return None


def _pose6_from_value(value):
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size >= 6:
        return [float(x) for x in arr[:6]]
    if arr.size == 3:
        # Assume [x, y, yaw] (deg) and fill the rest.
        out = [0.0] * 6
        out[0] = float(arr[0])
        out[1] = float(arr[1])
        out[4] = float(arr[2])
        return out
    return None


def _confidence_from_value(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def apply_pose_overrides(
    data_dict,
    *,
    override_map,
    sample_idx,
    pose_field="lidar_pose_pred_np",
    confidence_field="pose_confidence_np",
    apply_to="non-ego",
    freeze_ego=True,
):
    """
    Apply pose overrides from a precomputed map (JSON) to the current data_dict.

    Expected override map format (per sample):
      sample_idx -> {
        "cav_id_list": [...],
        "<pose_field>": [[x,y,z,roll,yaw,pitch], ...],
        "<confidence_field>": [conf, ...] (optional)
      }
    """
    entry = _resolve_pose_override_entry(override_map, sample_idx)
    if entry is None or not isinstance(entry, dict):
        return False

    drop_non_ego = bool(
        entry.get("force_ego_only", False)
        or entry.get("ego_only", False)
        or entry.get("drop_non_ego", False)
    )
    if drop_non_ego:
        for cav_id in list(data_dict.keys()):
            cav_content = data_dict.get(cav_id)
            if not isinstance(cav_content, dict):
                continue
            if cav_content.get("ego", False):
                continue
            data_dict.pop(cav_id, None)
        return True

    cav_ids = entry.get("cav_id_list") or entry.get("cav_ids") or entry.get("agent_ids")
    poses = entry.get(pose_field)
    confs = entry.get(confidence_field) if confidence_field else None

    pose_by_id = None
    if isinstance(poses, dict):
        pose_by_id = poses
    elif isinstance(cav_ids, list) and isinstance(poses, list):
        pose_by_id = {str(cid): poses[i] for i, cid in enumerate(cav_ids) if i < len(poses)}

    if not isinstance(pose_by_id, dict):
        return False

    conf_by_id = None
    if isinstance(confs, dict):
        conf_by_id = confs
    elif isinstance(cav_ids, list) and isinstance(confs, list):
        conf_by_id = {str(cid): confs[i] for i, cid in enumerate(cav_ids) if i < len(confs)}

    apply_to = str(apply_to or "non-ego").lower().strip()
    applied = False
    for cav_id, cav_content in data_dict.items():
        if not isinstance(cav_content, dict):
            continue
        if apply_to in {"non-ego", "non_ego", "cav", "other", "others"}:
            if cav_content.get("ego", False):
                continue
        if freeze_ego and cav_content.get("ego", False):
            continue
        pose_val = pose_by_id.get(str(cav_id))
        pose6 = _pose6_from_value(pose_val)
        if pose6 is None:
            continue
        params = cav_content.get("params") or {}
        if not isinstance(params, dict):
            continue
        params["lidar_pose"] = list(pose6)
        if conf_by_id is not None:
            conf_val = _confidence_from_value(conf_by_id.get(str(cav_id)))
            if conf_val is not None:
                params["pose_confidence"] = conf_val
        cav_content["params"] = params
        applied = True
    return applied


def load_pose_override_map(path):
    if not path:
        return {}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}

def generate_noise(pos_std, rot_std, pos_mean=0, rot_mean=0):
    """ Add localization error to the 6dof pose
        Noise includes position (x,y) and rotation (yaw).
        We use gaussian distribution to generate noise.
    
    Args:

        pos_std : float 
            std of gaussian dist, in meter

        rot_std : float
            std of gaussian dist, in degree

        pos_mean : float
            mean of gaussian dist, in meter

        rot_mean : float
            mean of gaussian dist, in degree
    
    Returns:
        pose_noise: np.ndarray, [6,]
            [x, y, z, roll, yaw, pitch]
    """

    xy = np.random.normal(pos_mean, pos_std, size=(2))
    yaw = np.random.normal(rot_mean, rot_std, size=(1))

    pose_noise = np.array([xy[0], xy[1], 0, 0, yaw[0], 0])

    
    return pose_noise



def generate_noise_laplace(pos_b, rot_b, pos_mu=0, rot_mu=0):
    """ Add localization error to the 6dof pose
        Noise includes position (x,y) and rotation (yaw).
        We use laplace distribution to generate noise.
    
    Args:

        pos_b : float 
            parameter b of laplace dist, in meter

        rot_b : float
            parameter b of laplace dist, in degree

        pos_mu : float
            mean of laplace dist, in meter

        rot_mu : float
            mean of laplace dist, in degree
    
    Returns:
        pose_noise: np.ndarray, [6,]
            [x, y, z, roll, yaw, pitch]
    """

    xy = np.random.laplace(pos_mu, pos_b, size=(2))
    yaw = np.random.laplace(rot_mu, rot_b, size=(1))

    pose_noise = np.array([xy[0], xy[1], 0, 0, yaw[0], 0])
    return pose_noise


def generate_noise_torch(pose, pos_std, rot_std, pos_mean=0, rot_mean=0):
    """ only used for v2vnet robust.
        rotation noise is sampled from von_mises distribution
    
    Args:
        pose : Tensor, [N. 6]
            including [x, y, z, roll, yaw, pitch]

        pos_std : float 
            std of gaussian dist, in meter

        rot_std : float
            std of gaussian dist, in degree

        pos_mean : float
            mean of gaussian dist, in meter

        rot_mean : float
            mean of gaussian dist, in degree
    
    Returns:
        pose_noisy: Tensor, [N, 6]
            noisy pose
    """

    N = pose.shape[0]
    noise = torch.zeros_like(pose, device=pose.device)
    concentration = (180 / (np.pi * rot_std)) ** 2

    noise[:, :2] = torch.normal(pos_mean, pos_std, size=(N, 2), device=pose.device)
    noise[:, 4] = dist.von_mises.VonMises(loc=rot_mean, concentration=concentration).sample((N,)).to(noise.device)


    return noise


def remove_z_axis(T):
    """ remove rotation/translation related to z-axis
    Args:
        T: np.ndarray
            [4, 4]
    Returns:
        T: np.ndarray
            [4, 4]
    """
    T[2,3] = 0 # z-trans
    T[0,2] = 0
    T[1,2] = 0
    T[2,0] = 0
    T[2,1] = 0
    T[2,2] = 1
    
    return T
