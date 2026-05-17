from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .path_utils import ensure_v2xreg_root_on_path


def _as_numpy(x) -> np.ndarray:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch

        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def corners_to_bbox3d_list(
    corners: np.ndarray,
    *,
    bbox_type: str = "detected",
    scores: Optional[Sequence[float]] = None,
    descriptors: Optional[Sequence[np.ndarray]] = None,
) -> List[object]:
    """
    Convert `(N, 8, 3)` corners into `BBox3d` objects from `v2x_calib`.
    """
    ensure_v2xreg_root_on_path()
    from v2x_calib.reader.BBox3d import BBox3d  # type: ignore

    corners_np = _as_numpy(corners)
    if corners_np is None:
        return []
    if corners_np.ndim != 3 or corners_np.shape[1:] != (8, 3):
        raise ValueError(f"corners must have shape (N,8,3), got {corners_np.shape}")

    scores_arr = None if scores is None else np.asarray(list(scores), dtype=float)
    if scores_arr is not None and scores_arr.shape[0] != corners_np.shape[0]:
        raise ValueError("scores length must match corners count")

    out: List[object] = []
    for i in range(corners_np.shape[0]):
        conf = float(scores_arr[i]) if scores_arr is not None else 1.0
        desc = None
        if descriptors is not None and i < len(descriptors):
            desc = descriptors[i]
        out.append(BBox3d(bbox_type=bbox_type, bbox_8_3=np.asarray(corners_np[i]), confidence=conf, descriptor=desc))
    return out


def bbox3d_to_state7(bbox) -> Tuple[np.ndarray, Tuple[float, float, float], float, np.ndarray]:
    """
    Convert a BBox3d-like object into state.
    Returns:
      center (3,), (l,w,h), yaw(rad), corners(8,3)
    """
    corners = np.asarray(bbox.get_bbox3d_8_3(), dtype=np.float64)
    center = corners.mean(axis=0)
    size = np.abs(corners[4] - corners[2])
    l, w, h = float(size[0]), float(size[1]), float(size[2])
    vec = corners[0] - corners[3]
    yaw = math.atan2(float(vec[1]), float(vec[0]))
    return center, (l, w, h), yaw, corners


def bbox_list_to_array7(bboxes: Iterable[object]) -> np.ndarray:
    """
    Convert list of BBox3d objects to (N,7) array: [x,y,z,h,w,l,yaw].
    """
    states = []
    for bbox in bboxes:
        center, (l, w, h), yaw, _ = bbox3d_to_state7(bbox)
        states.append([center[0], center[1], center[2], h, w, l, yaw])
    if not states:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(states, dtype=np.float32)


def svd_from_corresponded_corners(
    src_corners_list: List[np.ndarray],
    dst_corners_list: List[np.ndarray],
) -> Optional[np.ndarray]:
    """
    Estimate rigid transform T s.t. dst ≈ T * src using SVD.
    """
    if not src_corners_list or not dst_corners_list:
        return None
    src = np.vstack(src_corners_list)
    dst = np.vstack(dst_corners_list)
    if src.shape[0] < 3:
        return None
    src_cent = src.mean(axis=0)
    dst_cent = dst.mean(axis=0)
    src_centered = src - src_cent
    dst_centered = dst - dst_cent
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R_est = Vt.T @ U.T
    if np.linalg.det(R_est) < 0:
        Vt[-1, :] *= -1
        R_est = Vt.T @ U.T
    t_est = dst_cent - R_est @ src_cent
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_est
    T[:3, 3] = t_est
    return T


__all__ = [
    "corners_to_bbox3d_list",
    "bbox3d_to_state7",
    "bbox_list_to_array7",
    "svd_from_corresponded_corners",
]

