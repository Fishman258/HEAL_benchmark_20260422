from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def serialize_tfm(T: Any) -> Optional[List[float]]:
    if T is None:
        return None
    try:
        arr = np.asarray(T, dtype=np.float64)
    except Exception:
        return None
    if arr.shape != (4, 4):
        return None
    return [float(x) for x in arr.reshape(-1).tolist()]


def se2_from_tfm(T: Any) -> Optional[Dict[str, float]]:
    if T is None:
        return None
    try:
        arr = np.asarray(T, dtype=np.float64)
    except Exception:
        return None
    if arr.shape != (4, 4):
        return None
    try:
        dx = float(arr[0, 3])
        dy = float(arr[1, 3])
        yaw = float(np.degrees(np.arctan2(arr[1, 0], arr[0, 0])))
    except Exception:
        return None
    return {"dx_m": dx, "dy_m": dy, "yaw_deg": yaw}


def compute_match_pairs(
    src_boxes: Sequence[object],
    dst_boxes: Sequence[object],
    T_rel: Any,
    *,
    bbox_type: str,
    distance_threshold_m: float,
    topk_pairs: int = 0,
) -> Optional[Dict[str, Any]]:
    """Compute method-agnostic correspondence stats under a given relative transform.

    Returns a JSON-serializable dict:
      {precision, matched, pairs?}

    - precision follows CorrespondingDetector.get_distance_corresponding_precision
      (typically in [-3, 0], where 0 is best).
    - matched is CorrespondingDetector.get_matched_num.
    - pairs is optional: [{src, dst, score}] sorted by score desc.
    """
    if not src_boxes or not dst_boxes:
        return None
    try:
        T_np = np.asarray(T_rel, dtype=np.float64)
    except Exception:
        return None
    if T_np.shape != (4, 4):
        return None

    try:
        from opencood.utils.path_utils import ensure_v2xreg_root_on_path

        ensure_v2xreg_root_on_path()
        from opencood.registration.estimators.box_matching import CorrespondingDetector  # type: ignore
        from opencood.registration.utils.geometry import implement_T_3dbox_object_list  # type: ignore
    except Exception:
        return None

    bbox_key = str(bbox_type or "detected")
    threshold = {bbox_key: float(distance_threshold_m)}

    try:
        aligned_src = implement_T_3dbox_object_list(T_np, src_boxes)
        detector = CorrespondingDetector(aligned_src, dst_boxes, distance_threshold=threshold)
        precision = float(detector.get_distance_corresponding_precision())
        matched = int(detector.get_matched_num())
    except Exception:
        return None

    payload: Dict[str, Any] = {"precision": precision, "matched": matched}
    if int(topk_pairs or 0) > 0:
        pairs: List[Dict[str, Any]] = []
        try:
            matches_with_score = detector.get_matches_with_score()
        except Exception:
            matches_with_score = {}
        if isinstance(matches_with_score, dict):
            for pair, score in matches_with_score.items():
                try:
                    src_idx, dst_idx = pair
                    score_f = float(score)
                except Exception:
                    continue
                pairs.append({"src": int(src_idx), "dst": int(dst_idx), "score": score_f})
        pairs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        payload["pairs"] = pairs[: int(topk_pairs)]

    return payload
