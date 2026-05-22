from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Stage1BoxAlignPoseCorrector:
    """
    Apply the OpenCOOD `box_align_v2` pose refinement using cached stage-1 boxes.

    This is a thin wrapper that keeps pose-correction logic out of dataset code.
    """

    box_align_args: Dict[str, Any]

    def apply(
        self,
        *,
        sample_idx: int,
        cav_id_list: Sequence[Any],
        base_data_dict: MutableMapping[Any, Dict[str, Any]],
        stage1_result: Mapping[str, Any],
    ) -> bool:
        """
        Update `base_data_dict[cav_id]['params']['lidar_pose']` in-place.

        Returns:
            bool: True if refinement applied, False otherwise.
        """
        key = str(sample_idx)
        if key not in stage1_result:
            return False
        stage1_content = stage1_result.get(key)
        if stage1_content is None:
            return False

        from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np

        all_agent_id_list = stage1_content.get("cav_id_list") or []
        all_agent_corners_list = stage1_content.get("pred_corner3d_np_list") or []
        all_agent_uncertainty_list = stage1_content.get("uncertainty_np_list") or []

        if not all_agent_id_list or not all_agent_corners_list:
            return False

        cur_agent_pose = []
        cur_agent_in_all_agent = []
        for cav_id in cav_id_list:
            if cav_id not in base_data_dict:
                return False
            try:
                cur_agent_in_all_agent.append(all_agent_id_list.index(cav_id))
            except ValueError:
                return False
            cur_agent_pose.append(base_data_dict[cav_id]["params"]["lidar_pose"])

        cur_agent_pose = np.array(cur_agent_pose, dtype=np.float64)
        pred_corners_list: List[np.ndarray] = []
        uncertainty_list: List[np.ndarray] = []
        for idx_in_all in cur_agent_in_all_agent:
            pred_corners_list.append(np.array(all_agent_corners_list[idx_in_all], dtype=np.float64))
            uncertainty_list.append(np.array(all_agent_uncertainty_list[idx_in_all], dtype=np.float64))

        if sum(len(pred) for pred in pred_corners_list) == 0:
            return False

        refined_pose_dof3 = box_alignment_relative_sample_np(
            pred_corners_list,
            cur_agent_pose,
            uncertainty_list=uncertainty_list,
            **(self.box_align_args or {}),
        )
        cur_agent_pose[:, [0, 1, 4]] = refined_pose_dof3
        for i, cav_id in enumerate(cav_id_list):
            base_data_dict[cav_id]["params"]["lidar_pose"] = cur_agent_pose[i].tolist()
        return True


__all__ = ["Stage1BoxAlignPoseCorrector"]

