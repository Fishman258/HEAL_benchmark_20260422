# -*- coding: utf-8 -*-

from collections import OrderedDict

from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset


class V2V4REALBaseDataset(OPV2VBaseDataset):
    """
    V2V4Real uses the same OpenCDA/OpenCOOD folder convention as OPV2V.

    Paper setting (Table 2): each timestamp is evaluated from *both* ego views,
    effectively doubling the number of frames (e.g., 1993 -> 3986 on test).

    OpenCOOD's OPV2VBaseDataset assumes a single ego per timestamp (the first
    CAV in an OrderedDict). For V2V4Real we replicate each base sample for every
    non-negative CAV id and select that CAV as ego.
    """

    def __init__(self, params, visualize, train=True):
        # V2V4Real object labels are in the local LiDAR frame.
        self.force_vehicle_local_to_world = True
        # Enable multi-ego evaluation/training by default for V2V4Real to match the
        # official frame counts. Can be disabled in yaml via `v2v4real_multi_ego: false`.
        self.v2v4real_multi_ego = bool(params.get("v2v4real_multi_ego", True))
        self._v2v4real_index_map = None
        super().__init__(params, visualize, train)

    def reinitialize(self):
        super().reinitialize()

        if not self.v2v4real_multi_ego:
            self._v2v4real_index_map = None
            return

        # Non-invasive scenario filtering can yield an empty split (e.g. when a
        # scenario name exists only in train but not in val/test folders).
        if not getattr(self, "len_record", None):
            self._v2v4real_index_map = []
            return

        index_map = []
        # Map each "virtual" index -> (base_idx, ego_cav_id).
        for scenario_index, scenario_db in self.scenario_database.items():
            if not scenario_db:
                continue

            cav_ids = list(scenario_db.keys())
            if not cav_ids:
                continue

            first_cav = cav_ids[0]
            # scenario_db[first_cav] contains timestamp keys + a trailing 'ego' flag entry.
            timestamps = [k for k in scenario_db[first_cav].keys() if k != "ego"]

            base_offset = 0 if scenario_index == 0 else self.len_record[int(scenario_index) - 1]

            for ts_i, _ in enumerate(timestamps):
                base_idx = int(base_offset) + int(ts_i)
                for cav_id in cav_ids:
                    # Roadside units (negative IDs) should never be used as ego.
                    try:
                        if int(cav_id) < 0:
                            continue
                    except Exception:
                        pass
                    index_map.append((base_idx, cav_id))

        self._v2v4real_index_map = index_map
        base_len = int(self.len_record[-1]) if self.len_record else 0
        print("v2v4real_multi_ego enabled: base_len=%d, expanded_len=%d"
              % (base_len, len(self._v2v4real_index_map)))

    def __len__(self):
        if self.v2v4real_multi_ego and self._v2v4real_index_map is not None:
            return len(self._v2v4real_index_map)
        return super().__len__()

    def retrieve_base_data(self, idx):
        if not self.v2v4real_multi_ego or self._v2v4real_index_map is None:
            return super().retrieve_base_data(idx)

        base_idx, ego_cav_id = self._v2v4real_index_map[int(idx)]
        data = super().retrieve_base_data(int(base_idx))

        if ego_cav_id not in data:
            return data

        # Update ego flags and reorder so the ego CAV is the first entry.
        for cav_id, cav_content in data.items():
            cav_content["ego"] = cav_id == ego_cav_id

        reordered = OrderedDict()
        reordered[ego_cav_id] = data[ego_cav_id]
        for cav_id, cav_content in data.items():
            if cav_id == ego_cav_id:
                continue
            reordered[cav_id] = cav_content
        return reordered
