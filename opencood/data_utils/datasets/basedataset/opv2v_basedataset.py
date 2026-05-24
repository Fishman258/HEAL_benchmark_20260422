# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import re
from collections import OrderedDict
import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json
import random
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.transformation_utils import x_to_world
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.detection.preprocessing import build_preprocessor
from opencood.detection.postprocessing import build_postprocessor

class OPV2VBaseDataset(Dataset):
    @staticmethod
    def _order_cavs_by_scenario_ego_pair(cav_list, scenario_name: str):
        """
        Keep paired synthetic OPV2V views aligned with their folder contract.

        A folder named "...__ego_1045__pair_1054" must expose 1045 as the
        first CAV, otherwise OpenCOOD marks the wrong agent as ego after the
        usual sorted-directory ordering.
        """
        match = re.search(r"(?:^|__)ego_([^_]+)__pair_([^_]+)(?:__|$)", str(scenario_name))
        if not match:
            return cav_list
        preferred = [match.group(1), match.group(2)]
        out = []
        seen = set()
        for cav_id in preferred:
            if cav_id in cav_list and cav_id not in seen:
                out.append(cav_id)
                seen.add(cav_id)
        for cav_id in cav_list:
            if cav_id not in seen:
                out.append(cav_id)
                seen.add(cav_id)
        return out

    @staticmethod
    def _infer_split_name_from_root(root_dir: str, train: bool) -> str:
        """
        Best-effort split name inference for scenario filtering.

        OpenCOOD datasets typically use directory names {train, validate, test}.
        """
        try:
            base = os.path.basename(os.path.normpath(str(root_dir))).lower()
        except Exception:
            base = ""
        if base in {"train", "training"}:
            return "train"
        if base in {"val", "valid", "validate", "validation"}:
            return "val"
        if base in {"test", "testing"}:
            return "test"
        return "train" if train else "val"

    @staticmethod
    def _select_split_spec(spec, split_name: str):
        if spec is None:
            return None
        if isinstance(spec, dict):
            split_name = str(split_name).lower().strip()
            # Common synonyms.
            keys = [split_name]
            if split_name == "val":
                keys += ["validate", "validation"]
            if split_name == "train":
                keys += ["training"]
            if split_name == "test":
                keys += ["testing"]
            keys += ["all", "any", "default"]
            for k in keys:
                if k in spec:
                    return spec[k]
            return None
        return spec

    @staticmethod
    def _load_name_set(spec) -> set:
        """
        Load a set of scenario folder names from:
        - list/tuple/set of names
        - a repo-relative/absolute file path (.json/.txt/.yaml/.yml)
        """
        if spec is None:
            return set()
        if isinstance(spec, (list, tuple, set)):
            return {str(x) for x in spec}
        if not isinstance(spec, str):
            # Unknown type; ignore for backward compatibility.
            return set()

        from opencood.utils.path_utils import resolve_repo_path

        path = str(resolve_repo_path(spec))
        if path.endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
        elif path.endswith(".txt"):
            with open(path, "r") as f:
                data = [ln.strip() for ln in f.read().splitlines() if ln.strip() and not ln.strip().startswith("#")]
        elif path.endswith(".yaml") or path.endswith(".yml"):
            # Do not use OpenCOOD's load_yaml() here: it assumes the root is a dict
            # and may inject environment defaults. For scenario lists we want a raw load.
            import yaml as _yaml

            with open(path, "r") as f:
                data = _yaml.safe_load(f)
        else:
            # Treat as a single scenario name.
            data = [spec]

        if isinstance(data, dict):
            # Allow {"scenarios":[...]} or {"names":[...]}
            data = data.get("scenarios") or data.get("names") or data.get("scenario_list") or []
        if isinstance(data, (list, tuple, set)):
            return {str(x) for x in data}
        return set()

    @classmethod
    def _filter_scenario_folders(cls, scenario_folders, params: dict, split_name: str):
        """
        Apply non-invasive scenario filtering via yaml config.

        Supported keys (all optional):
        - scenario_list: list | path | {train/val/test: ...}
        - scenario_blacklist: list | path | {train/val/test: ...}
        """
        include_spec = cls._select_split_spec(params.get("scenario_list"), split_name)
        exclude_spec = cls._select_split_spec(params.get("scenario_blacklist"), split_name)

        include = cls._load_name_set(include_spec)
        exclude = cls._load_name_set(exclude_spec)

        if not include and not exclude:
            return scenario_folders

        out = []
        for p in scenario_folders:
            name = os.path.basename(os.path.normpath(p))
            if include and name not in include:
                continue
            if exclude and name in exclude:
                continue
            out.append(p)
        return out

    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.use_hdf5 = True

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        if 'data_augment' in params: # late and early
            self.data_augmentor = DataAugmentor(params['data_augment'], train)
        else: # intermediate
            self.data_augmentor = None

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        self.root_dir = root_dir 
        
        print("Dataset dir:", root_dir)

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                            else self.generate_object_center_camera
        self.generate_object_center_single = self.generate_object_center # will it follows 'self.generate_object_center' when 'self.generate_object_center' change?

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])

        split_name = self._infer_split_name_from_root(root_dir, bool(self.train))
        scenario_folders = self._filter_scenario_folders(scenario_folders, params, split_name)
        
        self.scenario_folders = scenario_folders
        self.reinitialize()


    def reinitialize(self):
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(self.scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            if self.train:
                cav_list = [x for x in os.listdir(scenario_folder)
                            if os.path.isdir(
                        os.path.join(scenario_folder, x))]
                # cav_list = sorted(cav_list)
                random.shuffle(cav_list)
            else:
                cav_list = sorted([x for x in os.listdir(scenario_folder)
                                   if os.path.isdir(
                        os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            """
            roadside unit data's id is always negative, so here we want to
            make sure they will be in the end of the list as they shouldn't
            be ego vehicle.
            """
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            """
            make the first cav to be ego modality
            """
            if getattr(self, "heterogeneous", False):
                scenario_name = scenario_folder.split("/")[-1]
                cav_list = self.adaptor.reorder_cav_list(cav_list, scenario_name)

            scenario_name = scenario_folder.split("/")[-1]
            cav_list = self._order_cavs_by_scenario_ego_pair(cav_list, scenario_name)


            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs reinitialize')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
                
                # this timestamp is not ready
                yaml_files = [x for x in yaml_files if not ("2021_08_20_21_10_24" in x and "000265" in x)]

                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()
                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    camera_files = self.find_camera_files(cav_path, 
                                                timestamp)
                    depth_files = self.find_depth_files(cav_path, timestamp)

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp]['cameras'] = \
                        camera_files
                    self.scenario_database[i][cav_id][timestamp]['depths'] = \
                        depth_files

                    if getattr(self, "heterogeneous", False):
                        scenario_name = scenario_folder.split("/")[-1]

                        cav_modality = self.adaptor.reassign_cav_modality(self.modality_assignment[scenario_name][cav_id] , j)

                        self.scenario_database[i][cav_id][timestamp]['modality_name'] = cav_modality

                        self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                            self.adaptor.switch_lidar_channels(cav_modality, lidar_file)


                   # load extra data
                    for file_extension in self.add_data_extension:
                        file_name = \
                            os.path.join(cav_path,
                                         timestamp + '_' + file_extension)

                        self.scenario_database[i][cav_id][timestamp][
                            file_extension] = file_name                  

                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the 
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False
        # Some non-invasive scenario filters can legitimately yield an empty split.
        # Avoid IndexError so callers get a clean 0-length dataset.
        if self.len_record:
            print("len:", self.len_record[-1])
        else:
            print("len: 0")

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            # load param file: json is faster than yaml
            json_file = cav_content[timestamp_key]['yaml'].replace("yaml", "json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data[cav_id]['params'] = json.load(f)
            else:
                data[cav_id]['params'] = \
                    load_yaml(cav_content[timestamp_key]['yaml'])

            # V2V4Real stores poses as 4x4 transformation matrices in yaml. The rest of the
            # OpenCOOD pipeline expects 6-DoF pose vectors [x,y,z,roll,yaw,pitch] in degrees.
            #
            # NOTE: V2V4Real object labels are in the *local LiDAR frame* (KITTI-style),
            # while OPV2V labels are typically in a global/world frame. When we detect a
            # 4x4 lidar_pose (V2V4Real), convert vehicle labels to world coordinates so the
            # existing OpenCOOD box projection code (world -> ego) works as intended.
            params = data[cav_id].get("params") or {}
            if isinstance(params, dict) and "lidar_pose" in params:
                pose = params.get("lidar_pose")
                pose_tfm = None
                try:
                    pose_arr = np.asarray(pose, dtype=np.float32)
                except Exception:
                    pose_arr = None
                if pose_arr is not None:
                    if pose_arr.shape == (4, 4):
                        pose_tfm = pose_arr
                        params["lidar_pose"] = np.asarray(tfm_to_pose(pose_arr), dtype=np.float32)
                    elif pose_arr.shape == (6,):
                        params["lidar_pose"] = pose_arr
                    elif pose_arr.shape == (3,):
                        full = np.zeros((6,), dtype=np.float32)
                        full[[0, 1, 4]] = pose_arr
                        params["lidar_pose"] = full

                # Convert per-frame vehicle labels from LiDAR-local to world when needed.
                #
                # V2V4Real uses KITTI-style labels in the *local LiDAR frame* while providing a
                # 4x4 `lidar_pose` transformation. OpenCOOD's downstream projection expects
                # world-frame objects, so we need to lift them to world coordinates.
                #
                # Some scenarios have small global translations (near the origin), where a
                # naive heuristic may fail. Allow dataset-specific override via
                # `force_vehicle_local_to_world` (set by V2V4REALBaseDataset).
                if pose_tfm is not None and isinstance(params.get("vehicles"), dict):
                    vehicles = params.get("vehicles") or {}
                    force_local_to_world = bool(getattr(self, "force_vehicle_local_to_world", False))
                    # Heuristic: if objects are near the origin while lidar pose is far from it,
                    # the labels are likely in the local LiDAR frame.
                    trans_norm = float(np.linalg.norm(np.asarray(pose_tfm[:2, 3], dtype=np.float32)))
                    local_like = False
                    if force_local_to_world:
                        local_like = True
                    else:
                        for v in vehicles.values():
                            loc = np.asarray((v or {}).get("location", [0, 0, 0]), dtype=np.float32)
                            if float(np.linalg.norm(loc[:2])) < 200.0 and trans_norm > 200.0:
                                local_like = True
                                break
                    if local_like and vehicles:
                        Tw_lidar = np.asarray(pose_tfm, dtype=np.float32)
                        vehicles_world = {}
                        for obj_id, obj in vehicles.items():
                            try:
                                obj = obj or {}
                                loc = np.asarray(obj.get("location", [0, 0, 0]), dtype=np.float32)
                                center = np.asarray(obj.get("center", [0, 0, 0]), dtype=np.float32)
                                ang = np.asarray(obj.get("angle", [0, 0, 0]), dtype=np.float32)
                                obj_pose_local = [
                                    float(loc[0] + center[0]),
                                    float(loc[1] + center[1]),
                                    float(loc[2] + center[2]),
                                    float(ang[0]),
                                    float(ang[1]),
                                    float(ang[2]),
                                ]
                                # Treat LiDAR frame as the 'world' to build T_lidar_obj.
                                T_lidar_obj = x_to_world(obj_pose_local)
                                T_world_obj = Tw_lidar @ T_lidar_obj
                                obj_pose_world = tfm_to_pose(T_world_obj)

                                new_obj = dict(obj)
                                new_obj["location"] = [float(obj_pose_world[0]), float(obj_pose_world[1]), float(obj_pose_world[2])]
                                new_obj["angle"] = [float(obj_pose_world[3]), float(obj_pose_world[4]), float(obj_pose_world[5])]
                                # Avoid double-applying 'center' after we've baked it into 'location'.
                                new_obj["center"] = [0, 0, 0]
                                vehicles_world[obj_id] = new_obj
                            except Exception:
                                vehicles_world[obj_id] = obj
                        params["vehicles"] = vehicles_world
            if isinstance(params, dict) and "true_ego_pos" in params:
                pose = params.get("true_ego_pos")
                try:
                    pose_arr = np.asarray(pose, dtype=np.float32)
                except Exception:
                    pose_arr = None
                if pose_arr is not None and pose_arr.shape == (4, 4):
                    params["true_ego_pos"] = np.asarray(tfm_to_pose(pose_arr), dtype=np.float32)
            data[cav_id]["params"] = params

            # load camera file: hdf5 is faster than png
            hdf5_file = cav_content[timestamp_key]['cameras'][0].replace("camera0.png", "imgs.hdf5")

            if self.use_hdf5 and os.path.exists(hdf5_file):
                with h5py.File(hdf5_file, "r") as f:
                    data[cav_id]['camera_data'] = []
                    data[cav_id]['depth_data'] = []
                    for i in range(4):
                        if self.load_camera_file:
                            data[cav_id]['camera_data'].append(Image.fromarray(f[f'camera{i}'][()]))
                        if self.load_depth_file:
                            data[cav_id]['depth_data'].append(Image.fromarray(f[f'depth{i}'][()]))
            else:
                if self.load_camera_file:
                    data[cav_id]['camera_data'] = \
                        load_camera_data(cav_content[timestamp_key]['cameras'])
                if self.load_depth_file:
                    data[cav_id]['depth_data'] = \
                        load_camera_data(cav_content[timestamp_key]['depths']) 

            # load lidar file
            if self.load_lidar_file or self.visualize:
                data[cav_id]['lidar_np'] = \
                    pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar'])

            if getattr(self, "heterogeneous", False):
                data[cav_id]['modality_name'] = cav_content[timestamp_key]['modality_name']

            for file_extension in self.add_data_extension:
                # if not find in the current directory
                # go to additional folder
                if not os.path.exists(cav_content[timestamp_key][file_extension]):
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("train","additional/train")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("validate","additional/validate")
                    cav_content[timestamp_key][file_extension] = cav_content[timestamp_key][file_extension].replace("test","additional/test")

                # Some OPV2V releases only provide `additional/train/*` (no test/validate).
                # Avoid noisy OpenCV warnings and let downstream code handle missing extras.
                if not os.path.exists(cav_content[timestamp_key][file_extension]):
                    data[cav_id][file_extension] = None
                    continue
                    
                if '.yaml' in file_extension:
                    data[cav_id][file_extension] = \
                        load_yaml(cav_content[timestamp_key][file_extension])
                else:
                    data[cav_id][file_extension] = \
                        cv2.imread(cav_content[timestamp_key][file_extension])


        return data

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    @staticmethod
    def find_camera_files(cav_path, timestamp, sensor="camera"):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        sensor : str
            "camera" or "depth" 

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    @staticmethod
    def find_depth_files(cav_path, timestamp):
        """
        Depth file naming differs across OPV2V releases.

        This repo historically expects png files named:
          {timestamp}_depth{0..3}.png (under OPV2V_Hetero)

        The provided depth pack may instead contain npy files named:
          {timestamp}_camera{0..3}_depth.npy (also under a parallel tree).

        We try multiple patterns and return the first existing path per camera.
        """
        # Try depth files under OPV2V_Hetero first, then fall back to the same tree.
        candidate_dirs = []
        hetero_dir = cav_path.replace("OPV2V", "OPV2V_Hetero")
        if hetero_dir != cav_path:
            candidate_dirs.append(hetero_dir)
        candidate_dirs.append(cav_path)

        patterns = [
            lambda i: f"{timestamp}_depth{i}.png",
            lambda i: f"{timestamp}_depth{i}.npy",
            lambda i: f"{timestamp}_camera{i}_depth.npy",
            lambda i: f"{timestamp}_camera{i}_depth.png",
        ]

        depth_files = []
        for i in range(4):
            chosen = None
            for d in candidate_dirs:
                for pat in patterns:
                    cand = os.path.join(d, pat(i))
                    if os.path.exists(cand):
                        chosen = cand
                        break
                if chosen is not None:
                    break
            # Keep a deterministic path for clearer error messages later.
            if chosen is None:
                chosen = os.path.join(candidate_dirs[0], f"{timestamp}_depth{i}.png")
            depth_files.append(chosen)

        return depth_files


    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask


    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self, 
                                cav_contents, 
                                reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.
        
        visibility_map : np.ndarray
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_visible_object_center(
            cav_contents, reference_lidar_pose
        )

    def get_ext_int(self, params, camera_id):
        camera_coords = np.array(params["camera%d" % camera_id]["cords"]).astype(
            np.float32)
        camera_to_lidar = x1_to_x2(
            camera_coords, params["lidar_pose_clean"]
        ).astype(np.float32)  # T_LiDAR_camera
        camera_to_lidar = camera_to_lidar @ np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32)  # UE4 coord to opencv coord
        camera_intrinsic = np.array(params["camera%d" % camera_id]["intrinsic"]).astype(
            np.float32
        )
        return camera_to_lidar, camera_intrinsic
