# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Transform points to voxels using sparse conv library
"""
import os
import sys

import numpy as np
import torch
from icecream import ic

from opencood.detection.preprocessing.base_preprocessor import \
    BasePreprocessor


class _NumpyVoxelGenerator:
    """
    Minimal CPU voxel generator used when `spconv` is unavailable.

    It mimics the output signature of spconv's VoxelGenerator:
      - voxels: (M, max_num_points, num_point_features)
      - coordinates: (M, 3) in (z, y, x) order
      - num_points_per_voxel: (M,)
    """

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.max_num_points = int(max_num_points)
        self.max_voxels = int(max_voxels)

        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size
        self.grid_size = np.floor(grid_size + 1e-6).astype(np.int64)

    def generate(self, points: np.ndarray):
        if points.size == 0:
            voxels = np.zeros((0, self.max_num_points, 4), dtype=np.float32)
            coors = np.zeros((0, 3), dtype=np.int32)
            num_points = np.zeros((0,), dtype=np.int32)
            return voxels, coors, num_points

        pts = np.asarray(points, dtype=np.float32)
        if pts.shape[1] < 4:
            raise ValueError(f"Expected point features >= 4 (x,y,z,intensity), got {pts.shape[1]}")

        pc_range_min = self.point_cloud_range[:3]
        pc_range_max = self.point_cloud_range[3:6]

        mask = (
            (pts[:, 0] >= pc_range_min[0])
            & (pts[:, 0] < pc_range_max[0])
            & (pts[:, 1] >= pc_range_min[1])
            & (pts[:, 1] < pc_range_max[1])
            & (pts[:, 2] >= pc_range_min[2])
            & (pts[:, 2] < pc_range_max[2])
        )
        pts = pts[mask]
        if pts.size == 0:
            voxels = np.zeros((0, self.max_num_points, 4), dtype=np.float32)
            coors = np.zeros((0, 3), dtype=np.int32)
            num_points = np.zeros((0,), dtype=np.int32)
            return voxels, coors, num_points

        coor_xyz = np.floor((pts[:, :3] - pc_range_min) / self.voxel_size).astype(np.int32)  # (x,y,z)
        valid = (
            (coor_xyz[:, 0] >= 0)
            & (coor_xyz[:, 0] < self.grid_size[0])
            & (coor_xyz[:, 1] >= 0)
            & (coor_xyz[:, 1] < self.grid_size[1])
            & (coor_xyz[:, 2] >= 0)
            & (coor_xyz[:, 2] < self.grid_size[2])
        )
        pts = pts[valid]
        coor_xyz = coor_xyz[valid]

        voxel_dict = {}
        voxel_list = []
        coors_list = []
        num_points_list = []

        for point, coor in zip(pts, coor_xyz):
            key = (int(coor[2]), int(coor[1]), int(coor[0]))  # (z,y,x)
            voxel_idx = voxel_dict.get(key)
            if voxel_idx is None:
                if len(voxel_list) >= self.max_voxels:
                    continue
                voxel_idx = len(voxel_list)
                voxel_dict[key] = voxel_idx
                voxel_list.append(np.zeros((self.max_num_points, 4), dtype=np.float32))
                coors_list.append(np.array(key, dtype=np.int32))
                num_points_list.append(0)

            curr_num = num_points_list[voxel_idx]
            if curr_num < self.max_num_points:
                voxel_list[voxel_idx][curr_num, :] = point[:4]
                num_points_list[voxel_idx] = curr_num + 1

        voxels = np.stack(voxel_list, axis=0) if voxel_list else np.zeros((0, self.max_num_points, 4), dtype=np.float32)
        coors = np.stack(coors_list, axis=0) if coors_list else np.zeros((0, 3), dtype=np.int32)
        num_points = np.asarray(num_points_list, dtype=np.int32) if num_points_list else np.zeros((0,), dtype=np.int32)
        return voxels, coors, num_points


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(SpVoxelPreprocessor, self).__init__(preprocess_params,
                                                  train)
        self.spconv = 0
        VoxelGenerator = None
        VoxelGeneratorGPU = None
        TorchPointToVoxel = None
        disable_spconv = str(os.environ.get("OPENCOOD_DISABLE_SPCONV", "") or "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        # NOTE: spconv CPU voxelizers can crash (SIGFPE) on some environments.
        # This escape hatch allows analysis scripts to force the safe numpy fallback.
        if not disable_spconv:
            try:
                # spconv v1.x
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator  # type: ignore
                self.spconv = 1
            except Exception:
                try:
                    # spconv v2.x
                    from cumm import tensorview as tv  # type: ignore
                    from spconv.utils import Point2VoxelCPU3d as VoxelGenerator  # type: ignore
                    from spconv.utils import Point2VoxelGPU3d as VoxelGeneratorGPU  # type: ignore
                    try:
                        from spconv.pytorch.utils import PointToVoxel as TorchPointToVoxel  # type: ignore
                    except Exception:
                        TorchPointToVoxel = None
                    self.tv = tv
                    self.spconv = 2
                except Exception:
                    # Fallback: pure numpy CPU voxelization.
                    self.spconv = 0
        self.lidar_range = self.params['cav_lidar_range']
        self.voxel_size = self.params['args']['voxel_size']
        self.max_points_per_voxel = self.params['args']['max_points_per_voxel']

        if train:
            self.max_voxels = self.params['args']['max_voxel_train']
        else:
            self.max_voxels = self.params['args']['max_voxel_test']

        grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        # use sparse conv library to generate voxel
        if self.spconv == 1:
            self.voxel_generator = VoxelGenerator(
                voxel_size=self.voxel_size,
                point_cloud_range=self.lidar_range,
                max_num_points=self.max_points_per_voxel,
                max_voxels=self.max_voxels
            )
        elif self.spconv == 2:
            env_flag = os.environ.get("OPENCOOD_VOXEL_GPU", "")
            use_gpu_flag = str(self.params.get("use_gpu_voxel", "")) if isinstance(self.params, dict) else ""
            use_gpu = str(env_flag or use_gpu_flag or "").lower() in {"1", "true", "yes", "on"}
            self.use_gpu = bool(use_gpu and torch.cuda.is_available() and VoxelGeneratorGPU is not None)
            self.device = torch.device("cuda" if self.use_gpu else "cpu")
            if self.use_gpu:
                if TorchPointToVoxel is not None:
                    self.voxel_generator = TorchPointToVoxel(
                        self.voxel_size,
                        self.lidar_range,
                        4,
                        self.max_voxels,
                        self.max_points_per_voxel,
                        device=self.device,
                    )
                    self.spconv = 4
                else:
                    # Fallback to low-level GPU voxelizer if torch wrapper is unavailable.
                    # Point2VoxelGPU3d expects positional args:
                    # (vsize_xyz, coors_range_xyz, num_point_features, max_num_voxels, max_num_points_per_voxel)
                    self.voxel_generator = VoxelGeneratorGPU(
                        self.voxel_size,
                        self.lidar_range,
                        4,
                        self.max_voxels,
                        self.max_points_per_voxel,
                    )
                    self.spconv = 3
            else:
                self.voxel_generator = VoxelGenerator(
                    vsize_xyz=self.voxel_size,
                    coors_range_xyz=self.lidar_range,
                    max_num_points_per_voxel=self.max_points_per_voxel,
                    num_point_features=4,
                    max_num_voxels=self.max_voxels
                )
        else:
            self.voxel_generator = _NumpyVoxelGenerator(
                voxel_size=self.voxel_size,
                point_cloud_range=self.lidar_range,
                max_num_points=self.max_points_per_voxel,
                max_voxels=self.max_voxels,
            )

    def preprocess(self, pcd_np):
        data_dict = {}
        voxel_output = None
        if self.spconv == 4:
            pts = torch.as_tensor(pcd_np, device=self.device, dtype=torch.float32)
            voxels, coordinates, num_points = self.voxel_generator(pts)
        elif self.spconv == 3:
            pts = torch.as_tensor(pcd_np, device=self.device, dtype=torch.float32)
            voxel_output = self.voxel_generator.point_to_voxel_hash(pts)
        elif self.spconv == 2:
            pcd_tv = self.tv.from_numpy(pcd_np)
            voxel_output = self.voxel_generator.point_to_voxel(pcd_tv)
        else:
            voxel_output = self.voxel_generator.generate(pcd_np)
        if voxel_output is not None:
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], \
                    voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output

        if self.spconv == 2:
            voxels = voxels.numpy()
            coordinates = coordinates.numpy()
            num_points = num_points.numpy()

        data_dict['voxel_features'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points

        return data_dict

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = []
        voxel_num_points = []
        voxel_coords = []

        for i in range(len(batch)):
            voxel_features.append(batch[i]['voxel_features'])
            voxel_num_points.append(batch[i]['voxel_num_points'])
            coords = batch[i]['voxel_coords']
            if isinstance(coords, torch.Tensor):
                pad = torch.full((coords.shape[0], 1), i, dtype=coords.dtype, device=coords.device)
                voxel_coords.append(torch.cat([pad, coords], dim=1))
            else:
                voxel_coords.append(
                    np.pad(coords, ((0, 0), (1, 0)),
                           mode='constant', constant_values=i))

        if isinstance(voxel_features[0], torch.Tensor):
            voxel_num_points = torch.cat(voxel_num_points, dim=0)
            voxel_features = torch.cat(voxel_features, dim=0)
            voxel_coords = torch.cat(voxel_coords, dim=0)
        else:
            voxel_num_points = torch.from_numpy(np.concatenate(voxel_num_points))
            voxel_features = torch.from_numpy(np.concatenate(voxel_features))
            voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}

    @staticmethod
    def collate_batch_dict(batch: dict):
        """
        Collate batch if the batch is a dictionary,
        eg: {'voxel_features': [feature1, feature2...., feature n]}

        Parameters
        ----------
        batch : dict

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        if isinstance(batch['voxel_features'][0], torch.Tensor):
            voxel_features = torch.cat(batch['voxel_features'], dim=0)
            voxel_num_points = torch.cat(batch['voxel_num_points'], dim=0)
        else:
            voxel_features = \
                torch.from_numpy(np.concatenate(batch['voxel_features']))
            voxel_num_points = \
                torch.from_numpy(np.concatenate(batch['voxel_num_points']))
        coords = batch['voxel_coords']
        voxel_coords = []

        for i in range(len(coords)):
            if isinstance(coords[i], torch.Tensor):
                pad = torch.full((coords[i].shape[0], 1), i, dtype=coords[i].dtype, device=coords[i].device)
                voxel_coords.append(torch.cat([pad, coords[i]], dim=1))
            else:
                voxel_coords.append(
                    np.pad(coords[i], ((0, 0), (1, 0)),
                           mode='constant', constant_values=i))
        if isinstance(voxel_coords[0], torch.Tensor):
            voxel_coords = torch.cat(voxel_coords, dim=0)
        else:
            voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}
