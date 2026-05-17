#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
from collections import OrderedDict

import numpy as np
import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import box_utils
from opencood.utils.common_utils import limit_period


def _parse_args():
    parser = argparse.ArgumentParser(description="Export per-CAV stage1 boxes.")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--nms_pre_topk",
        type=int,
        default=0,
        help="Keep only top-k scored boxes per CAV before rotated NMS. <=0 disables it.",
    )
    return parser.parse_args()


def _resolve_split_hypes(hypes, split):
    hypes = copy.deepcopy(hypes)
    split = str(split or "test").lower()
    if split == "test":
        hypes["validate_dir"] = hypes["test_dir"]
        train = False
    elif split in {"val", "validate", "validation"}:
        train = False
        split = "val"
    elif split == "train":
        train = True
    else:
        raise ValueError("Unsupported split: {}".format(split))
    return hypes, train, split


def _to_jsonable(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _load_checkpoint_or_path(checkpoint_path, model):
    try:
        loaded_state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(loaded_state_dict, strict=False)
        return model
    except Exception:
        _, model = train_utils.load_saved_model(os.path.dirname(str(checkpoint_path)), model)
        return model


def _post_process_stage1(output_dict, post_processor, anchor_box, nms_pre_topk=0):
    cls_preds = output_dict["cls_preds"]
    reg_preds = output_dict["reg_preds"]

    cls_prob = torch.sigmoid(cls_preds.permute(0, 2, 3, 1).contiguous())
    batch_box3d = post_processor.delta_to_boxes3d(reg_preds, anchor_box)
    mask = torch.gt(cls_prob, post_processor.params["target_args"]["score_threshold"])
    batch_num_box_count = [int(m.sum()) for m in mask]

    mask_flat = mask.view(1, -1)
    mask_reg = mask_flat.unsqueeze(2).repeat(1, 1, 7)
    boxes3d = torch.masked_select(batch_box3d.view(-1, 7), mask_reg[0]).view(-1, 7)
    scores = torch.masked_select(cls_prob.view(-1), mask_flat[0])

    if "dir_preds" in output_dict and len(boxes3d) != 0:
        dir_preds = output_dict["dir_preds"]
        dir_offset = post_processor.params["dir_args"]["dir_offset"]
        num_bins = post_processor.params["dir_args"]["num_bins"]
        dir_cls_preds = dir_preds.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins)
        dir_cls_preds = dir_cls_preds[mask_flat]
        dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
        period = 2 * np.pi / num_bins
        dir_rot = limit_period(boxes3d[..., 6] - dir_offset, 0, period)
        boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(boxes3d.dtype)
        boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi)

    if len(boxes3d) == 0:
        return [], [], [], []

    pred_box3d_original = boxes3d.detach()

    pred_corner_list = []
    pred_box_list = []
    score_list = []
    uncertainty_list = []
    cur_idx = 0
    for n in batch_num_box_count:
        cur_boxes = pred_box3d_original[cur_idx : cur_idx + n]
        cur_scores = scores[cur_idx : cur_idx + n]
        cur_idx += n
        if n == 0:
            pred_corner_list.append([])
            pred_box_list.append([])
            score_list.append([])
            uncertainty_list.append([])
            continue
        if int(nms_pre_topk) > 0 and int(cur_scores.numel()) > int(nms_pre_topk):
            topk_index = torch.topk(cur_scores, k=int(nms_pre_topk)).indices
            cur_boxes = cur_boxes[topk_index]
            cur_scores = cur_scores[topk_index]
        cur_corners = box_utils.boxes_to_corners_3d(cur_boxes, order=post_processor.params["order"])
        keep_index = box_utils.nms_rotated(cur_corners, cur_scores, post_processor.params["nms_thresh"])
        keep_corners = cur_corners[keep_index]
        keep_boxes = cur_boxes[keep_index]
        keep_scores = cur_scores[keep_index]
        pred_corner_list.append(keep_corners.detach().cpu().numpy().tolist())
        pred_box_list.append(keep_boxes.detach().cpu().numpy().tolist())
        score_list.append(keep_scores.detach().cpu().numpy().tolist())
        uncertainty_list.append([[0.0, 0.0] for _ in range(int(keep_scores.numel()))])

    return pred_corner_list, pred_box_list, score_list, uncertainty_list


def main():
    opt = _parse_args()
    if int(opt.num_shards) <= 0:
        raise ValueError("--num_shards must be >= 1")
    if int(opt.shard_index) < 0 or int(opt.shard_index) >= int(opt.num_shards):
        raise ValueError("--shard_index must be in [0, num_shards)")
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    hypes, train_flag, split_name = _resolve_split_hypes(hypes, opt.split)

    dataset = build_dataset(hypes, visualize=False, train=train_flag)
    model = train_utils.create_model(hypes)
    model = _load_checkpoint_or_path(opt.checkpoint_path, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    anchor_box = torch.from_numpy(dataset.post_processor.generate_anchor_box()).to(device)
    total_len = len(dataset)
    max_samples = opt.max_samples if opt.max_samples is not None else total_len
    stage1 = OrderedDict()

    with torch.no_grad():
        for idx in range(min(int(max_samples), total_len)):
            if int(opt.num_shards) > 1 and (idx % int(opt.num_shards)) != int(opt.shard_index):
                continue
            base_data_dict = dataset.retrieve_base_data(idx)
            cav_id_list = list(base_data_dict.keys())
            pred_corner3d_np_list = []
            pred_box3d_np_list = []
            pred_score_np_list = []
            uncertainty_np_list = []

            for cav_id in cav_id_list:
                single_base = copy.deepcopy(base_data_dict[cav_id])
                single_base["ego"] = True
                params = single_base.setdefault("params", {})
                if "lidar_pose_clean" not in params:
                    params["lidar_pose_clean"] = params.get("lidar_pose")
                processed = dataset.get_item_single_car(single_base)
                batch = dataset.collate_batch_train([{"ego": processed}])
                batch = train_utils.to_device(batch, device)
                output = model(batch["ego"])
                corners, boxes, scores, uncertainty = _post_process_stage1(
                    output, dataset.post_processor, anchor_box, opt.nms_pre_topk
                )
                pred_corner3d_np_list.append(corners[0] if corners else [])
                pred_box3d_np_list.append(boxes[0] if boxes else [])
                pred_score_np_list.append(scores[0] if scores else [])
                uncertainty_np_list.append(uncertainty[0] if uncertainty else [])

            rec = OrderedDict()
            rec["pred_corner3d_np_list"] = pred_corner3d_np_list
            rec["pred_box3d_np_list"] = pred_box3d_np_list
            rec["pred_score_np_list"] = pred_score_np_list
            rec["uncertainty_np_list"] = uncertainty_np_list
            rec["lidar_pose_np"] = [
                _to_jsonable(np.asarray(base_data_dict[cav_id]["params"].get("lidar_pose"), dtype=np.float32))
                for cav_id in cav_id_list
            ]
            rec["lidar_pose_clean_np"] = [
                _to_jsonable(
                    np.asarray(
                        base_data_dict[cav_id]["params"].get(
                            "lidar_pose_clean",
                            base_data_dict[cav_id]["params"].get("lidar_pose"),
                        ),
                        dtype=np.float32,
                    )
                )
                for cav_id in cav_id_list
            ]
            rec["cav_id_list"] = [str(x) for x in cav_id_list]
            rec["veh_frame_id"] = None
            rec["infra_frame_id"] = None
            rec["bev_range"] = _to_jsonable(np.asarray(dataset.params["model"]["args"]["lidar_range"], dtype=np.float32))
            stage1[str(idx)] = rec

            if opt.log_interval and len(stage1) % int(opt.log_interval) == 0:
                print(
                    "[stage1-export] shard {}/{} processed {} samples (idx={} / {})".format(
                        int(opt.shard_index), int(opt.num_shards), len(stage1), idx, total_len
                    ),
                    flush=True,
                )

    output_dir = os.path.join(opt.output_dir, split_name)
    os.makedirs(output_dir, exist_ok=True)
    if int(opt.num_shards) > 1:
        filename = "stage1_boxes_shard{:02d}of{:02d}.json".format(int(opt.shard_index), int(opt.num_shards))
    else:
        filename = "stage1_boxes.json"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stage1, f, sort_keys=True)
    print("Wrote {} samples to {}".format(len(stage1), output_path), flush=True)


if __name__ == "__main__":
    main()
