#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_STAGE1 = Path("/data2/pubmap_full_training/stage1_cache/pubmap_pointpillar_bestval51/test/stage1_boxes.json")
DEFAULT_CONTRACT = Path("/data2/pubmap_full_training/contracts/test.json")
DEFAULT_SOURCE_DATASET = Path("/data2/pubmap_full_training/datasets/heal_pointpillar_opv2v/test")
DEFAULT_OPV2V_ROOT = Path("/data2/OPV2V/test")
DEFAULT_OUTPUT_ROOT = Path("/data2/pubmap_full_training/paired_benchmark_inputs")


PER_CAV_FIELDS = (
    "pred_corner3d_np_list",
    "pred_box3d_np_list",
    "pred_score_np_list",
    "uncertainty_np_list",
    "lidar_pose_clean_np",
    "lidar_pose_np",
)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")


def _frame_name(frame: Mapping[str, Any]) -> str:
    main_agent = str(frame["main_agent"])
    coop_agents = [str(x) for x in frame.get("coop_agents") or []]
    pair_agents = [x for x in coop_agents if x != main_agent]
    if not pair_agents:
        raise ValueError(f"frame has no non-ego pair agent: {frame}")
    return f"{frame['sequence']}__frame_{frame['frame']}__ego_{main_agent}__pair_{pair_agents[0]}"


def _pair_agent(frame: Mapping[str, Any]) -> str:
    main_agent = str(frame["main_agent"])
    coop_agents = [str(x) for x in frame.get("coop_agents") or []]
    for cav_id in coop_agents:
        if cav_id != main_agent:
            return cav_id
    raise ValueError(f"frame has no non-ego pair agent: {frame}")


def _single_value(entry: Mapping[str, Any], field: str, *, idx: int) -> Any:
    values = entry.get(field)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"stage1[{idx}] field {field} must be a single-item list")
    return values[0]


def _build_index(frames: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, str], int]:
    out: Dict[Tuple[str, str, str], int] = {}
    for idx, frame in enumerate(frames):
        key = (str(frame["sequence"]), str(frame["frame"]), str(frame["main_agent"]))
        if key in out:
            raise ValueError(f"duplicate contract key: {key}")
        out[key] = int(idx)
    return out


def _merge_stage1_entry(
    *,
    sample_idx: int,
    pair_idx: int,
    stage1: Mapping[str, Any],
    ego_id: str,
    pair_id: str,
) -> Dict[str, Any]:
    ego_entry = stage1.get(str(sample_idx))
    pair_entry = stage1.get(str(pair_idx))
    if not isinstance(ego_entry, Mapping):
        raise ValueError(f"missing stage1 entry for sample {sample_idx}")
    if not isinstance(pair_entry, Mapping):
        raise ValueError(f"missing stage1 reverse entry for sample {pair_idx}")

    ego_cavs = [str(x) for x in ego_entry.get("cav_id_list") or []]
    pair_cavs = [str(x) for x in pair_entry.get("cav_id_list") or []]
    if ego_cavs != [str(ego_id)]:
        raise ValueError(f"sample {sample_idx} cav_id_list={ego_cavs}, expected [{ego_id}]")
    if pair_cavs != [str(pair_id)]:
        raise ValueError(f"sample {pair_idx} cav_id_list={pair_cavs}, expected [{pair_id}]")

    merged = dict(ego_entry)
    merged["cav_id_list"] = [str(ego_id), str(pair_id)]
    for field in PER_CAV_FIELDS:
        merged[field] = [
            _single_value(ego_entry, field, idx=sample_idx),
            _single_value(pair_entry, field, idx=pair_idx),
        ]
    merged["source_stage1_indices"] = [int(sample_idx), int(pair_idx)]
    return merged


def _symlink_force(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def _copy_or_symlink_file(src: Path, dst: Path, *, materialize_pcd_from_source_dataset: Optional[Path]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if materialize_pcd_from_source_dataset is not None and src.suffix == ".pcd":
        shutil.copy2(src, dst)
        return
    dst.symlink_to(src)


def _build_dataset_view(
    *,
    frames: Sequence[Mapping[str, Any]],
    output_test_dir: Path,
    opv2v_root: Path,
    source_dataset: Optional[Path],
    materialize_ego_pcd: bool,
) -> Dict[str, Any]:
    created = 0
    linked_files = 0
    copied_files = 0
    missing: List[str] = []

    if output_test_dir.exists():
        shutil.rmtree(output_test_dir)
    output_test_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        ego_id = str(frame["main_agent"])
        pair_id = _pair_agent(frame)
        seq = str(frame["sequence"])
        timestamp = str(frame["frame"])
        scenario_dir = output_test_dir / _frame_name(frame)

        for cav_id in (ego_id, pair_id):
            cav_dir = scenario_dir / cav_id
            cav_dir.mkdir(parents=True, exist_ok=True)
            for suffix in (".yaml", ".pcd"):
                src = opv2v_root / seq / cav_id / f"{timestamp}{suffix}"
                if not src.exists():
                    missing.append(str(src))
                    continue

                materialize_src = None
                if materialize_ego_pcd and cav_id == ego_id and suffix == ".pcd" and source_dataset is not None:
                    source_pcd = source_dataset / _frame_name(frame) / ego_id / f"{timestamp}.pcd"
                    if source_pcd.exists():
                        materialize_src = source_pcd
                before_copied = materialize_src is not None
                _copy_or_symlink_file(materialize_src or src, cav_dir / f"{timestamp}{suffix}", materialize_pcd_from_source_dataset=materialize_src)
                if before_copied:
                    copied_files += 1
                else:
                    linked_files += 1
        created += 1

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"missing {len(missing)} OPV2V source files, first items:\n{preview}")

    return {
        "scenarios": int(created),
        "linked_files": int(linked_files),
        "copied_files": int(copied_files),
    }


def _load_contract_frames(path: Path) -> List[Mapping[str, Any]]:
    payload = _read_json(path)
    frames = payload.get("frames") if isinstance(payload, Mapping) else payload
    if not isinstance(frames, list):
        raise ValueError(f"contract must be a list or contain frames list: {path}")
    return frames


def build_inputs(opt: argparse.Namespace) -> Path:
    frames = _load_contract_frames(opt.contract)
    stage1 = _read_json(opt.stage1)
    if not isinstance(stage1, Mapping):
        raise ValueError(f"stage1 must be a JSON object: {opt.stage1}")
    if len(stage1) != len(frames):
        raise ValueError(f"stage1/contract sample count mismatch: stage1={len(stage1)} contract={len(frames)}")

    out_dir = opt.output_root / f"pubmap_paired_opv2v_{_timestamp()}"
    stage1_out = out_dir / "stage1_cache" / "test" / "stage1_boxes.json"
    dataset_out = out_dir / "datasets" / "heal_pointpillar_opv2v_paired" / "test"

    index = _build_index(frames)
    paired_stage1: Dict[str, Any] = {}
    for sample_idx, frame in enumerate(frames):
        ego_id = str(frame["main_agent"])
        pair_id = _pair_agent(frame)
        reverse_idx = index.get((str(frame["sequence"]), str(frame["frame"]), pair_id))
        if reverse_idx is None:
            raise ValueError(f"missing reverse frame for sample {sample_idx}: ego={ego_id} pair={pair_id}")
        paired_stage1[str(sample_idx)] = _merge_stage1_entry(
            sample_idx=sample_idx,
            pair_idx=reverse_idx,
            stage1=stage1,
            ego_id=ego_id,
            pair_id=pair_id,
        )

    _write_json(stage1_out, paired_stage1)
    dataset_stats = _build_dataset_view(
        frames=frames,
        output_test_dir=dataset_out,
        opv2v_root=opt.opv2v_root,
        source_dataset=opt.source_dataset if opt.source_dataset else None,
        materialize_ego_pcd=bool(opt.materialize_ego_pcd),
    )

    manifest = {
        "schema_version": "pubmap_paired_opv2v_inputs_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_stage1": str(opt.stage1),
        "source_contract": str(opt.contract),
        "source_dataset": "" if opt.source_dataset is None else str(opt.source_dataset),
        "opv2v_root": str(opt.opv2v_root),
        "stage1_output": str(stage1_out),
        "dataset_output": str(dataset_out),
        "samples": int(len(frames)),
        "dataset_stats": dataset_stats,
        "cav_count_distribution": {"2": int(len(paired_stage1))},
        "notes": [
            "stage1 entries are paired as [ego, pair] using contract reverse-frame mapping",
            "dataset scenario folders are named __ego_<id>__pair_<id> and contain two CAV dirs",
        ],
    }
    _write_json(out_dir / "manifest.json", manifest)

    if opt.update_latest:
        latest = opt.output_root / "latest_pubmap_paired_opv2v"
        if latest.is_symlink() or latest.exists():
            if latest.is_dir() and not latest.is_symlink():
                shutil.rmtree(latest)
            else:
                latest.unlink()
        latest.symlink_to(out_dir)

    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired OPV2V inputs from PubMap single-CAV stage1 cache.")
    parser.add_argument("--stage1", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--opv2v-root", type=Path, default=DEFAULT_OPV2V_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--materialize-ego-pcd",
        action="store_true",
        help="Copy ego PCDs from the source single-CAV dataset instead of symlinking original OPV2V ego PCDs.",
    )
    parser.add_argument("--update-latest", action="store_true")
    return parser.parse_args()


def main() -> None:
    opt = parse_args()
    out_dir = build_inputs(opt)
    print(out_dir)


if __name__ == "__main__":
    main()
