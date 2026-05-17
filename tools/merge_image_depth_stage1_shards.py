#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="Merge sharded image-depth stage1 cache files.")
    parser.add_argument("--shard-dir", type=str, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--expected-samples", type=int, default=0)
    parser.add_argument("--require-contiguous-keys", action="store_true")
    return parser.parse_args()


def main():
    opt = _parse_args()
    shard_dir = Path(opt.shard_dir)
    merged = {}
    for shard_idx in range(int(opt.num_shards)):
        shard_path = shard_dir / "stage1_boxes_image_depth_camera_model_shard{:02d}of{:02d}.json".format(
            shard_idx,
            int(opt.num_shards),
        )
        if not shard_path.exists():
            raise FileNotFoundError(str(shard_path))
        with shard_path.open("r", encoding="utf-8") as f:
            shard = json.load(f)
        for key, value in shard.items():
            if key in merged:
                raise ValueError("Duplicate sample key {} from {}".format(key, shard_path))
            merged[str(key)] = value

    ordered = OrderedDict((str(k), merged[str(k)]) for k in sorted(map(int, merged.keys())))
    if int(opt.expected_samples) > 0 and len(ordered) != int(opt.expected_samples):
        raise ValueError("Expected {} samples, got {}".format(int(opt.expected_samples), len(ordered)))
    if opt.require_contiguous_keys:
        expected = [str(i) for i in range(len(ordered))]
        actual = list(ordered.keys())
        if actual != expected:
            missing = sorted(set(expected) - set(actual), key=int)[:20]
            extra = sorted(set(actual) - set(expected), key=int)[:20]
            raise ValueError("Non-contiguous keys. Missing first: {} Extra first: {}".format(missing, extra))

    out = Path(opt.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(str(out))
    with out.open("w", encoding="utf-8") as f:
        json.dump(ordered, f)
    print("Merged {} samples to {}".format(len(ordered), out), flush=True)


if __name__ == "__main__":
    main()
