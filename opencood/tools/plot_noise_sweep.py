import argparse
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_xlim(raw: str) -> Optional[Tuple[float, float]]:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--xlim must be 'min,max' (e.g., 0,10)")
    return float(parts[0]), float(parts[1])


def _extract_x(
    dump: Mapping[str, Any],
    *,
    x_key: str,
) -> List[float]:
    pos_list = dump.get("pos_std_list") or []
    rot_list = dump.get("rot_std_list") or []
    rel_stats = dump.get("rel_error_stats") or []

    if x_key == "pos_std":
        return [float(x) for x in pos_list]
    if x_key == "rot_std":
        return [float(x) for x in rot_list]

    def _get_rel(metric: str, stat: str) -> List[float]:
        out = []
        for entry in rel_stats:
            if not isinstance(entry, dict):
                out.append(float("nan"))
                continue
            blob = entry.get(metric) or {}
            if not isinstance(blob, dict):
                out.append(float("nan"))
                continue
            value = blob.get(stat)
            try:
                out.append(float(value))
            except Exception:
                out.append(float("nan"))
        return out

    if x_key.startswith("rel_trans_"):
        stat = x_key.split("_")[-1]
        return _get_rel("rel_trans_m", stat)
    if x_key.startswith("rel_yaw_"):
        stat = x_key.split("_")[-1]
        return _get_rel("rel_yaw_deg", stat)
    raise ValueError(f"Unsupported --x {x_key}")


def _is_grid(pos_list: Sequence[float], rot_list: Sequence[float]) -> bool:
    pos_unique = sorted({float(v) for v in pos_list})
    rot_unique = sorted({float(v) for v in rot_list})
    if len(pos_unique) <= 1 or len(rot_unique) <= 1:
        return False
    return len(pos_list) == len(rot_list) == len(pos_unique) * len(rot_unique)


def _plot_curve(
    *,
    series: List[Tuple[str, List[float], List[float]]],
    x_label: str,
    y_label: str,
    title: str,
    xlim: Optional[Tuple[float, float]],
    out_path: str,
    dpi: int,
) -> None:
    plt.figure(figsize=(7, 4))
    for label, xs, ys in series:
        points = []
        for x, y in zip(xs, ys):
            if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                continue
            points.append((float(x), float(y)))
        points.sort(key=lambda p: p[0])
        if not points:
            continue
        x_sorted = [p[0] for p in points]
        y_sorted = [p[1] for p in points]
        plt.plot(x_sorted, y_sorted, marker="o", linewidth=1.5, markersize=3.5, label=label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(True, linestyle="--", alpha=0.35)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def _plot_heatmap(
    *,
    label: str,
    pos_list: Sequence[float],
    rot_list: Sequence[float],
    values: Sequence[float],
    metric: str,
    title: str,
    out_path: str,
    dpi: int,
) -> None:
    pos_unique = sorted({float(v) for v in pos_list})
    rot_unique = sorted({float(v) for v in rot_list})
    pos_to_idx = {v: i for i, v in enumerate(pos_unique)}
    rot_to_idx = {v: i for i, v in enumerate(rot_unique)}
    grid = np.full((len(rot_unique), len(pos_unique)), np.nan, dtype=np.float64)
    for p, r, v in zip(pos_list, rot_list, values):
        try:
            i = rot_to_idx[float(r)]
            j = pos_to_idx[float(p)]
            grid[i, j] = float(v)
        except Exception:
            continue

    plt.figure(figsize=(7, 4))
    im = plt.imshow(
        grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[min(pos_unique), max(pos_unique), min(rot_unique), max(rot_unique)],
    )
    plt.colorbar(im, label=metric)
    plt.xlabel("pos_std (m)")
    plt.ylabel("rot_std (deg)")
    final_title = title or f"{label}: {metric} heatmap"
    plt.title(final_title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AP curves/heatmaps from inference_w_noise YAML dumps.")
    parser.add_argument("--yaml", nargs="+", required=True, help="One or more AP030507_*.yaml files.")
    parser.add_argument("--label", nargs="*", default=None, help="Optional labels (same count as --yaml).")
    parser.add_argument("--metric", choices=["ap30", "ap50", "ap70"], default="ap50")
    parser.add_argument(
        "--x",
        choices=[
            "pos_std",
            "rot_std",
            "rel_trans_mean",
            "rel_trans_median",
            "rel_trans_p90",
            "rel_yaw_mean",
            "rel_yaw_median",
            "rel_yaw_p90",
        ],
        default="pos_std",
        help="X axis source. Use rel_* to plot against post-correction pose error.",
    )
    parser.add_argument(
        "--slice",
        choices=["all", "trans", "rot"],
        default="all",
        help="Optional slice over (pos_std, rot_std) pairs: "
             "trans keeps rot_std==0; rot keeps pos_std==0; all keeps everything.",
    )
    parser.add_argument("--xlim", type=str, default="0,10", help="x-axis limits as 'min,max'. Use empty to disable.")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--out", type=str, default="", help="Output PNG path. Default: next to first YAML.")
    parser.add_argument("--dpi", type=int, default=150)
    opt = parser.parse_args()

    labels = opt.label or []
    if labels and len(labels) != len(opt.yaml):
        raise ValueError("--label must match the number of --yaml inputs (or be omitted).")

    series = []
    grid_candidates = []
    for idx, path in enumerate(opt.yaml):
        dump = _load_yaml(path)
        label = labels[idx] if labels else (dump.get("pose_correction") or os.path.basename(path))
        pos_list = [float(v) for v in (dump.get("pos_std_list") or [])]
        rot_list = [float(v) for v in (dump.get("rot_std_list") or [])]
        y_list = [float(v) for v in (dump.get(opt.metric) or [])]
        x_list = [float(v) for v in _extract_x(dump, x_key=opt.x)]

        indices = list(range(min(len(pos_list), len(rot_list), len(y_list), len(x_list))))
        if opt.slice != "all":
            tol = 1e-6
            kept = []
            for i in indices:
                p = float(pos_list[i])
                r = float(rot_list[i])
                if opt.slice == "trans" and abs(r) > tol:
                    continue
                if opt.slice == "rot" and abs(p) > tol:
                    continue
                kept.append(i)
            indices = kept

        pos_list = [pos_list[i] for i in indices]
        rot_list = [rot_list[i] for i in indices]
        y_list = [y_list[i] for i in indices]
        x_list = [x_list[i] for i in indices]
        series.append((str(label), x_list, [float(v) for v in y_list]))
        grid_candidates.append((str(label), pos_list, rot_list, y_list, path))

    xlim = _parse_xlim(opt.xlim)
    out_path = opt.out
    if not out_path:
        base_dir = os.path.dirname(os.path.abspath(opt.yaml[0]))
        stem = os.path.splitext(os.path.basename(opt.yaml[0]))[0]
        out_path = os.path.join(base_dir, f"{stem}_{opt.metric}_vs_{opt.x}.png")

    if len(opt.yaml) == 1:
        label, pos_list, rot_list, y_list, _ = grid_candidates[0]
        if _is_grid(pos_list, rot_list) and opt.x in {"pos_std", "rot_std"}:
            heat_out = out_path.replace(".png", "_heatmap.png")
            _plot_heatmap(
                label=label,
                pos_list=[float(v) for v in pos_list],
                rot_list=[float(v) for v in rot_list],
                values=[float(v) for v in y_list],
                metric=opt.metric,
                title=opt.title,
                out_path=heat_out,
                dpi=int(opt.dpi),
            )
            return

    _plot_curve(
        series=series,
        x_label=opt.x,
        y_label=opt.metric,
        title=opt.title,
        xlim=xlim,
        out_path=out_path,
        dpi=int(opt.dpi),
    )


if __name__ == "__main__":
    main()
