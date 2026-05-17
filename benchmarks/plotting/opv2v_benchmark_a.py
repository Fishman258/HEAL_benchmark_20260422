from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def write_plots(
    run_dir: Path,
    point_rows: Sequence[Mapping[str, Any]],
    profile_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"error": f"matplotlib unavailable: {type(exc).__name__}: {exc}"}

    line_order = ["baseline", "single", "oracle", "cbm", "freealign", "v2xregpp", "vips"]
    colors = {
        "baseline": "#4C566A",
        "single": "#A3BE8C",
        "oracle": "#D08770",
        "cbm": "#5E81AC",
        "freealign": "#B48EAD",
        "v2xregpp": "#BF616A",
        "vips": "#EBCB8B",
    }
    paths: Dict[str, str] = {}
    for metric in ("ap30", "ap50", "ap70"):
        plt.figure(figsize=(8, 4.6))
        for line in line_order:
            rows = [r for r in point_rows if r.get("series_id") == line and r.get(metric) not in {None, ""}]
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: float(r["noise"]))
            plt.plot(
                [float(r["noise"]) for r in rows],
                [float(r[metric]) for r in rows],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=line,
                color=colors.get(line),
            )
        plt.xlabel("pose noise std (m / deg, paired)")
        plt.ylabel(metric.upper())
        plt.title(f"OPV2V Benchmark A {metric.upper()}")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.xlim(0, 10)
        plt.ylim(0, 1.0)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        out = plots_dir / f"benchmarkA_{metric}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        paths[f"benchmarkA_{metric}"] = str(out)

    def _mean_for_line(line: str, key: str, *, include_solver: bool = False) -> float:
        vals = []
        for r in profile_rows:
            if r.get("line_id") != line:
                continue
            if r.get("job_type") == "solver" and not include_solver:
                continue
            val = r.get(key)
            if val not in {None, ""}:
                vals.append(float(val))
        return sum(vals) / len(vals) if vals else 0.0

    def _bar_plot(name: str, ylabel: str, values: Dict[str, float]) -> None:
        labels = [x for x in line_order if x in values]
        plt.figure(figsize=(8, 4.2))
        plt.bar(labels, [values[x] for x in labels], color=[colors.get(x, "#888888") for x in labels])
        plt.ylabel(ylabel)
        plt.xticks(rotation=25, ha="right")
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        out = plots_dir / f"{name}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        paths[name] = str(out)

    payload_values = {}
    for line in line_order:
        rows = [r for r in point_rows if r.get("series_id") == line]
        vals = [float(r.get("total_payload_bytes_per_sample_proxy") or 0.0) for r in rows]
        if vals:
            payload_values[line] = sum(vals) / len(vals)
    _bar_plot("payload_proxy_bytes_per_sample", "payload proxy bytes / sample", payload_values)

    time_values = {}
    solver_by_key = {
        (str(r.get("line_id")), float(r.get("noise", 0.0))): r
        for r in profile_rows
        if r.get("job_type") == "solver"
    }
    for line in line_order:
        vals = []
        for r in profile_rows:
            if r.get("line_id") != line or r.get("job_type") not in {"bound", "downstream"}:
                continue
            samples = float(r.get("samples") or 0.0)
            if samples <= 0:
                continue
            total = float(r.get("infer_sec") or 0.0)
            solver = solver_by_key.get((line, float(r.get("noise", 0.0))))
            if solver:
                total += float(solver.get("wall_sec") or 0.0)
            vals.append(total / samples)
        if vals:
            time_values[line] = sum(vals) / len(vals)
    _bar_plot("pipeline_sec_per_sample", "pipeline seconds / sample", time_values)

    mem_values = {
        line: _mean_for_line(line, "cuda_peak_allocated_bytes", include_solver=True) / (1024.0 * 1024.0)
        for line in line_order
    }
    mem_values = {k: v for k, v in mem_values.items() if v > 0}
    _bar_plot("cuda_peak_allocated_mib", "CUDA peak allocated MiB", mem_values)
    return paths
