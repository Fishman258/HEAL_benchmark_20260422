#!/usr/bin/env python3
"""Common helpers for stage-based benchmark artifacts.

The pipeline layer owns experiment organization.  Core algorithms stay under
`opencood/`; these helpers only create stable run directories and records.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

try:
    import yaml
except Exception:  # pragma: no cover - yaml is already a project dependency.
    yaml = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "pipelines"


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))
    return cleaned.strip("_") or "run"


def resolve_path(value: Any, *, root: Path = ROOT) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def shell_join(parts: Sequence[Any]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


def load_mapping(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            payload = json.load(f)
        else:
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML configs")
            payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping config: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_yaml_or_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json" or yaml is None:
        write_json(path, payload)
        return
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(payload), f, sort_keys=False)


def make_run_dir(name: str, *, output_root: Optional[Path] = None, note: str = "") -> Path:
    root = output_root or DEFAULT_OUTPUT_ROOT
    suffix = safe_name(note) if note else safe_name(name)
    run_dir = root / safe_name(name) / f"run_{timestamp()}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_stage_dirs(run_dir: Path, stage: str) -> Dict[str, Path]:
    stage_dir = run_dir / safe_name(stage)
    output_dir = stage_dir / "output"
    result_dir = stage_dir / "result"
    log_dir = stage_dir / "logs"
    for path in (output_dir, result_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"stage": stage_dir, "output": output_dir, "result": result_dir, "logs": log_dir}


def run_command(
    cmd: Sequence[Any],
    *,
    cwd: Path = ROOT,
    env: Optional[Mapping[str, str]] = None,
    dry_run: bool = False,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    started = time.time()
    command = [str(x) for x in cmd]
    record: Dict[str, Any] = {
        "command": command,
        "command_str": shell_join(command),
        "cwd": str(cwd),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": bool(dry_run),
    }
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(record["command_str"] + "\n", encoding="utf-8")
        output_log_path = log_path.with_suffix(".log")
        record["stdout_path"] = str(output_log_path)
    else:
        output_log_path = None
    if dry_run:
        record.update({"returncode": None, "elapsed_sec": 0.0, "status": "dry_run"})
        return record
    proc_env = os.environ.copy()
    if env:
        proc_env.update({str(k): str(v) for k, v in env.items()})
    tail = []
    with subprocess.Popen(
        command,
        cwd=str(cwd),
        env=proc_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        out_fp = output_log_path.open("w", encoding="utf-8") if output_log_path else None
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                if out_fp:
                    out_fp.write(line)
                    out_fp.flush()
                tail.append(line.rstrip("\n"))
                if len(tail) > 120:
                    tail.pop(0)
        finally:
            if out_fp:
                out_fp.close()
        returncode = proc.wait()
    record.update(
        {
            "returncode": int(returncode),
            "elapsed_sec": float(time.time() - started),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "completed" if returncode == 0 else "failed",
            "stdout_tail": tail,
        }
    )
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return record
