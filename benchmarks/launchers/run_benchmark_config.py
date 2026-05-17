#!/usr/bin/env python3
"""Run an existing benchmark launcher from a YAML/JSON config.

This is a compatibility layer: it converts a readable config file into the
same CLI arguments consumed by the current scripts under `scripts/`.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def _resolve_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def _cli_key(key: str) -> str:
    return "--" + str(key).strip().replace("_", "-")


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return str(value)


def _append_args(cmd: List[str], args: Mapping[str, Any]) -> None:
    for key, value in args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(_cli_key(key))
            continue
        cmd.extend([_cli_key(key), _format_value(value)])


def _parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        pass
    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return raw


def _set_nested(config: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        current = target.get(part)
        if current is None:
            current = {}
            target[part] = current
        if not isinstance(current, dict):
            raise ValueError(f"cannot set {dotted_key}: {part} is not a mapping")
        target = current
    target[parts[-1]] = value


def _apply_overrides(config: MutableMapping[str, Any], overrides: Iterable[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        _set_nested(config, key.strip(), _parse_scalar(raw_value.strip()))


def _shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


def build_command(config: Mapping[str, Any], *, force_dry_run: bool = False) -> List[str]:
    launcher = config.get("launcher")
    if not launcher:
        raise ValueError("config missing required key: launcher")

    launcher_path = _resolve_path(launcher)
    if not launcher_path.exists():
        raise FileNotFoundError(f"launcher not found: {launcher_path}")

    python_bin = config.get("python") or config.get("launcher_python") or sys.executable
    cmd = [str(python_bin), str(launcher_path)]

    args = config.get("args") or {}
    if not isinstance(args, dict):
        raise ValueError("config key 'args' must be a mapping")
    _append_args(cmd, args)

    if force_dry_run and "--dry-run" not in cmd:
        cmd.append("--dry-run")

    extra_args = config.get("extra_args") or []
    if not isinstance(extra_args, list):
        raise ValueError("config key 'extra_args' must be a list")
    cmd.extend(str(x) for x in extra_args)
    return cmd


def build_env(config: Mapping[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    extra = config.get("environment") or {}
    if not isinstance(extra, dict):
        raise ValueError("config key 'environment' must be a mapping")
    for key, value in extra.items():
        env[str(key)] = str(value)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmark launcher from YAML/JSON config.")
    parser.add_argument("config", type=Path, help="Path to benchmark config.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config value, e.g. args.max-eval-samples=4.")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to the underlying launcher.")
    parser.add_argument("--print-only", action="store_true", help="Print the resolved command and exit.")
    return parser.parse_args()


def main() -> None:
    opt = parse_args()
    config = _load_config(opt.config)
    _apply_overrides(config, opt.overrides)
    cmd = build_command(config, force_dry_run=bool(opt.dry_run))
    print(_shell_join(cmd), flush=True)
    if opt.print_only:
        return
    subprocess.run(cmd, cwd=str(ROOT), env=build_env(config), check=True)


if __name__ == "__main__":
    main()

