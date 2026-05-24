#!/usr/bin/env python3
"""Shared stdlib-only validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


class ValidationError(RuntimeError):
    """Raised when an artifact violates its lightweight contract."""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def numeric_keys(payload: Mapping[Any, Any]) -> Optional[List[int]]:
    keys: List[int] = []
    for key in payload.keys():
        try:
            keys.append(int(key))
        except Exception:
            return None
    return keys


def validate_count_and_keys(
    payload: Mapping[Any, Any],
    *,
    expected_samples: Optional[int],
    require_contiguous_keys: bool,
    errors: List[str],
) -> None:
    if expected_samples is not None:
        require(
            len(payload) == int(expected_samples),
            "expected {} samples, found {}".format(int(expected_samples), len(payload)),
            errors,
        )
    keys = numeric_keys(payload)
    if require_contiguous_keys:
        require(keys is not None, "top-level keys must be numeric when contiguous keys are required", errors)
        if keys:
            sorted_keys = sorted(keys)
            expected = list(range(sorted_keys[0], sorted_keys[0] + len(sorted_keys)))
            require(sorted_keys == expected, "top-level numeric keys are not contiguous", errors)


def limited_items(mapping: Mapping[Any, Any], sample_limit: int) -> Iterable[Any]:
    items = list(mapping.items())
    if sample_limit and sample_limit > 0:
        return items[: int(sample_limit)]
    return items


def is_pose6(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return len(value) >= 6 and all(isinstance(x, (int, float)) for x in value[:6])


def print_result(path: Path, summary: Dict[str, Any], errors: List[str]) -> None:
    if errors:
        print("FAIL {}".format(path))
        for err in errors:
            print("- {}".format(err))
        raise SystemExit(1)
    print("PASS {}".format(path))
    print(json.dumps(summary, indent=2, sort_keys=True))
