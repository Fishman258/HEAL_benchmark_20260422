import sys
from pathlib import Path
from typing import Union


def ensure_v2xreg_root_on_path() -> Path:
    """
    Ensure the mono-repo root (containing `calib/` and `legacy/`) is importable.

    This HEAL checkout lives in `<root>/HEAL`. Most calibration code lives in
    `<root>/calib` and `<root>/legacy`, so we add `<root>` to `sys.path` when
    needed.
    """
    here = Path(__file__).resolve()
    root = here.parents[3]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def resolve_repo_path(path: Union[str, Path]) -> Path:
    """
    Resolve a possibly repo-relative path.

    When running from `<root>/HEAL`, many configs/data live under `<root>/...`.
    This helper first tries `path` as-is, then tries `<root>/path`.
    """
    raw = Path(path)
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw.resolve()
    root = ensure_v2xreg_root_on_path()
    candidate = (root / raw).resolve()
    return candidate


__all__ = ["ensure_v2xreg_root_on_path", "resolve_repo_path"]
