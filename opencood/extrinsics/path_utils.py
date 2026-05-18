import sys
from pathlib import Path
from typing import Union


def ensure_v2xreg_root_on_path() -> Path:
    """
    Ensure the HEAL benchmark root is importable.

    Pose-estimation code now lives under `opencood/extrinsics/pose_estimation`.
    The project root is still added so compatibility shims such as `calib` and
    `v2x_calib` keep working for older imports.
    """
    here = Path(__file__).resolve()
    root = here.parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def resolve_repo_path(path: Union[str, Path]) -> Path:
    """
    Resolve a possibly repo-relative path.

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
