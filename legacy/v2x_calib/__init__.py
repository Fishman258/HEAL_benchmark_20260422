"""Compatibility shim for code moved to opencood.extrinsics.pose_estimation."""

import importlib
import sys

_MODULE = importlib.import_module("opencood.extrinsics.pose_estimation.v2x_calib")

globals().update({k: v for k, v in _MODULE.__dict__.items() if not k.startswith("_")})
__all__ = getattr(_MODULE, "__all__", [k for k in globals().keys() if not k.startswith("_")])
__path__ = _MODULE.__path__  # type: ignore[attr-defined]
sys.modules[__name__] = _MODULE
