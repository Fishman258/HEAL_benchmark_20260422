"""Compatibility wrapper for relocated legacy config helpers."""

import importlib
import sys

_MODULE = importlib.import_module("opencood.extrinsics.pose_estimation.configs.legacy_api")

globals().update({k: v for k, v in _MODULE.__dict__.items() if not k.startswith("_")})
__all__ = getattr(_MODULE, "__all__", [k for k in globals().keys() if not k.startswith("_")])
sys.modules[__name__] = _MODULE
