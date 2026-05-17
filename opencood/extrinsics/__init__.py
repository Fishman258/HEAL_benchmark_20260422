"""
Extrinsic estimation & pose alignment utilities.

This package provides lightweight wrappers for the calibration algorithms
implemented in the parent repo (V2X-Reg++/CBM/VIPS) and a small, unified API
to run them from HEAL/OpenCOOD code.
"""

from .types import ExtrinsicEstimate, ExtrinsicInit, MethodContext

__all__ = [
    "ExtrinsicEstimate",
    "ExtrinsicInit",
    "MethodContext",
]

