"""
Registration utilities for estimating and applying inter-agent extrinsics.

This package groups the benchmark runtime adapters, concrete registration
estimators, and shared geometry helpers used by HEAL/OpenCOOD code.
"""

from .utils.types import ExtrinsicEstimate, ExtrinsicInit, MethodContext

__all__ = [
    "ExtrinsicEstimate",
    "ExtrinsicInit",
    "MethodContext",
]
