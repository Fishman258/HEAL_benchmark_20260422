from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class ExtrinsicInit:
    """
    Initial transform guess (source -> target).
    """

    T_init: np.ndarray  # (4,4)
    source: str = "unknown"
    init_RE: Optional[float] = None
    init_TE: Optional[float] = None


@dataclass
class ExtrinsicEstimate:
    """
    Extrinsic estimation output (source -> target).
    """

    T: Optional[np.ndarray]
    success: bool
    method: str
    stability: float = 0.0
    matches: List[Dict[str, Any]] = field(default_factory=list)
    RE: Optional[float] = None
    TE: Optional[float] = None
    time_sec: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodContext:
    """
    Common optional context shared across methods.
    """

    T_true: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


__all__ = ["ExtrinsicInit", "ExtrinsicEstimate", "MethodContext"]

