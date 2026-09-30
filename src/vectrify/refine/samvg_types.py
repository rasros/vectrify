"""Shared value objects used by SAMVG's segmentation and SVG stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskLayer:
    """One painted segmentation mask, in document compositing order."""

    mask: np.ndarray
    colour: tuple[int, int, int]
    impact: float
