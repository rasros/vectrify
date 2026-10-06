"""Separate material interiors and persistent voids from alpha edge coverage.

The half-modal contour is a coverage hypothesis, never a paint substitution.
Native color/alpha scores still see every source pixel. Independent faint
components retain the policy's original mass checks. Uniform weak rings also
retain holes even when they are attached to a stronger material component.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    find_objects,
    label,
)

VISIBLE = 0.5 / 255
CORE_FRACTION = 0.5
PLATEAU_RATIO = 1.25
COVERAGE_VERSION = 1


@dataclass(frozen=True)
class Coverage:
    interior: np.ndarray
    holes: np.ndarray
    material: np.ndarray
    plateau_holes: int


def from_alpha(alpha: np.ndarray, visible: np.ndarray) -> Coverage:
    """Fixed native support, using only source opacity and bounded histograms."""
    components, count = label(visible)
    material = np.zeros_like(visible)
    for index, box in enumerate(find_objects(components, max_label=count), 1):
        if box is None:
            continue
        own = components[box] == index
        samples = alpha[box][own]
        levels = np.rint(samples * 255).astype(np.int32)
        modal = int(np.bincount(levels, minlength=256).argmax())
        opacity = float(np.median(samples[levels == modal]))
        threshold = max(VISIBLE, opacity * CORE_FRACTION)
        material[box] |= own & (alpha[box] >= threshold)

    # Voids remain enclosed after removing the weakest coverage halo. This
    # includes intentional reduced-opacity holes, not only fully empty ones.
    holes = np.asarray(binary_fill_holes(material)) & ~material
    native = np.asarray(binary_fill_holes(visible)) & ~visible
    native_labels, count = label(native)
    plateau_holes = 0
    height, width = alpha.shape
    for index, box in enumerate(find_objects(native_labels, max_label=count), 1):
        if box is None:
            continue
        own = native_labels[box] == index
        if own.sum() < 4 or holes[box][own].all():
            continue
        expanded = (
            slice(max(0, box[0].start - 1), min(height, box[0].stop + 1)),
            slice(max(0, box[1].start - 1), min(width, box[1].stop + 1)),
        )
        void = native_labels[expanded] == index
        rim = binary_dilation(void) & ~void & visible[expanded]
        samples = alpha[expanded][rim]
        if not len(samples):
            continue
        low, high = np.quantile(samples, (0.1, 0.9))
        # The existing hole tolerance is two alpha bytes. A coherent enclosing
        # plateau above it can support a weak intentional loop. One strong
        # attachment need not disqualify the rest of that enclosing ring.
        if low > 2 / 255 and high <= low * PLATEAU_RATIO + VISIBLE:
            holes[box] |= own
            plateau_holes += 1

    interior = binary_erosion(material, iterations=2) | (holes & visible)
    for array in (interior, holes, material):
        array.flags.writeable = False
    return Coverage(interior, holes, material, plateau_holes)
