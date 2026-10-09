"""Bounded width hypotheses after physical source discovery has finished.

Original geometry, paint, caps and ports remain fixed. Narrower bodies compete
only when they retain more complete chains under native source absence. Actual
compound bodies, carrier footprints and ownership still decide model export.
This is an explicit competitor; its original-width interpretation remains an
independent candidate, rather than a width rule imposed on the final drawing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vectrify.document import Geometry
from vectrify.refine.cel_plan.model import StageInterruptedError

MAX_STYLES = 8
MAX_RUNS = 128
MAX_NODES = 16_384
MAX_FITS = 256
FACTORS = (0.9, 0.8, 0.7, 0.6)


def _check(work):
    if work.interrupted:
        raise StageInterruptedError("Source width fitting interrupted")


class SourceWidths:
    def __init__(self, absence):
        self.absence = absence
        self.diagnostics: dict[str, Any] = {
            "fits": 0,
            "bounded": 0,
            "changed_styles": 0,
            "styles": [],
        }

    def fit(self, groups, widths, scale, work):
        """Return a complete width interpretation, never a partly fitted style set.

        The widest width with the most absence-valid original bodies wins.
        No gain keeps the original width. A capacity failure keeps every
        original width; interruption publishes no interpretation. Widths only
        narrow, so they cannot exceed the original exact carrier ceilings.
        """
        _check(work)
        self.diagnostics.update(fits=0, bounded=0, changed_styles=0, styles=[])
        if (
            len(groups) != len(widths)
            or not np.isfinite(scale)
            or scale <= 0
            or any(not np.isfinite(w) or w <= 0 for w in widths)
        ):
            raise ValueError("Source widths require finite aligned styles")
        original = tuple(widths)
        if not len(self.absence.points):
            return original
        if (
            len(groups) > MAX_STYLES
            or sum(len(g["contours"]) for g in groups) > MAX_RUNS
            or sum(len(s.nodes) for g in groups for s in g["contours"]) > MAX_NODES
        ):
            self.diagnostics["bounded"] += 1
            return original
        selected, records = [], []
        for group, width in zip(groups, widths, strict=True):
            geometries = tuple(
                Geometry("source-width", (s,)) for s in group["contours"]
            )
            values = tuple(
                dict.fromkeys(
                    (width, *(min(width, max(0.8 / scale, width * f)) for f in FACTORS))
                )
            )
            best_width, best_count, initial = width, -1, 0
            for value in values:
                _check(work)
                count = 0
                for geometry in geometries:
                    _check(work)
                    if self.diagnostics["fits"] >= MAX_FITS:
                        self.diagnostics["bounded"] += 1
                        return original
                    self.diagnostics["fits"] += 1
                    count += self.absence.permits(geometry, value, group["cap"], work)
                if value == width:
                    initial = count
                if count > best_count:
                    best_width, best_count = value, count
            selected.append(best_width)
            records.append(
                {
                    "original": width,
                    "fitted": best_width,
                    "original_complete_bodies": initial,
                    "fitted_complete_bodies": best_count,
                }
            )
        _check(work)
        self.diagnostics["styles"] = records
        self.diagnostics["changed_styles"] = sum(
            a != b for a, b in zip(original, selected, strict=True)
        )
        return tuple(selected)
