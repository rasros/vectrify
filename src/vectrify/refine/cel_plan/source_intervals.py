"""Recover editable intervals separated by observed raw-source absence.

No owner clipping, human repair or nearby endpoint can split a source chain.
Only its own measured internal gaps authorize new caps; actual native bodies
and exact carrier footprints must still pass. Short or ambiguous pieces stay
filled. These inspected engineering bounds are not calibrated release gates.
"""

from __future__ import annotations

import numpy as np

from vectrify.document import Geometry
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.model import StageInterruptedError
from vectrify.refine.crossings import crossings

MAX_INTERVALS = 8
MAX_FITS = 256
MAX_RUNS = 128
MAX_NODES = 16_384
CAP_MARGINS = (1.0, 2.0, 3.0)
MIN_SUPPORT = 0.45


def _check(work):
    if work.interrupted:
        raise StageInterruptedError("Source interval reconstruction interrupted")


class SourceIntervals:
    def __init__(self, absence):
        if absence.source_breaks is None:
            raise ValueError("Source intervals require retained raw observations")
        self.absence = absence
        self.source_breaks = absence.source_breaks
        self.diagnostics = {
            "profiles": 0,
            "intervals": 0,
            "fits": 0,
            "recovered_runs": 0,
            "nodes": 0,
            "closed_exclusions": 0,
        }

    def recover(self, original, profile, width, cap, options, work, carried):
        """Publish all recovered intervals only after this chain completes.

        Surviving original terminals and junctions stay exact. A new internal
        cap starts at a qualified raw-source sample inside an observed broken
        interval. Bounded setbacks pay for cap radius and native antialiasing;
        no fully supported host is shortened to satisfy a neighbouring gap.
        """
        _check(work)
        if not np.isfinite(width) or width <= 0 or cap not in {"round", "butt"}:
            raise ValueError("Source intervals require a finite supported stroke style")
        if profile is None:
            return ()
        observed = self.source_breaks(profile, work=work)
        if observed is None or not observed.gaps.any():
            return ()
        if original.closed or np.array_equal(observed.points[0], observed.points[-1]):
            self.diagnostics["closed_exclusions"] += 1
            return ()
        endpoints = np.array((original.nodes[0].endpoint, original.nodes[-1].endpoint))
        if not np.array_equal(endpoints, observed.points[[0, -1]]):
            return ()
        self.diagnostics["profiles"] += 1
        points, qualified, gaps = observed.points, observed.qualified, observed.gaps
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        keep = ~gaps
        starts = np.flatnonzero(keep & ~np.r_[False, keep[:-1]])
        stops = np.flatnonzero(keep & ~np.r_[keep[1:], False]) + 1
        if len(starts) > MAX_INTERVALS:
            raise ValueError("Source intervals exceed the per-chain interval bound")
        result = []
        for raw_start, raw_stop in zip(starts, stops, strict=True):
            _check(work)
            self.diagnostics["intervals"] += 1
            for margin in CAP_MARGINS:
                _check(work)
                start, stop = int(raw_start), int(raw_stop)
                if start:
                    start = int(
                        np.searchsorted(
                            distance, distance[start - 1] + width / 2 + margin
                        )
                    )
                if stop < len(points):
                    stop = int(
                        np.searchsorted(
                            distance, distance[stop] - width / 2 - margin, side="right"
                        )
                    )
                if start < raw_start or stop > raw_stop or stop - start < 4:
                    continue
                # A new cap needs positive native evidence at its centre. Keep
                # original junctions even if their side probes lie inside ink.
                supported = np.flatnonzero(qualified[start:stop])
                if not len(supported):
                    continue
                if raw_start:
                    start += int(supported[0])
                    if (
                        distance[start] - distance[raw_start - 1]
                        > width / 2 + CAP_MARGINS[-1] + 0.5
                    ):
                        continue
                if raw_stop < len(points):
                    stop = start + int(np.flatnonzero(qualified[start:stop])[-1]) + 1
                    if (
                        distance[raw_stop] - distance[stop - 1]
                        > width / 2 + CAP_MARGINS[-1] + 0.5
                    ):
                        continue
                if (
                    stop - start < 4
                    or distance[stop - 1] - distance[start] < max(8, 4 * width)
                    or qualified[start:stop].mean() < MIN_SUPPORT
                ):
                    continue
                if self.diagnostics["fits"] >= MAX_FITS:
                    raise ValueError("Source intervals exceed the fit bound")
                self.diagnostics["fits"] += 1
                part = fitted(
                    points[start:stop], min(options.tolerance or 0.75, 0.25)
                ).contour
                _check(work)
                geometry = Geometry("source-interval", (part,))
                if crossings(geometry) or not self.absence.permits(
                    geometry, width, cap, work
                ):
                    continue
                if not carried(geometry):
                    _check(work)
                    continue
                _check(work)
                nodes = len(part.nodes)
                if (
                    self.diagnostics["recovered_runs"] >= MAX_RUNS
                    or self.diagnostics["nodes"] + nodes > MAX_NODES
                ):
                    raise ValueError("Source intervals exceed the output bound")
                self.diagnostics["recovered_runs"] += 1
                self.diagnostics["nodes"] += nodes
                result.append(part)
                break
        _check(work)
        return tuple(result)
