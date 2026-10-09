"""Recover source-supported runs from a measured cycle, independently of owners.

Chunk seams are observation packaging, never terminals. All own native gaps
remain constraints; only those gaps can authorize new caps. Full-cycle coverage
and complete recovery precede publication of a bounded, source-ranked set.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from vectrify.document import Geometry
from vectrify.refine import cel
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.line_fidelity import SourceBreaks, SourceProfile
from vectrify.refine.cel_plan.model import StageInterruptedError
from vectrify.refine.cel_plan.source_intervals import (
    CAP_MARGINS,
    MAX_FITS,
    MAX_INTERVALS,
    MAX_NODES,
    MAX_RUNS,
    MIN_SUPPORT,
)
from vectrify.refine.crossings import crossings

PROFILE_POINTS = 1024
PROFILE_OVERLAP = 256
MAX_RAW_POINTS = 16_384


def _check(work):
    if work.interrupted:
        raise StageInterruptedError("Source cycle recovery interrupted")


class SourceCycle:
    def __init__(self, complete):
        if (
            len(complete.points) < 5
            or len(complete.points) > MAX_RAW_POINTS + 1
            or not np.array_equal(complete.points[0], complete.points[-1])
            or complete.terminals != (False, False)
        ):
            raise ValueError(
                "Source cycles require a bounded closed nonterminal profile"
            )
        self.complete = complete
        self.dense = complete.dense()
        self.steps = np.maximum(
            1, np.ceil(2 * np.linalg.norm(np.diff(complete.points, axis=0), axis=1))
        ).astype(int)
        self.cumulative = np.r_[0, np.cumsum(self.steps)]
        self.steps.flags.writeable = False
        self.cumulative.flags.writeable = False
        count = len(complete.points) - 1
        overlap = min(PROFILE_OVERLAP, count)
        indices = np.arange(-overlap, count + overlap)
        fields = tuple(
            getattr(complete, name)[indices % count]
            for name in ("points", "sides", "direction", "tolerance")
        )
        chunks, origins = [], []
        for start in range(0, len(indices) - 1, PROFILE_POINTS - PROFILE_OVERLAP):
            end = min(len(indices), start + PROFILE_POINTS)
            arrays = tuple(a[start:end] for a in fields)
            for a in arrays:
                a.flags.writeable = False
            chunks.append(
                SourceProfile(
                    arrays[0],
                    arrays[1],
                    arrays[2],
                    arrays[3],
                    complete.component,
                    (False, False),
                )
            )
            raw = int(indices[start])
            origins.append(
                int((raw // count) * self.cumulative[-1] + self.cumulative[raw % count])
            )
        self.profiles = tuple(chunks)
        self.origins = tuple(origins)
        self.diagnostics = dict.fromkeys(
            (
                "observations",
                "gaps",
                "intervals",
                "selected_intervals",
                "fits",
                "recovered_runs",
                "nodes",
                "unobserved_samples",
                "geometry_exclusions",
                "absence_exclusions",
            ),
            0,
        )

    def observations(self, absence, work):
        """Merge only identity-bound own observations on the original cyclic grid."""
        _check(work)
        if absence.source_breaks is None:
            raise ValueError("Cycle recovery requires retained native observations")
        count = len(self.dense.points) - 1
        seen = np.zeros(count, bool)
        qualified = np.zeros(count, bool)
        gaps = np.zeros(count, bool)
        for profile, origin in zip(self.profiles, self.origins, strict=True):
            _check(work)
            observed = absence.source_breaks(profile, work=work)
            if observed is None:
                continue
            indices = (origin + np.arange(len(observed.points))) % count
            if not np.array_equal(observed.points, self.dense.points[indices]):
                raise ValueError(
                    "Cycle observations do not match the original native grid"
                )
            seen[indices] = True
            np.logical_or.at(qualified, indices, observed.qualified)
            # A duplicate observation can add a constraint, never remove one.
            np.logical_or.at(gaps, indices, observed.gaps)
            self.diagnostics["observations"] += len(observed.points)
        self.diagnostics["unobserved_samples"] = int((~seen).sum())
        if not seen.all():
            raise ValueError(
                "Cycle observations do not cover the complete source cycle"
            )
        qualified &= ~gaps
        self.diagnostics["gaps"] = int(gaps.sum())
        for a in (qualified, gaps):
            a.flags.writeable = False
        _check(work)
        return SourceBreaks(self.dense.points[:-1], qualified, gaps)

    def recover(self, absence, width, cap, options, work):
        """Offer complete own-gap intervals, with no cap at a serialization seam.

        Every interval is enumerated before at most eight longest supported runs
        are fitted. Remaining runs retain their fills. Capacity/interruption
        errors publish no prefix, and unsupported endpoints are never moved to
        satisfy an owner or carrier. Actual native bodies/caps decide retention.
        """
        _check(work)
        if not np.isfinite(width) or width <= 0 or cap not in {"round", "butt"}:
            raise ValueError("Source cycle recovery requires a finite stroke style")
        observed = self.observations(absence, work)
        positions = np.flatnonzero(observed.gaps)
        if not len(positions):
            return ()
        # Canonical serialization starts inside an observed gap. Rotating an
        # otherwise identical input cycle cannot create or remove a terminal.
        first = int(positions[np.lexsort(observed.points[positions].T[::-1])[0]])
        points = np.roll(observed.points, -first, axis=0)
        points = np.vstack((points, points[0]))
        qualified = np.r_[np.roll(observed.qualified, -first), False]
        gaps = np.r_[np.roll(observed.gaps, -first), True]
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        keep = ~gaps
        starts = np.flatnonzero(keep & ~np.r_[False, keep[:-1]])
        stops = np.flatnonzero(keep & ~np.r_[keep[1:], False]) + 1
        self.diagnostics["intervals"] = len(starts)
        if len(starts) > MAX_RUNS:
            raise ValueError("Source cycles exceed the complete interval bound")
        intervals = [
            (int(start), int(stop))
            for start, stop in zip(starts, stops, strict=True)
            if stop - start >= 4
            and distance[stop - 1] - distance[start] >= max(8, 4 * width)
            and qualified[start:stop].mean() >= MIN_SUPPORT
        ]
        intervals.sort(
            key=lambda p: (
                -(distance[p[1] - 1] - distance[p[0]]),
                tuple(points[p[0]]),
                tuple(points[p[1] - 1]),
            )
        )
        intervals = intervals[:MAX_INTERVALS]
        self.diagnostics["selected_intervals"] = len(intervals)
        result = []
        for raw_start, raw_stop in intervals:
            for margin in CAP_MARGINS:
                _check(work)
                start = int(
                    np.searchsorted(
                        distance, distance[raw_start - 1] + width / 2 + margin
                    )
                )
                stop = int(
                    np.searchsorted(
                        distance, distance[raw_stop] - width / 2 - margin, side="right"
                    )
                )
                if start < raw_start or stop > raw_stop or stop - start < 4:
                    continue
                supported = np.flatnonzero(qualified[start:stop])
                if not len(supported):
                    continue
                start += int(supported[0])
                stop = start + int(np.flatnonzero(qualified[start:stop])[-1]) + 1
                if (
                    distance[start] - distance[raw_start - 1]
                    > width / 2 + CAP_MARGINS[-1] + 0.5
                    or distance[raw_stop] - distance[stop - 1]
                    > width / 2 + CAP_MARGINS[-1] + 0.5
                    or stop - start < 4
                    or distance[stop - 1] - distance[start] < max(8, 4 * width)
                ):
                    continue
                original = points[start:stop]
                contour = None
                for sigma, tolerance in ((0, 0), (0, 0.25), (1, 0), (1, 0.25)):
                    _check(work)
                    candidate = original
                    if sigma:
                        candidate = gaussian_filter1d(
                            original, sigma, axis=0, mode="nearest"
                        )
                        candidate[[0, -1]] = original[[0, -1]]
                        if np.linalg.norm(candidate - original, axis=1).max() > min(
                            1, width / 2
                        ):
                            continue
                    if self.diagnostics["fits"] >= MAX_FITS:
                        raise ValueError("Source cycles exceed the fit bound")
                    self.diagnostics["fits"] += 1
                    trial = fitted(
                        cel.simplify(candidate, tolerance) if tolerance else candidate,
                        min(options.tolerance or 0.75, 0.25),
                    ).contour
                    geometry = Geometry("source-cycle-interval", (trial,))
                    _check(work)
                    if crossings(geometry):
                        self.diagnostics["geometry_exclusions"] += 1
                        continue
                    if not absence.permits(geometry, width, cap, work):
                        self.diagnostics["absence_exclusions"] += 1
                        continue
                    contour = trial
                    break
                if contour is None:
                    continue
                nodes = len(contour.nodes)
                if self.diagnostics["nodes"] + nodes > MAX_NODES:
                    raise ValueError("Source cycles exceed the output node bound")
                self.diagnostics["nodes"] += nodes
                self.diagnostics["recovered_runs"] += 1
                result.append(contour)
                break
        _check(work)
        return tuple(result)
