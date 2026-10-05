"""A fitted stroke can supply nearby fill curves in a joint proposal.

Correspondences are inferred from geometry in reference pixels. De Casteljau
weights copy a subcurve of a later selected stroke cubic; least-squares weights
fit across source knots without adding points. Their image gradients reach the
stroke's actual controls, with the original editor movement bounds retained.
This is a candidate model, not a persistent drawing constraint. The ordinary
joint fit remains available and exact rendering decides which result to keep.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vectrify.refine.crossings import bezier
from vectrify.refine.support import SAMPLES, _closest, _split

REACH = 3.0
MAX_SOURCE_SEGMENTS = 256


class StrokeSupports:
    def __init__(
        self,
        coordinates: list[Any],
        reference_scale,
        shared_rows=(),
        deadline: float = float("inf"),
    ):
        self.coordinates = coordinates
        self.entries = []
        shared = set(shared_rows)
        scale = np.asarray(reference_scale)
        for a, target in enumerate(coordinates):
            if target.stroke_only:
                continue
            for b, source in enumerate(coordinates[a + 1 :], a + 1):
                if time.monotonic() >= deadline:
                    return
                if (
                    not source.stroke_only
                    or len(source.geometry.subpaths) != 1
                    or source.geometry.subpaths[0].closed
                    or not 0 < len(source.mapping.gather_index) <= MAX_SOURCE_SEGMENTS
                    or any(i == b for i, _row in shared)
                ):
                    continue
                linear = source.mapping.linear.cpu().numpy()
                offset = source.mapping.offset.cpu().numpy()
                controls = [
                    (source.local[indices.cpu().numpy()] @ linear.T + offset) * scale
                    for indices in source.mapping.gather_index
                ]
                samples = np.vstack(
                    [
                        bezier(c, np.linspace(0, 1, SAMPLES + 1)[:-1])[0]
                        for c in controls
                    ]
                    + [controls[-1][-1:]]
                )
                for segment, indices in enumerate(target.mapping.gather_index):
                    if time.monotonic() >= deadline:
                        return
                    if bool(target.mapping.straight_mask[segment].any()):
                        # An implicit closure has derived controls, not four
                        # independent coordinates for copying a cubic.
                        continue
                    if not bool(target.mapping.movable[indices].all()) or any(
                        (a, int(i)) in shared for i in indices
                    ):
                        continue
                    rows = indices.cpu().numpy()
                    old = (
                        target.local[rows] @ target.mapping.linear.cpu().numpy().T
                        + target.mapping.offset.cpu().numpy()
                    ) * scale
                    probes = bezier(old, np.linspace(0, 1, 13))[0]
                    at = np.array([_closest(p, controls, samples) for p in probes])
                    low, high = sorted((at[0], at[-1]))
                    owner = min(int(low + 1e-7), len(controls) - 1)
                    if any(
                        np.linalg.norm(self._point(controls, t) - p) > REACH
                        for p, t in zip(probes, at, strict=True)
                    ):
                        continue
                    monotone = (
                        np.maximum.accumulate(at)
                        if at[-1] >= at[0]
                        else np.minimum.accumulate(at)
                    )
                    if any(
                        np.linalg.norm(self._point(controls, t) - p) > REACH
                        for p, t in zip(probes, monotone, strict=True)
                    ):
                        continue
                    if high <= low + 1e-8:
                        continue
                    if high <= owner + 1 + 1e-7:
                        start, end = (
                            np.clip(low - owner, 0, 1),
                            np.clip(high - owner, 0, 1),
                        )
                        weights = _split(np.eye(4), end)[0] if end < 1 else np.eye(4)
                        if start > 0:
                            weights = _split(weights, start / end)[1]
                        if at[-1] < at[0]:
                            weights = weights[::-1]
                        source_rows = source.mapping.gather_index[owner]
                        source_controls = controls[owner]
                    else:
                        # A target crossing a source knot keeps its existing
                        # topology. Fit one cubic to the corresponding samples,
                        # rather than adding an editor point at every knot.
                        # Its linear weights also couple both source segments.
                        weights = self._weights(controls, at)
                        source_rows = source.mapping.gather_index.reshape(-1)
                        source_controls = np.vstack(controls)
                    projected = (
                        weights @ source_controls / scale
                        - target.mapping.offset.cpu().numpy()
                    ) @ target.mapping.inverse.cpu().numpy().T
                    if (
                        np.max(np.linalg.norm(projected - target.local[rows], axis=1))
                        > target.mapping.displacement
                    ):
                        continue
                    self.entries.append(
                        (
                            a,
                            b,
                            indices,
                            source_rows,
                            target.original.new_tensor(weights.copy()),
                        )
                    )

    @staticmethod
    def _point(controls, value):
        index = min(int(value), len(controls) - 1)
        return bezier(controls[index], np.array([value - index]))[0][0]

    @staticmethod
    def _weights(controls, at):
        source = np.zeros((len(at), len(controls) * 4))
        for row, value in enumerate(at):
            index = min(int(value), len(controls) - 1)
            t = value - index
            source[row, index * 4 : (index + 1) * 4] = (
                (1 - t) ** 3,
                3 * (1 - t) ** 2 * t,
                3 * (1 - t) * t**2,
                t**3,
            )
        t = np.linspace(0, 1, len(at))
        basis = np.stack(
            ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3), -1
        )
        residual = source - basis[:, :1] * source[:1] - basis[:, 3:] * source[-1:]
        middle = np.linalg.lstsq(basis[:, 1:3], residual, rcond=None)[0]
        return np.vstack((source[0], middle, source[-1]))

    @property
    def active(self):
        return bool(self.entries)

    def __call__(self, values):
        if not self.entries:
            return values
        result = [value.clone() for value in values]
        for a, b, target_rows, source_rows, weights in self.entries:
            target, source = self.coordinates[a], self.coordinates[b]
            root = (
                values[b][source_rows] @ source.mapping.linear.T + source.mapping.offset
            )
            projected = (
                weights @ root - target.mapping.offset
            ) @ target.mapping.inverse.T
            # Coupling must not enlarge the editor's movement allowance. A
            # capped point can depart from the model; it stays a valid proposal
            # and the exact render still has to improve before it is retained.
            delta = projected - target.original[target_rows]
            length = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            result[a][target_rows] = target.original[target_rows] + delta * (
                target.mapping.displacement / length
            ).clamp(max=1)
        return result
