"""Bounded native source-line contracts, independent of human vector geometry.

Inspect measured source troughs, including chains which failed stroke export.
These engineering thresholds are diagnostic controls, not calibrated release
criteria. No geometry is generated or repaired by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import map_coordinates

from vectrify.refine.cel_plan.ink import _colors
from vectrify.refine.cel_plan.model import StageInterruptedError
from vectrify.refine.cel_plan.opacity import VISIBLE

LINE_FIDELITY_VERSION = 3
MAX_SAMPLES = 131_072
MAX_GAP_PROFILES = 128
MAX_PROFILES = 512


def _check(work):
    if work is not None and work.interrupted:
        raise StageInterruptedError("Source line fidelity interrupted")


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class SourceBreaks:
    """Copied native observations for one original physical source profile."""

    points: np.ndarray
    qualified: np.ndarray
    gaps: np.ndarray
    anchors: np.ndarray | None = None


@dataclass(frozen=True)
class SourceProfile:
    """Native centres and directional probes copied from original source ink."""

    points: np.ndarray
    sides: np.ndarray
    direction: np.ndarray
    tolerance: np.ndarray
    component: tuple[str, int] | None = None
    terminals: tuple[bool, bool] = (True, True)

    @classmethod
    def from_ink(cls, original, ink, evidence, *, component=None, cyclic=False):
        # A short link's proof is densified before measuring; ordinary proofs
        # retain the original skeleton tangents instead of centred jitter.
        original = original if len(original) == len(ink.points) else ink.points
        if cyclic:
            if len(original) < 5 or not np.array_equal(original[0], original[-1]):
                raise ValueError("Cyclic source profiles require a closed perimeter")
            source = original[:-1]
            tangent = np.roll(source, -2, axis=0) - np.roll(source, 2, axis=0)
            tangent = np.vstack((tangent, tangent[0]))
        else:
            tangent = np.gradient(original, axis=0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
        native = normal / evidence.scale
        length = np.linalg.norm(native, axis=1)
        return cls(
            _readonly(ink.points / evidence.scale + evidence.offset),
            _readonly(native * max(3, 2.5 * ink.width)),
            _readonly(native / np.maximum(length[:, None], 1e-6)),
            _readonly(np.clip(ink.width * length / 2, 0.5, 2)),
            component,
            (False, False) if cyclic else (True, True),
        )

    @classmethod
    def at(cls, points, width):
        """Build a native profile for controlled source/renderer comparisons."""
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("Source line requires at least two planar centres")
        if not np.isfinite(width) or width <= 0 or not np.isfinite(points).all():
            raise ValueError("Source line geometry must be finite with positive width")
        tangent = np.gradient(points, axis=0)
        direction = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
        return cls(
            _readonly(points),
            _readonly(direction * max(3, 2.5 * width)),
            _readonly(direction),
            _readonly(np.full(len(points), np.clip(width / 2, 0.5, 2))),
        )

    def dense(self):
        if (
            self.points.ndim != 2
            or self.points.shape[1] != 2
            or len(self.points) < 2
            or self.sides.shape != self.points.shape
            or self.direction.shape != self.points.shape
            or self.tolerance.shape != (len(self.points),)
            or not all(
                np.isfinite(v).all()
                for v in (self.points, self.sides, self.direction, self.tolerance)
            )
            or np.any(self.tolerance < 0)
        ):
            raise ValueError("Source line profile fields must be finite and aligned")
        steps = np.maximum(
            1, np.ceil(2 * np.linalg.norm(np.diff(self.points, axis=0), axis=1))
        )
        if steps.sum() + 1 > MAX_SAMPLES:
            raise ValueError("Source line exceeds the native sample limit")
        steps = steps.astype(int)

        # Sample every intervening source interval: sparse centres cannot hide
        # a bright/transparent gap. Preserve every supplied source anchor.
        def interpolate(values):
            return _readonly(
                np.concatenate(
                    [
                        *[
                            np.linspace(a, b, n, endpoint=False)
                            for a, b, n in zip(
                                values[:-1], values[1:], steps, strict=True
                            )
                        ],
                        values[-1:],
                    ]
                )
            )

        return SourceProfile(
            interpolate(self.points),
            interpolate(self.sides),
            interpolate(self.direction),
            interpolate(self.tolerance),
            self.component,
            self.terminals,
        )


def _sample(rgba, profile, shifts, visible=None):
    displacements = (
        np.asarray(shifts)[None, :, None]
        * profile.tolerance[:, None, None]
        * profile.direction[:, None, :]
    )
    centers = profile.points[:, None, :] + displacements
    points = (
        centers[:, :, None, :]
        + np.stack(
            (-profile.sides, np.zeros_like(profile.sides), profile.sides), axis=1
        )[:, None, :, :]
    )
    coordinates = np.stack((points[..., 1] - 0.5, points[..., 0] - 0.5)).reshape(2, -1)
    if visible is None:
        visible = rgba[..., 3] > VISIBLE
    colors, seen = _colors(rgba[..., :3], coordinates, visible)
    shape = (len(profile.points), len(shifts), 3)
    light = colors.reshape(*shape, 3).max(axis=-1) * 255
    seen = seen.reshape(shape)
    surface = np.minimum(
        np.where(seen[..., 0] >= 0.5, light[..., 0], np.inf),
        np.where(seen[..., 2] >= 0.5, light[..., 2], np.inf),
    )
    opacity = map_coordinates(
        rgba[..., 3], coordinates, order=1, mode="constant", cval=0
    ).reshape(shape)[..., 1]
    valid = np.isfinite(surface) & (seen[..., 1] >= 0.5) & (light[..., 1] <= 150)
    return surface - light[..., 1], opacity, valid, light[..., 1]


@dataclass(frozen=True)
class ProfileError:
    samples: int
    missing: int
    maximum_missing_span: float
    gap_samples: int
    gap_completed: int


@dataclass(frozen=True, eq=False)
class LineObservation:
    """Complete immutable painted observations bound to one source bank.

    This is a comparison parent, not a replacement for the guard's fixed
    established baseline. It retains bounded sampled values, never the raster.
    """

    authority: object
    profiles: tuple[ProfileError, ...]
    gaps: tuple[np.ndarray, ...]


def _gap_state(prepared, actual, visible):
    profile, _reference, _opacity, _qualified, gaps = prepared
    contrast, alpha, valid, _light = _sample(actual, profile, (0,), visible)
    return gaps & valid[:, 0] & (contrast[:, 0] >= 12) & (alpha[:, 0] > VISIBLE)


def _error(prepared, actual, visible):
    profile, reference, opacity, qualified, gaps = prepared
    contrast, alpha, valid, _light = _sample(
        actual, profile, np.linspace(-1, 1, 9), visible
    )
    present = (
        valid
        & (contrast >= np.minimum(12, 0.5 * reference)[:, None])
        & (alpha >= 0.5 * opacity[:, None])
    ).any(axis=1)
    missing = qualified & ~present
    weights = np.linalg.norm(np.diff(profile.points, axis=0), axis=1)
    weights = (np.r_[weights[0], weights] + np.r_[weights, weights[-1]]) / 2
    peak = current = 0.0
    for absent, weight in zip(missing, weights, strict=True):
        current = current + weight if absent else 0.0
        peak = max(peak, current)
    # A gap is tested at its source position. Search tolerance for a retained
    # line cannot borrow a nearby source line to excuse filling this gap.
    bridged = _gap_state(prepared, actual, visible)
    return ProfileError(
        int(qualified.sum()),
        int(missing.sum()),
        peak,
        int(gaps.sum()),
        int(bridged.sum()),
    )


def _prepare(
    truth, profile, visible
) -> tuple[SourceProfile, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    contrast, opacity, valid, light = _sample(truth, profile, (0,), visible)
    contrast, opacity, valid, light = (
        v[:, 0] for v in (contrast, opacity, valid, light)
    )
    qualified = valid & (contrast >= 12)
    gaps = np.zeros(len(qualified), bool)
    if qualified.any():
        indices = np.arange(len(qualified))
        left = np.maximum.accumulate(np.where(qualified, indices, -1))
        right = np.minimum.accumulate(np.where(qualified, indices, len(indices))[::-1])[
            ::-1
        ]
        distance = np.r_[
            0,
            np.cumsum(np.linalg.norm(np.diff(profile.points, axis=0), axis=1)),
        ]
        span = (
            distance[np.minimum(right, len(indices) - 1)]
            - distance[np.maximum(left, 0)]
        )
        width = np.median(np.linalg.norm(profile.sides, axis=1)) / 2.5
        gaps = (
            ~qualified
            & (left >= 0)
            & (right < len(indices))
            & (span <= max(12, 8 * width))
            & (
                (opacity <= VISIBLE)
                | ((contrast < 6) & (light >= np.median(light[qualified]) + 24))
            )
        )
        if gaps.any():
            # The positive matcher permits a small normal displacement. A
            # bright measured centre beside that same real source trough is
            # not negative evidence. Require absence throughout the matching
            # window before protecting a gap; candidate matching still tests
            # that proved gap at its exact source position.
            nearby, alpha, seen, _light = _sample(
                truth, profile, np.linspace(-1, 1, 9), visible
            )
            supported = seen & (nearby >= 6) & (alpha > VISIBLE)
            gaps &= ~supported.any(axis=1)
        return (
            profile,
            _readonly(contrast),
            _readonly(opacity),
            _readonly(qualified),
            _readonly(gaps),
        )
    return None


def _endpoint_gap(a, b, end_a, end_b):
    """Observe a short, facing gap; never create a stroke connection."""
    pa, pb = a[0], b[0]
    if (
        not pa.terminals[0 if end_a == 0 else 1]
        or not pb.terminals[0 if end_b == 0 else 1]
    ):
        return None
    near_a = a[3][:8] if end_a == 0 else a[3][-8:]
    near_b = b[3][:8] if end_b == 0 else b[3][-8:]
    if not near_a.any() or not near_b.any():
        return None
    first, last = pa.points[end_a], pb.points[end_b]
    delta = last - first
    distance = float(np.linalg.norm(delta))
    width_a = float(np.linalg.norm(pa.sides[end_a])) / 2.5
    width_b = float(np.linalg.norm(pb.sides[end_b])) / 2.5
    if not 0.5 < distance <= min(12, max(4, 4 * max(width_a, width_b))):
        return None
    if max(width_a, width_b) > 1.6 * min(width_a, width_b):
        return None
    before_a = (
        pa.points[min(4, len(pa.points) - 1)]
        if end_a == 0
        else pa.points[max(-5, -len(pa.points))]
    )
    before_b = (
        pb.points[min(4, len(pb.points) - 1)]
        if end_b == 0
        else pb.points[max(-5, -len(pb.points))]
    )
    outward_a, outward_b = first - before_a, last - before_b
    unit = delta / distance
    if (
        float(outward_a @ unit) < 0.8 * np.linalg.norm(outward_a)
        or float(outward_b @ -unit) < 0.8 * np.linalg.norm(outward_b)
        or np.linalg.norm(outward_a) < 1e-6
        or np.linalg.norm(outward_b) < 1e-6
    ):
        return None
    return SourceProfile.at(np.array((first, last)), (width_a + width_b) / 2).dense()


class FittingLines:
    """Crop penalties for proposal fitting, never complete acceptance evidence."""

    def __init__(self, prepared, bounds):
        self._profiles, self.bounds = prepared, bounds
        self._authority = object()
        self.shape = (bounds[3] - bounds[1], bounds[2] - bounds[0], 4)

    def observe(self, actual, *, work=None):
        if (
            actual.shape != self.shape
            or not np.isfinite(actual).all()
            or np.any((actual < 0) | (actual > 1))
        ):
            raise ValueError("Line fitting raster must match its finite native crop")
        visible = actual[..., 3] > VISIBLE
        errors, gaps = [], []
        for record in self._profiles:
            _check(work)
            errors.append(_error(record, actual, visible))
            gaps.append(_readonly(_gap_state(record, actual, visible)))
        _check(work)
        return LineObservation(self._authority, tuple(errors), tuple(gaps))

    def penalty(self, before, after, *, work=None):
        if (
            not isinstance(before, LineObservation)
            or before.authority is not self._authority
        ):
            raise ValueError("Line fitting observation belongs to another crop")
        current = self.observe(after, work=work)
        value = 0.0
        for old, now, a, b in zip(
            before.profiles, current.profiles, before.gaps, current.gaps, strict=True
        ):
            _check(work)
            value += 1000 * max(
                0, now.missing - old.missing - max(1, round(now.samples * 0.02))
            )
            value += 1000 * max(
                0, now.maximum_missing_span - old.maximum_missing_span - 2
            )
            value += 10000 * int((b & ~a).sum())
        return float(value)


class FittingBody:
    """Positive source support supplied by one isolated editable stroke.

    Queries use frozen qualified observations and their raw-positive normal
    matching windows. Painted backgrounds cannot supply missing stroke support.
    This contract does not authorize endpoint movement or prove fill removal.
    """

    SUPPORT_ALPHA = 0.05

    def __init__(self, profile, qualified, allowed, bounds):
        shifts = np.linspace(-1, 1, 9)
        points = (
            profile.points[:, None, :]
            + shifts[None, :, None]
            * profile.tolerance[:, None, None]
            * profile.direction[:, None, :]
            - np.asarray(bounds[:2])
            - 0.5
        )
        self._points = _readonly(points[qualified].reshape(-1, 2))
        self._allowed = _readonly(allowed[qualified])
        self.shape = (bounds[3] - bounds[1], bounds[2] - bounds[0])

    def observe(self, alpha, *, work=None):
        _check(work)
        if (
            alpha.shape != self.shape
            or not np.isfinite(alpha).all()
            or np.any((alpha < 0) | (alpha > 1))
        ):
            raise ValueError("Stroke support requires finite aligned native alpha")
        sampled = map_coordinates(
            alpha,
            [self._points[:, 1], self._points[:, 0]],
            order=1,
            mode="constant",
            cval=0,
        ).reshape(-1, 9)
        support = np.max(np.where(self._allowed, sampled, 0), axis=1)
        missing = support < self.SUPPORT_ALPHA
        _check(work)
        return {
            "qualified_samples": len(support),
            "supported_samples": int((~missing).sum()),
            "missing_samples": int(missing.sum()),
            "penalty": 1000 * float(np.maximum(0, self.SUPPORT_ALPHA - support).sum()),
            "support_alpha": self.SUPPORT_ALPHA,
            "raw_positive_matching_windows": True,
        }


class SourceLineGuard:
    """Immutable source supports and one fixed, per-chain baseline allowance.

    Protects inspected measured troughs, not every possible line in the image.
    Sparse source discovery and uncalibrated thresholds remain explicit limits.
    """

    def __init__(self, truth, profiles, *, work=None):
        self.shape = truth.shape
        self._validate(truth)
        prepared = []
        source, body_windows = [], []
        count = 0
        visible = truth[..., 3] > VISIBLE
        for index, raw in enumerate(profiles):
            _check(work)
            if index >= MAX_PROFILES:
                raise ValueError("Source lines exceed the bounded profile limit")
            profile = raw.dense()
            count += len(profile.points)
            if count > MAX_SAMPLES:
                raise ValueError("Source lines exceed the native sample limit")
            measured = _prepare(truth, profile, visible)
            if measured is not None:
                prepared.append(measured)
                contrast, alpha, valid, _light = _sample(
                    truth, profile, np.linspace(-1, 1, 9), visible
                )
                body_windows.append(
                    _readonly(
                        valid & (contrast >= 6) & (alpha >= 0.5 * alpha[:, 4, None])
                    )
                )
                source.append(
                    (
                        raw,
                        SourceBreaks(
                            profile.points,
                            measured[3],
                            measured[4],
                            _readonly(raw.points),
                        ),
                    )
                )
        original = tuple(prepared)
        self.gap_profiles = 0
        for i, a in enumerate(original):
            for b in original[i + 1 :]:
                _check(work)
                if a[0].component is not None and a[0].component == b[0].component:
                    continue
                for end_a in (0, -1):
                    for end_b in (0, -1):
                        probe = _endpoint_gap(a, b, end_a, end_b)
                        if probe is None:
                            continue
                        count += len(probe.points)
                        if count > MAX_SAMPLES:
                            raise ValueError(
                                "Source lines exceed the native sample limit"
                            )
                        measured = _prepare(truth, probe, visible)
                        if measured is None or not measured[-1].any():
                            continue
                        self.gap_profiles += 1
                        if self.gap_profiles > MAX_GAP_PROFILES:
                            raise ValueError(
                                "Source gaps exceed the bounded profile limit"
                            )
                        # Gap probes protect source absence only. Positive ends
                        # already belong to their complete original profiles.
                        prepared.append(
                            (
                                measured[0],
                                measured[1],
                                measured[2],
                                _readonly(np.zeros(len(probe.points), bool)),
                                measured[4],
                            )
                        )
        _check(work)
        self._source = tuple(source)
        self._body_windows = tuple(body_windows)
        self._profiles = tuple(prepared)
        self._baseline = None
        self._baseline_gaps = None
        self._authority = object()
        self.sampled = count

    def _validate(self, rgba):
        if (
            rgba.shape != self.shape
            or rgba.ndim != 3
            or rgba.shape[-1] != 4
            or not np.isfinite(rgba).all()
            or np.any((rgba < 0) | (rgba > 1))
        ):
            raise ValueError("Source line raster must be finite aligned RGBA in [0, 1]")

    def assess(self, actual, *, work=None):
        self._validate(actual)
        result = []
        visible = actual[..., 3] > VISIBLE
        for prepared in self._profiles:
            _check(work)
            result.append(_error(prepared, actual, visible))
        _check(work)
        return tuple(result)

    def source_breaks(self, profile, *, work=None):
        """Own raw gaps only; facing endpoint probes cannot split a host chain.

        Identity binds observations to the original measured physical profile.
        Unknown, copied or unsupported profiles cannot authorize reconstruction.
        Arrays are immutable copies made during source preparation.
        """
        for original, observed in self._source:
            _check(work)
            if original is profile:
                return observed
        _check(work)
        return None

    def original_profiles(self, *, work=None):
        """Original identities for queries of copied physical observations.

        Consumers must use ``source_breaks`` for coordinates and qualification;
        the original profile is an identity, not a mutable replacement witness.
        Facing endpoint probes cannot authorize extending a physical chain.
        """
        _check(work)
        return tuple(original for original, _observed in self._source)

    def fitting_body(self, profile, bounds, *, work=None):
        """Copy a bounded stroke-only contract from this original source bank."""
        _check(work)
        bounds = np.asarray(bounds)
        if (
            bounds.shape != (4,)
            or not np.isfinite(bounds).all()
            or not np.array_equal(bounds, np.floor(bounds))
            or np.any(bounds[:2] < 0)
            or np.any(bounds[2:] > self.shape[1::-1])
            or np.any(bounds[2:] <= bounds[:2])
        ):
            raise ValueError("Stroke support requires an integer native crop")
        for i, (original, observed) in enumerate(self._source):
            _check(work)
            if original is profile:
                if len(observed.points) > 16_384 or not observed.qualified.any():
                    return None
                # Prepared profiles, unlike original identity handles, are
                # immutable copies of the measured native directions/centres.
                return FittingBody(
                    self._profiles[i][0],
                    observed.qualified,
                    self._body_windows[i],
                    tuple(map(int, bounds)),
                )
        _check(work)
        return None

    def fitting_crop(self, bounds, *, work=None):
        """A bounded optimization hint using the unchanged original observations.

        Qualification and negative evidence are copied, never remeasured against
        a cropped source. This cannot accept an edit: publication still requires
        complete native comparison against this original guard.
        """
        _check(work)
        bounds = np.asarray(bounds)
        if (
            bounds.shape != (4,)
            or not np.isfinite(bounds).all()
            or not np.array_equal(bounds, np.floor(bounds))
            or np.any(bounds[:2] < 0)
            or np.any(bounds[2:] > self.shape[1::-1])
            or np.any(bounds[2:] <= bounds[:2])
        ):
            raise ValueError("Line fitting requires an integer native crop")
        prepared, count = [], 0
        for record in self._profiles:
            _check(work)
            profile = record[0]
            inside = (
                (profile.points >= bounds[:2]) & (profile.points < bounds[2:])
            ).all(axis=1)
            if not inside.any():
                continue
            count += len(profile.points)
            if count > 16_384 or len(prepared) >= 32:
                return None
            prepared.append(
                (
                    replace(profile, points=_readonly(profile.points - bounds[:2])),
                    *record[1:],
                )
            )
        _check(work)
        return FittingLines(tuple(prepared), tuple(map(int, bounds)))

    def gap_centres(self, *, limit=MAX_SAMPLES, work=None):
        """Copy inspected raw absence positions for a bounded body constraint.

        These are source observations, including failed source chains and
        facing endpoint gaps. They do not authorize moving an endpoint or
        generating a link, and their engineering thresholds are uncalibrated.
        Check the requested allocation before concatenating the observations.
        """
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("Source absence requires a nonnegative point limit")
        count = 0
        pieces = []
        for profile, _contrast, _alpha, _qualified, gaps in self._profiles:
            _check(work)
            count += int(gaps.sum())
            if count > limit:
                raise ValueError("Source absence exceeds the bounded point limit")
            pieces.append(profile.points[gaps])
        _check(work)
        return _readonly(np.concatenate(pieces) if pieces else np.empty((0, 2)))

    def establish(self, actual, *, work=None):
        if self._baseline is not None:
            raise ValueError("Source line baseline cannot change")
        measured = self.assess(actual, work=work)
        samples = sum(r.samples for r in measured)
        if samples and sum(r.missing for r in measured) == samples:
            raise ValueError("Source line baseline cannot lose every inspected line")
        gaps = self.gap_states(actual, work=work)
        _check(work)
        self._baseline, self._baseline_gaps = measured, gaps

    def gap_states(self, actual, *, work=None):
        """Immutable observations at each original gap, never aggregate counts.

        Representation changes can uncover one old gap while completing another.
        Equal profile totals do not establish preservation of either position.
        This remains the inspected, uncalibrated photometric contract.
        """
        self._validate(actual)
        visible = actual[..., 3] > VISIBLE
        result = []
        for prepared in self._profiles:
            _check(work)
            result.append(_readonly(_gap_state(prepared, actual, visible)))
        _check(work)
        return tuple(result)

    def compare(self, before, after, *, work=None):
        """Compare a complete painted edit without changing the fixed baseline."""
        return self.compare_observed(self.observe(before, work=work), after, work=work)

    def observe(self, actual, *, work=None):
        """Measure one parent once for its bounded geometry/width alternatives."""
        profiles = self.assess(actual, work=work)
        gaps = self.gap_states(actual, work=work)
        _check(work)
        return LineObservation(self._authority, profiles, gaps)

    def compare_observed(self, before, after, *, work=None):
        """Compare against a complete observation from this exact source bank."""
        _check(work)
        if (
            not isinstance(before, LineObservation)
            or before.authority is not self._authority
        ):
            raise ValueError("Painted line observation belongs to another source bank")
        current = self.assess(after, work=work)
        new_gaps = self.gap_states(after, work=work)
        changes = tuple(
            int((new & ~old).sum())
            for old, new in zip(before.gaps, new_gaps, strict=True)
        )
        rejected = self._rejections(before.profiles, current, changes)
        _check(work)
        return {
            "version": LINE_FIDELITY_VERSION,
            "scope": "inspected-measured-source-troughs",
            "calibrated_release_gate": False,
            "new_gap_completed": sum(changes),
            "qualified_samples": sum(r.samples for r in current),
            "missing_samples": sum(r.missing for r in current),
            "rejections": rejected,
        }

    @staticmethod
    def _rejections(initial, current, changes):
        rejected = []
        for index, (now, old, new_gaps) in enumerate(
            zip(current, initial, changes, strict=True)
        ):
            allowance = max(1, round(now.samples * 0.02))
            if (
                now.missing > old.missing + allowance
                or now.maximum_missing_span > old.maximum_missing_span + 2
            ):
                rejected.append({"profile": index, "reason": "source-line-lost"})
            if new_gaps:
                rejected.append(
                    {"profile": index, "reason": "source-line-gap-completed"}
                )
        return rejected

    def metrics(self, actual, *, work=None):
        if self._baseline is None:
            raise ValueError("Source line baseline must be established")
        result = self.assess(actual, work=work)
        assert self._baseline_gaps is not None
        gaps = self.gap_states(actual, work=work)
        changes = tuple(
            int((now & ~old).sum())
            for old, now in zip(self._baseline_gaps, gaps, strict=True)
        )
        rejected = self._rejections(self._baseline, result, changes)
        samples = sum(r.samples for r in result)
        return {
            "version": LINE_FIDELITY_VERSION,
            "scope": "inspected-measured-source-troughs",
            "calibrated_release_gate": False,
            "sampled_positions": self.sampled,
            "endpoint_gap_profiles": self.gap_profiles,
            "qualified_samples": samples,
            "missing_samples": sum(r.missing for r in result),
            "missing_share": sum(r.missing for r in result) / max(1, samples),
            "gap_samples": sum(r.gap_samples for r in result),
            "gap_completed": sum(r.gap_completed for r in result),
            "new_gap_completed": sum(changes),
            "profiles": [dict(vars(r)) for r in result],
            "baseline": [dict(vars(r)) for r in self._baseline],
            "rejections": rejected,
        }
