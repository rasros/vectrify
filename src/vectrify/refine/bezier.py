"""Cubic Bezier arithmetic: weights, points, splits, fits and a path's segments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, overload

import numpy as np

if TYPE_CHECKING:
    from vectrify.document import PathNode


def basis(t: Any) -> np.ndarray:
    """The cubic's Bernstein weights at each of *t*, on a new last axis."""
    u = 1 - t
    return np.stack((u**3, 3 * u**2 * t, 3 * u * t**2, t**3), axis=-1)


def evaluate(control: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Points at each of *t* along a line (two controls) or cubic (four).

    Controls (..., k, 2) give points (..., len(t), 2); *t* may also carry the
    leading axes, a row of parameters for each curve.
    """
    if control.shape[-2] == 2:
        start = control[..., :1, :]
        return start + (control[..., 1:, :] - start) * t[..., None]
    return np.einsum("...sk,...kc->...sc", basis(t), control)


def derivative(control: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The derivative at each of *t* along a line (two controls) or cubic
    (four), shaped as :func:`evaluate`'s points."""
    t = t[..., None]
    if control.shape[-2] == 2:
        step = control[..., 1:, :] - control[..., :1, :]
        return np.repeat(step, t.shape[-2], axis=-2)
    a, b, c, d = (control[..., k, None, :] for k in range(4))
    u = 1 - t
    return 3 * u**2 * (b - a) + 6 * u * t * (c - b) + 3 * t**2 * (d - c)


def split(control: np.ndarray, t: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """The controls of *control*'s two parts either side of *t*, by de
    Casteljau, for a curve of any degree."""
    levels = [control]
    while len(levels[-1]) > 1:
        previous = levels[-1]
        levels.append(previous[:-1] + (previous[1:] - previous[:-1]) * t)
    return np.array([level[0] for level in levels]), np.array(
        [level[-1] for level in reversed(levels)]
    )


def straight(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """The straight cubic from *start* to *end*, its handles at the thirds."""
    step = end - start
    return np.stack([start, start + step / 3, start + 2 * step / 3, end], axis=-2)


@overload
def fit_cubic(
    points: np.ndarray,
    t: np.ndarray,
    *,
    ends: tuple[np.ndarray, np.ndarray] | None = None,
    leave: np.ndarray | None = None,
    arrive: np.ndarray | None = None,
    clamp: bool = False,
) -> np.ndarray: ...


@overload
def fit_cubic(
    points: np.ndarray,
    t: None = None,
    *,
    ends: tuple[np.ndarray, np.ndarray] | None = None,
    leave: np.ndarray | None = None,
    arrive: np.ndarray | None = None,
    clamp: bool = False,
) -> np.ndarray | None: ...


def fit_cubic(
    points: np.ndarray,
    t: np.ndarray | None = None,
    *,
    ends: tuple[np.ndarray, np.ndarray] | None = None,
    leave: np.ndarray | None = None,
    arrive: np.ndarray | None = None,
    clamp: bool = False,
) -> np.ndarray | None:
    """The controls of the cubic from *ends*, else *points*' first and last,
    that passes closest to *points*, each at its parameter in *t*.

    By default *t* goes by length along the points, and None comes back when
    they have none. Given the unit directions *leave* and *arrive*, only the
    handles' lengths are fitted, a third of the chord each where that fails;
    with *clamp*, neither longer than the chord.
    """
    start, end = (points[0], points[-1]) if ends is None else ends
    if t is None:
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total = steps.sum()
        if total < 1e-9:
            return None
        t = np.concatenate([[0], np.cumsum(steps)]) / total
    weights = basis(t)
    if leave is None or arrive is None:
        fixed = weights[:, :1] * start + weights[:, 3:] * end
        handles, *_ = np.linalg.lstsq(weights[:, 1:3], points - fixed, rcond=None)
        return np.vstack([start, handles, end])
    fixed = (weights[:, 0] + weights[:, 1])[:, None] * start + (
        weights[:, 2] + weights[:, 3]
    )[:, None] * end
    matrix = np.stack(
        [weights[:, 1:2] * leave, weights[:, 2:3] * arrive], axis=-1
    ).reshape(-1, 2)
    lengths, *_ = np.linalg.lstsq(matrix, (points - fixed).reshape(-1), rcond=None)
    chord = float(np.linalg.norm(end - start))
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        lengths = np.array([chord / 3, chord / 3])
    if clamp:
        lengths = np.minimum(lengths, chord)
    return np.vstack(
        [start, start + lengths[0] * leave, end + lengths[1] * arrive, end]
    )


def segment_controls(
    nodes: Sequence[PathNode],
    pixels: Callable[[tuple[float, ...]], np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Each segment's controls along *nodes*, two for a line and four for a
    cubic, through *pixels* (a frame's) when given. A closed subpath's
    closing line is not among them."""
    if pixels is None:

        def pixels(values):
            return np.asarray(values, dtype=np.float64).reshape(-1, 2)

    if not nodes:
        return []
    controls = []
    start = pixels(nodes[0].values)[-1]
    for node in nodes[1:]:
        points = pixels(node.values)
        controls.append(np.vstack([start, points]))
        start = points[-1]
    return controls
