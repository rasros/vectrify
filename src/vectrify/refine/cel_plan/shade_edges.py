"""Bounded source fitting of two opaque materials and their shared shade edge.

Pixel coverage belongs to geometry; it is not a third material or intrinsic
paint opacity. These hypotheses require an independently proved opaque core
before export as one whole-family base and a complementary shade overlay.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from vectrify.refine.cel_plan.surface_models import prediction
from vectrify.refine.cel_plan.surface_splits import fit

MAX_SAMPLES = 4096
MAX_EVALUATIONS = 24
MAX_ANGLE = np.pi / 32
MAX_SHIFT = 4.0


def coverage(xy, normal, rho):
    """Exact unit-square area left of a line, independent of normal signs.

    The sum of two uniform coordinates has a piecewise quadratic cumulative
    distribution. Evaluate only its transition band to avoid cancellation on
    distant coordinates. Native rasterization remains the admission authority.
    """
    a, b = sorted(np.abs(normal), reverse=True)
    if not np.isfinite((a, b, rho)).all() or a <= 0:
        raise ValueError("Shade coverage needs a finite nonzero line")
    distance = rho - xy @ normal
    if b < a * 1e-8:
        return np.clip(distance / a + 0.5, 0, 1)
    result = (distance >= 0).astype(float)
    transition = np.abs(distance) < (a + b) / 2
    d = distance[transition]
    result[transition] = (
        np.maximum(d + (a + b) / 2, 0) ** 2
        - np.maximum(d + (a - b) / 2, 0) ** 2
        - np.maximum(d + (-a + b) / 2, 0) ** 2
        + np.maximum(d - (a + b) / 2, 0) ** 2
    ) / (2 * a * b)
    return np.clip(result, 0, 1)


def predict(paints, xy, normal, rho, *, extend):
    weight = coverage(xy, normal, rho)[:, None]
    return weight * prediction(paints[0], xy, extend=extend) + (
        1 - weight
    ) * prediction(paints[1], xy, extend=extend)


def paints(xy, rgba, normal, rho, *, gradients):
    """Fit pure-side colors, excluding the shared edge's mixed pixels.

    Input alpha is deliberately unused: callers must prove that child paint is
    opaque inside its current group. Genuine material translucency is outside
    this model, and must keep the adjacent RGBA interpretation.
    """
    weight = coverage(xy, normal, rho)
    pure = (weight >= 1 - 1e-6, weight <= 1e-6)
    if min(int(side.sum()) for side in pure) < 16:
        return None
    opaque = np.column_stack((rgba[:, :3], np.ones(len(rgba))))
    return tuple(fit(xy[side], opaque[side], gradients=gradients) for side in pure)


def refine(xy, rgba, normal, rho, work, *, gradients, diagnostics):
    """Alternate pure-color fits and bounded subpixel line optimization.

    Only the planning fit samples; complete owner support is screened later.
    Angle and offset are local to the source vote, never a geometry-generation
    network. Cancellation cannot publish a partly optimized interpretation.
    """
    if work.interrupted:
        return None
    stride = max(1, (len(xy) + MAX_SAMPLES - 1) // MAX_SAMPLES)
    points, values = xy[::stride], rgba[::stride]
    center = points.mean(axis=0)
    relative = points - center
    theta = float(np.arctan2(normal[1], normal[0]))
    offset = float(rho - center @ normal)
    initial = np.array((theta, offset))
    parameters = initial.copy()
    lower = initial - (MAX_ANGLE, MAX_SHIFT)
    upper = initial + np.array((MAX_ANGLE, MAX_SHIFT))
    for _ in range(2):
        if work.interrupted:
            return None
        n = np.array((np.cos(parameters[0]), np.sin(parameters[0])))
        r = float(parameters[1] + center @ n)
        models = paints(points, values, n, r, gradients=gradients)
        if models is None:
            diagnostics["shade_fit_exclusions"] += 1
            return None
        fields = tuple(prediction(p, points, extend=True) for p in models)

        def residual(candidate, fields=fields):
            if work.interrupted:
                return np.zeros(3 * len(points))
            diagnostics["shade_fit_evaluations"] += 1
            direction = np.array((np.cos(candidate[0]), np.sin(candidate[0])))
            mix = coverage(relative, direction, candidate[1])[:, None]
            return (mix * fields[0] + (1 - mix) * fields[1] - values[:, :3]).ravel()

        solved = least_squares(
            residual,
            parameters,
            bounds=(lower, upper),
            max_nfev=MAX_EVALUATIONS,
            loss="soft_l1",
            f_scale=16 / 255,
            x_scale=(0.01, 1),
        )
        if work.interrupted or not np.isfinite(solved.x).all():
            return None
        parameters = solved.x
    direction = np.array((np.cos(parameters[0]), np.sin(parameters[0])))
    intercept = float(parameters[1] + center @ direction)
    models = paints(xy, rgba, direction, intercept, gradients=gradients)
    if models is None or work.interrupted:
        return None
    diagnostics["shade_fits"] += 1
    return direction, intercept, models
