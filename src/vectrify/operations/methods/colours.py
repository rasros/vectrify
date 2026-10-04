"""Improve: fit the fills of the selected objects, geometry locked.

Compositing is linear in an object's fill colour: every pixel of the rendered
region equals ``backdrop + coverage * fill``, where coverage already includes
the object's antialiasing, opacity, clipping and everything painted in front.
Rendering the region twice, once with the fill black and once white, measures
both terms exactly, so the fill that best matches the reference is a closed-form
least-squares solution per channel. Objects are fitted back to front, each
against the drawing as already refitted. No GPU and no search are involved.

With ``fill: "linear"`` the fill may vary across the object: each channel is
fitted as an affine field ``a + b x + c y`` by the same weighted least squares,
the fields' common direction (the first singular vector of their 3x2 gradient)
becomes the gradient's axis. The ramp may start and stop inside the object,
flat beyond its ends (the gradient's padding): where it starts and ends along
the axis is searched on a grid over the covered extent and refined, with the
axis turned a few degrees either way, each candidate's two end colours solved
in closed form. Ends past the object's edges paint it the same as ends at its
edges with the colours there, so the search stays within it. A ramp that
changes by less than about two levels of 255 stays a flat fill.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np

from vectrify.document import Document, DocumentError
from vectrify.document.join import path_style
from vectrify.document.paint import GradientStop, LinearGradient, hex_colour
from vectrify.document.redraw import root_matrix
from vectrify.image_utils import preview_urls
from vectrify.operations.contract import (
    OperationRequest,
    OperationResult,
    Proposal,
    RunContext,
    register,
)
from vectrify.operations.generate import Region, error, render_region, target_region
from vectrify.operations.settings import Setting, read_settings

DRAWABLE = {"path", "rect", "circle", "ellipse", "use"}
# What can take a gradient: an instance's user space is its source's.
GRADIENT_DRAWABLE = {"path", "rect", "circle", "ellipse"}
SETTINGS = {
    "passes": Setting(int, 1, minimum=1, maximum=5),
    "resolution": Setting(int, 256, minimum=32, maximum=1024, label="resolution"),
    "fill": Setting(str, "flat", choices=("flat", "linear"), label="fill kind"),
}
# A gradient whose ends differ by less than this, per channel, is flat.
FLAT = 2 / 255
# Pixels that count as covered when placing a gradient's ends.
COVERED = 0.05


def targets(document: Document, request: OperationRequest) -> list[str]:
    """Selected drawables with a fill, in paint order (back to front)."""
    selected = document.selection_ids(request.snapshot.selection)
    found = []
    for element in document.elements():
        if element.id not in selected or element.tag not in DRAWABLE:
            continue
        if any(a.tag in {"defs", "clipPath"} for a in document.ancestry(element.id)):
            continue
        if path_style(document, element)["fill"] != "none":
            found.append(element.id)
    return found


def _with_fill(document: Document, oid: str, fill: str) -> Document:
    element = document.element(oid)
    attributes = dict(element.attributes)
    attributes["fill"] = fill
    return document.replace_element(
        replace(element, attributes=tuple(attributes.items()))
    )


def _array(document: Document, region: Region) -> np.ndarray:
    return np.asarray(render_region(document, region), dtype=np.float64) / 255


def _terms(
    document: Document, oid: str, region: Region
) -> tuple[np.ndarray, np.ndarray] | None:
    """The backdrop and coverage of *oid* in the region, or None if hidden."""
    dark = _array(_with_fill(document, oid, "#000000"), region)
    light = _array(_with_fill(document, oid, "#ffffff"), region)
    coverage = light - dark
    if float(np.sum(coverage * coverage)) < 1e-6:
        return None
    return dark, coverage


def _flat(dark: np.ndarray, coverage: np.ndarray, target: np.ndarray) -> str:
    channels = np.sum(coverage * (target - dark), axis=(0, 1)) / np.sum(
        coverage * coverage, axis=(0, 1)
    ).clip(1e-12)
    return hex_colour(tuple(float(c) for c in np.clip(channels, 0, 1)))


def _solve(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Least squares of *values* (n,) on *design* (n, k), robust to rank loss."""
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _pixel_points(region: Region) -> tuple[np.ndarray, np.ndarray]:
    """Root user-space coordinates of every pixel centre of the region."""
    width, height = region.image.size
    xs = region.x + (np.arange(width) + 0.5) * region.width / width
    ys = region.y + (np.arange(height) + 0.5) * region.height / height
    return np.meshgrid(xs, ys)


@dataclass(frozen=True)
class Ramp:
    """A fitted linear ramp in root user space: colours at *start* and *end*."""

    start: tuple[float, float]
    end: tuple[float, float]
    colours: tuple[tuple[float, ...], tuple[float, ...]]

    def flat(self) -> bool:
        a, b = self.colours
        return max(abs(x - y) for x, y in zip(a, b, strict=True)) < FLAT


def fit_ramp(
    dark: np.ndarray, coverage: np.ndarray, target: np.ndarray, region: Region
) -> Ramp | None:
    """The linear ramp that best fits the region, or None if it has no extent."""
    xs, ys = _pixel_points(region)
    weight = coverage.mean(axis=2)
    covered = weight > COVERED
    if covered.sum() < 3:
        return None
    # Centre the coordinates so the normal equations are well conditioned.
    cx, cy = float(xs[covered].mean()), float(ys[covered].mean())
    x, y = (xs - cx).ravel(), (ys - cy).ravel()
    slopes = np.zeros((3, 2))
    for c in range(3):
        cov = coverage[..., c].ravel()
        design = np.stack([cov, cov * x, cov * y], axis=1)
        slopes[c] = _solve(design, (target - dark)[..., c].ravel())[1:]
    _, singular, vt = np.linalg.svd(slopes)
    if singular[0] < 1e-12:
        return None
    # The ramp may start and stop inside the object, flat beyond its ends:
    # search where along the axis, and the axis a few degrees either way.
    keep = coverage.mean(axis=2).ravel() > 0
    cov = coverage.reshape(-1, 3)[keep]
    values = (target - dark).reshape(-1, 3)[keep]
    x, y = x[keep], y[keep]
    inside = covered.ravel()[keep]
    base = float(np.arctan2(vt[0][1], vt[0][0]))
    best = None
    for turn in TURNS:
        angle = base + np.radians(turn)
        direction = (float(np.cos(angle)), float(np.sin(angle)))
        s = direction[0] * x + direction[1] * y
        low, high = float(s[inside].min()), float(s[inside].max())
        if high - low < 1e-9:
            continue
        found = _ends(s, cov, values, low, high)
        if best is None or found[0] < best[0]:
            best = (found[0], direction, found[1], found[2], found[3])
    if best is None:
        return None
    _, direction, t0, t1, colours = best
    ends = tuple((cx + direction[0] * t, cy + direction[1] * t) for t in (t0, t1))
    clipped = tuple(
        tuple(float(np.clip(colours[i][c], 0, 1)) for c in range(3)) for i in (0, 1)
    )
    return Ramp(ends[0], ends[1], (clipped[0], clipped[1]))


# Turns of the fitted axis tried, in degrees, and the grid of ends searched.
TURNS = (0.0, -1.0, 1.0, -2.5, 2.5, -4.0, 4.0)
STEPS = 16


def _ramp_error(s, cov, values, t0, t1):
    """The squared error and end colours of the ramp from *t0* to *t1*, flat
    beyond: each channel's two end colours in closed form."""
    u = np.clip((s - t0) / (t1 - t0), 0, 1)[:, None]
    p, q = cov * (1 - u), cov * u
    pp, pq, qq = (p * p).sum(0), (p * q).sum(0), (q * q).sum(0)
    pv, qv = (p * values).sum(0), (q * values).sum(0)
    det = pp * qq - pq * pq
    # Where the two ends can't be told apart, one colour fills it.
    single = np.abs(det) < 1e-12
    safe = np.where(single, 1.0, det)
    flat = pv / np.maximum(pp + qq, 1e-12)
    c0 = np.where(single, flat, (qq * pv - pq * qv) / safe)
    c1 = np.where(single, flat, (pp * qv - pq * pv) / safe)
    residual = values - p * c0 - q * c1
    return float((residual * residual).sum()), (c0, c1)


def _ramp_errors(s, cov, values, t0, t1) -> np.ndarray:
    """:func:`_ramp_error`'s squared error for each pair of *t0* and *t1*,
    many at once: the residual expanded in the sums the two end colours are
    solved from, so no pair needs its own pass over the pixels' residuals."""
    u = np.clip((s[None] - t0[:, None]) / (t1 - t0)[:, None], 0, 1)
    # Per pair and channel: the sums of p p, p q, q q, p v, q v with
    # p = cov (1 - u), q = cov u.
    cc, cv = cov * cov, cov * values
    uu = u * u
    a = cc.sum(0)[None]
    b = u @ cc
    c = uu @ cc
    pp, pq, qq = a - 2 * b + c, b - c, c
    av = cv.sum(0)[None]
    qv = u @ cv
    pv = av - qv
    det = pp * qq - pq * pq
    single = np.abs(det) < 1e-12
    safe = np.where(single, 1.0, det)
    flat = pv / np.maximum(pp + qq, 1e-12)
    c0 = np.where(single, flat, (qq * pv - pq * qv) / safe)
    c1 = np.where(single, flat, (pp * qv - pq * pv) / safe)
    # |v - p c0 - q c1|^2, summed over the pixels.
    vv = (values * values).sum(0)[None]
    error = (
        vv - 2 * c0 * pv - 2 * c1 * qv + c0 * c0 * pp + 2 * c0 * c1 * pq + c1 * c1 * qq
    )
    return error.sum(1)


# Pairs of ends whose errors are worked out together, times the pixels.
BATCH = 4_000_000


def _ends(s, cov, values, low, high):
    """The best ends along the axis: a coarse grid over the covered extent,
    then twice a grid four times finer around the best pair."""
    step = (high - low) / STEPS
    shortest = (high - low) / 200

    def search(starts, ends, best):
        pairs = np.array(
            [(t0, t1) for t0 in starts for t1 in ends if t1 - t0 >= shortest]
        )
        if not len(pairs):
            return best
        chunk = max(1, BATCH // max(len(s), 1))
        errors = np.concatenate(
            [
                _ramp_errors(s, cov, values, part[:, 0], part[:, 1])
                for part in np.array_split(pairs, -(-len(pairs) // chunk))
            ]
        )
        i = int(np.argmin(errors))
        t0, t1 = float(pairs[i, 0]), float(pairs[i, 1])
        err, colours = _ramp_error(s, cov, values, t0, t1)
        return (err, t0, t1, colours) if err < best[0] else best

    grid = [low + step * i for i in range(STEPS + 1)]
    best = search(grid, grid, (np.inf, low, high, None))

    def around(t, step):
        return sorted({min(high, max(low, t + step * k)) for k in range(-4, 5)})

    for _ in range(2):
        step /= 4
        best = search(around(best[1], step), around(best[2], step), best)
    err, t0, t1, colours = best
    assert colours is not None
    return err, t0, t1, (tuple(colours[0]), tuple(colours[1]))


def local_gradient(document: Document, oid: str, ramp: Ramp) -> LinearGradient:
    """*ramp*, as a gradient in *oid*'s own user space.

    The ramp's parameter is affine in root coordinates, so it is affine in the
    object's too, but under a skew or uneven scale its level lines are no
    longer perpendicular to the axis there. The ends are chosen so the
    gradient's own perpendicular level lines are exactly the ramp's.
    """
    a, b, c, d, e, f = root_matrix(document, oid)
    (x0, y0), (x1, y1) = ramp.start, ramp.end
    dx, dy = x1 - x0, y1 - y0
    length = dx * dx + dy * dy
    # t = beta . p + alpha in root space, mapped through p = M q + m.
    beta = (dx / length, dy / length)
    alpha = -(beta[0] * x0 + beta[1] * y0)
    local = (a * beta[0] + b * beta[1], c * beta[0] + d * beta[1])
    alpha += beta[0] * e + beta[1] * f
    norm = local[0] ** 2 + local[1] ** 2
    start = (-alpha * local[0] / norm, -alpha * local[1] / norm)
    end = (start[0] + local[0] / norm, start[1] + local[1] / norm)
    stops = tuple(
        GradientStop(offset, hex_colour(colour))
        for offset, colour in zip((0.0, 1.0), ramp.colours, strict=True)
    )
    return LinearGradient(start, end, stops)


class ColourFit:
    action: ClassVar[str] = "improve"
    name: ClassVar[str] = "colours"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        read_settings(request.settings, SETTINGS, "colour-fit")
        if not request.permissions.paint:
            raise DocumentError("Allow paint changes to fit colours")
        if not targets(request.snapshot.document, request):
            raise DocumentError("Select objects with a solid fill or a gradient")
        target_region(request)

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        settings = read_settings(request.settings, SETTINGS, "colour-fit")
        linear = settings["fill"] == "linear"
        full = target_region(request)
        scale = min(1, settings["resolution"] / max(full.image.size))
        size = (
            max(1, round(full.image.width * scale)),
            max(1, round(full.image.height * scale)),
        )
        region = replace(full, image=full.image.convert("RGB").resize(size))
        target = np.asarray(region.image, dtype=np.float64) / 255
        document = request.snapshot.document
        ids = targets(document, request)
        total = len(ids) * settings["passes"]
        tx = request.transaction("Fit gradients" if linear else "Fit colours")
        fitted: set[str] = set()
        gradients: set[str] = set()
        step = 0
        for _ in range(settings["passes"]):
            for oid in ids:
                if context.stop.is_set():
                    break
                context.progress(step, f"Fitting {step + 1} of {total}…", total=total)
                step += 1
                terms = _terms(tx.preview, oid, region)
                if terms is None:
                    continue
                fill: str | LinearGradient = _flat(*terms, target)
                if linear and tx.preview.element(oid).tag in GRADIENT_DRAWABLE:
                    ramp = fit_ramp(*terms, target, region)
                    if ramp is not None and not ramp.flat():
                        fill = local_gradient(tx.preview, oid, ramp)
                if self._apply(tx, oid, fill):
                    fitted.add(oid)
                if isinstance(fill, LinearGradient):
                    gradients.add(oid)
                else:
                    gradients.discard(oid)
        before_image = render_region(document, full)
        after_image = render_region(tx.preview, full)
        reference = full.image.convert("RGB")
        return OperationResult(
            Proposal(
                tx,
                bool(fitted),
                metrics={
                    "before": {"error": error(before_image, reference)},
                    "after": {"error": error(after_image, reference)},
                    "objects": len(fitted),
                    "gradients": len(gradients & fitted),
                    "considered": len(ids),
                },
                previews=preview_urls(full.image, before_image, after_image),
            ),
            message=None if fitted else "The colours already fit the reference",
        )

    @staticmethod
    def _apply(tx, oid: str, fill: str | LinearGradient) -> bool:
        """Set *oid*'s fill (and a stroke painted like it); whether it changed."""
        before = tx.preview
        element = before.element(oid)
        style = path_style(before, element)
        # An outline painted in the fill colour belongs to the shape.
        outlined = style["stroke"] != "none" and style["stroke"] == style["fill"]
        if isinstance(fill, LinearGradient):
            tx.set_fill(oid, fill)
            if outlined:
                tx.set_attributes(oid, {"stroke": tx.preview.element(oid).get("fill")})
        else:
            if outlined:
                tx.set_attributes(oid, {"stroke": fill})
            tx.set_fill(oid, fill)
        return tx.preview != before


register(ColourFit())
