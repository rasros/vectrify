"""Opacity evidence and paint proposals for adjacent, nonoverlapping surfaces.

Transparent ink is a filled surface in this representation. It does not add a
second opacity over an already translucent fill. The same canonical graph can
merge surfaces and propose a common-axis RGBA gradient; native validation
decides whether the result preserves the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    find_objects,
    label,
    maximum_filter,
)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from vectrify.document import import_svg
from vectrify.document.paint import GradientStop, LinearGradient, hex_colour
from vectrify.refine import cel
from vectrify.refine.cel_plan.bases import propose as propose_bases
from vectrify.refine.cel_plan.geometry import Boundaries
from vectrify.refine.cel_plan.model import (
    Evidence,
    Options,
    StageInterruptedError,
    Work,
)
from vectrify.refine.cel_plan.ownership import exported as surface_ownership
from vectrify.refine.colour_regions import fit_palette
from vectrify.refine.crossings import crossings

VISIBLE = 0.5 / 255
SEED_RATIO = 1.125


def needed(alpha: np.ndarray) -> bool:
    """Distinguish translucent content from an opaque shape's antialias fringe."""
    support = alpha > VISIBLE
    partial = support & (alpha < 1 - 1 / 255)
    if not partial.any():
        return False
    interior = binary_erosion(support, iterations=2)
    away_from_opaque = maximum_filter(alpha, size=5) < 1 - 1 / 255
    return bool((partial & (interior | away_from_opaque)).any())


def components(values: np.ndarray) -> np.ndarray:
    """Four-connected equal-value labels through a sparse horizontal-run graph.

    The graph has one vertex per horizontal run, and one edge per vertically
    overlapping equal pair, rather than one full-image label pass per RGBA
    palette value. Each transparent component retains its own hidden label.
    """
    starts = np.ones(values.shape, dtype=bool)
    starts[:, 1:] = values[:, 1:] != values[:, :-1]
    runs = np.cumsum(starts.ravel(), dtype=np.int32).reshape(values.shape) - 1
    upper, lower = runs[:-1], runs[1:]
    same = values[:-1] == values[1:]
    overlap = np.ones(same.shape, dtype=bool)
    overlap[:, 1:] = (upper[:, 1:] != upper[:, :-1]) | (lower[:, 1:] != lower[:, :-1])
    overlap &= same
    a, b = upper[overlap], lower[overlap]
    count = int(runs[-1, -1]) + 1
    graph = csr_matrix((np.ones(len(a), dtype=np.uint8), (a, b)), shape=(count, count))
    _, labels = connected_components(graph, directed=False)
    return labels[runs].astype(np.int32)


def labels(
    target: np.ndarray,
    alpha: np.ndarray,
    shown: np.ndarray,
    count: int,
    work: Work,
    *,
    ink: np.ndarray | None = None,
    smooth: np.ndarray | None = None,
) -> np.ndarray:
    """Retain ink, opacity discontinuities and tiny connected color components."""
    values = np.zeros(shown.shape, dtype=np.int32)
    if shown.any():
        colors = np.zeros(shown.shape, dtype=np.int32)
        marked = shown & ink if ink is not None else np.zeros_like(shown)
        for support, limit, offset in (
            (shown & ~marked, count, 0),
            (marked, 8, count),
        ):
            if support.any():
                colors[support] = (
                    offset
                    + fit_palette(
                        (target if offset or smooth is None else smooth)[support][
                            :, None, :
                        ],
                        min(limit, int(support.sum())),
                        16,
                        gpu=False,
                    ).ravel()
                )
        # Relative bands retain low-alpha marks and discontinuities without
        # treating each almost-opaque byte as a different surface. Paint fits
        # use original alpha, and full native validation accepts the partition.
        opacity = np.floor(
            np.log(np.maximum(alpha[shown] * 255, 1)) / np.log(SEED_RATIO)
        ).astype(np.int32)
        values[shown] = 1 + colors[shown] * 256 + opacity
    if work.interrupted:
        # Whole arrays remain a complete partition. The orchestrator will not
        # export them after a deadline or cancellation before its checkpoint.
        return values
    return components(values)


def ink_width(
    target: np.ndarray,
    smooth: np.ndarray,
    drawn: np.ndarray,
    shown: np.ndarray,
    width: float,
    scale: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """An explicit width applied to filled ink, in source-reference pixels."""
    skeleton = cel.thin(drawn)
    if not skeleton.any():
        return target, drawn
    sx, sy = scale
    distance, nearest = cast(
        tuple[np.ndarray, np.ndarray],
        distance_transform_edt(
            ~skeleton, sampling=(1 / sy, 1 / sx), return_indices=True
        ),
    )
    pixel = float(np.sqrt(1 / (sx * sy)))
    coverage = np.clip((width / 2 + pixel / 2 - distance) / pixel, 0, 1)
    ink = (coverage > 0) & shown
    changed = target.copy()
    changed[drawn] = smooth[drawn]
    weight = coverage[ink, None]
    changed[ink] = (
        changed[ink] * (1 - weight) + target[nearest[0][ink], nearest[1][ink]] * weight
    )
    return changed, ink


@dataclass(frozen=True)
class Paint:
    color: str
    opacity: float
    gradient: LinearGradient | None = None


def fit(
    target: np.ndarray,
    alpha: np.ndarray,
    mask: np.ndarray,
    *,
    origin: tuple[int, int] = (0, 0),
    gradients: bool = True,
    gradient_price: float = 0,
) -> Paint:
    """Flat RGBA versus a shared-axis color/opacity ramp, bounded sample count."""
    y, x = np.nonzero(mask)
    step = max(1, (len(x) + 4095) // 4096)
    x, y = x[::step], y[::step]
    rgba = np.column_stack((target[y, x] / 255, alpha[y, x])).astype(np.float64)
    median = np.median(rgba, axis=0)
    flat = Paint(hex_colour(tuple(median[:3])), float(median[3]))
    if not gradients or len(x) < 4:
        return flat
    xy = np.column_stack((x + origin[0] + 0.5, y + origin[1] + 0.5))
    center = xy.mean(axis=0)
    design = np.column_stack((np.ones(len(x)), xy - center))
    weights = np.maximum(rgba[:, 3], 1 / 255)
    coefficients = np.linalg.lstsq(
        design * weights[:, None], rgba * weights[:, None], rcond=None
    )[0]
    _, singular, directions = np.linalg.svd(coefficients[1:].T)
    if singular[0] < 1e-8:
        return flat
    axis = directions[0]
    along = (xy - center) @ axis
    low, high = float(along.min()), float(along.max())
    if high - low < 1e-6:
        return flat
    ramp_design = np.column_stack((np.ones(len(x)), along))
    coefficients = np.linalg.lstsq(
        ramp_design * weights[:, None], rgba * weights[:, None], rcond=None
    )[0]
    ends = (
        np.clip(np.array([1, low]) @ coefficients, 0, 1),
        np.clip(np.array([1, high]) @ coefficients, 0, 1),
    )
    # Color variation can tilt the shared axis. Opacity must not extrapolate
    # beyond its observed support merely to follow that color direction.
    for end in ends:
        end[3] = np.clip(end[3], rgba[:, 3].min(), rgba[:, 3].max())
    if float(np.max(np.abs(ends[0] - ends[1]))) < 2 / 255:
        return flat
    u = ((along - low) / (high - low))[:, None]
    fitted = ends[0] + u * (ends[1] - ends[0])

    def error(values):
        return (
            np.square(values[:, :3] * values[:, 3:] - rgba[:, :3] * rgba[:, 3:]).mean()
            + np.square(values[:, 3] - rgba[:, 3]).mean()
        )

    before, after = error(np.repeat(median[None], len(x), axis=0)), error(fitted)
    if after >= before * 0.9 or before - after < max(1 / 255**2, gradient_price):
        return flat
    start, end = center + axis * low, center + axis * high
    gradient = LinearGradient(
        tuple(start),
        tuple(end),
        tuple(
            GradientStop(offset, hex_colour(tuple(value[:3])), float(value[3]))
            for offset, value in zip((0, 1), ends, strict=True)
        ),
    )
    return Paint(flat.color, 1.0, gradient)


def export(
    evidence: Evidence,
    labels: np.ndarray,
    options: Options,
    work: Work,
    *,
    structure: bool,
    conservative: bool,
    tolerance: float,
    cost_normalizer: float | None = None,
    layers: bool = False,
):
    """Export RGBA surfaces, optionally inside a validated opacity-core group."""
    assert evidence.opacity is not None

    def check():
        if work.interrupted:
            raise StageInterruptedError("Opacity surface fitting interrupted")

    models = Boundaries() if structure and not conservative else None
    hidden = {int(i) for i in np.unique(labels[evidence.empty])}
    boxes = find_objects(labels + 1)
    # Color/opacity partition strips are not mandatory thin features. Only a
    # visible alpha component without an eroded core gets this topology hold;
    # narrow shading within a broad component remains a scored interpretation.
    visible_components, component_count = label(~evidence.empty)
    core = binary_erosion(~evidence.empty, iterations=2)
    has_core = np.bincount(visible_components[core], minlength=component_count + 1) > 0
    thin_components = ~has_core
    thin_components[0] = False
    thin = {
        int(index) for index in np.unique(labels[thin_components[visible_components]])
    }
    bases = (
        propose_bases(evidence, labels, visible_components, work)
        if layers and not conservative
        else ()
    )
    base_for_region = {index: base for base in bases for index in base.members}
    alpha_levels = np.zeros(len(boxes))
    for index, box in enumerate(boxes):
        check()
        if index not in hidden and box is not None:
            alpha_levels[index] = np.median(evidence.opacity[box][labels[box] == index])
    fixed_ink_regions = (
        {int(index) for index in np.unique(labels[evidence.drawn])}
        if options.line_width > 0
        else set()
    )
    topology_regions: set[int] = set()
    rejected_regions: set[int] = set()
    fitted_regions: set[int] = set()

    def sides(points):
        middle = (points[0] + points[1]) / 2
        delta = points[1] - points[0]
        normal = np.array([-delta[1], delta[0]]) * 0.25
        result = []
        for point in (middle + normal, middle - normal):
            x, y = np.floor(point).astype(int)
            result.append(
                int(labels[y, x])
                if 0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]
                else -1
            )
        return result

    def boundary(points, bound):
        check()
        regions = sides(points)
        native = any(
            index == -1
            or index in hidden
            or index in thin
            or index in fixed_ink_regions
            for index in regions
        )
        if not native:
            low, high = sorted(alpha_levels[index] for index in regions)
            native = high > low * 1.25 + 1e-7
        repaired = any(index in rejected_regions for index in regions)
        if conservative or native or repaired:
            # A compact inner boundary cannot change native transparent space
            # or consume a thin component. Shared chains still have one owner.
            polygon_bound = tolerance * min(evidence.scale) if conservative else 0
            if native or repaired:
                topology_regions.update(
                    index for index in regions if index not in hidden and index >= 0
                )
            points = cel.simplify(points, polygon_bound)
            return [("L", tuple(float(v) for v in point)) for point in points[1:]]
        fitted_regions.update(
            index for index in regions if index >= 0 and index not in hidden
        )
        if models is not None:
            return models(points, bound)
        return cel.curve_nodes(points, bound, smooth=cel.FILL_SMOOTH, fit=cel.FILL_FIT)

    check()
    outlines = cel.region_outlines(
        labels,
        options.boundary_tolerance * float(np.sqrt(np.prod(evidence.scale))),
        fit_boundary=boundary,
        check=check,
    )
    if not conservative:
        # One bad interior fit must not discard all other compact surfaces.
        # Restore every canonical edge of the offending region, including the
        # neighbor's copy, then let the whole-candidate checks decide retention.
        for index, data in outlines.items():
            check()
            if index not in fitted_regions:
                continue
            drawing = import_svg(f'<svg><path id="check" d="{data}"/></svg>')
            if crossings(drawing.geometry_for("check")):
                rejected_regions.add(index)
        if rejected_regions:
            models = Boundaries() if structure else None
            outlines = cel.region_outlines(
                labels,
                options.boundary_tolerance * float(np.sqrt(np.prod(evidence.scale))),
                fit_boundary=boundary,
                check=check,
            )
    definitions, parts, constraints, paint_constraints = [], [], [], []
    grouped: dict[int, list[str]] = {base.component: [] for base in bases}
    area = max(1, int((~evidence.empty).sum()))
    for index, data in sorted(outlines.items()):
        check()
        box = boxes[index]
        if index in hidden or box is None:
            continue
        own = labels[box] == index
        paint = fit(
            evidence.target[box],
            evidence.opacity[box],
            own,
            origin=(box[1].start, box[0].start),
            gradients=options.gradients and not conservative,
            # A local evidence estimate prioritizes a paint model. Retention
            # still requires the same complete native objective and hard checks.
            gradient_price=(
                options.detail_cost
                * 12
                / cost_normalizer
                * area
                / max(1, int(own.sum()))
                if cost_normalizer is not None
                else 0
            ),
        )
        fill, opacity = paint.color, f' fill-opacity="{paint.opacity:.9g}"'
        base = base_for_region.get(index)
        parent_opacity = base.opacity if base is not None else 1.0
        opacity = f' fill-opacity="{min(1.0, paint.opacity / parent_opacity):.9g}"'
        if paint.gradient is not None:
            import xml.etree.ElementTree as ET

            gradient_id = f"cel-alpha-ramp-{index}"
            node = ET.Element(
                "linearGradient",
                {"id": gradient_id, **dict(paint.gradient.attributes())},
            )
            for stop in paint.gradient.stops:
                normalized = GradientStop(
                    stop.offset, stop.colour, min(1.0, stop.opacity / parent_opacity)
                )
                ET.SubElement(
                    node, "stop", dict(normalized.element("unused").attributes)
                )
            definitions.append(ET.tostring(node, encoding="unicode"))
            fill, opacity = f"url(#{gradient_id})", ""
        part = (
            f'<path id="cel-fill-{index}" d="{data}" fill="{fill}"{opacity} '
            'fill-rule="evenodd"/>'
        )
        (grouped[base.component] if base is not None else parts).append(part)
        fixed_ink = options.line_width > 0 and evidence.drawn[box][own].any()
        if structure or fixed_ink or index in topology_regions:
            # Whole-path holds preserve accepted compact geometry and explicit
            # filled-ink widths until parameterized fitting can refine them.
            constraints.append(f"cel-fill-{index}")
        if fixed_ink:
            paint_constraints.append(f"cel-fill-{index}")
    for base in bases:
        check()
        x, y = base.origin
        parts.append(
            f'<g id="cel-component-{base.component}" opacity="{base.opacity:.9g}">'
            f'<path id="cel-base-{base.component}" d="{base.data}" '
            f'transform="translate({x} {y})" fill="{base.color}" fill-rule="evenodd"/>'
            f"{''.join(grouped[base.component])}</g>"
        )
        constraints.append(f"cel-base-{base.component}")
    check()
    sx, sy = evidence.scale
    x, y = evidence.offset
    width, height = evidence.source_size
    defs = f"<defs>{''.join(definitions)}</defs>" if definitions else ""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'{defs}<g transform="translate({x} {y}) scale({1 / sx} {1 / sy})">'
        f"{''.join(parts)}</g></svg>"
    )
    return svg, {
        "planning_surfaces": surface_ownership(
            evidence,
            labels,
            frozenset(index for index in outlines if index not in hidden),
            work,
            bases=bases,
        ),
        "regions": len(outlines) - len(hidden),
        "gradients": len(definitions),
        "alpha_model": "opacity-core-layers" if bases else "adjacent-rgba-surfaces",
        "base_models": [
            {
                "model": "uniform-alpha-core",
                "component": base.component,
                "opacity": base.opacity,
                "core_fraction": base.core_fraction,
                "members": list(base.members),
            }
            for base in bases
        ],
        "alpha_seed_levels": int(np.ceil(np.log(255) / np.log(SEED_RATIO))) + 1,
        "alpha_seed_ratio": SEED_RATIO,
        "paint_cost_normalizer": cost_normalizer,
        "boundary_tolerance": options.boundary_tolerance,
        "conservative_geometry": conservative,
        "conservative_tolerance": tolerance,
        "geometry_constraints": constraints,
        "paint_constraints": paint_constraints,
        "native_alpha_regions": len(topology_regions),
        "repaired_crossing_regions": sorted(rejected_regions),
        "geometry_models": models.decisions if models else [],
        "overlay_models": [],
        "line_paths": 0,
        "line_style": "filled",
        "outline": False,
        "outline_width": options.line_width,
    }
