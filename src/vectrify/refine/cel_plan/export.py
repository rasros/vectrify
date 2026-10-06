"""Export canonical shared fill boundaries and fitted ink into editable SVG."""

from __future__ import annotations

import time

import numpy as np

from vectrify.refine import cel
from vectrify.refine.cel_plan.geometry import Boundaries
from vectrify.refine.cel_plan.layers import continued
from vectrify.refine.cel_plan.model import (
    Evidence,
    Options,
    StageInterruptedError,
    Work,
)
from vectrify.refine.cel_plan.ownership import exported as surface_ownership
from vectrify.refine.cel_plan.strokes import strokes
from vectrify.refine.colour_regions import colour


def export(
    evidence: Evidence,
    labels: np.ndarray,
    options: Options,
    work: Work,
    *,
    structure: bool = False,
    layers: bool = False,
    conservative: bool = False,
    conservative_tolerance: float = 0,
    cost_normalizer: float | None = None,
):
    started = time.monotonic()

    def check():
        if work.interrupted:
            raise StageInterruptedError("Planning stage interrupted")

    check()
    if evidence.empty.all():
        width, height = evidence.source_size
        work.timings["geometry"] = time.monotonic() - started
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}"/>',
            {"regions": 0, "gradients": 0, "line_paths": 0, "line_pieces": 0},
        )
    if evidence.opacity is not None:
        from vectrify.refine.cel_plan.opacity import export as rgba_export

        result = rgba_export(
            evidence,
            labels,
            options,
            work,
            structure=structure,
            conservative=conservative,
            tolerance=conservative_tolerance,
            cost_normalizer=cost_normalizer,
            layers=layers,
        )
        work.timings["geometry"] = time.monotonic() - started
        return result
    scale = float(np.sqrt(np.prod(evidence.scale)))
    tolerance = options.boundary_tolerance * scale
    original_labels = labels
    input_labels = labels
    overlays = ()
    if layers:
        labels, overlays = continued(evidence, labels, options, work)
    hidden = set(np.unique(labels[evidence.empty]).tolist())
    fills = cel.region_medians(evidence.smooth, labels, evidence.line)
    boundary_models = Boundaries() if structure else None
    constrained_regions: set[int] = set()

    def pixels(
        points: np.ndarray, _tolerance: float
    ) -> list[tuple[str, tuple[float, ...]]]:
        # Linear fits retain canonical chain ends and have no curve overshoot.
        # A positive bound is in native pixels; the raw fallback uses zero.
        check()
        bound = conservative_tolerance * min(evidence.scale)
        return [("L", (float(x), float(y))) for x, y in cel.simplify(points, bound)[1:]]

    def fitted_boundary(
        points: np.ndarray, tolerance: float
    ) -> list[tuple[str, tuple[float, ...]]]:
        check()
        if boundary_models is not None:
            result = boundary_models(points, tolerance)
            if boundary_models.decisions[-1]["model"] != "curve":
                middle = (points[0] + points[1]) / 2
                direction = points[1] - points[0]
                normal = np.array([-direction[1], direction[0]]) * 0.25
                for side in (middle + normal, middle - normal):
                    x, y = np.floor(side).astype(int)
                    if 0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]:
                        constrained_regions.add(int(labels[y, x]))
            return result
        return cel.curve_nodes(
            points, tolerance, smooth=cel.FILL_SMOOTH, fit=cel.FILL_FIT
        )

    outlines = cel.region_outlines(
        labels,
        tolerance,
        fit_boundary=pixels if conservative else fitted_boundary,
        check=check,
    )
    order = np.argsort(-np.bincount(labels.ravel()))
    parts = []
    line_parts, line_metrics = strokes(
        evidence,
        options,
        labels=labels if structure else None,
        overlays=overlays,
        conservative=conservative,
        conservative_tolerance=conservative_tolerance,
    )
    check()
    cover, painted = cel.line_layer(line_parts, labels.shape[1], labels.shape[0])
    fitted = cel.fitted_fills(evidence.target, labels, cover, painted, fills)
    ramps = {}
    if options.gradients and not work.interrupted:
        ramps = cel.ramps(evidence.target, labels, cover, painted, fitted)
    check()
    definitions = [cel._gradient(f"ramp{i}", ramp) for i, ramp in ramps.items()]
    if definitions:
        parts.append(f"<defs>{''.join(definitions)}</defs>")
    for value in order:
        index = int(value)
        if index not in hidden and index in outlines:
            paint = f"url(#ramp{index})" if index in ramps else colour(fitted[index])
            parts.append(
                f'<path id="cel-fill-{index}" d="{outlines[index]}" '
                f'fill="{paint}" fill-rule="evenodd"/>'
            )
    if overlays:
        original_labels = original_labels.copy()
        for overlay in overlays:
            original_labels[np.isin(original_labels, overlay.members)] = overlay.region
        overlay_fills = cel.region_medians(
            evidence.smooth, original_labels, evidence.line
        )
        overlay_fills = cel.fitted_fills(
            evidence.target, original_labels, cover, painted, overlay_fills
        )
        # Overlays are fitted as surfaces after their geometry is chosen. A
        # flat paint remains a competitor rather than inheriting noisy ramps.
        parts.append("<g>")
        for overlay in overlays:
            paint = colour(overlay_fills[overlay.region])
            parts.append(
                f'<path id="cel-overlay-{overlay.region}" d="{overlay.data}" '
                f'fill="{paint}"/>'
            )
        parts.append("</g>")
    parts.extend(
        part.replace("<path ", f'<path id="cel-ink-{index}" ', 1)
        for index, part in enumerate(line_parts)
    )
    sx, sy = evidence.scale
    x, y = evidence.offset
    width, height = evidence.source_size
    backdrop = ""
    if evidence.background is not None:
        paint = colour(np.array(evidence.background) * 255)
        backdrop = (
            f'<rect id="cel-backdrop" width="{width}" height="{height}" '
            f'fill="{paint}"/>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'{backdrop}<g transform="translate({x} {y}) scale({1 / sx} {1 / sy})">'
        f"<g>{''.join(parts)}</g></g></svg>"
    )
    work.timings["geometry"] = time.monotonic() - started
    return svg, {
        "planning_surfaces": surface_ownership(
            evidence,
            input_labels,
            frozenset(index for index in outlines if index not in hidden),
            work,
            overlays=overlays,
        ),
        "regions": len(outlines) - len(hidden),
        "gradients": len(ramps),
        "boundary_tolerance": options.boundary_tolerance,
        "conservative_geometry": conservative,
        "conservative_tolerance": conservative_tolerance,
        "geometry_constraints": [
            *(f"cel-fill-{index}" for index in sorted(constrained_regions - hidden)),
            *(f"cel-overlay-{overlay.region}" for overlay in overlays),
            *(
                f"cel-ink-{index}"
                for index in line_metrics.get("constrained_lines", ())
            ),
        ],
        "geometry_models": boundary_models.decisions if boundary_models else [],
        "overlay_models": [
            {
                "operator": "base-overlay",
                "region": overlay.region,
                "members": list(overlay.members),
                "model": overlay.model.kind,
                "residual": overlay.model.residual / scale
                if overlay.model.residual is not None
                else None,
            }
            for overlay in overlays
        ],
        **line_metrics,
    }
