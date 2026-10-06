"""Export canonical shared fill boundaries and fitted ink into editable SVG."""

from __future__ import annotations

import time

import numpy as np

from vectrify.refine import cel
from vectrify.refine.cel_plan.geometry import Boundaries
from vectrify.refine.cel_plan.layers import continued
from vectrify.refine.cel_plan.model import Evidence, Options, Work
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
):
    started = time.monotonic()
    if evidence.empty.all():
        width, height = evidence.source_size
        work.timings["geometry"] = time.monotonic() - started
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}"/>',
            {"regions": 0, "gradients": 0, "line_paths": 0, "line_pieces": 0},
        )
    scale = float(np.sqrt(np.prod(evidence.scale)))
    tolerance = options.boundary_tolerance * scale
    original_labels = labels
    overlays = ()
    if layers:
        labels, overlays = continued(evidence, labels, options, work)
    hidden = set(np.unique(labels[evidence.empty]).tolist())
    fills = cel.region_medians(evidence.smooth, labels, evidence.line)
    boundary_models = Boundaries() if structure else None
    outlines = cel.region_outlines(labels, tolerance, fit_boundary=boundary_models)
    order = np.argsort(-np.bincount(labels.ravel()))
    parts = []
    line_parts, line_metrics = strokes(
        evidence, options, labels=labels if structure else None, overlays=overlays
    )
    cover, painted = cel.line_layer(line_parts, labels.shape[1], labels.shape[0])
    fitted = cel.fitted_fills(evidence.target, labels, cover, painted, fills)
    ramps = {}
    if options.gradients and not work.interrupted:
        ramps = cel.ramps(evidence.target, labels, cover, painted, fitted)
    definitions = [cel._gradient(f"ramp{i}", ramp) for i, ramp in ramps.items()]
    if definitions:
        parts.append(f"<defs>{''.join(definitions)}</defs>")
    for value in order:
        index = int(value)
        if index not in hidden and index in outlines:
            paint = f"url(#ramp{index})" if index in ramps else colour(fitted[index])
            parts.append(
                f'<path d="{outlines[index]}" fill="{paint}" fill-rule="evenodd"/>'
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
            parts.append(f'<path d="{overlay.data}" fill="{paint}"/>')
        parts.append("</g>")
    parts.extend(line_parts)
    sx, sy = evidence.scale
    x, y = evidence.offset
    width, height = evidence.source_size
    backdrop = ""
    if evidence.background is not None:
        paint = colour(np.array(evidence.background) * 255)
        backdrop = f'<rect width="{width}" height="{height}" fill="{paint}"/>'
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'{backdrop}<g transform="translate({x} {y}) scale({1 / sx} {1 / sy})">'
        f"<g>{''.join(parts)}</g></g></svg>"
    )
    work.timings["geometry"] = time.monotonic() - started
    return svg, {
        "regions": len(outlines) - len(hidden),
        "gradients": len(ramps),
        "boundary_tolerance": options.boundary_tolerance,
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
