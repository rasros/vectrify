"""Export canonical shared fill boundaries and fitted ink into editable SVG."""

from __future__ import annotations

import time

import numpy as np

from vectrify.refine import cel
from vectrify.refine.cel_plan.model import Evidence, Options, Work
from vectrify.refine.cel_plan.strokes import strokes
from vectrify.refine.colour_regions import colour


def export(evidence: Evidence, labels: np.ndarray, options: Options, work: Work):
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
    hidden = set(np.unique(labels[evidence.empty]).tolist())
    fills = cel.region_medians(evidence.smooth, labels, evidence.line)
    outlines = cel.region_outlines(labels, tolerance)
    order = np.argsort(-np.bincount(labels.ravel()))
    parts = []
    line_parts, line_metrics = strokes(evidence, options)
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
        **line_metrics,
    }
