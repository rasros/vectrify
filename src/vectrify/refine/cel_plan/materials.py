"""Bounded coherent flat/linear material proposals before SVG initialization.

Sufficient statistics fit a common-axis premultiplied RGBA model over complete
source atoms. This only proposes a partition: export refits actual SVG paint,
then the ordinary native policy and frontier decide whether to retain it.
"""

from __future__ import annotations

import heapq
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, find_objects, label

from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.local import Box
from vectrify.refine.cel_plan.model import (
    Evidence,
    Graph,
    Options,
    StageInterruptedError,
    Work,
)
from vectrify.refine.cel_plan.opacity import fit
from vectrify.refine.cel_plan.score import render
from vectrify.refine.colour_regions import simplified_indices

MAX_REGIONS = 16_384
MAX_EDGES = 65_536
MAX_MODELS = 32_768
CHUNK_PIXELS = 65_536
MAX_THIN_PAINT_PIXELS = 8192


@dataclass(frozen=True)
class Loss:
    error: float
    gradient: bool
    alpha_residual: float


def model(gram: np.ndarray, *, gradients: bool, gradient_price: float) -> Loss:
    """Least-squares flat or rank-one spatial RGBA model, without pixel copies."""
    size = float(gram[0, 0])
    if size <= 0:
        return Loss(0, False, 0)
    centered = gram[1:, 1:] - np.outer(gram[0, 1:], gram[0, 1:]) / size
    residual = max(0.0, float(np.trace(centered[2:, 2:])))
    alpha_variance = max(0.0, float(centered[-1, -1]))
    flat = Loss(residual, False, float(alpha_variance > 1e-10))
    if not gradients:
        return flat
    values, vectors = np.linalg.eigh(centered[:2, :2])
    keep = values > max(1e-10, float(values.max()) * 1e-8)
    if not keep.any():
        return flat
    whiten = vectors[:, keep] / np.sqrt(values[keep])
    projected = whiten.T @ centered[:2, 2:]
    energy, directions = np.linalg.eigh(projected @ projected.T)
    saved = max(0.0, min(residual, float(energy[-1])))
    if saved <= gradient_price + 1e-10:
        return flat
    axis = whiten @ directions[:, -1]
    explained_alpha = float((axis @ centered[:2, -1]) ** 2)
    alpha_residual = max(0.0, alpha_variance - explained_alpha) / max(
        alpha_variance, 1e-10
    )
    return Loss(residual - saved + gradient_price, True, alpha_residual)


def moments(
    evidence: Evidence, graph: Graph, options: Options, work: Work
) -> np.ndarray:
    """Stream weighted [1,x,y,premultiplied RGBA] outer products per source atom."""
    result = np.zeros((len(graph.regions), 7, 7), dtype=np.float64)
    feature_weight = np.array(
        [
            1 + 8 * options.protection * r.feature * (1 - r.texture)
            for r in graph.regions
        ]
    )
    height, width = graph.labels.shape
    scale = max(height, width, 1)
    for chunk in Box(0, 0, width, height).chunks(CHUNK_PIXELS):
        if work.interrupted:
            raise StageInterruptedError("Material statistics interrupted")
        box = chunk.slices
        shown = ~evidence.empty[box]
        y, x = np.nonzero(shown)
        ids = graph.labels[box][shown]
        alpha = (
            evidence.opacity[box][shown]
            if evidence.opacity is not None
            else np.ones(len(ids))
        )
        colors = evidence.target[box][shown] / 255 * alpha[:, None]
        basis = np.column_stack(
            (
                np.ones(len(ids)),
                (x + chunk.x + 0.5) / scale,
                (y + chunk.y + 0.5) / scale,
                colors,
                alpha,
            )
        )
        weights = feature_weight[ids]
        for a in range(7):
            for b in range(a, 7):
                if work.interrupted:
                    raise StageInterruptedError("Material statistics interrupted")
                values = np.bincount(
                    ids,
                    weights=weights * basis[:, a] * basis[:, b],
                    minlength=len(graph.regions),
                )
                result[:, a, b] += values
                if a != b:
                    result[:, b, a] += values
    return result


def retain_thin_paint(
    evidence: Evidence,
    labels: np.ndarray,
    options: Options,
    work: Work,
    *,
    normalizer: float,
) -> tuple[np.ndarray, dict]:
    """Screen actual export paint; restore original atoms for unsupported marks.

    A linear growth estimate is not necessarily the exported paint. Thin
    components have no opacity base, and a cheaper median flat can destroy a
    faint alpha ramp. Render its actual paint in a bounded rectangle before export.
    This is conservative proposal repair, not a geometric or native proof.
    """
    details = {
        "checked": 0,
        "restored_families": 0,
        "bounded_families": 0,
        "paint_evaluation": "native-bounded-rectangle",
    }
    if evidence.opacity is None:
        return labels, details
    visible = ~evidence.empty
    components, count = label(visible)
    core = binary_erosion(visible, iterations=2)
    has_core = np.bincount(components[core], minlength=count + 1) > 0
    thin = ~has_core
    thin[0] = False
    ids = {int(i) for i in np.unique(labels[thin[components]])}
    if not ids:
        return labels, details
    area = max(1, int(visible.sum()))
    restored = []
    for index, box in enumerate(find_objects(labels + 1)):
        if work.interrupted:
            raise StageInterruptedError("Thin material paint interrupted")
        if index not in ids or box is None:
            continue
        own = labels[box] == index
        size = int(own.sum())
        if own.size > MAX_THIN_PAINT_PIXELS:
            details["bounded_families"] += 1
            restored.append(index)
            continue
        details["checked"] += 1
        paint = fit(
            evidence.target[box],
            evidence.opacity[box],
            own,
            origin=(box[1].start, box[0].start),
            gradients=options.gradients,
            gradient_price=options.detail_cost * 12 / normalizer * area / max(1, size),
        )
        original = evidence.opacity[box][own]
        y, x = box[0].start, box[1].start
        height, width = own.shape
        definitions = ""
        if paint.gradient is None:
            paint_attributes = (
                f'fill="{paint.color}" fill-opacity="{paint.opacity:.9g}"'
            )
        else:
            gradient = paint.gradient
            node = ET.Element(
                "linearGradient", {"id": "paint", **dict(gradient.attributes())}
            )
            for stop in gradient.stops:
                ET.SubElement(node, "stop", dict(stop.element("unused").attributes))
            definitions = f"<defs>{ET.tostring(node, encoding='unicode')}</defs>"
            paint_attributes = 'fill="url(#paint)"'
        svg = (
            f'<svg width="{width}" height="{height}" '
            f'viewBox="{x} {y} {width} {height}">'
            f'{definitions}<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f"{paint_attributes}/></svg>"
        )
        predicted = render(svg, (width, height))[..., 3][own]
        original_mass = float(original.sum())
        if (
            float(np.minimum(predicted, original).sum()) < original_mass * 0.95
            or float(predicted.sum()) > original_mass * 1.25 + size / 255
        ):
            restored.append(index)
    if not restored:
        return labels, details
    # Keep complete source atoms, including when growth started from a coarser
    # validated partition. Distinct namespaces prevent label identity clashes.
    repair = np.isin(labels, restored)
    keys = labels + int(evidence.labels.max()) + 1
    keys[repair] = evidence.labels[repair]
    _, mapping = np.unique(keys, return_inverse=True)
    repaired = mapping.reshape(labels.shape).astype(np.int32)
    details["restored_families"] = len(restored)
    return repaired, details


def coherent_labels(
    evidence: Evidence, graph: Graph, options: Options, work: Work, *, normalizer: float
) -> tuple[np.ndarray, dict]:
    """Agglomerate connected source atoms using competing material models.

    Supported/unresolved ridges and explicit ink widths remain barriers. A
    broad alpha range requires a linear model explaining at least 98% of alpha
    variance; a step or faint mark cannot use the gradient interpretation just
    because its absolute pixel loss is small. Full native validation follows.
    """
    began = time.monotonic()
    count = len(graph.regions)
    details: dict = {
        "version": 1,
        "growth_model": "flat-or-linear-lower-bound",
        "paint_cost_acceptance": "native-frontier",
        "status": "complete",
        "source_regions": count,
        "merges": 0,
        "model_evaluations": 0,
        "alpha_exclusions": 0,
        "model_limit": MAX_MODELS,
        "model_limit_hit": False,
    }
    if count > MAX_REGIONS or len(graph.boundaries) > MAX_EDGES:
        return graph.labels, {**details, "status": "graph-limit"}
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("Material proposals require a fixed positive normalizer")
    gram = moments(evidence, graph, options, work)
    regions = graph.regions
    area = max(1, sum(r.area for r in regions))
    price = options.detail_cost / normalizer
    # A small atom cannot amortize a whole SVG gradient yet. Charging that
    # activation cost at every pair creates a local minimum even for a perfect
    # broad ramp. Grow with a common-axis fit lower bound, then export charges
    # actual gradients against flats and the full native frontier pays their
    # representation cost. This estimate never accepts the drawing.
    gradient_price = 0.0
    losses = []
    for g in gram:
        if work.interrupted:
            raise StageInterruptedError("Material seed fitting interrupted")
        losses.append(
            model(g, gradients=options.gradients, gradient_price=gradient_price)
        )
    parent = np.arange(count)
    versions = np.zeros(count, dtype=np.int32)
    alpha_range = np.array([r.opacity_range or (r.opacity, r.opacity) for r in regions])
    ink_pixels = np.bincount(
        graph.labels.ravel(), weights=evidence.drawn.ravel(), minlength=count
    )
    ink = ink_pixels >= np.array([max(1, r.area) for r in regions]) * 0.6
    meter = Families(evidence, graph, options)
    # Edges also retain blocked contacts. A weak alternate route must not
    # merge around a supported ridge and then silently erase that ridge.
    edges: list[dict[int, tuple[float, bool]]] = [{} for _ in regions]
    for edge in graph.boundaries:
        if work.interrupted:
            raise StageInterruptedError("Material adjacency interrupted")
        a, b = edge.left, edge.right
        if (
            min(a, b) < 0
            or a in graph.hidden
            or b in graph.hidden
            or regions[a].component != regions[b].component
        ):
            continue
        blocked = regions[a].fixed or regions[b].fixed or ink[a] != ink[b]
        if not blocked and edge.line_support > 0.5:
            blocked = meter._protected(edge, work)
        saved, was_blocked = edges[a].get(b, (0, False))
        value = (
            saved + 2 + len(simplified_indices(edge.points, 2)),
            blocked or was_blocked,
        )
        edges[a][b] = edges[b][a] = value

    def proposal(a, b):
        if edges[a][b][1] or details["model_evaluations"] >= MAX_MODELS:
            return None
        details["model_evaluations"] += 1
        proposed = model(
            gram[a] + gram[b],
            gradients=options.gradients,
            gradient_price=gradient_price,
        )
        low = min(alpha_range[a, 0], alpha_range[b, 0])
        high = max(alpha_range[a, 1], alpha_range[b, 1])
        if high > low * 1.25 + 1e-7 and (
            not proposed.gradient or proposed.alpha_residual > 0.02
        ):
            details["alpha_exclusions"] += 1
            return None
        change = (proposed.error - losses[a].error - losses[b].error) / area
        change -= price * edges[a][b][0]
        return change, proposed

    heap = []
    for a in range(count):
        if work.interrupted:
            raise StageInterruptedError("Material model initialization interrupted")
        for b in edges[a]:
            if a < b:
                candidate = proposal(a, b)
                if candidate is not None:
                    heap.append((candidate[0], a, b, 0, 0, candidate[1]))
    heapq.heapify(heap)
    while heap and not work.interrupted and details["model_evaluations"] < MAX_MODELS:
        change, a, b, va, vb, fitted = heapq.heappop(heap)
        if versions[a] != va or versions[b] != vb or b not in edges[a]:
            continue
        if change >= 0:
            break
        if len(edges[a]) < len(edges[b]):
            a, b = b, a
        parent[b] = a
        gram[a] += gram[b]
        losses[a] = fitted
        alpha_range[a] = (
            min(alpha_range[a, 0], alpha_range[b, 0]),
            max(alpha_range[a, 1], alpha_range[b, 1]),
        )
        del edges[a][b]
        for c, (saving, blocked) in edges[b].items():
            if c == a:
                continue
            del edges[c][b]
            old_saving, old_blocked = edges[a].get(c, (0, False))
            value = saving + old_saving, blocked or old_blocked
            edges[a][c] = edges[c][a] = value
        edges[b] = {}
        versions[a] += 1
        versions[b] += 1
        details["merges"] += 1
        for c in edges[a]:
            candidate = proposal(a, c)
            if candidate is not None:
                heapq.heappush(
                    heap, (candidate[0], a, c, versions[a], versions[c], candidate[1])
                )
    for index in range(count):
        root = index
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = int(parent[root])
        parent[index] = root
    _, mapping = np.unique(parent, return_inverse=True)
    labels = mapping[graph.labels].astype(np.int32)
    growth_regions = len(np.unique(labels[~evidence.empty]))
    if not work.interrupted:
        labels, paint_repair = retain_thin_paint(
            evidence, labels, options, work, normalizer=normalizer
        )
    else:
        paint_repair = {"status": "interrupted"}
    details.update(
        {
            "status": "interrupted" if work.interrupted else "complete",
            "model_limit_hit": details["model_evaluations"] >= MAX_MODELS,
            "result_regions": len(np.unique(labels[~evidence.empty])),
            "growth_result_regions": growth_regions,
            "thin_paint": paint_repair,
            "linear_estimate_families": sum(
                losses[i].gradient for i in np.unique(parent)
            ),
            "ridge_diagnostics": meter.diagnostics,
            "seconds": time.monotonic() - began,
        }
    )
    return labels, details
