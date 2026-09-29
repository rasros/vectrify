"""Improve: Optimize nodes, a search over the selected paths' points.

The general tool for reshaping paths, and the fallback where the GPU path fit
cannot go (several paths, strokes, open contours, no GPU). Each checkbox adds
a kind of move: nudging points and handles, splitting segments, removing
points, moving whole paths, scaling strokes. It is scored on the selection's
surroundings against the reference, or, without one, against the paths as
they were, so on its own Simplify removes points while keeping the look.
Colour is left to Fit colours.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import ClassVar

import numpy as np

from vectrify.document import DocumentError
from vectrify.document.join import path_style
from vectrify.image_utils import preview_urls, resize_long_side
from vectrify.operations.candidates import region_svg
from vectrify.operations.contract import (
    OperationRequest,
    OperationResult,
    Proposal,
    RunContext,
    register,
)
from vectrify.operations.generate import (
    drawing_region,
    render_region,
    target_region,
)
from vectrify.operations.settings import Setting, read_settings

DEFAULT_TASKS = 600
LABEL = "Optimize nodes"
MOVES = ("shape", "detail", "simplify", "strokes", "position")

SETTINGS = {
    "shape": Setting(bool, True),
    "detail": Setting(bool, False),
    "simplify": Setting(bool, False),
    "strokes": Setting(bool, False),
    "position": Setting(bool, False),
    # How much of the fit simplifying may give up, as a percentage of what the
    # selected paths contribute: the difference between the region without
    # them and with them as they started.
    "tolerance": Setting(float, 2.0, minimum=0.0, maximum=50.0, label="tolerance"),
    "workers": Setting(int, 2, minimum=1, maximum=max(1, os.cpu_count() or 1)),
    "resolution": Setting(int, 256, minimum=64, maximum=1024, label="resolution"),
}


def selected_paths(request: OperationRequest) -> list[str]:
    """The selected paths, and the paths inside selected groups."""
    document = request.snapshot.document
    selection = request.snapshot.selection
    if selection.whole_document or not selection.object_ids:
        raise DocumentError("Select the paths to optimize")
    found: list[str] = []
    pending = [document.element(oid) for oid in sorted(selection.object_ids)]
    while pending:
        element = pending.pop(0)
        if element.tag == "path" and element.id not in found:
            found.append(element.id)
        pending.extend(element.children)
    if not found:
        raise DocumentError("Select the paths to optimize")
    for oid in found:
        geometry = document.geometry_for(oid)
        if document.geometry_users(geometry.id) != {oid}:
            raise DocumentError("Detach shared geometry before optimizing its nodes")
    return found


def needed_permissions(settings) -> set[str]:
    kinds = set()
    if settings["shape"] or settings["position"]:
        kinds.add("geometry")
    if settings["detail"] or settings["simplify"]:
        kinds |= {"geometry", "structure"}
    if settings["strokes"]:
        kinds.add("paint")
    return kinds


class OptimizeNodes:
    action: ClassVar[str] = "improve"
    name: ClassVar[str] = "nodes"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        settings = read_settings(request.settings, SETTINGS, "Optimize nodes")
        if not any(settings[move] for move in MOVES):
            raise DocumentError("Choose at least one thing to optimize")
        selected_paths(request)
        if request.reference is None:
            if settings["detail"]:
                raise DocumentError("Add a reference image to add detail")
            if not settings["simplify"]:
                raise DocumentError(
                    "Add a reference image to fit against, or choose Simplify"
                )
        missing = needed_permissions(settings) - request.permissions.allowed
        if missing:
            raise DocumentError(f"Allow {', '.join(sorted(missing))} changes")

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        from vectrify.image_utils import rasterize_svg_to_png_bytes
        from vectrify.score.compare import compare, prepare
        from vectrify.vector.nodes import Paths, frozen
        from vectrify.vector.search import SearchSettings, run_search
        from vectrify.vector.worker import Renderer, WorkerContext

        settings = read_settings(request.settings, SETTINGS, "Optimize nodes")
        tasks = request.budget.steps or DEFAULT_TASKS
        document = request.snapshot.document
        oids = selected_paths(request)
        region = (
            target_region(request)
            if request.reference is not None
            else drawing_region(request, settings["resolution"])
        )
        target = resize_long_side(region.image, settings["resolution"])
        size = target.size
        context.progress(0, "Starting the workers…", total=tasks)
        svg, _ = region_svg(request, region, size)

        start = Paths(
            {oid: document.geometry_for(oid) for oid in oids},
            {
                oid: float(style["stroke-width"])
                for oid in oids
                if (style := path_style(document, document.element(oid)))["stroke"]
                != "none"
            },
        )
        worker = WorkerContext(svg, size, frozen(document, start))
        # Scored at the crop's own size: the stock scorer shrinks everything
        # to 256 px first, which on a small region blurs away the edges a
        # point is being moved onto.
        reference = prepare(target)

        def score(image: bytes | np.ndarray) -> float:
            value = compare(reference, image).blend()
            return value if np.isfinite(value) else 1.0

        # What the selected paths are worth to the fit: the region scored
        # without them, against the region scored as they start.
        without = ET.fromstring(svg)
        for parent in list(without.iter()):
            for child in list(parent):
                if child.get("id") in start.geometries:
                    parent.remove(child)
        empty = score(
            rasterize_svg_to_png_bytes(
                ET.tostring(without, encoding="unicode"), out_w=size[0], out_h=size[1]
            )
        )
        initial = score(Renderer(worker)(start))
        tolerance = settings["tolerance"] / 100 * max(empty - initial, 0.0)

        moves = tuple(
            move
            for move in ("shape", "detail", "position", "strokes")
            if settings[move]
        )
        outcome = run_search(
            start,
            score,
            worker,
            SearchSettings(
                moves=moves,
                simplify=settings["simplify"],
                tolerance=tolerance,
                workers=settings["workers"],
                max_total_tasks=tasks,
                max_wall_seconds=request.budget.seconds,
            ),
            stop=context.stop,
            progress=lambda p: context.progress(
                p.tasks_completed,
                f"Optimizing · {p.tasks_completed:,}/{tasks:,} tries"
                f" · {p.nodes:,} points",
            ),
        )
        best = outcome.best.state
        tx = request.transaction(LABEL)
        for oid, geometry in best.geometries.items():
            if geometry != start.geometries[oid]:
                tx.reshape_path(oid, geometry)
        for oid, width in best.strokes.items():
            if width != start.strokes[oid]:
                tx.set_attributes(oid, {"stroke-width": f"{width:.4g}"})
        changed = best.key() != start.key()
        before = render_region(document, region)
        return OperationResult(
            Proposal(
                tx,
                changed,
                metrics={
                    "before": {
                        "difference": outcome.start.score,
                        "nodes": start.nodes(),
                    },
                    "after": {
                        "difference": outcome.best.score,
                        "nodes": best.nodes(),
                    },
                    "tasks": outcome.tasks_completed,
                    "accepted": outcome.accepted,
                    "reference": request.reference is not None,
                },
                previews=preview_urls(
                    region.image, before, render_region(tx.preview, region)
                ),
            ),
            message=None if changed else "No change improved the paths",
        )


register(OptimizeNodes())
