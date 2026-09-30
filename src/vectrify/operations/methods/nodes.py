"""Improve: Optimize nodes, which fits the selected paths to the reference.

It mixes three steps and picks, round by round, whichever helps:

- Shape fits the points and handles by gradient descent (the path fit),
  on the GPU when there is one and on the CPU otherwise.
- Snap puts the points on the reference's nearest edges; with Add detail it
  also adds points where a piece of the shape is missing or too much.
- Simplify removes the points the outline does not need, within a tolerance
  in the reference's pixels.

Every round tries each chosen step on the paths as they stand and keeps the
one that lowers the difference to the reference most. When none does,
Simplify gets its turn, and once nothing changes the run ends. So a rough
shape can be snapped, fitted, thinned and fitted again, in whatever order
works. With several workers a round's steps run side by side, but only one
path fit runs at a time.

Without a reference only Simplify runs, judged against the drawing itself.
Colour is left to Fit colours.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import ClassVar

from PIL import Image

from vectrify.document import Document, DocumentError, Geometry, Selection
from vectrify.image_utils import preview_urls
from vectrify.operations.contract import (
    OperationRequest,
    OperationResult,
    Proposal,
    RunContext,
    register,
)
from vectrify.operations.generate import (
    Region,
    drawing_region,
    error,
    render_region,
    target_region,
)
from vectrify.operations.settings import Setting, read_settings

LABEL = "Optimize nodes"
STEPS = ("shape", "snap", "simplify")
DEFAULT_ROUNDS = 8
# A step has to lower the difference by this share to count as helping.
GAIN = 0.005

SETTINGS = {
    "shape": Setting(bool, True),
    "snap": Setting(bool, False),
    # Snap may add points where the path misses the shape.
    "detail": Setting(bool, False),
    "simplify": Setting(bool, False),
    # How far Simplify may move an outline, in the reference's pixels.
    "tolerance": Setting(float, 1.0, minimum=0.0, maximum=20.0, label="tolerance"),
    # Each path fit: gradient steps, how far a point may move in SVG units,
    # and the size it works at.
    "steps": Setting(int, 40, minimum=1, maximum=1000, label="steps"),
    "movement": Setting(float, 2.0, minimum=0.0, maximum=100.0, label="movement"),
    "resolution": Setting(int, 768, minimum=64, maximum=2048, label="resolution"),
    "workers": Setting(int, 2, minimum=1, maximum=max(1, os.cpu_count() or 1)),
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
    if settings["shape"] or settings["snap"]:
        kinds.add("geometry")
    if (settings["snap"] and settings["detail"]) or settings["simplify"]:
        kinds |= {"geometry", "structure"}
    return kinds


@dataclass(frozen=True)
class _Task:
    """What a step needs, sent to a worker whole."""

    document: Document
    # The difference is measured against the region's image: the reference,
    # or the drawing as it started when there is none.
    region: Region
    settings: dict
    oids: tuple[str, ...]
    reference: Image.Image | None


def _with(document: Document, geometries: dict[str, Geometry]) -> Document:
    for geometry in geometries.values():
        document = document.replace_geometry(geometry)
    return document


def _paths(document: Document, oids):
    from vectrify.refine.frozen import Paths

    return Paths({oid: document.geometry_for(oid) for oid in oids})


def _difference(document: Document, region: Region) -> float:
    return error(render_region(document, region), region.image)


def _run_step(step: str, task: _Task, stop=None, progress=None):
    """(document after *step*, its difference, why paths were skipped)."""
    document, region, settings = task.document, task.region, task.settings
    skipped: dict[str, str] = {}
    if step == "shape":
        document, skipped = _fit(task, stop, progress)
    else:
        from vectrify.refine.frozen import frozen

        paths = _paths(document, task.oids)
        fixed = frozen(document, paths)
        if step == "snap":
            from vectrify.refine.snap import snap

            paths = snap(document, paths, region, fixed, detail=settings["detail"])
        else:
            from vectrify.refine.simplify import simplify

            paths = simplify(document, paths, region, fixed, settings["tolerance"])
        document = _with(document, dict(paths.geometries))
    return document, _difference(document, region), skipped


def _fit(task: _Task, stop, progress) -> tuple[Document, dict[str, str]]:
    """Fit each path in turn by gradient descent."""
    from vectrify.refine.selected import FitOptions, fit_selected_path

    document, settings = task.document, task.settings
    assert task.reference is not None
    options = FitOptions(
        steps=settings["steps"],
        displacement=settings["movement"],
        resolution=settings["resolution"],
    )
    skipped: dict[str, str] = {}
    for oid in task.oids:
        if stop is not None and stop.is_set():
            break
        try:
            fit = fit_selected_path(
                document,
                Selection(object_ids=frozenset({oid})),
                task.reference,
                options,
                stop=stop,
                progress=progress,
            )
        except DocumentError as exc:
            skipped[oid] = str(exc)
            continue
        if not fit.values:
            continue
        geometry = document.geometry_for(oid)
        document = document.replace_geometry(
            replace(
                geometry,
                subpaths=tuple(
                    replace(
                        s,
                        nodes=tuple(
                            replace(n, values=fit.values[n.id])
                            if n.id in fit.values
                            else n
                            for n in s.nodes
                        ),
                    )
                    for s in geometry.subpaths
                ),
            )
        )
    return document, skipped


class OptimizeNodes:
    action: ClassVar[str] = "improve"
    name: ClassVar[str] = "nodes"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        settings = read_settings(request.settings, SETTINGS, LABEL)
        if not any(settings[step] for step in STEPS):
            raise DocumentError("Choose at least one step")
        selected_paths(request)
        if request.reference is None:
            if settings["shape"]:
                raise DocumentError("Add a reference image to fit the shape to")
            if settings["snap"]:
                raise DocumentError("Add a reference image to snap to")
        if settings["shape"]:
            from vectrify.refine.selected import fit_problem

            problem = fit_problem()
            if problem:
                raise DocumentError(problem)
        missing = needed_permissions(settings) - request.permissions.allowed
        if missing:
            raise DocumentError(f"Allow {', '.join(sorted(missing))} changes")

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        settings = read_settings(request.settings, SETTINGS, LABEL)
        rounds = request.budget.steps or DEFAULT_ROUNDS
        start = request.snapshot.document
        oids = tuple(selected_paths(request))
        region = (
            target_region(request)
            if request.reference is not None
            else drawing_region(request, settings["resolution"])
        )
        steps = [s for s in STEPS if settings[s]]
        document = start
        before = current = _difference(document, region)
        points = _count(document, oids)
        taken: list[str] = []
        skipped: dict[str, str] = {}
        workers = min(settings["workers"], len(steps))
        pool = (
            ProcessPoolExecutor(
                max_workers=workers - 1 if "shape" in steps else workers,
                mp_context=mp.get_context("spawn"),
            )
            if workers > 1
            else None
        )
        try:
            for done in range(rounds):
                if context.stop.is_set():
                    break
                heading = f"Round {done + 1}/{rounds} · {points} points"
                context.progress(done, heading, total=rounds)

                def report(_step, message, done=done, heading=heading):
                    context.progress(done, f"{heading} · {message}")

                task = _Task(document, region, settings, oids, request.reference)
                results = _round(steps, task, pool, context.stop, report)
                for _doc, _diff, why in results.values():
                    skipped.update(why)
                chosen = _choose(results, current, points, oids)
                if chosen is None:
                    break
                taken.append(chosen)
                document, current, _ = results[chosen]
                points = _count(document, oids)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)

        tx = request.transaction(LABEL)
        for oid in oids:
            geometry = document.geometry_for(oid)
            if geometry != start.geometry_for(oid):
                tx.reshape_path(oid, geometry)
        changed = bool(taken)
        message = None
        if not changed:
            message = next(iter(skipped.values()), "No step improved the paths")
        return OperationResult(
            Proposal(
                tx,
                changed,
                metrics={
                    "before": {"difference": before, "nodes": _count(start, oids)},
                    "after": {"difference": current, "nodes": points},
                    "steps": taken,
                    "skipped": skipped,
                    "reference": request.reference is not None,
                },
                previews=preview_urls(
                    region.image,
                    render_region(start, region),
                    render_region(tx.preview, region),
                ),
            ),
            message=message,
        )


def _count(document: Document, oids) -> int:
    return sum(
        len(s.nodes) for oid in oids for s in document.geometry_for(oid).subpaths
    )


def _round(steps, task: _Task, pool, stop, report):
    """Each step's outcome from the same start. The path fit runs here, the
    others on the workers alongside it."""
    pending: dict[str, Future] = {}
    if pool is not None:
        for step in steps:
            if step != "shape":
                pending[step] = pool.submit(_run_step, step, task)
    results = {}
    for step in steps:
        if step not in pending:
            results[step] = _run_step(step, task, stop, report)
    for step, future in pending.items():
        results[step] = future.result()
    return results


def _choose(results, current: float, points: int, oids) -> str | None:
    """The step to keep: the one that lowers the difference most, or else
    Simplify if it removed points."""
    helping = [
        (difference, step)
        for step, (_doc, difference, _why) in results.items()
        if step != "simplify" and difference < current * (1 - GAIN)
    ]
    if helping:
        return min(helping)[1]
    if "simplify" in results:
        simpler, _difference, _why = results["simplify"]
        if _count(simpler, oids) < points:
            return "simplify"
    return None


register(OptimizeNodes())
