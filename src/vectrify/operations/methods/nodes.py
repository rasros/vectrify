"""Improve: Optimize nodes, a quick tidy of the selected paths' points.

It mixes three steps and picks, round by round, whichever helps:

- Snap puts the points on the reference's nearest edges; with Add detail it
  also adds points where a piece of the shape is missing or too much.
- Simplify removes the points the outline does not need, within a tolerance
  in the reference's pixels.
- Shape fits the points and handles by gradient descent (the path fit),
  on the GPU when there is one and on the CPU otherwise. It is off unless
  asked for: Redraw outline reshapes a path far faster.

Every round tries each chosen step on the paths as they stand and keeps the
one that lowers the difference to the reference most, if it fixes enough of
the difference where it acted: over the pixels it changed and a thin band
around them, so a small fix on a large selection counts as much as on a
small one. When none does, Simplify gets its turn. No step is kept that
leaves the difference where the run has acted (against the paths as they
started) worse than a small allowance, so a Tidy never trades the match
for fewer points beyond it; once nothing qualifies the run ends. A step
that leaves an outline crossing itself more than before, a twist or a
curve looped over itself, is never kept, however close it gets; concave
outlines are fine. With several workers a round's steps run
side by side, but only one path fit runs at a time.

Every run has a time limit. Each round checks it and gives each step a share
of what is left, keeping one back to judge them; a slow step stops at its
share, handing back how far it got. A run out of time keeps the best it
found, as Stop does.

Without a reference only Simplify runs, judged against the drawing itself.
Colour is left to Fit colours.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
from PIL import Image
from scipy import ndimage

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
    render_region,
    target_region,
)
from vectrify.operations.settings import Setting, read_settings

LABEL = "Optimize nodes"
STEPS = ("shape", "snap", "simplify")
DEFAULT_ROUNDS = 4
# How far, in reference pixels, the pixels a step changed are widened to
# judge it by the difference where it acted.
BAND = 2
# How far, in reference pixels, a fitted curve's handles may sit from its
# line and still be drawn as the line.
STRAIGHT = 0.25

SETTINGS = {
    "shape": Setting(bool, False),
    "snap": Setting(bool, True),
    # Snap may add points where the path misses the shape, each of which has
    # to fix this many reference pixels.
    "detail": Setting(bool, False),
    # How far past the selection, as a share of its size in percent, the
    # reference is read: how far a snapped or added point can reach.
    "margin": Setting(float, 10.0, minimum=0.0, maximum=200.0, label="margin"),
    "detail_gain": Setting(
        float, 12.0, minimum=1.0, maximum=500.0, label="detail gain"
    ),
    "simplify": Setting(bool, True),
    # How much Simplify may raise the difference where it acts, in percent:
    # it removes points only while the match stays within this.
    "budget": Setting(float, 1.0, minimum=0.0, maximum=100.0, label="error budget"),
    # The most Simplify may move an outline at any point, in the
    # reference's pixels: a cap, as the error budget decides how far it goes;
    # without a reference it decides alone, and is NO_REFERENCE unless set.
    "tolerance": Setting(float, 3.0, minimum=0.0, maximum=20.0, label="tolerance"),
    # Each path fit: gradient steps, how far a point may move in SVG units,
    # and the size it works at.
    "steps": Setting(int, 40, minimum=1, maximum=1000, label="steps"),
    "movement": Setting(float, 2.0, minimum=0.0, maximum=100.0, label="movement"),
    "resolution": Setting(int, 768, minimum=64, maximum=2048, label="resolution"),
    "workers": Setting(int, 1, minimum=1, maximum=max(1, os.cpu_count() or 1)),
    # How much of the difference where a step acted it has to fix to be
    # kept, in percent.
    "gain": Setting(float, 1.0, minimum=0.0, maximum=50.0, label="minimum improvement"),
    # How much worse than at the start, in percent, the difference where the
    # run has acted may get for a step to be kept: a step that saves points
    # but costs more than this is not.
    "allowance": Setting(float, 1.0, minimum=0.0, maximum=100.0, label="allowance"),
    # The most the whole run may take, in seconds.
    "seconds": Setting(float, 10.0, minimum=0.5, maximum=3600.0, label="time limit"),
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
    # When the run's time is up (time.monotonic()), and how long of it each
    # step may take from when it starts.
    deadline: float = float("inf")
    share: float = float("inf")


class _Until(threading.Event):
    """Set once *stop* is, or once *deadline* passes: a step's own stop."""

    def __init__(self, deadline: float, stop: threading.Event | None = None):
        super().__init__()
        self.deadline = deadline
        self.stop = stop

    def is_set(self) -> bool:
        return (
            super().is_set()
            or (self.stop is not None and self.stop.is_set())
            or time.monotonic() >= self.deadline
        )


def _with(document: Document, geometries: dict[str, Geometry]) -> Document:
    for geometry in geometries.values():
        document = document.replace_geometry(geometry)
    return document


def _paths(document: Document, oids):
    from vectrify.refine.frozen import Paths

    return Paths({oid: document.geometry_for(oid) for oid in oids})


def _pixels(document: Document, region: Region) -> np.ndarray:
    return np.asarray(render_region(document, region))


@dataclass(frozen=True)
class _Scored:
    """A render of the region and how far each of its pixels is off."""

    pixels: np.ndarray
    # Squared difference per pixel, summed over RGB in 0-1.
    off: np.ndarray

    @classmethod
    def of(cls, pixels: np.ndarray, region: Region) -> _Scored:
        target = np.asarray(region.image.convert("RGB"), dtype=np.float64) / 255
        off = ((pixels.astype(np.float64) / 255 - target) ** 2).sum(axis=-1)
        return cls(pixels, off)

    @property
    def difference(self) -> float:
        """The mean squared difference over the region, as generate.error."""
        return float(self.off.sum() / (3 * self.off.size))

    def fixed(self, before: _Scored) -> float:
        """The share of *before*'s difference fixed where the two differ.

        Only the pixels whose colour changed, and a band of BAND pixels
        around them, count: what a step did is judged by where it acted,
        not diluted by the rest of a large selection.
        """
        changed = np.any(self.pixels != before.pixels, axis=-1)
        if not changed.any():
            return 0.0
        near = ndimage.binary_dilation(changed, iterations=BAND)
        base, now = float(before.off[near].sum()), float(self.off[near].sum())
        if base <= 0:
            # Nothing was off there: a change can only make it worse.
            return 0.0 if now <= 0 else float("-inf")
        return (base - now) / base


def _run_step(step: str, task: _Task, stop=None, progress=None):
    """(document after *step*, its render of the region, why paths were
    skipped). The step stops at its share of the time, keeping how far it
    got."""
    document, region, settings = task.document, task.region, task.settings
    deadline = min(task.deadline, time.monotonic() + task.share)
    skipped: dict[str, str] = {}
    if step == "shape":
        document, skipped = _fit(task, _Until(deadline, stop), progress)
    else:
        from vectrify.refine.frozen import frozen

        paths = _paths(document, task.oids)
        fixed = frozen(paths)
        if step == "snap":
            from vectrify.refine.snap import snap

            paths = snap(
                document,
                paths,
                region,
                fixed,
                detail=settings["detail"],
                split_gain=settings["detail_gain"],
                deadline=deadline,
            )
        else:
            paths = _simplified(task, paths, fixed, deadline)
        document = _with(document, dict(paths.geometries))
    return document, _pixels(document, region), skipped


# How many tolerances, up to the set one, Simplify's error budget picks from.
LADDER = 8
# Simplify's tolerance without a reference, unless one is set: no budget
# judges it then.
NO_REFERENCE = 1.0


def _simplified(task: _Task, paths, fixed, deadline: float):
    """*paths* simplified within the tolerance and, with a reference, within
    the error budget: at the largest of LADDER tolerances up to the set one
    whose result raises the difference where it acted by no more than the
    budget, found by bisection. A larger tolerance removes more points."""
    from vectrify.refine.simplify import simplify

    document, region, settings = task.document, task.region, task.settings

    def at(tolerance: float):
        return simplify(document, paths, region, fixed, tolerance, deadline)

    top = settings["tolerance"]
    if task.reference is None or top <= 0:
        return at(top)
    budget = settings["budget"] / 100
    start = _Scored.of(_pixels(document, region), region)

    def within(candidate) -> bool:
        after = _Scored.of(
            _pixels(_with(document, dict(candidate.geometries)), region), region
        )
        return after.fixed(start) >= -budget

    best = None
    low, high = 0, LADDER
    while low < high and time.monotonic() < deadline:
        middle = (low + high + 1) // 2
        candidate = at(top * middle / LADDER)
        if within(candidate):
            best, low = candidate, middle
        else:
            high = middle - 1
    # At no tolerance only points that change nothing go.
    return best if best is not None else at(0.0)


def _fit(task: _Task, stop, progress) -> tuple[Document, dict[str, str]]:
    """Fit each path in turn by gradient descent.

    Straight segments are fitted as curves, so the fit can bend one where
    the reference needs; those it leaves straight go back to lines.
    """
    from vectrify.refine.selected import FitOptions, fit_selected_path
    from vectrify.refine.simplify import curved, straightened
    from vectrify.refine.snap import _frame

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
        original = document.geometry_for(oid)
        document = document.replace_geometry(curved(original))
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
            document = document.replace_geometry(original)
            continue
        if not fit.values:
            document = document.replace_geometry(original)
            continue
        geometry = document.geometry_for(oid)
        fitted = replace(
            geometry,
            subpaths=tuple(
                replace(
                    s,
                    nodes=tuple(
                        replace(n, values=fit.values[n.id]) if n.id in fit.values else n
                        for n in s.nodes
                    ),
                )
                for s in geometry.subpaths
            ),
        )
        # Lines the fit bent by less than it can tell apart go back to lines.
        frame = _frame(document, oid, task.region, task.region.image.size)
        if frame is not None:
            fitted = straightened(fitted, STRAIGHT, frame)
        document = document.replace_geometry(fitted)
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
        if request.reference is None and "tolerance" not in request.settings:
            settings["tolerance"] = NO_REFERENCE
        rounds = request.budget.steps or DEFAULT_ROUNDS
        start = request.snapshot.document
        oids = tuple(selected_paths(request))
        margin = settings["margin"] / 100
        region = (
            target_region(request, margin)
            if request.reference is not None
            else drawing_region(request, settings["resolution"], margin)
        )
        steps = [s for s in STEPS if settings[s]]
        began = time.monotonic()
        deadline = began + settings["seconds"]
        document = start
        current = _Scored.of(_pixels(document, region), region)
        first, before = current.pixels, current.difference
        points = _count(document, oids)
        taken: list[str] = []
        skipped: dict[str, str] = {}
        # How often each step's result crossed itself more and was not kept.
        folded: dict[str, int] = {}
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
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                heading = f"Round {done + 1}/{rounds} · {points} points"
                context.progress(done, heading, total=rounds)

                def report(_step, message, done=done, heading=heading):
                    context.progress(done, f"{heading} · {message}")

                task = _Task(
                    document,
                    region,
                    settings,
                    oids,
                    request.reference,
                    deadline,
                    # One share is kept back for rendering and judging them.
                    left / (len(steps) + 1),
                )
                results = _round(steps, task, pool, context.stop, report)
                for _doc, _pixels_after, why in results.values():
                    skipped.update(why)
                for step in _folding(results, _crossings(document, oids), oids):
                    del results[step]
                    folded[step] = folded.get(step, 0) + 1
                scored = {
                    step: (after, _Scored.of(pixels, region))
                    for step, (after, pixels, _why) in results.items()
                }
                chosen = _choose(
                    scored,
                    current,
                    points,
                    oids,
                    settings["gain"] / 100,
                    # Without a reference Simplify is judged against the
                    # drawing itself, which any change makes worse.
                    _Scored.of(first, region)
                    if request.reference is not None
                    else None,
                    settings["allowance"] / 100,
                )
                if chosen is None:
                    break
                taken.append(chosen)
                document, current = scored[chosen]
                points = _count(document, oids)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)

        spent = time.monotonic() - began
        out_of_time = not context.stop.is_set() and time.monotonic() >= deadline
        tx = request.transaction(LABEL)
        for oid in oids:
            geometry = document.geometry_for(oid)
            if geometry != start.geometry_for(oid):
                tx.reshape_path(oid, geometry)
        changed = bool(taken)
        message = None
        if not changed:
            message = next(
                iter(skipped.values()),
                "Every step that helped made a path cross itself"
                if folded
                else "No step improved the paths"
                + (" within the time limit" if out_of_time else ""),
            )
        return OperationResult(
            Proposal(
                tx,
                changed,
                metrics={
                    "before": {"difference": before, "nodes": _count(start, oids)},
                    "after": {"difference": current.difference, "nodes": points},
                    "steps": taken,
                    "seconds": round(spent, 3),
                    # The time limit ended the run, not the steps running out.
                    "out_of_time": out_of_time,
                    "skipped": skipped,
                    "folded": folded,
                    "reference": request.reference is not None,
                },
                # The renders the steps were judged by.
                previews=preview_urls(
                    region.image,
                    Image.fromarray(first),
                    Image.fromarray(current.pixels),
                ),
            ),
            message=message,
        )


def _count(document: Document, oids) -> int:
    return sum(
        len(s.nodes) for oid in oids for s in document.geometry_for(oid).subpaths
    )


def _crossings(document: Document, oids) -> dict[str, int]:
    """How many times each path's outline crosses itself."""
    from vectrify.refine.crossings import crossings

    return {oid: crossings(document.geometry_for(oid)) for oid in oids}


def _folding(results, before: dict[str, int], oids) -> list[str]:
    """The steps whose result has a path crossing itself more than *before*.

    However much closer it looks, an outline folded over itself is no
    improvement; one that already crossed itself may keep its crossings.
    """
    return [
        step
        for step, (document, _pixels, _why) in results.items()
        if any(n > before[oid] for oid, n in _crossings(document, oids).items())
    ]


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


def _choose(
    scored,
    current: _Scored,
    points: int,
    oids,
    gain: float,
    start: _Scored | None = None,
    allowance: float = 0.0,
) -> str | None:
    """The step to keep: the one that lowers the difference most, fixing at
    least *gain* of it where it acted, or else Simplify if it removed points.

    With *start*, no step is kept that leaves the difference where the run
    has acted, against *start*, worse by more than *allowance* of it: a run
    never makes the match worse than that, however many points it saves.
    """

    def allowed(after: _Scored) -> bool:
        return start is None or after.fixed(start) >= -allowance

    helping = [
        (after.difference, step)
        for step, (_doc, after) in scored.items()
        if step != "simplify"
        and after.difference < current.difference
        and after.fixed(current) >= gain
        and allowed(after)
    ]
    if helping:
        return min(helping)[1]
    if "simplify" in scored:
        simpler, after = scored["simplify"]
        if _count(simpler, oids) < points and allowed(after):
            return "simplify"
    return None


register(OptimizeNodes())
