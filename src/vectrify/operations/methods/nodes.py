"""Improve: Optimize nodes, a quick tidy of the selected paths' points.

Fit path combines a bounded edge-seeking proposal with gradient fitting of
points and handles, keeping the better exact render. Add detail explicitly
allows adding points. Strokes first seek the middle and width of their ink;
round strokes also support gradient fitting on CPU and CUDA. Without PyTorch,
edge-seeking remains available. Simplify removes unnecessary points within
an error budget and a tolerance in reference pixels. The legacy snap-only
setting remains available to API callers.

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

Every run has a time limit. Each round checks it and gives the inexpensive
steps a share of what is left. The fit reclaims the unused time, reserving a
short interval for rendering and judging, and shares it among the paths.
A run out of time keeps the best it found, as Stop does.

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

from vectrify.document import Document, DocumentError, Geometry, Selection, export_svg
from vectrify.image_utils import on_white, preview_urls
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
from vectrify.svg_render import cached_rendering, render_image

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
    "shape": Setting(bool, True),
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
    "steps": Setting(int, 60, minimum=1, maximum=1000, label="steps"),
    "movement": Setting(float, 2.0, minimum=0.0, maximum=100.0, label="movement"),
    "resolution": Setting(int, 384, minimum=64, maximum=2048, label="resolution"),
    # A path fit stops once a check, every ten steps, improves its match by
    # less than this, in percent: it has stalled.
    "stall": Setting(float, 0.1, minimum=0.0, maximum=50.0, label="stall"),
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
    # Where a selected path shares an edge with a neighbour, the neighbour's
    # edge moves with it. Joint fitting may additionally overlap selected
    # layers in painter order, keeping only an exact-render improvement.
    "shared": Setting(bool, True, label="move shared edges together"),
    # Only what lies in this area is tidied: [x, y, width, height] or a
    # polygon [[x, y], ...] in document units.
    "region": Setting(list, None, label="region"),
}


def effective_settings(request: OperationRequest) -> dict:
    settings = read_settings(request.settings, SETTINGS, LABEL)
    if request.reference is None and "tolerance" not in request.settings:
        settings["tolerance"] = NO_REFERENCE
    return settings


def area(settings) -> list[tuple[float, float]] | None:
    """The polygon the *settings* confine Tidy to, or None for whole paths."""
    if settings["region"] is None:
        return None
    from vectrify.document.regions import region_polygon

    return region_polygon(settings["region"])


def selected_paths(request: OperationRequest, polygon=None) -> list[str]:
    """The selected paths, and the paths inside selected groups; within
    *polygon*, those of them, or of the whole drawing when nothing is
    selected, that paint inside it."""
    document = request.snapshot.document
    selection = request.snapshot.selection
    if polygon is not None:
        return _paths_in(request, polygon)
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
    chosen = set(found)
    return [e.id for e in document.elements() if e.id in chosen]


def _paths_in(request: OperationRequest, polygon) -> list[str]:
    """The paths painting inside *polygon*: of the selection, or of the
    whole drawing without one, leaving out those whose geometry is shared
    or locked."""
    from shapely.geometry import Polygon

    from vectrify.document.hit_test import HitIndex
    from vectrify.document.model import EditKind

    document = request.snapshot.document
    selection = request.snapshot.selection
    roots = (
        [document.element(oid) for oid in sorted(selection.object_ids)]
        if selection.object_ids and not selection.whole_document
        else [document.root]
    )
    pool: list[str] = []
    while roots:
        element = roots.pop(0)
        if element.tag in {"defs", "clipPath", "mask"}:
            continue
        if element.tag == "path" and element.id not in pool:
            pool.append(element.id)
        roots.extend(element.children)
    shape = Polygon(polygon)
    index = HitIndex(document)
    found = []
    for oid in pool:
        painted = index.area(oid)
        if painted is None or not painted.intersects(shape):
            continue
        if document.geometry_users(document.geometry_for(oid).id) != {oid}:
            continue
        if any(EditKind.GEOMETRY in a.locks for a in document.ancestry(oid)):
            continue
        found.append(oid)
    if not found:
        raise DocumentError("No path to tidy paints inside the region")
    chosen = set(found)
    return [e.id for e in document.elements() if e.id in chosen]


def _outside(document: Document, oids, polygon) -> frozenset[str]:
    """The points of *oids* outside *polygon*: they stay as they are."""
    import shapely
    from shapely.geometry import Polygon

    from vectrify.document.regions import object_matrix

    shape = Polygon(polygon)
    held = set()
    for oid in oids:
        a, b, c, d, e, f = object_matrix(document, oid)
        nodes = [n for s in document.geometry_for(oid).subpaths for n in s.nodes]
        if not nodes:
            continue
        x, y = np.array([n.values[-2:] for n in nodes], dtype=np.float64).T
        inside = shapely.contains_xy(shape, a * x + c * y + e, b * x + d * y + f)
        held.update(n.id for n, i in zip(nodes, inside, strict=True) if not i)
    return frozenset(held)


def _area_region(request: OperationRequest, polygon, margin: float, long_side: int):
    """The region Tidy judges a region's tidy by: *polygon*'s bounds, widened
    by *margin* of their size, on the artboard, cropped from the reference
    at whole pixels, or the drawing rendered there without one."""
    from vectrify.image_utils import on_white

    document = request.snapshot.document
    vx, vy, vw, vh = document.artboard()
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    pad = margin * max(max(xs) - min(xs), max(ys) - min(ys))
    left, top = max(min(xs) - pad, vx), max(min(ys) - pad, vy)
    right, bottom = min(max(xs) + pad, vx + vw), min(max(ys) + pad, vy + vh)
    if right <= left or bottom <= top:
        raise DocumentError("The region is outside the artboard")
    if request.reference is None:
        scale = long_side / max(right - left, bottom - top)
        size = (
            max(1, round((right - left) * scale)),
            max(1, round((bottom - top) * scale)),
        )
        blank = Region(left, top, right - left, bottom - top, Image.new("RGB", size))
        return replace(blank, image=render_region(document, blank))
    reference = on_white(request.reference)
    sx, sy = reference.width / vw, reference.height / vh
    box = (
        round((left - vx) * sx),
        round((top - vy) * sy),
        round((right - vx) * sx),
        round((bottom - vy) * sy),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise DocumentError("The region is smaller than one reference pixel")
    alpha = None
    if request.reference.has_transparency_data:
        opacity = np.asarray(
            request.reference.convert("RGBA").getchannel("A").crop(box)
        )
        if opacity.min() < 255:
            alpha = opacity.astype(np.float32) / 255
    return Region(
        vx + box[0] / sx,
        vy + box[1] / sy,
        (box[2] - box[0]) / sx,
        (box[3] - box[1]) / sy,
        reference.crop(box),
        alpha,
    )


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
    # Points no step may move or remove: those outside a region's tidy and
    # where a shared edge ends.
    held: frozenset[str] = frozenset()
    # The edges the selected paths share with neighbours, which follow them.
    shared: tuple = ()
    # Whether edge-seeking may set stroked lines' widths: a paint change.
    widths: bool = False
    # The current render, shared with steps that only need to compare to it.
    pixels: np.ndarray | None = None
    # Inferred shared junctions hold endpoints; they may bend between them
    # during overlapping joint fits. Region holds still protect all controls.
    junctions: frozenset[str] = frozenset()


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
    from vectrify.document.features import held_nodes

    for geometry in geometries.values():
        original = document.geometry(geometry.id)
        held = held_nodes(original)
        if held:
            old = {n.id: n for sp in original.subpaths for n in sp.nodes}
            geometry = replace(
                geometry,
                subpaths=tuple(
                    replace(
                        sp,
                        nodes=tuple(old[n.id] if n.id in held else n for n in sp.nodes),
                    )
                    for sp in geometry.subpaths
                ),
            )
        document = document.replace_geometry(geometry)
    return document


def _paths(document: Document, oids):
    from vectrify.refine.frozen import Paths

    return Paths({oid: document.geometry_for(oid) for oid in oids})


def _pixels(document: Document, region: Region) -> np.ndarray:
    if region.alpha is not None:
        return np.asarray(
            render_image(
                export_svg(document),
                (region.x, region.y, region.width, region.height),
                region.image.size,
                alpha=True,
            )
        )
    return np.asarray(render_region(document, region))


@dataclass(frozen=True)
class _Scored:
    """A render of the region and how far each of its pixels is off."""

    pixels: np.ndarray
    # Squared difference per pixel, over white-backed RGB and, when supplied,
    # reference opacity. White paint and transparent gaps must be distinguishable.
    off: np.ndarray
    normalizer: int = 3

    @classmethod
    def of(cls, pixels: np.ndarray, region: Region) -> _Scored:
        target = np.asarray(region.image.convert("RGB"), dtype=np.float64) / 255
        rgb = (
            np.asarray(on_white(Image.fromarray(pixels)))
            if pixels.shape[-1] == 4
            else pixels
        )
        off = ((rgb.astype(np.float64) / 255 - target) ** 2).sum(axis=-1)
        normalizer = 3
        if region.alpha is not None:
            opacity = pixels[:, :, 3].astype(np.float64) / 255
            # Balance mean RGB error and opacity error equally.
            off += 3 * (opacity - region.alpha) ** 2
            normalizer = 6
        return cls(pixels, off, normalizer)

    @property
    def difference(self) -> float:
        """The mean squared difference over the region, as generate.error."""
        return float(self.off.sum() / (self.normalizer * self.off.size))

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


@cached_rendering()
def _run_step(step: str, task: _Task, stop=None, progress=None):
    """(document after *step*, its render of the region, why paths were
    skipped). The step stops at its share of the time, keeping how far it
    got."""
    document, region, settings = task.document, task.region, task.settings
    deadline = min(task.deadline, time.monotonic() + task.share)
    skipped: dict[str, str] = {}
    if step == "shape":
        from vectrify.refine.lines import is_line

        initial = tuple(oid for oid in task.oids if is_line(document, oid))
        if settings["snap"] and initial:
            # Edge-seeking is the initializer of the fit, not a competing
            # round result. A stopped gradient fit still retains this proposal.
            proposed, pixels, _ = _run_step(
                "snap",
                replace(task, oids=initial, deadline=deadline, share=task.share / 4),
                stop,
                progress,
            )
            watched = tuple(set(task.oids) | {link.neighbour for link in task.shared})
            if not _folding(
                {"initial": (proposed, pixels, {})},
                _crossings(document, watched),
                watched,
            ):
                document = proposed
                task = replace(task, document=document, pixels=pixels)
        document, skipped = _fit(task, _Until(deadline, stop), progress)
    else:
        from vectrify.refine.frozen import Frozen, frozen

        paths = _paths(document, task.oids)
        fixed = Frozen(frozen(paths).endpoints | task.held)
        if step == "snap":
            from vectrify.refine.lines import fit_lines, is_line
            from vectrify.refine.snap import snap

            # Stroked lines go onto their ink's middle; fills onto edges.
            lines = [oid for oid in task.oids if is_line(document, oid)]
            if lines:
                document = fit_lines(document, lines, region, fixed, task.widths)
            fills = [oid for oid in task.oids if oid not in lines]
            paths = snap(
                document,
                _paths(document, fills),
                region,
                fixed,
                detail=settings["detail"],
                split_gain=settings["detail_gain"],
                deadline=deadline,
            )
            candidate = _with(document, dict(paths.geometries))
            document = task.document
            current = _Scored.of(
                task.pixels if task.pixels is not None else _pixels(document, region),
                region,
            )
            # One badly snapped fill must not discard improvements to other
            # paths or strokes. Keep each actual, followed change independently.
            for oid in task.oids:
                proposed = document.replace_geometry(candidate.geometry_for(oid))
                width = candidate.element(oid).get("stroke-width")
                if width != document.element(oid).get("stroke-width"):
                    proposed = proposed.replace_element(candidate.element(oid))
                document, current = _improvement(task, document, proposed, current)
        else:
            paths = _simplified(task, paths, fixed, deadline)
            document = _with(document, dict(paths.geometries))
    if task.shared:
        from vectrify.refine.shared import follow

        document, _ = follow(
            document,
            list(_fit_shared(task, document) if step == "shape" else task.shared),
        )
    pixels = (
        task.pixels
        if task.pixels is not None and document == task.document
        else _pixels(document, region)
    )
    return document, pixels, skipped


def _improvement(task: _Task, before: Document, candidate: Document, current: _Scored):
    """Keep a path's actual improvement after shared edges have followed."""
    from vectrify.refine.shared import follow

    candidate, _ = follow(candidate, list(task.shared))
    if candidate == before:
        return before, current
    watched = set(task.oids) | {link.neighbour for link in task.shared}
    changed = [
        oid
        for oid in watched
        if before.geometry_for(oid) != candidate.geometry_for(oid)
    ]
    crossings_before = _crossings(before, changed)
    if any(
        count > crossings_before[oid]
        for oid, count in _crossings(candidate, changed).items()
    ):
        return before, current
    after = _Scored.of(_pixels(candidate, task.region), task.region)
    if after.difference < current.difference and after.fixed(current) >= 0:
        return candidate, after
    return before, current


def _fit_shared(task: _Task, document: Document):
    """Retain external links and selected edges that still match exactly."""
    from vectrify.refine.shared import intact

    selected = set(task.oids)
    return tuple(
        link
        for link in task.shared
        if link.path not in selected
        or link.neighbour not in selected
        or intact(document, link)
    )


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
    from vectrify.refine.support import supported

    document, region, settings = task.document, task.region, task.settings
    initial_costs = {}

    def at(tolerance: float):
        return simplify(
            document,
            paths,
            region,
            fixed,
            tolerance,
            deadline,
            initial_costs=initial_costs,
            cost_bound=settings["tolerance"],
        )

    top = settings["tolerance"]
    if task.reference is None or top <= 0:
        return at(top)
    budget = settings["budget"] / 100
    start = _Scored.of(
        task.pixels if task.pixels is not None else _pixels(document, region), region
    )
    judged = []

    def within(candidate) -> bool:
        if candidate.geometries == paths.geometries:
            return True
        for previous, allowed in judged:
            if candidate.geometries == previous.geometries:
                return allowed
        from vectrify.refine.shared import follow

        proposed, _ = follow(
            _with(document, dict(candidate.geometries)), list(task.shared)
        )
        after = _Scored.of(_pixels(proposed, region), region)
        allowed = after.fixed(start) >= -budget
        judged.append((candidate, allowed))
        return allowed

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
    best = best if best is not None else at(0.0)
    # A nearby stroke is a second simplification model, judged run by run.
    # Independently simplifying that proposal again would discard its exact
    # curves before the reference could judge their usefulness.
    candidate = supported(document, paths, region, fixed, top, deadline, accept=within)

    def complexity(value):
        return (
            value.nodes(),
            sum(
                len(n.values) // 2 - 1
                for g in value.geometries.values()
                for s in g.subpaths
                for n in s.nodes
            ),
        )

    return candidate if complexity(candidate) < complexity(best) else best


def _fit(task: _Task, stop, progress) -> tuple[Document, dict[str, str]]:
    """Fit paths in order, then jointly polish compatible sibling paths.

    Straight segments are fitted as curves, so the fit can bend one where
    the reference needs; those it leaves straight go back to lines.
    """
    from vectrify.refine.parameters import line_knots
    from vectrify.refine.selected import FitOptions, fit_selected_path
    from vectrify.refine.simplify import curved, straightened
    from vectrify.refine.snap import _frame

    document, settings = task.document, task.settings
    reference = task.reference
    assert reference is not None
    options = FitOptions(
        steps=settings["steps"],
        displacement=settings["movement"],
        resolution=settings["resolution"],
        stall=settings["stall"] / 100,
        snap=settings["snap"],
    )
    skipped: dict[str, str] = {}
    from vectrify.refine.lines import is_line
    from vectrify.refine.paths import UnsupportedPathError

    pending = list(task.oids)
    current = _Scored.of(
        task.pixels if task.pixels is not None else _pixels(document, task.region),
        task.region,
    )
    for index, oid in enumerate(pending):
        if stop is not None and stop.is_set():
            break
        original = document.geometry_for(oid)
        # In a region's tidy only the points inside it move.
        movable = frozenset(
            n.id for s in original.subpaths for n in s.nodes if n.id not in task.held
        )
        if not movable:
            continue
        # Every remaining fill gets a share of the shape step's remaining
        # time. One expensive early path must not starve the rest of a selection.
        path_stop = stop
        if isinstance(stop, _Until):
            now = time.monotonic()
            path_stop = _Until(
                now + (stop.deadline - now) / (len(pending) - index), stop
            )
        before = document
        prepared = curved(original)
        if settings["detail"] and not is_line(document, oid):
            from vectrify.refine.detail import densified

            frame = _frame(document, oid, task.region, task.region.image.size)
            if frame is not None:
                prepared = densified(prepared, frame, task.held)
                movable |= frozenset(
                    n.id
                    for s in prepared.subpaths
                    for n in s.nodes
                    if n.id not in task.held
                )
        document = document.replace_geometry(prepared)
        polygon = area(settings)
        if settings["detail"] and polygon is not None:
            # A curve can leave the view even when both original endpoints
            # are inside it. New knots outside it inherit the same region hold.
            task = replace(task, held=task.held | _outside(document, (oid,), polygon))
            movable -= task.held
        try:
            fit = fit_selected_path(
                document,
                Selection(
                    object_ids=frozenset({oid}),
                    node_ids=movable if task.held else frozenset(),
                ),
                reference,
                replace(options, snap=False) if is_line(document, oid) else options,
                stop=path_stop,
                progress=progress,
                corners=line_knots(original),
            )
        except (DocumentError, UnsupportedPathError) as exc:
            # A path the fit cannot take, a gradient's say, is left as it is.
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
        # Prefer a line when it retains the fitted improvement. Straightening
        # can introduce crossings or lose useful subpixel curvature; fall back
        # to the raw fit if the straightened result is rejected.
        raw = document.replace_geometry(fitted)
        frame = _frame(document, oid, task.region, task.region.image.size)
        if frame is not None and not settings["detail"]:
            fitted = straightened(fitted, STRAIGHT, frame)
        document = document.replace_geometry(fitted)
        # Judge the actual curves and followed neighbours before fitting the
        # next path. Straightening and copying an edge can change the result
        # the single-path fitter judged in its frozen surrounding artwork.
        simplified, simplified_score = _improvement(task, before, document, current)
        if simplified == before and raw != document:
            fitted_document, fitted_score = _improvement(task, before, raw, current)
            if fitted_score.difference < simplified_score.difference:
                document, current = fitted_document, fitted_score
                continue
        document, current = simplified, simplified_score
    if settings["snap"]:
        document, current = _span_lines(task, document, current, stop)
    if stop is None or not stop.is_set():
        from vectrify.refine.joint import polish

        before = document
        raw = polish(
            document,
            task.oids,
            reference,
            options,
            held=task.held,
            shared=task.shared,
            overlaps=True,
            junctions=task.junctions,
            score=lambda candidate: (
                _Scored.of(_pixels(candidate, task.region), task.region).difference
            ),
            stop=stop,
            progress=progress,
        )
        candidate = raw
        for oid in task.oids:
            geometry = raw.geometry_for(oid)
            if geometry == before.geometry_for(oid):
                continue
            frame = _frame(raw, oid, task.region, task.region.image.size)
            if frame is not None and not settings["detail"]:
                candidate = candidate.replace_geometry(
                    straightened(geometry, STRAIGHT, frame)
                )
        final_task = replace(task, shared=_fit_shared(task, raw))
        document, current = _improvement(final_task, before, candidate, current)
        if document == before and raw != candidate:
            document, current = _improvement(final_task, before, raw, current)
    return document, skipped


def _span_lines(task: _Task, document: Document, current: _Scored, stop):
    """Try whole-curve ink readings after the individual fits, before polish.

    Unlike the midpoint initializer, these readings can separate opposite
    handle errors. Background colours can bias them, so keep only an actual
    improvement with the selection's current paint and followed neighbours.
    """
    from vectrify.refine.frozen import Frozen, Paths, frozen
    from vectrify.refine.lines import fit_lines, is_line

    for oid in task.oids:
        if stop is not None and stop.is_set():
            break
        if not is_line(document, oid):
            continue
        geometry = document.geometry_for(oid)
        if not any(n.command == "C" for s in geometry.subpaths for n in s.nodes):
            continue
        fixed = Frozen(frozen(Paths({oid: geometry})).endpoints | task.held)
        proposed = fit_lines(
            document, [oid], task.region, fixed, task.widths, span=True
        )
        bounded = []
        for sub, new_sub in zip(
            geometry.subpaths, proposed.geometry_for(oid).subpaths, strict=True
        ):
            nodes = []
            for node, new in zip(sub.nodes, new_sub.nodes, strict=True):
                values = np.asarray(node.values).reshape(-1, 2)
                delta = np.asarray(new.values).reshape(-1, 2) - values
                if node.id in task.held:
                    delta[:] = 0
                elif node.pinned:
                    delta[-1] = 0
                length = np.linalg.norm(delta, axis=1)
                delta *= np.minimum(
                    1, task.settings["movement"] / np.maximum(length, 1e-12)
                )[:, None]
                nodes.append(
                    replace(
                        new, values=tuple(float(v) for v in (values + delta).ravel())
                    )
                )
            bounded.append(replace(new_sub, nodes=tuple(nodes)))
        proposed = proposed.replace_geometry(replace(geometry, subpaths=tuple(bounded)))
        document, current = _improvement(task, document, proposed, current)
    return document, current


class OptimizeNodes:
    action: ClassVar[str] = "improve"
    name: ClassVar[str] = "nodes"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        settings = read_settings(request.settings, SETTINGS, LABEL)
        if settings["detail"] and not settings["snap"]:
            raise DocumentError("detail requires snap=true")
        if not any(settings[step] for step in STEPS):
            raise DocumentError("Choose at least one step")
        selected_paths(request, area(settings))
        if request.reference is None:
            if settings["shape"]:
                raise DocumentError("Add a reference image to fit the shape to")
            if settings["snap"]:
                raise DocumentError("Add a reference image to snap to")
        if settings["shape"]:
            from vectrify.refine.selected import fit_problem

            problem = fit_problem()
            # Without PyTorch the other steps still run, if there are any.
            if problem and not any(settings[s] for s in STEPS if s != "shape"):
                raise DocumentError(problem)
        missing = needed_permissions(settings) - request.permissions.allowed
        if missing:
            raise DocumentError(f"Allow {', '.join(sorted(missing))} changes")

    @cached_rendering()
    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        settings = effective_settings(request)
        rounds = request.budget.steps or DEFAULT_ROUNDS
        start = request.snapshot.document
        polygon = area(settings)
        oids = tuple(selected_paths(request, polygon))
        margin = settings["margin"] / 100
        if polygon is not None:
            region = _area_region(request, polygon, margin, settings["resolution"])
            held = _outside(start, oids, polygon)
        else:
            region = (
                target_region(request, margin)
                if request.reference is not None
                else drawing_region(request, settings["resolution"], margin)
            )
            held = frozenset()
        shared = _shared_edges(start, oids) if settings["shared"] else []
        from vectrify.document.features import held_nodes

        held |= frozenset(
            n for oid in oids for n in held_nodes(start.geometry_for(oid))
        )
        region_held = held
        if shared:
            from vectrify.refine.shared import frozen_points

            held |= frozen_points(start, shared)
        neighbours = tuple(sorted({link.neighbour for link in shared} - set(oids)))
        # The paths a step may change: the selected ones and, along shared
        # edges, their neighbours.
        watched = oids + neighbours
        steps = [s for s in STEPS if settings[s]]
        if "shape" in steps:
            from vectrify.refine.selected import fit_problem

            if fit_problem():
                steps.remove("shape")
        if "shape" in steps and "snap" in steps:
            steps.remove("snap")
        began = time.monotonic()
        deadline = began + settings["seconds"]
        document = start
        current = _Scored.of(_pixels(document, region), region)
        start_score = current
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
                if shared:
                    from vectrify.refine.shared import frozen_points, intact

                    # Accepted overlaps are no longer exact shared runs.
                    shared = [link for link in shared if intact(document, link)]
                    held = region_held | frozen_points(document, shared)
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
                    held,
                    tuple(shared),
                    "paint" in request.permissions.allowed,
                    current.pixels,
                    held - region_held,
                )
                results = _round(steps, task, pool, context.stop, report)
                for _doc, _pixels_after, why in results.values():
                    skipped.update(why)
                for step in _folding(results, _crossings(document, watched), watched):
                    del results[step]
                    folded[step] = folded.get(step, 0) + 1
                scored = {
                    step: (after, _Scored.of(pixels, region))
                    for step, (after, pixels, _why) in results.items()
                }
                from vectrify.document.features import violations

                for step, (candidate, _) in list(scored.items()):
                    problems = violations(start, candidate)
                    if problems:
                        skipped[step] = (
                            f"Protected feature {problems[0]['node']}: "
                            f"{problems[0]['reason']}"
                        )
                        del scored[step]
                chosen = _choose(
                    scored,
                    current,
                    points,
                    oids,
                    settings["gain"] / 100,
                    # Without a reference Simplify is judged against the
                    # drawing itself, which any change makes worse.
                    start_score if request.reference is not None else None,
                    settings["allowance"] / 100,
                    handles=_handles(document, oids),
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
        followed = [
            oid
            for oid in neighbours
            if document.geometry_for(oid) != start.geometry_for(oid)
        ]
        tx = (
            request.transaction(LABEL)
            if polygon is None and not followed
            # A region's tidy, or one whose neighbours followed, edits and
            # then selects the paths it changed along with the selection.
            else request.editor.transaction(
                LABEL,
                selection=Selection(
                    object_ids=frozenset(oids)
                    | frozenset(followed)
                    | (
                        request.snapshot.selection.object_ids
                        if polygon is None
                        else frozenset()
                    )
                ),
                allowed=request.permissions.allowed,
                base=request.snapshot,
            )
        )
        for oid in (*oids, *followed):
            geometry = document.geometry_for(oid)
            if geometry != start.geometry_for(oid):
                tx.reshape_path(oid, geometry)
            width = document.element(oid).get("stroke-width")
            if width != start.element(oid).get("stroke-width"):
                tx.set_attributes(oid, {"stroke-width": width})
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
                    "before": {
                        "difference": before,
                        "nodes": _count(start, oids),
                        "handles": _handles(start, oids),
                    },
                    "after": {
                        "difference": current.difference,
                        "nodes": points,
                        "handles": _handles(document, oids),
                    },
                    "steps": taken,
                    "seconds": round(spent, 3),
                    # The time limit ended the run, not the steps running out.
                    "out_of_time": out_of_time,
                    "skipped": skipped,
                    "folded": folded,
                    # Neighbours whose shared edges moved with the paths.
                    "followed": len(followed),
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


def _shared_edges(document: Document, oids) -> list:
    """The edges *oids* share with other paths whose geometry may change:
    their own, unlocked, filled or stroked."""
    from vectrify.document.model import EditKind
    from vectrify.refine.shared import links

    chosen = set(oids)
    candidates = [
        e.id
        for e in document.elements()
        if e.tag == "path"
        and not any(
            a.tag in {"defs", "clipPath", "mask"} for a in document.ancestry(e.id)
        )
        and document.geometry_users(document.geometry_for(e.id).id) == {e.id}
        and not any(EditKind.GEOMETRY in a.locks for a in document.ancestry(e.id))
    ]
    order = {e.id: i for i, e in enumerate(document.elements())}
    return [
        link
        for link in links(document, oids, candidates)
        if link.neighbour not in chosen or order[link.path] < order[link.neighbour]
    ]


def _count(document: Document, oids) -> int:
    return sum(
        len(s.nodes) for oid in oids for s in document.geometry_for(oid).subpaths
    )


def _handles(document: Document, oids) -> int:
    return sum(
        len(n.values) // 2 - 1
        for oid in oids
        for s in document.geometry_for(oid).subpaths
        for n in s.nodes
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
        if step != "shape" and step not in pending:
            results[step] = _run_step(step, task, stop, report)
    if "shape" in steps:
        # Cheap steps often finish well before their share. Give the fit the
        # remaining run time rather than reserving unused shares for them.
        # All candidates still start from the same drawing and are judged
        # together. Leave a small interval for the final render and scoring.
        remaining = max(0.0, task.deadline - time.monotonic() - 0.25)
        fitting = replace(task, share=remaining)
        results["shape"] = _run_step("shape", fitting, stop, report)
    for step, future in pending.items():
        results[step] = future.result()
    return {step: results[step] for step in steps}


def _choose(
    scored,
    current: _Scored,
    points: int,
    oids,
    gain: float,
    start: _Scored | None = None,
    allowance: float = 0.0,
    *,
    handles: int | None = None,
) -> str | None:
    """The step to keep: the one that lowers the difference most, fixing at
    least *gain* of it where it acted, or else Simplify if it removed points
    or unnecessary handles.

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
        reduced = _count(simpler, oids) < points or (
            handles is not None and _handles(simpler, oids) < handles
        )
        if reduced and allowed(after):
            return "simplify"
    return None


register(OptimizeNodes())
