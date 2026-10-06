"""Connected region families compete as one flat or gradient RGBA surface.

The canonical source graph proposes membership. Curve union preserves current
exterior geometry and holes, and leaves neighboring paths untouched. Native
local/full validation decides whether removing the subdivisions is worthwhile.
No human geometry or semantic recognition enters these proposal generators.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from cairosvg.colors import color
from scipy.ndimage import find_objects, gaussian_filter

from vectrify.document import Editor, Selection
from vectrify.document.join import path_style, union_geometry
from vectrify.document.paint import GradientStop, LinearGradient
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine import cel
from vectrify.refine.cel_plan.constraints import discard
from vectrify.refine.cel_plan.ink import measure
from vectrify.refine.cel_plan.local import (
    MAX_CROP_PIXELS,
    Box,
    LocalLimitError,
    tile_boxes,
)
from vectrify.refine.cel_plan.model import Boundary, Evidence, Graph, Options, Work
from vectrify.refine.cel_plan.opacity import Paint, fit_samples
from vectrify.refine.cel_plan.ownership import Surface
from vectrify.refine.cel_plan.search import Proposal, State

MAX_PATHS = 128
MAX_NODES = 6_000
MAX_FAMILIES = 24
MAX_GROUPS = 96
MAX_BOUNDARY_PROOFS = 512
MAX_BOUNDARY_POINTS = 2_048
THRESHOLDS = (12, 28, 56)


def _opacity(document, oid):
    return float(
        np.prod(
            [float(a.get("opacity", "1") or "1") for a in document.ancestry(oid)[:-1]]
        )
    )


def _gradient(paint: Paint, evidence: Evidence, document, oid, opacity):
    assert paint.gradient is not None
    inverse = inverse_matrix(root_matrix(document, oid))
    a, b, c, d, e, f = inverse
    sx, sy = evidence.scale
    ox, oy = evidence.offset

    def local(point):
        x, y = point[0] / sx + ox, point[1] / sy + oy
        return a * x + c * y + e, b * x + d * y + f

    return LinearGradient(
        local(paint.gradient.start),
        local(paint.gradient.end),
        tuple(
            GradientStop(stop.offset, stop.colour, min(1.0, stop.opacity / opacity))
            for stop in paint.gradient.stops
        ),
    )


class Families:
    def __init__(self, evidence: Evidence, graph: Graph, options: Options):
        self.evidence, self.graph, self.options = evidence, graph, options
        self.boxes = find_objects(graph.labels + 1, max_label=len(graph.regions))
        self.ridges: dict[int, bool] = {}
        self.light: np.ndarray | None = None
        self.diagnostics = {
            "ridge_proofs": 0,
            "supported_ridges": 0,
            "shade_boundaries": 0,
            "unresolved_boundaries": 0,
        }

    def _protected(self, edge: Boundary, work: Work) -> bool:
        """Coarse ink can be a shade step; only bounded ridge tests relax it.

        Short, large or unexamined chains keep the conservative restriction.
        Evidence is immutable, so cached decisions survive ownership changes.
        Absence of a ridge permits a proposal, never acceptance of its pixels.
        """
        if edge.line_support <= 0.5:
            return False
        if edge.id in self.ridges:
            return self.ridges[edge.id]
        points = edge.points
        if (
            not 4 <= len(points) <= MAX_BOUNDARY_POINTS
            or np.linalg.norm(np.diff(points, axis=0), axis=1).sum() < 8
            or len(self.ridges) >= MAX_BOUNDARY_PROOFS
            or work.interrupted
        ):
            self.diagnostics["unresolved_boundaries"] += 1
            return True
        if self.light is None:
            self.light = gaussian_filter(cel.lightness(self.evidence.target), 0.5)
        if work.interrupted:
            return True
        proof = measure(points, self.evidence.target, 1.5, light=self.light)
        if work.interrupted:
            return True
        protected = proof is not None
        self.ridges[edge.id] = protected
        self.diagnostics["ridge_proofs"] += 1
        self.diagnostics["supported_ridges" if protected else "shade_boundaries"] += 1
        return protected

    def _groups(self, state: State, work: Work, *, thresholds=THRESHOLDS):
        partition = state.partition
        if partition is None:
            return
        document = state.document
        owners = partition.owners
        constrained = set(state.details.get("paint_constraints", ()))
        eligible = {}
        nodes = {}
        parents = {}
        components = {}
        colors = {}
        for surface in partition.surfaces:
            if work.interrupted:
                return
            if (
                surface.role != "surface"
                or surface.covered
                or surface.id in constrained
            ):
                continue
            regions = [self.graph.regions[index] for index in surface.members]
            if (
                any(region.fixed for region in regions)
                or len({r.component for r in regions}) != 1
            ):
                continue
            element = document.element(surface.id)
            style = path_style(document, element)
            if (
                style["stroke"] != "none"
                or float(element.get("opacity", "1") or "1") != 1
                or element.get("clip-path", "none") != "none"
            ):
                continue
            count = sum(
                len(sub.nodes) for sub in document.geometry_for(surface.id).subpaths
            )
            if count > MAX_NODES:
                continue
            weights = np.array([max(1, r.area) for r in regions])
            colors[surface.id] = np.average(
                np.array(
                    [
                        (*[value * r.opacity for value in r.paint], r.opacity * 255)
                        for r in regions
                    ]
                ),
                weights=weights,
                axis=0,
            )
            eligible[surface.id] = surface
            nodes[surface.id] = count
            parents[surface.id] = document.ancestry(surface.id)[-2].id
            components[surface.id] = regions[0].component
        neighbors: dict[str, set[str]] = {oid: set() for oid in eligible}
        barriers: dict[str, set[str]] = {oid: set() for oid in eligible}
        for edge in self.graph.boundaries:
            if work.interrupted:
                return
            left, right = owners.get(edge.left), owners.get(edge.right)
            if left not in eligible or right not in eligible or left == right:
                continue
            if parents[left] != parents[right] or components[left] != components[right]:
                continue
            if document.element(left).get("transform") != document.element(right).get(
                "transform"
            ):
                continue
            # No threshold can place colors farther than this in one family.
            # Avoid spending ridge work on pairs that cannot co-occur anyway.
            if float(np.linalg.norm(colors[left] - colors[right])) > 2 * max(
                thresholds
            ):
                continue
            if self._protected(edge, work):
                barriers[left].add(right)
                barriers[right].add(left)
                continue
            neighbors[left].add(right)
            neighbors[right].add(left)
        seeds = sorted(eligible, key=lambda oid: (-nodes[oid], oid))
        seen = set()
        emitted = 0
        for seed in seeds:
            for threshold in thresholds:
                if work.interrupted or emitted >= MAX_GROUPS:
                    return
                family = [seed]
                queued = {seed}
                queue = deque(sorted(neighbors[seed]))
                total = nodes[seed]
                while queue and len(family) < MAX_PATHS:
                    if work.interrupted:
                        return
                    oid = queue.popleft()
                    if oid in queued:
                        continue
                    queued.add(oid)
                    if total + nodes[oid] > MAX_NODES:
                        continue
                    if float(np.linalg.norm(colors[oid] - colors[seed])) > threshold:
                        continue
                    # A weak alternate route must not erase a supported ridge.
                    if barriers[oid].intersection(family):
                        continue
                    family.append(oid)
                    total += nodes[oid]
                    queue.extend(sorted(neighbors[oid] - queued))
                ids = tuple(sorted(family))
                if len(ids) > 1 and ids not in seen:
                    seen.add(ids)
                    emitted += 1
                    yield ids, threshold

    def _paints(self, state: State, work: Work):
        """Prioritize bounded fits before curve union or exact evaluation.

        Samples compare predicted RGBA with the current native raster, using a
        fixed content denominator and a rough representation saving. This omits
        layers, filters and feature terms and cannot accept an edit.
        """
        from vectrify.refine.cel_plan.proposals import bounds

        assert state.partition is not None
        surfaces = {surface.id: surface for surface in state.partition.surfaces}
        candidates = []
        area = max(1, sum(region.area for region in self.graph.regions))
        estimate_work = Work(
            min(work.deadline, time.monotonic() + work.remaining * 0.25),
            work.stop,
            work.timings,
        )
        for ids, threshold in self._groups(state, estimate_work):
            if estimate_work.interrupted:
                break
            box = bounds(state.document, state.document, ids)
            try:
                tile_boxes(box, state.snapshot.canvas.root.shape)
            except LocalLimitError:
                continue
            members = tuple(
                sorted(member for oid in ids for member in surfaces[oid].members)
            )
            samples = self.samples(members, estimate_work)
            if samples is None:
                break
            xy, rgba, size = samples
            if not size:
                continue
            paint = fit_samples(xy, rgba, gradients=self.options.gradients)
            variants = [paint]
            if paint.gradient:
                variants.append(fit_samples(xy, rgba, gradients=False))
            sx, sy = self.evidence.scale
            ox, oy = self.evidence.offset
            nx = np.clip(
                np.floor(xy[:, 0] / sx + ox).astype(int),
                0,
                state.snapshot.canvas.root.shape[1] - 1,
            )
            ny = np.clip(
                np.floor(xy[:, 1] / sy + oy).astype(int),
                0,
                state.snapshot.canvas.root.shape[0] - 1,
            )
            before = state.snapshot.canvas.samples(nx, ny)

            def error(values, truth=rgba):
                return float(
                    np.square(
                        values[:, :3] * values[:, 3:] - truth[:, :3] * truth[:, 3:]
                    ).mean()
                    + np.square(values[:, 3] - truth[:, 3]).mean()
                )

            before_error = error(before)
            saved = (len(ids) - 1) * 8
            for variant in variants:
                if variant.gradient:
                    gradient = variant.gradient
                    axis = np.array(gradient.end) - gradient.start
                    u = np.clip(
                        (xy - gradient.start) @ axis / float(axis @ axis), 0, 1
                    )[:, None]
                    ends = np.array(
                        [
                            (*color(stop.colour)[:3], stop.opacity)
                            for stop in gradient.stops
                        ]
                    )
                    predicted = ends[0] * (1 - u) + ends[1] * u
                else:
                    predicted = np.repeat(
                        np.array([(*color(variant.color)[:3], variant.opacity)]),
                        len(xy),
                        axis=0,
                    )
                delta = (error(predicted) - before_error) * size / area
                priority = delta - self.options.detail_cost * saved / max(
                    1, state.snapshot.evaluation.cost
                )
                candidates.append(
                    (priority, ids, threshold, members, variant, delta, -saved)
                )
        candidates.sort(
            key=lambda value: (
                value[0],
                value[1],
                value[2],
                value[4].gradient is not None,
            )
        )
        yield from candidates[:MAX_FAMILIES]

    def samples(self, members: tuple[int, ...], work: Work):
        """Uniform row-major sampling without an unbounded family mask/crop."""
        boxes = [
            self.boxes[index] for index in members if self.boxes[index] is not None
        ]
        if not boxes:
            return np.empty((0, 2)), np.empty((0, 4)), 0
        bounds = Box(
            min(box[1].start for box in boxes),
            min(box[0].start for box in boxes),
            max(box[1].stop for box in boxes),
            max(box[0].stop for box in boxes),
        )
        size = sum(self.graph.regions[index].area for index in members)
        step = max(1, (size + 4095) // 4096)
        offset = 0
        positions, colors = [], []
        for chunk in bounds.chunks(MAX_CROP_PIXELS):
            if work.interrupted:
                return None
            own = (
                np.isin(self.graph.labels[chunk.slices], members)
                & ~self.evidence.empty[chunk.slices]
            )
            indices = np.flatnonzero(own)
            keep = indices[(-offset) % step :: step]
            offset += len(indices)
            y, x = np.unravel_index(keep, own.shape)
            alpha = (
                self.evidence.opacity[chunk.slices][y, x]
                if self.evidence.opacity is not None
                else np.ones(len(x))
            )
            positions.append(np.column_stack((x + chunk.x + 0.5, y + chunk.y + 0.5)))
            colors.append(
                np.column_stack((self.evidence.target[chunk.slices][y, x] / 255, alpha))
            )
        if work.interrupted:
            return None
        return np.concatenate(positions), np.concatenate(colors), offset

    def __call__(self, state: State, work: Work):
        # Import here keeps the bounds helper and the operator factory acyclic.
        from vectrify.refine.cel_plan.proposals import bounds

        partition = state.partition
        if partition is None:
            return
        for priority, ids, threshold, members, paint, delta, saving in self._paints(
            state, work
        ):
            if work.interrupted:
                return
            document = state.document
            parent = document.ancestry(ids[0])[-2]
            selected = [child for child in parent.children if child.id in ids]
            survivor = selected[-1].id
            styles = [path_style(document, element) for element in selected]
            geometry = union_geometry(
                [document.geometry_for(p.id) for p in selected], styles
            )
            if work.interrupted:
                return
            parent_opacity = _opacity(document, survivor)
            if parent_opacity <= 0:
                continue
            changed = partition.replace(ids, (Surface(survivor, members),))
            if work.interrupted:
                return
            editor = Editor(document, selection=Selection(whole_document=True))
            with editor.transaction("Propose coherent planned surface") as transaction:
                transaction.replace_geometry(survivor, geometry)
                transaction.delete_objects(frozenset(set(ids) - {survivor}))
                transaction.set_fill(
                    survivor,
                    _gradient(paint, self.evidence, document, survivor, parent_opacity)
                    if paint.gradient
                    else paint.color,
                )
                transaction.set_attributes(
                    survivor,
                    {
                        "fill-rule": "nonzero",
                        "fill-opacity": "1"
                        if paint.gradient
                        else repr(min(1.0, paint.opacity / parent_opacity)),
                    },
                )
            proposed = editor.snapshot.document
            holds = set(state.details.get("geometry_constraints", ()))
            preserve = bool(holds.intersection(ids))
            holds.difference_update(ids)
            if preserve:
                holds.add(survivor)
            yield Proposal(
                "family-surface",
                ids,
                ("gradient" if paint.gradient else "flat", threshold, members),
                state.key,
                proposed,
                bounds(document, proposed, ids),
                estimate=saving,
                details={
                    "regions": sum(s.role != "underlay" for s in changed.surfaces),
                    "geometry_constraints": sorted(holds),
                    "chain_constraints": discard(
                        state.details.get("chain_constraints"), ids
                    ),
                    "family_estimate": {"priority": priority, "paint_delta": delta},
                },
                dependencies=(parent.id,),
                partition=changed,
            )
