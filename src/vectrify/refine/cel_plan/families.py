"""Connected region families compete as one flat or gradient RGBA surface.

The canonical source graph proposes membership. Curve union preserves current
exterior geometry and holes, and leaves neighboring paths untouched. Native
local/full validation decides whether removing the subdivisions is worthwhile.
No human geometry or semantic recognition enters these proposal generators.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import replace

import numpy as np
import pathops
from cairosvg.colors import color
from scipy.ndimage import find_objects, gaussian_filter

from vectrify.document import Editor, Selection
from vectrify.document.join import path_style, union_geometry
from vectrify.document.paint import GradientStop, LinearGradient
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine import cel
from vectrify.refine.cel_plan.boundary_evidence import shade_fragment
from vectrify.refine.cel_plan.constraints import discard, merged
from vectrify.refine.cel_plan.ink import measure
from vectrify.refine.cel_plan.layer_order import ordered
from vectrify.refine.cel_plan.local import (
    MAX_CROP_PIXELS,
    Box,
    LocalLimitError,
    tile_boxes,
)
from vectrify.refine.cel_plan.model import Boundary, Evidence, Graph, Options, Work
from vectrify.refine.cel_plan.nested import enclosed, in_core
from vectrify.refine.cel_plan.opacity import Paint, fit_samples
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.search import MAX_EDIT_OBJECTS, Proposal, State

MAX_PATHS = 128
MAX_NODES = 6_000
MAX_FAMILIES = 24
MAX_GROUPS = 96
MAX_BOUNDARY_PROOFS = 512
MAX_BOUNDARY_POINTS = 2_048
MAX_FRAGMENT_PROOFS = 4_096
MAX_INK_CONTACTS = 4_096
MAX_INK_SAMPLE_PIXELS = 65_536
INK_COLOR_SPREAD = 24.0
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
        self.fragments: dict[int, bool] = {}
        self.ink_contacts: set[int] = set()
        self.ink_consistency: dict[int, bool] = {}
        ink_pixels = np.bincount(
            graph.labels.ravel(),
            weights=evidence.drawn.ravel(),
            minlength=len(graph.regions),
        )
        self.ink_regions = (
            ink_pixels >= np.array([max(1, r.area) for r in graph.regions]) * 0.6
        )
        self.light: np.ndarray | None = None
        self.diagnostics = {
            "ridge_proofs": 0,
            "supported_ridges": 0,
            "shade_boundaries": 0,
            "unresolved_boundaries": 0,
            "fragment_proofs": 0,
            "shade_fragments": 0,
            "protected_fragments": 0,
            "compatible_ink_contacts": 0,
            "nested_proposals": 0,
            "nested_crop_limits": 0,
            "nested_core_exclusions": 0,
            "nested_geometry_limits": 0,
            "nested_order_exclusions": 0,
            "order_proofs": 0,
            "order_proof_limits": 0,
        }
        self.nesting_rejections: dict[str, int] = {}
        self._surface_models = None
        self._split_models = None
        self._piecewise_models = None
        self._core_models = None
        self._joint_models = None
        self.band_planner = None

    def _same_ink(self, edge: Boundary, work: Work) -> bool:
        """An internal paint partition is not a gap in a continuous dark mark.

        Complete native atom samples must agree with their dark paint model.
        A small dark ridge hidden inside a broad shade atom therefore cannot
        use its median color to claim that erasing the ridge is harmless.
        """
        indices = (edge.left, edge.right)
        if any(
            i < 0
            or i in self.graph.hidden
            or not self.ink_regions[i]
            or self.graph.regions[i].fixed
            for i in indices
        ):
            return False
        paints = np.array([self.graph.regions[i].paint for i in indices])
        if (
            cel.lightness(paints).max() > 96
            or np.linalg.norm(paints[0] - paints[1]) > INK_COLOR_SPREAD
        ):
            return False
        for i, paint in zip(indices, paints, strict=True):
            if work.interrupted:
                return False
            if i not in self.ink_consistency:
                box = self.boxes[i]
                if (
                    box is None
                    or (box[0].stop - box[0].start) * (box[1].stop - box[1].start)
                    > MAX_INK_SAMPLE_PIXELS
                ):
                    self.ink_consistency[i] = False
                else:
                    own = (self.graph.labels[box] == i) & ~self.evidence.empty[box]
                    samples = self.evidence.target[box][own]
                    consistent = bool(
                        len(samples)
                        and (
                            np.linalg.norm(samples - paint, axis=1) <= INK_COLOR_SPREAD
                        ).all()
                    )
                    if work.interrupted:
                        return False
                    self.ink_consistency[i] = consistent
            if not self.ink_consistency[i]:
                return False
        return True

    def _protected(self, edge: Boundary, work: Work) -> bool:
        """Coarse ink can be a shade step; only bounded ridge tests relax it.

        Short chains require complete monotone native cross-sections to relax.
        Large or unexamined chains keep the conservative restriction.
        Evidence is immutable, so cached decisions survive ownership changes.
        Absence of a ridge permits a proposal, never acceptance of its pixels.
        """
        if edge.line_support <= 0.5:
            return False
        if edge.id in self.ink_contacts:
            return False
        if (
            len(self.ink_contacts) < MAX_INK_CONTACTS
            and not work.interrupted
            and self._same_ink(edge, work)
        ):
            if work.interrupted:
                return True
            self.ink_contacts.add(edge.id)
            self.diagnostics["compatible_ink_contacts"] += 1
            return False
        if edge.id in self.ridges:
            return self.ridges[edge.id]
        if edge.id in self.fragments:
            return self.fragments[edge.id]
        points = edge.points
        short = (
            len(points) < 4 or np.linalg.norm(np.diff(points, axis=0), axis=1).sum() < 8
        )
        if short and len(points) <= MAX_BOUNDARY_POINTS:
            if len(self.fragments) >= MAX_FRAGMENT_PROOFS or work.interrupted:
                self.diagnostics["unresolved_boundaries"] += 1
                return True
            if self.light is None:
                self.light = gaussian_filter(cel.lightness(self.evidence.target), 0.5)
            protected = not shade_fragment(points, self.evidence, self.light)
            if work.interrupted:
                return True
            self.fragments[edge.id] = protected
            self.diagnostics["fragment_proofs"] += 1
            self.diagnostics[
                "protected_fragments" if protected else "shade_fragments"
            ] += 1
            return protected
        if (
            not 4 <= len(points) <= MAX_BOUNDARY_POINTS
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

    def _continued(self, state, ids, members, survivor, geometry, work):
        """An adjacent family also competes as a base under enclosed marks.

        Preserve all actual mark paint/geometry. Primary ownership is unchanged;
        only a proved base receives secondary coverage and a local order edge.
        """
        boxes = [self.boxes[i] for i in members if self.boxes[i] is not None]
        box = Box(
            min(b[1].start for b in boxes),
            min(b[0].start for b in boxes),
            max(b[1].stop for b in boxes),
            max(b[0].stop for b in boxes),
        ).expand(1, self.graph.labels.shape)
        if box.area > MAX_CROP_PIXELS:
            self.diagnostics["nested_crop_limits"] += 1
            return None
        own = (
            np.isin(self.graph.labels[box.slices], members)
            & ~self.evidence.empty[box.slices]
        )
        nesting = enclosed(
            self.evidence,
            self.graph,
            state,
            ids,
            box,
            own,
            survivor,
            work,
            rejections=self.nesting_rejections,
        )
        if nesting is None or not nesting.ids or work.interrupted:
            return None
        continued = nesting.continued(geometry)
        if work.interrupted:
            return None
        if sum(len(s.nodes) for s in continued.subpaths) > MAX_NODES:
            self.diagnostics["nested_geometry_limits"] += 1
            return None
        if self.evidence.opacity is not None and not in_core(
            state,
            tuple(sorted((*members, *nesting.members))),
            survivor,
            continued,
            work,
        ):
            self.diagnostics["nested_core_exclusions"] += int(not work.interrupted)
            return None
        return continued, nesting

    @property
    def surface_models(self):
        from vectrify.refine.cel_plan.surface_models import MaterialSurfaces

        if self._surface_models is None:
            self._surface_models = MaterialSurfaces(self, self.options)
        return self._surface_models

    def __call__(self, state: State, work: Work):
        from vectrify.refine.cel_plan.core_cells import CoreCells
        from vectrify.refine.cel_plan.joint_cells import JointCells
        from vectrify.refine.cel_plan.piecewise_surfaces import PiecewiseSurfaces
        from vectrify.refine.cel_plan.surface_splits import SurfaceSplits

        if self._split_models is None:
            self._split_models = SurfaceSplits(self, self.options)
        if self._piecewise_models is None:
            self._piecewise_models = PiecewiseSurfaces(self, self.options)
        if self._core_models is None:
            # Small fragment sets already have the existing local competitors.
            # Reserve the source-contour rebuild for substantial fragmentation.
            self._core_models = CoreCells(self, self.options, minimum_paths=32)
        if self._joint_models is None and self.options.quality == "high":
            self._joint_models = JointCells(self, self.options, bands=self.band_planner)
        # A High-quality seed beyond the small local-edit object bound gets
        # one joint opportunity before broad surface scans consume its window.
        # Smaller seeds keep the ordinary surface opportunity first. The
        # original interpretation precedes the fitted alternative in both cases.
        # All hypotheses occupy the existing family slot, so other operators
        # retain their reserved turns and the deadline is unchanged.
        cursors = [
            iter(self._joint_models(state, work))
            if self._joint_models is not None
            and state.partition is not None
            and len(state.partition.surfaces) >= 32
            else iter(()),
            iter(self.surface_models(state, work)),
            iter(self._core_models(state, work))
            if state.partition is not None and len(state.partition.surfaces) >= 32
            else iter(()),
            iter(self._piecewise_models(state, work)),
            iter(self.adjacent(state, work)),
            iter(self._split_models(state, work)),
        ]
        if state.partition is None or len(state.partition.surfaces) <= MAX_EDIT_OBJECTS:
            cursors[0], cursors[1] = cursors[1], cursors[0]
        alive = set(range(len(cursors)))
        try:
            while alive and not work.interrupted:
                for index in tuple(sorted(alive)):
                    try:
                        proposal = next(cursors[index], None)
                    except pathops.PathOpsError:
                        self.diagnostics["family_boolean_failures"] = (
                            self.diagnostics.get("family_boolean_failures", 0) + 1
                        )
                        alive.remove(index)
                        continue
                    if proposal is None:
                        alive.remove(index)
                    else:
                        yield proposal
                    if work.interrupted:
                        return
        finally:
            for cursor in cursors:
                close = getattr(cursor, "close", None)
                if close is not None:
                    close()

    def adjacent(self, state: State, work: Work):
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
                    "chain_constraints": merged(
                        state.details.get("chain_constraints"),
                        document,
                        proposed,
                        ids,
                        survivor,
                        work,
                    ),
                    "family_estimate": {"priority": priority, "paint_delta": delta},
                },
                dependencies=(parent.id,),
                partition=changed,
            )
            if work.interrupted:
                return
            continuation = self._continued(
                state, ids, members, survivor, geometry, work
            )
            if continuation is None:
                continue
            continued, nesting = continuation
            editor = Editor(proposed, selection=Selection(whole_document=True))
            with editor.transaction(
                "Continue family beneath its owned marks"
            ) as transaction:
                transaction.replace_geometry(survivor, continued)
            nested_document = ordered(
                editor.snapshot.document,
                survivor,
                set(),
                nesting.ids,
                continued,
                work,
                self.diagnostics,
            )
            if work.interrupted:
                return
            if nested_document is None:
                self.diagnostics["nested_order_exclusions"] += 1
                continue
            nested_partition = Partition(
                tuple(
                    replace(s, covered=nesting.members) if s.id == survivor else s
                    for s in changed.surfaces
                ),
                changed.atoms,
            )
            edited = (*ids, *nesting.ids)
            self.diagnostics["nested_proposals"] += 1
            yield Proposal(
                "family-surface",
                edited,
                (
                    "gradient" if paint.gradient else "flat",
                    threshold,
                    members,
                    "continued",
                ),
                state.key,
                nested_document,
                bounds(document, nested_document, edited),
                estimate=saving,
                details={
                    "regions": sum(
                        s.role != "underlay" for s in nested_partition.surfaces
                    ),
                    "geometry_constraints": sorted(holds | {survivor}),
                    "chain_constraints": discard(
                        state.details.get("chain_constraints"), edited
                    ),
                    "family_estimate": {"priority": priority, "paint_delta": delta},
                    "nested_surface": {
                        "marks": nesting.ids,
                        "covered_members": nesting.members,
                    },
                },
                dependencies=(parent.id,),
                partition=nested_partition,
            )
