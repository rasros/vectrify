"""Owned closed shapes compete above restored neighboring RGBA material.

Native region masks supply bounded perimeter/paint evidence. Existing opacity
cores must geometrically cover both the old and proposed footprint. Exact
local evaluation and full checkpoints choose between these and subdivisions.
"""

from __future__ import annotations

import time
from dataclasses import replace
from itertools import zip_longest

import numpy as np

from vectrify.document import Editor, Geometry, Selection
from vectrify.document.join import (
    path_style,
    transformed_geometry,
    union_geometry,
)
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine import cel
from vectrify.refine.cel_plan.constraints import discard
from vectrify.refine.cel_plan.families import Families, _gradient, _opacity
from vectrify.refine.cel_plan.geometry import Model, ellipse
from vectrify.refine.cel_plan.ink_replace import InkReplacement, identified
from vectrify.refine.cel_plan.layer_order import ordered
from vectrify.refine.cel_plan.model import Evidence, Graph, Options, Work
from vectrify.refine.cel_plan.nested import enclosed
from vectrify.refine.cel_plan.opacity import fit_samples
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.search import Proposal, State
from vectrify.refine.crossings import crossings
from vectrify.refine.tracing import _loops

MAX_GROUPS = 32
MAX_PERIMETER = 4_096
MAX_NODES = 6_000
MIN_AREA = 64


class ClosedOverlays:
    def __init__(
        self,
        evidence: Evidence,
        graph: Graph,
        options: Options,
        *,
        families: Families | None = None,
        restoration: InkReplacement | None = None,
    ):
        self.evidence, self.graph, self.options = evidence, graph, options
        self.families = (
            families if families is not None else Families(evidence, graph, options)
        )
        self.restoration = (
            restoration
            if restoration is not None
            else InkReplacement(evidence, graph, options)
        )
        self.diagnostics = dict.fromkeys(
            (
                "groups",
                "family_groups",
                "single_groups",
                "area_exclusions",
                "source_areas_bounded",
                "topology_exclusions",
                "perimeter_bounded",
                "no_restoration",
                "order_exclusions",
                "order_proofs",
                "order_proof_limits",
                "nested_exclusions",
                "nested_mark_outside",
                "nested_proposals",
                "ellipse_proposals",
                "contour_proposals",
                "time_bounded",
            ),
            0,
        )
        self.nesting_rejections: dict[str, int] = {}

    def groups(self, state: State, work: Work):
        partition = state.partition
        if partition is None:
            return
        surfaces = {s.id: s for s in partition.surfaces}
        total = sum(r.area for r in self.graph.regions if r.id not in self.graph.hidden)
        seen = set()
        count = 0

        def families():
            singles = []
            for surface in partition.surfaces:
                if work.interrupted:
                    return
                if (
                    surface.role != "surface"
                    or surface.covered
                    or surface.id in state.details.get("paint_constraints", ())
                ):
                    continue
                if any(self.graph.regions[i].fixed for i in surface.members):
                    continue
                regions = [self.graph.regions[i] for i in surface.members]
                if (
                    not MIN_AREA <= sum(r.area for r in regions) <= total * 0.05
                    or len({r.component for r in regions}) != 1
                ):
                    self.diagnostics["area_exclusions"] += 1
                    continue
                style = path_style(state.document, state.document.element(surface.id))
                if (
                    style["stroke"] != "none"
                    or float(style["opacity"]) != 1
                    or state.document.element(surface.id).get("clip-path", "none")
                    != "none"
                ):
                    continue
                nodes = sum(
                    len(sub.nodes)
                    for sub in state.document.geometry_for(surface.id).subpaths
                )
                if nodes <= MAX_NODES:
                    singles.append((-nodes, surface.id))
            groups = self.families._groups(state, work, thresholds=(24, 56, 96))
            try:
                for family, single in zip_longest(groups, sorted(singles)):
                    if family is not None:
                        yield family[0]
                    if single is not None:
                        yield (single[1],)
            finally:
                groups.close()

        for ids in families():
            if work.interrupted or count >= MAX_GROUPS:
                return
            if ids in seen:
                continue
            seen.add(ids)
            members = tuple(sorted(i for oid in ids for i in surfaces[oid].members))
            regions = [self.graph.regions[i] for i in members]
            size = sum(r.area for r in regions)
            if (
                not MIN_AREA <= size <= total * 0.05
                or len({r.component for r in regions}) != 1
            ):
                self.diagnostics["area_exclusions"] += 1
                continue
            count += 1
            self.diagnostics["family_groups" if len(ids) > 1 else "single_groups"] += 1
            yield ids, members

    def __call__(self, state: State, work: Work):
        bounded = Work(
            min(work.deadline, time.monotonic() + work.remaining * 0.25),
            work.stop,
            work.timings,
        )
        try:
            yield from self.proposals(state, bounded)
        finally:
            self.diagnostics["time_bounded"] += int(bounded.interrupted)

    def proposals(self, state: State, work: Work):
        from vectrify.refine.cel_plan.proposals import bounds

        partition = state.partition
        if partition is None:
            return
        document = state.document
        for ids, members in self.groups(state, work):
            self.diagnostics["groups"] += 1
            cropped = self.restoration.mask(members)
            if work.interrupted:
                return
            if cropped is None:
                self.diagnostics["source_areas_bounded"] += 1
                continue
            box, own = cropped
            parent = document.ancestry(ids[0])[-2]
            selected = [child for child in parent.children if child.id in ids]
            survivor = selected[-1].id
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
            if work.interrupted:
                return
            if nesting is None:
                self.diagnostics["nested_exclusions"] += 1
                continue
            loops = _loops(nesting.mask)
            if len(loops) != 1:
                self.diagnostics["topology_exclusions"] += 1
                continue
            if len(loops[0]) > MAX_PERIMETER:
                self.diagnostics["perimeter_bounded"] += 1
                continue
            points = np.array([*loops[0], loops[0][0]], dtype=float)
            points += (box.x, box.y)
            points = points / self.evidence.scale + self.evidence.offset
            if work.interrupted:
                return
            compact = ellipse(points, self.options.boundary_tolerance)
            models = [] if compact is None else [compact]
            models.append(
                Model(
                    cel._contour(points, self.options.boundary_tolerance),
                    "contour",
                    None,
                )
            )
            original = union_geometry(
                [document.geometry_for(p.id) for p in selected],
                [path_style(document, p) for p in selected],
            )
            if (not nesting.ids and len(original.subpaths) != 1) or not all(
                s.closed for s in original.subpaths
            ):
                self.diagnostics["topology_exclusions"] += 1
                continue
            samples = self.families.samples(members, work)
            if samples is None or work.interrupted:
                return
            xy, rgba, size = samples
            if not size:
                continue
            paint = fit_samples(xy, rgba, gradients=self.options.gradients)
            paints = [paint]
            if paint.gradient:
                paints.append(fit_samples(xy, rgba, gradients=False))
            opacity = _opacity(document, survivor)
            if opacity <= 0:
                continue
            inverse = inverse_matrix(root_matrix(document, survivor))
            for model in models:
                if work.interrupted:
                    return
                shape = identified(
                    transformed_geometry(Geometry("closed", (model.contour,)), inverse),
                    survivor,
                )
                if crossings(shape):
                    self.diagnostics["topology_exclusions"] += 1
                    continue
                if not nesting.contains(shape, work):
                    self.diagnostics["nested_mark_outside"] += 1
                    continue
                restored = self.restoration.restorations(
                    state,
                    members,
                    original,
                    box,
                    own,
                    survivor,
                    work,
                    coverage=shape,
                    require_core=self.evidence.opacity is not None,
                    continue_neighbors=True,
                    ignored_neighbors=nesting.members,
                )
                if work.interrupted:
                    return
                if restored is None:
                    self.diagnostics["no_restoration"] += 1
                    continue
                continuations, neighbors = restored
                changed = partition.replace(
                    ids, (Surface(survivor, members, "overlay"),)
                )
                continued_ids = {e.id for e, _ in continuations}
                changed = Partition(
                    tuple(
                        replace(s, covered=tuple(sorted(set(s.covered) | set(members))))
                        if s.id in continued_ids
                        else replace(s, covered=nesting.members)
                        if s.id == survivor
                        else s
                        for s in changed.surfaces
                    )
                )
                for paint in paints:
                    if work.interrupted:
                        return
                    editor = Editor(document, selection=Selection(whole_document=True))
                    with editor.transaction(
                        "Propose compact closed overlay"
                    ) as transaction:
                        transaction.delete_objects(frozenset(set(ids) - {survivor}))
                        transaction.replace_geometry(survivor, shape)
                        transaction.set_fill(
                            survivor,
                            _gradient(paint, self.evidence, document, survivor, opacity)
                            if paint.gradient
                            else paint.color,
                        )
                        transaction.set_attributes(
                            survivor,
                            {
                                "fill-rule": "nonzero",
                                "stroke": "none",
                                "fill-opacity": "1"
                                if paint.gradient
                                else repr(min(1.0, paint.opacity / opacity)),
                            },
                        )
                        for element, geometry in continuations:
                            transaction.replace_geometry(element.id, geometry)
                            transaction.set_attributes(
                                element.id, {"fill-rule": "nonzero"}
                            )
                    proposed = editor.snapshot.document
                    footprint = union_geometry(
                        [original, shape], [{"fill-rule": "nonzero"}] * 2
                    )
                    proposed = ordered(
                        proposed,
                        survivor,
                        continued_ids,
                        nesting.ids,
                        footprint,
                        work,
                        self.diagnostics,
                    )
                    if work.interrupted:
                        return
                    if proposed is None:
                        self.diagnostics["order_exclusions"] += 1
                        continue
                    self.diagnostics[f"{model.kind}_proposals"] += 1
                    edited = (*ids, *(e.id for e, _ in continuations), *nesting.ids)
                    self.diagnostics["nested_proposals"] += int(bool(nesting.ids))
                    holds = set(state.details.get("geometry_constraints", ())) - set(
                        ids
                    )
                    holds.update(e.id for e, _ in continuations)
                    holds.add(survivor)
                    yield Proposal(
                        "closed-overlay",
                        edited,
                        (model.kind, "gradient" if paint.gradient else "flat", members),
                        state.key,
                        proposed,
                        bounds(document, proposed, edited),
                        details={
                            "regions": sum(
                                s.role != "underlay" for s in changed.surfaces
                            ),
                            "geometry_constraints": sorted(holds),
                            "chain_constraints": discard(
                                state.details.get("chain_constraints"), edited
                            ),
                            "closed_overlay": {
                                "model": model.kind,
                                "residual": model.residual,
                                "removed_paths": len(ids) - 1,
                                "continued_neighbors": len(continuations),
                                "source_members": members,
                                "nested_marks": nesting.ids,
                                "covered_members": nesting.members,
                            },
                        },
                        dependencies=(parent.id, *neighbors),
                        partition=changed,
                    )
