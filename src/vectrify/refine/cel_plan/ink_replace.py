"""Replace owned ink fragments with filled marks or supported strokes.

Stroke proposals restore locally adjoining paint inside the old ink footprint.
An opaque material, or an existing isolated opacity core, is required for that
overlap. Exact local scoring and independent full checkpoints choose edits.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import replace
from typing import cast

import numpy as np
import pathops
from scipy.ndimage import distance_transform_edt, find_objects, gaussian_filter

from vectrify.document import Editor, Element, Geometry, PathNode, Selection, Subpath
from vectrify.document.hit_test import multiply
from vectrify.document.join import (
    curve_path,
    path_geometry,
    path_style,
    transformed_geometry,
    union_geometry,
)
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine import cel
from vectrify.refine.cel_plan.constraints import discard
from vectrify.refine.cel_plan.families import Families, _opacity
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.ink import measure
from vectrify.refine.cel_plan.ink_contours import rim
from vectrify.refine.cel_plan.ink_underpaint import compact as compact_underpaint
from vectrify.refine.cel_plan.layer_order import ordered, ordered_surfaces
from vectrify.refine.cel_plan.local import MAX_CROP_PIXELS, Box
from vectrify.refine.cel_plan.model import Evidence, Graph, Options, Work
from vectrify.refine.cel_plan.nested import opaque_fill
from vectrify.refine.cel_plan.opacity import fit_samples
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.search import Proposal, State
from vectrify.refine.crossings import crossings
from vectrify.refine.tracing import _loops

MAX_GROUPS = 16
MAX_PATHS = 128
MAX_NODES = 6_000
MAX_NEIGHBORS = 4
MAX_CONTINUATION_NEIGHBORS = 64
MAX_RUNS = 16


def identified(geometry: Geometry, oid: str) -> Geometry:
    return replace(
        geometry,
        id=f"{oid}-geometry",
        subpaths=tuple(
            replace(
                sub,
                id=f"{oid}-contour-{i}",
                nodes=tuple(
                    replace(node, id=f"{oid}-node-{i}-{j}")
                    for j, node in enumerate(sub.nodes)
                ),
            )
            for i, sub in enumerate(geometry.subpaths)
        ),
    )


class InkReplacement:
    def __init__(self, evidence: Evidence, graph: Graph, options: Options):
        self.evidence, self.graph, self.options = evidence, graph, options
        self.boxes = find_objects(graph.labels + 1, max_label=len(graph.regions))
        self.samples = Families(evidence, graph, options)
        self.ink_pixels = np.bincount(
            graph.labels.ravel(),
            weights=evidence.drawn.ravel(),
            minlength=len(graph.regions),
        )
        self.light: np.ndarray | None = None
        self.restoration_rejections: dict[str, int] = {}
        self.diagnostics = {
            "eligible_paths_peak": 0,
            "groups": 0,
            "source_areas_bounded": 0,
            "no_ridge": 0,
            "variable_width": 0,
            "stroke_without_restoration": 0,
            "stroke_proposals": 0,
            "filled_proposals": 0,
            "time_bounded": 0,
            "restoration_neighbors_peak": 0,
            "rim_topology_exclusions": 0,
            "rim_perimeter_limits": 0,
            "rim_no_compaction": 0,
            "rim_without_restoration": 0,
            "rim_order_exclusions": 0,
            "rim_proposals": 0,
            "rim_compact_underpaints": 0,
            "rim_explicit_width_exclusions": 0,
            "order_proofs": 0,
            "order_proof_limits": 0,
        }

    def groups(self, state: State, work: Work):
        partition = state.partition
        if partition is None:
            return
        eligible, nodes, parents, colors, components = {}, {}, {}, {}, {}
        for surface in partition.surfaces:
            if work.interrupted:
                return
            if surface.role != "surface" or surface.covered:
                continue
            regions = [self.graph.regions[i] for i in surface.members]
            size = sum(r.area for r in regions)
            if (
                not size
                or len({r.component for r in regions}) != 1
                or sum(self.ink_pixels[r.id] for r in regions) < size * 0.6
            ):
                continue
            color = np.average(
                [r.paint for r in regions], axis=0, weights=[r.area for r in regions]
            )
            if float(cel.lightness(color)) > 160:
                continue
            element = state.document.element(surface.id)
            style = path_style(state.document, element)
            count = sum(
                len(s.nodes) for s in state.document.geometry_for(surface.id).subpaths
            )
            if (
                style["stroke"] != "none"
                or float(style["opacity"]) != 1
                or element.get("clip-path", "none") != "none"
                or count > MAX_NODES
            ):
                continue
            eligible[surface.id] = surface
            nodes[surface.id] = count
            parents[surface.id] = state.document.ancestry(surface.id)[-2].id
            colors[surface.id] = color
            components[surface.id] = regions[0].component
        self.diagnostics["eligible_paths_peak"] = max(
            self.diagnostics["eligible_paths_peak"], len(eligible)
        )
        neighbors = {oid: set() for oid in eligible}
        owners = partition.owners
        for edge in self.graph.boundaries:
            if work.interrupted:
                return
            a, b = owners.get(edge.left), owners.get(edge.right)
            if a not in eligible or b not in eligible or a == b:
                continue
            if (
                parents[a] == parents[b]
                and components[a] == components[b]
                and state.document.element(a).get("transform")
                == state.document.element(b).get("transform")
                and np.linalg.norm(colors[a] - colors[b]) <= 32
            ):
                neighbors[a].add(b)
                neighbors[b].add(a)
        seen = set()
        emitted = 0
        for seed in sorted(eligible, key=lambda oid: (-nodes[oid], oid)):
            if seed in seen or work.interrupted:
                continue
            queue = deque([seed])
            ids, count = [], 0
            while queue and len(ids) < MAX_PATHS:
                if work.interrupted:
                    return
                oid = queue.popleft()
                if oid in seen:
                    continue
                if (
                    count + nodes[oid] > MAX_NODES
                    or np.linalg.norm(colors[oid] - colors[seed]) > 32
                ):
                    continue
                seen.add(oid)
                ids.append(oid)
                count += nodes[oid]
                queue.extend(sorted(neighbors[oid] - seen))
            if len(ids) < 2:
                continue
            members = tuple(sorted(i for oid in ids for i in eligible[oid].members))
            yield tuple(sorted(ids)), members
            emitted += 1
            if emitted >= MAX_GROUPS:
                return

    def mask(self, members):
        boxes = [self.boxes[i] for i in members if self.boxes[i] is not None]
        box = Box(
            min(b[1].start for b in boxes),
            min(b[0].start for b in boxes),
            max(b[1].stop for b in boxes),
            max(b[0].stop for b in boxes),
        ).expand(32, self.graph.labels.shape)
        if box.area > MAX_CROP_PIXELS:
            return None
        own = (
            np.isin(self.graph.labels[box.slices], members)
            & ~self.evidence.empty[box.slices]
        )
        return box, own

    def proof(self, box, own, work):
        depth = np.asarray(distance_transform_edt(own))
        skeleton = cel.thin(own)
        if work.interrupted or not skeleton.any():
            return None
        runs = cel.line_runs(skeleton, spur=0, depth=depth)
        if not runs or len(runs) > MAX_RUNS:
            return None
        widths = 2 * depth[skeleton]
        ratio = float(np.percentile(widths, 90) / max(np.percentile(widths, 10), 0.5))
        typical = max(0.8, float(np.median(widths)))
        proofs = []
        if self.light is None:
            self.light = gaussian_filter(cel.lightness(self.evidence.target), 0.5)
        for run in runs:
            if work.interrupted:
                return None
            points = run + np.array((box.x, box.y))
            ink = measure(points, self.evidence.target, typical, light=self.light)
            if ink is None:
                return None
            proofs.append(ink)
        return proofs, ratio

    def restorations(
        self,
        state,
        members,
        geometry,
        box,
        own,
        survivor,
        work,
        *,
        coverage=None,
        require_core=False,
        continue_neighbors=False,
        ignored_neighbors=(),
    ):
        def reject(reason):
            self.restoration_rejections[reason] = (
                self.restoration_rejections.get(reason, 0) + 1
            )

        partition = state.partition
        assert partition is not None
        document = state.document
        parent = document.ancestry(survivor)[-2]
        selected = set(members)
        surrounding = set()
        for edge in self.graph.boundaries:
            if work.interrupted:
                return None
            if edge.left in selected and edge.right not in selected:
                surrounding.add(edge.right)
            if edge.right in selected and edge.left not in selected:
                surrounding.add(edge.left)
        surrounding.difference_update(ignored_neighbors)
        if -1 in surrounding or surrounding.intersection(self.graph.hidden):
            return reject("silhouette-or-hole-contact")
        if surrounding - partition.owners.keys():
            return reject("missing-neighbor-ownership")
        neighbors = sorted(
            {partition.owners[i] for i in surrounding if i in partition.owners}
        )
        self.diagnostics["restoration_neighbors_peak"] = max(
            self.diagnostics["restoration_neighbors_peak"], len(neighbors)
        )
        limit = MAX_CONTINUATION_NEIGHBORS if continue_neighbors else MAX_NEIGHBORS
        if not 1 <= len(neighbors) <= limit:
            return reject("neighbor-count")
        if (
            continue_neighbors
            and sum(
                len(sub.nodes)
                for oid in neighbors
                for sub in document.geometry_for(oid).subpaths
            )
            > MAX_NODES
        ):
            return reject("neighbor-node-limit")
        primary = {s.id: s for s in partition.surfaces if s.role != "underlay"}
        original = curve_path(geometry, "nonzero")
        required = original
        if coverage is not None:
            required = pathops.op(
                original, curve_path(coverage, "nonzero"), pathops.PathOp.UNION
            )
        inverse = inverse_matrix(root_matrix(document, survivor))
        covered = False
        for surface in partition.surfaces:
            if work.interrupted:
                return None
            if (
                not selected.issubset(
                    surface.members if surface.role == "underlay" else surface.covered
                )
                or document.ancestry(surface.id)[-2].id != parent.id
            ):
                continue
            base = document.element(surface.id)
            style = path_style(document, base)
            if (
                float(style["fill-opacity"]) != 1
                or float(style["opacity"]) != 1
                or not opaque_fill(document, style["fill"])
            ):
                continue
            # Membership alone does not prove that a partial opacity core covers
            # this mark. Verify its complete old footprint in the same frame.
            shape = transformed_geometry(
                document.geometry_for(surface.id),
                multiply(inverse, root_matrix(document, surface.id)),
            )
            outside = pathops.op(
                required,
                curve_path(shape, style["fill-rule"]),
                pathops.PathOp.DIFFERENCE,
            )
            if not list(outside):
                covered = True
                break
        if require_core and not covered:
            return reject("unproved-core-coverage")
        for oid in neighbors:
            element = document.element(oid)
            style = path_style(document, element)
            if any(a.locks for a in document.ancestry(oid)) or any(
                node.pinned
                for sub in document.geometry_for(oid).subpaths
                for node in sub.nodes
            ):
                return reject("protected-neighbor")
            if (
                primary[oid].role != "surface"
                or document.ancestry(oid)[-2].id != parent.id
                or element.get("transform")
                != document.element(survivor).get("transform")
                or style["stroke"] != "none"
                or float(style["opacity"]) != 1
                or element.get("clip-path", "none") != "none"
                or (not covered and float(style["fill-opacity"]) != 1)
            ):
                return reject("neighbor-style-or-frame")
        # Overlap is also safe inside an isolated uniform material whose
        # children are opaque. Otherwise adjacent translucency would double.
        material_opacity = _opacity(document, survivor)
        if (
            not covered
            and self.evidence.opacity is not None
            and any(
                abs(self.graph.regions[i].opacity - material_opacity) > 0.5 / 255
                for i in (*members, *surrounding)
            )
        ):
            return reject("uncovered-variable-alpha")
        labels = self.graph.labels[box.slices]
        assignment = np.zeros(labels.shape, dtype=np.int32)
        for i, oid in enumerate(neighbors, 1):
            assignment[np.isin(labels, primary[oid].members)] = i
        valid = (
            (assignment > 0)
            & ~self.evidence.drawn[box.slices]
            & ~self.evidence.empty[box.slices]
        )
        if not valid.any():
            return reject("no-neighbor-paint-samples")
        nearest = cast(
            np.ndarray,
            distance_transform_edt(~valid, return_distances=False, return_indices=True),
        )
        extended = assignment[tuple(nearest)]
        counts = np.bincount(extended[own], minlength=len(neighbors) + 1)
        dominant = int(counts.argmax())
        if not dominant:
            return reject("no-neighbor-assignment")
        sx, sy = self.evidence.scale
        ox, oy = self.evidence.offset
        a, b, c, d, e, f = inverse
        matrix = (
            a / sx,
            b / sx,
            c / sy,
            d / sy,
            a * ox + c * oy + e,
            b * ox + d * oy + f,
        )
        result = []
        node_count = 0
        order = [
            dominant,
            *(i for i in range(1, len(neighbors) + 1) if i != dominant and counts[i]),
        ]
        for index in order:
            if work.interrupted:
                return None
            oid = f"{survivor}-under-{index}"
            if index == dominant and not continue_neighbors:
                shape = geometry
            else:
                contours = []
                support = (extended == index) & (True if continue_neighbors else own)
                for i, loop in enumerate(_loops(support)):
                    points = cel.simplify(np.array([*loop, loop[0]]), 0)[:-1]
                    contours.append(
                        Subpath(
                            f"s{i}",
                            tuple(
                                PathNode(
                                    f"n{i}-{j}",
                                    "M" if j == 0 else "L",
                                    (float(p[0] + box.x), float(p[1] + box.y)),
                                )
                                for j, p in enumerate(points)
                            ),
                            True,
                        )
                    )
                if not contours:
                    continue
                shape = transformed_geometry(
                    Geometry("continuation", tuple(contours)), matrix
                )
                shape = path_geometry(
                    pathops.op(
                        curve_path(shape, "evenodd"),
                        original,
                        pathops.PathOp.INTERSECTION,
                    )
                )
            if continue_neighbors:
                oid = neighbors[index - 1]
                shape = union_geometry(
                    [document.geometry_for(oid), shape],
                    [
                        path_style(document, document.element(oid)),
                        {"fill-rule": "nonzero"},
                    ],
                )
            shape = identified(shape, oid)
            node_count += sum(len(s.nodes) for s in shape.subpaths)
            if node_count > MAX_NODES:
                return reject("continuation-node-limit")
            attrs = path_style(document, document.element(neighbors[index - 1]))
            attrs.update({"stroke": "none", "fill-rule": "nonzero"})
            transform = document.element(survivor).get("transform")
            if transform:
                attrs["transform"] = transform
            result.append(
                (
                    Element(oid, "path", tuple(attrs.items()), geometry_id=shape.id),
                    shape,
                )
            )
        return result, neighbors

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

    def rims(
        self,
        state,
        ids,
        members,
        original,
        box,
        own,
        survivor,
        paint,
        inks,
        ratio,
        work,
    ):
        from vectrify.refine.cel_plan.proposals import bounds

        if (
            self.options.line_width
            or self.evidence.filled_line_width
            or any(self.graph.regions[i].fixed for i in members)
        ):
            self.diagnostics["rim_explicit_width_exclusions"] += 1
            return
        document = state.document
        compact = rim(
            document,
            survivor,
            self.evidence,
            box,
            own,
            original,
            self.options.boundary_tolerance,
            work,
            self.diagnostics,
        )
        if compact is None or work.interrupted:
            return
        shape, models = compact
        shape = identified(shape, survivor)
        restored = self.restorations(
            state,
            members,
            original,
            box,
            own,
            survivor,
            work,
            coverage=shape,
            require_core=True,
            continue_neighbors=True,
        )
        if work.interrupted:
            return
        if restored is None:
            self.diagnostics["rim_without_restoration"] += 1
            return
        continuations, neighbors = restored
        compact = compact_underpaint(
            state,
            self.evidence,
            self.graph,
            ids,
            members,
            own,
            box,
            original,
            shape,
            survivor,
            neighbors,
            self.options.boundary_tolerance,
            work,
            self.diagnostics,
        )
        if work.interrupted:
            return
        hidden = {element.id: members for element, _ in continuations}
        underpaint_model = "traced"
        intrinsic_opacity = min(1.0, paint.opacity / _opacity(document, survivor))
        if compact is not None:
            continuations, hidden, underpaint_model = compact
            continuations = [(e, identified(g, e.id)) for e, g in continuations]
            # Compact underpaint requires the same intrinsically opaque ink
            # already proved for every old fragment. Group opacity stays exact.
            intrinsic_opacity = 1.0
        continued_ids = {element.id for element, _ in continuations}
        assert state.partition is not None
        changed = state.partition.replace(ids, (Surface(survivor, members, "overlay"),))
        changed = Partition(
            tuple(
                replace(s, covered=tuple(sorted(set(s.covered) | set(hidden[s.id]))))
                if s.id in continued_ids
                else s
                for s in changed.surfaces
            ),
            changed.atoms,
        )
        editor = Editor(document, selection=Selection(whole_document=True))
        with editor.transaction("Fit closed ink and continue adjacent paint") as tx:
            tx.delete_objects(frozenset(set(ids) - {survivor}))
            tx.replace_geometry(survivor, shape)
            tx.set_fill(survivor, paint.color)
            tx.set_attributes(
                survivor,
                {
                    "fill-rule": "nonzero",
                    "stroke": "none",
                    "fill-opacity": repr(intrinsic_opacity),
                },
            )
            for element, geometry in continuations:
                tx.replace_geometry(element.id, geometry)
                tx.set_attributes(element.id, {"fill-rule": "nonzero"})
        footprint = union_geometry([original, shape], [{"fill-rule": "nonzero"}] * 2)
        if compact is not None:
            outer, inner = (element.id for element, _ in continuations)
            inner_footprint = union_geometry(
                [
                    document.geometry_for(inner),
                    editor.snapshot.document.geometry_for(inner),
                ],
                [
                    path_style(document, document.element(inner)),
                    {"fill-rule": "nonzero"},
                ],
            )
            proposed = ordered_surfaces(
                editor.snapshot.document,
                (inner, survivor),
                {outer},
                (),
                (inner_footprint, footprint),
                work,
                self.diagnostics,
            )
        else:
            proposed = ordered(
                editor.snapshot.document,
                survivor,
                continued_ids,
                (),
                footprint,
                work,
                self.diagnostics,
            )
        if work.interrupted:
            return
        if proposed is None:
            self.diagnostics["rim_order_exclusions"] += 1
            return
        self.diagnostics["rim_proposals"] += 1
        self.diagnostics["filled_proposals"] += 1
        edited = (*ids, *(element.id for element, _ in continuations))
        holds = (
            (set(state.details.get("geometry_constraints", ())) - set(ids))
            | continued_ids
            | {survivor}
        )
        yield Proposal(
            "ink-replacement",
            edited,
            (
                "filled",
                members,
                ratio,
                "source-rim",
                models,
                underpaint_model,
                hashlib.sha256(shape.path_data().encode()).hexdigest(),
            ),
            state.key,
            proposed,
            bounds(document, proposed, edited),
            details={
                "regions": sum(s.role != "underlay" for s in changed.surfaces),
                "geometry_constraints": sorted(holds),
                "chain_constraints": discard(
                    state.details.get("chain_constraints"), edited
                ),
                "paint_constraints": sorted(
                    (set(state.details.get("paint_constraints", ())) - set(ids))
                    | (
                        {survivor}
                        if set(state.details.get("paint_constraints", ())).intersection(
                            ids
                        )
                        else set()
                    )
                ),
                "ink_replacement": {
                    "model": "filled",
                    "boundary_model": "source-rim",
                    "boundary_models": models,
                    "underpaint_model": underpaint_model,
                    "intrinsic_opacity": intrinsic_opacity,
                    "removed_paths": len(ids) - 1,
                    "continued_neighbors": len(continuations),
                    "underlays": 0,
                    "width_ratio": ratio,
                    "support": min(ink.support for ink in inks),
                    "peak_gap": max(ink.peak_gap for ink in inks),
                    "original_nodes": sum(len(s.nodes) for s in original.subpaths),
                    "fitted_nodes": sum(len(s.nodes) for s in shape.subpaths),
                },
            },
            dependencies=(document.ancestry(survivor)[-2].id,),
            partition=changed,
        )

    def proposals(self, state: State, work: Work):
        from vectrify.refine.cel_plan.proposals import bounds

        partition = state.partition
        if partition is None:
            return
        for ids, members in self.groups(state, work):
            self.diagnostics["groups"] += 1
            cropped = self.mask(members)
            if work.interrupted:
                return
            if cropped is None:
                self.diagnostics["source_areas_bounded"] += 1
                continue
            box, own = cropped
            proof = self.proof(box, own, work)
            if work.interrupted:
                return
            if proof is None:
                self.diagnostics["no_ridge"] += 1
                continue
            inks, ratio = proof
            sampled = self.samples.samples(members, work)
            if sampled is None:
                return
            xy, rgba, _size = sampled
            paint = fit_samples(xy, rgba, gradients=False)
            document = state.document
            parent = document.ancestry(ids[0])[-2]
            selected = [child for child in parent.children if child.id in ids]
            survivor = selected[-1].id
            geometry = union_geometry(
                [document.geometry_for(p.id) for p in selected],
                [path_style(document, p) for p in selected],
            )
            opacity = _opacity(document, survivor)
            if work.interrupted or opacity <= 0:
                return
            yield from self.rims(
                state,
                ids,
                members,
                geometry,
                box,
                own,
                survivor,
                paint,
                inks,
                ratio,
                work,
            )
            variants = [("filled", geometry, [], (), 0.0)]
            # Width variation is a reason to retain a filled mark. A user width
            # remains a competing fixed-width interpretation under exact checks.
            if ratio <= 1.6 or self.options.line_width:
                contours = []
                widths = []
                for ink in inks:
                    points = (
                        ink.points / np.array(self.evidence.scale)
                        + self.evidence.offset
                    )
                    model = fitted(points, self.options.boundary_tolerance)
                    contours.append(model.contour)
                    widths.append(
                        ink.width / float(np.sqrt(np.prod(self.evidence.scale)))
                    )
                stroke = identified(Geometry("ink", tuple(contours)), survivor)
                width = self.options.line_width or float(np.median(widths))
                restored = self.restorations(
                    state, members, geometry, box, own, survivor, work
                )
                if restored is not None and not crossings(stroke):
                    underlays, neighbors = restored
                    variants.insert(
                        0, ("stroke", stroke, underlays, tuple(neighbors), width)
                    )
                else:
                    self.diagnostics["stroke_without_restoration"] += 1
            else:
                self.diagnostics["variable_width"] += 1
            for kind, shape, underlays, dependencies, width in variants:
                if work.interrupted:
                    return
                changed = partition.replace(
                    ids, (Surface(survivor, members, "overlay"),)
                )
                changed = Partition(
                    (
                        *changed.surfaces,
                        *(
                            Surface(element.id, members, "underlay")
                            for element, _ in underlays
                        ),
                    ),
                    changed.atoms,
                )
                editor = Editor(document, selection=Selection(whole_document=True))
                with editor.transaction(
                    "Propose planned ink replacement"
                ) as transaction:
                    transaction.delete_objects(frozenset(set(ids) - {survivor}))
                    transaction.replace_geometry(survivor, shape)
                    transaction.set_fill(
                        survivor, "none" if kind == "stroke" else paint.color
                    )
                    attributes: dict[str, str | None] = {
                        "fill-rule": "nonzero",
                        "stroke": "none",
                        "fill-opacity": repr(min(1.0, paint.opacity / opacity)),
                    }
                    if kind == "stroke":
                        # Centerlines and widths use native pixels. Map the
                        # whole stroke into its parent's coordinate frame.
                        inverse_parent = inverse_matrix(
                            root_matrix(document, parent.id)
                        )
                        matrix_text = " ".join(str(v) for v in inverse_parent)
                        attributes.update(
                            {
                                "stroke": paint.color,
                                "stroke-width": repr(width),
                                "stroke-opacity": repr(
                                    min(1.0, paint.opacity / opacity)
                                ),
                                "stroke-linecap": "round",
                                "stroke-linejoin": "round",
                                "transform": f"matrix({matrix_text})",
                            }
                        )
                    transaction.set_attributes(survivor, attributes)
                    remaining = [
                        child
                        for child in parent.children
                        if child.id not in set(ids) - {survivor}
                    ]
                    position = next(
                        i for i, child in enumerate(remaining) if child.id == survivor
                    )
                    for element, shape in underlays:
                        transaction.insert_object(
                            parent.id, element, index=position, geometries=(shape,)
                        )
                        position += 1
                proposed = editor.snapshot.document
                self.diagnostics[f"{kind}_proposals"] += 1
                edited = (*ids, *(element.id for element, _ in underlays))
                holds = set(state.details.get("geometry_constraints", ())) - set(ids)
                holds.update(element.id for element, _ in underlays)
                holds.add(survivor)
                yield Proposal(
                    "ink-replacement",
                    edited,
                    (kind, members, ratio),
                    state.key,
                    proposed,
                    bounds(document, proposed, edited),
                    details={
                        "regions": sum(s.role != "underlay" for s in changed.surfaces),
                        "geometry_constraints": sorted(holds),
                        "chain_constraints": discard(
                            state.details.get("chain_constraints"), ids
                        ),
                        "paint_constraints": sorted(
                            (set(state.details.get("paint_constraints", ())) - set(ids))
                            | (
                                {survivor}
                                if set(
                                    state.details.get("paint_constraints", ())
                                ).intersection(ids)
                                else set()
                            )
                        ),
                        "ink_replacement": {
                            "model": kind,
                            "removed_paths": len(ids) - 1,
                            "underlays": len(underlays),
                            "width_ratio": ratio,
                            "support": min(ink.support for ink in inks),
                            "peak_gap": max(ink.peak_gap for ink in inks),
                        },
                    },
                    dependencies=(parent.id, *dependencies),
                    partition=changed,
                )
