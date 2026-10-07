"""Closed source ridges cross palette boundaries through exact ownership cuts.

Only existing drawn pixels can enter a ridge band. Complete two-sided ridge
support, an actual opacity core and the ordinary native evaluator are required.
Mixed owners keep their unselected geometry and source atoms; the fitted rim
replaces their selected fragments in one atomic proposal.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
import pathops
from scipy.ndimage import binary_fill_holes, distance_transform_edt, find_objects, label

from vectrify.document import Editor, Geometry, PathNode, Selection, Subpath
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
from vectrify.refine.cel_plan.atoms import MAX_PIXELS, Atoms
from vectrify.refine.cel_plan.constraints import discard
from vectrify.refine.cel_plan.geometry import ellipse
from vectrify.refine.cel_plan.ink_replace import (
    MAX_NODES,
    MAX_PATHS,
    InkReplacement,
    identified,
)
from vectrify.refine.cel_plan.local import MAX_CROP_PIXELS, Box
from vectrify.refine.cel_plan.model import (
    Evidence,
    Graph,
    Options,
    StageInterruptedError,
    Work,
)
from vectrify.refine.cel_plan.opacity import fit_samples
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.search import Proposal, State
from vectrify.refine.tracing import _loops

MAX_CAVITIES = 32
MAX_CAVITY_COMPONENTS = 8192
MAX_BANDS = 8
MAX_RADIUS = 12
MAX_CAVITY_AREA = 8192
MIN_CAVITY_AREA = 32


class GraphBranch(Protocol):
    graph: Graph


@dataclass(frozen=True)
class RidgeCut:
    state: State
    evidence: Evidence
    graph: Graph
    ids: tuple[str, ...]
    members: tuple[int, ...]
    before: tuple[str, ...]
    branch: GraphBranch | None = None

    def __iter__(self):
        yield from (
            self.state,
            self.evidence,
            self.graph,
            self.ids,
            self.members,
            self.before,
        )


def mask_geometry(own, box, evidence, document, oid):
    contours = []
    for i, loop in enumerate(_loops(own)):
        points = cel.simplify(np.array([*loop, loop[0]]), 0)[:-1]
        contours.append(
            Subpath(
                f"ridge-{i}",
                tuple(
                    PathNode(
                        f"ridge-{i}-{j}",
                        "M" if j == 0 else "L",
                        (float(p[0] + box.x), float(p[1] + box.y)),
                    )
                    for j, p in enumerate(points)
                ),
                True,
            )
        )
    sx, sy = evidence.scale
    ox, oy = evidence.offset
    a, b, c, d, e, f = inverse_matrix(root_matrix(document, oid))
    return transformed_geometry(
        Geometry("ridge-mask", tuple(contours)),
        (a / sx, b / sx, c / sy, d / sy, a * ox + c * oy + e, b * ox + d * oy + f),
    )


def cut_geometry(document, oid, members, graph, evidence, classification, clip, work):
    """Keep complete disconnected contour groups exact on either side.

    Pixel-grid clipping a fully selected curved fragment leaves meaningless
    coverage slivers. A group can move whole only when ALL its source-owner
    samples agree. Nested or intersecting contour boxes stay together.
    """
    original = document.geometry_for(oid)
    groups = []
    for sub in original.subpaths:
        if work.interrupted:
            raise StageInterruptedError("Ridge contour cutting interrupted")
        shape = Geometry("part", (sub,))
        box = curve_path(shape).bounds
        joined = [
            g
            for g in groups
            if not (
                box[2] < g[0][0]
                or g[0][2] < box[0]
                or box[3] < g[0][1]
                or g[0][3] < box[1]
            )
        ]
        subs = [sub]
        for bounds, contours in joined:
            groups.remove((bounds, contours))
            box = (
                min(box[0], bounds[0]),
                min(box[1], bounds[1]),
                max(box[2], bounds[2]),
                max(box[3], bounds[3]),
            )
            subs.extend(contours)
        # A newly enlarged box can encompass a previously separate group.
        while True:
            more = [
                g
                for g in groups
                if not (
                    box[2] < g[0][0]
                    or g[0][2] < box[0]
                    or box[3] < g[0][1]
                    or g[0][3] < box[1]
                )
            ]
            if not more:
                break
            for bounds, contours in more:
                groups.remove((bounds, contours))
                box = (
                    min(box[0], bounds[0]),
                    min(box[1], bounds[1]),
                    max(box[2], bounds[2]),
                    max(box[3], bounds[3]),
                )
                subs.extend(contours)
        groups.append((box, subs))
    selected, retained = [], []
    rule = path_style(document, document.element(oid))["fill-rule"]
    for _box, contours in groups:
        if work.interrupted:
            raise StageInterruptedError("Ridge contour cutting interrupted")
        group = Geometry("part", tuple(contours))
        native = transformed_geometry(group, root_matrix(document, oid))
        bounds = np.array(curve_path(native, rule).bounds).reshape(2, 2)
        bounds = (bounds - evidence.offset) * evidence.scale
        # Include the complete source samples in this contour group's bounds,
        # with a conservative one-pixel margin for fitted exterior coverage.
        a, b = np.floor(bounds[0]).astype(int) - 1
        c, d = np.ceil(bounds[1]).astype(int) + 1
        roi = Box(
            max(0, a),
            max(0, b),
            min(graph.labels.shape[1], c),
            min(graph.labels.shape[0], d),
        )
        values = np.array([], bool)
        if 0 < roi.area <= MAX_CROP_PIXELS:
            own = np.isin(graph.labels[roi.slices], members)
            values = classification[roi.slices][own]
        if values.size and values.all():
            selected.extend(contours)
        elif values.size and not values.any():
            retained.extend(contours)
        else:
            shape = curve_path(group, rule)
            selected.extend(
                path_geometry(
                    pathops.op(shape, clip, pathops.PathOp.INTERSECTION)
                ).subpaths
            )
            retained.extend(
                path_geometry(
                    pathops.op(shape, clip, pathops.PathOp.DIFFERENCE)
                ).subpaths
            )
    return Geometry("ink-cut", tuple(selected)), Geometry(
        "material-cut", tuple(retained)
    )


class SourceRidges:
    def __init__(
        self,
        evidence: Evidence,
        graph: Graph,
        options: Options,
        *,
        resolver: Callable[[Partition, Work], GraphBranch] | None = None,
    ):
        self.evidence, self.graph, self.options = evidence, graph, options
        self.resolver = resolver
        self.restoration_rejections: dict[str, int] = {}
        self.rim_diagnostics: dict[str, int] = {}
        self.underpaint_rejections: dict[str, int] = {}
        self.diagnostics = dict.fromkeys(
            (
                "cavities",
                "bands",
                "topology_exclusions",
                "ridge_exclusions",
                "owner_exclusions",
                "atom_exclusions",
                "geometry_exclusions",
                "restoration_exclusions",
                "proposals",
                "cuts",
                "time_bounded",
            ),
            0,
        )

    def bands(self, work):
        evidence = self.evidence
        if work.interrupted or evidence.drawn.size > MAX_PIXELS:
            return
        # Do not close a detected gap or treat an alpha void as an RGB cavity.
        cavities, count = label(binary_fill_holes(evidence.drawn) & ~evidence.drawn)
        if work.interrupted:
            return
        if count > MAX_CAVITY_COMPONENTS:
            self.diagnostics["cavity_component_limits"] = (
                self.diagnostics.get("cavity_component_limits", 0) + 1
            )
            return
        areas = np.bincount(cavities.ravel(), minlength=count + 1)
        boxes = find_objects(cavities, max_label=count)
        candidates = sorted(
            (
                i
                for i in range(1, count + 1)
                if MIN_CAVITY_AREA <= areas[i] <= MAX_CAVITY_AREA
            ),
            key=lambda i: (-areas[i], i),
        )[:MAX_CAVITIES]
        ranked = []
        scanned = 0
        for i in candidates:
            if work.interrupted:
                return
            slices = boxes[i - 1]
            box = Box(slices[1].start, slices[0].start, slices[1].stop, slices[0].stop)
            box = box.expand(MAX_RADIUS + 2, evidence.drawn.shape)
            if box.area > MAX_CROP_PIXELS:
                continue
            cavity = cavities[box.slices] == i
            if evidence.empty[box.slices][cavity].any():
                continue
            self.diagnostics["cavities"] += 1
            drawn = evidence.drawn[box.slices] & ~evidence.empty[box.slices]
            distance = np.asarray(distance_transform_edt(~cavity))
            depth = np.asarray(distance_transform_edt(drawn))
            adjacent = drawn & (distance <= 2)
            if not adjacent.any():
                continue
            typical = max(2.0, 2 * float(np.median(depth[adjacent])))
            radii = sorted(
                {
                    min(MAX_RADIUS, max(2, round(typical * r)))
                    for r in (0.75, 1.0, 1.5, 3.0)
                }
                | {MAX_RADIUS}
            )
            seen = set()
            for radius in radii:
                if work.interrupted:
                    return
                parts, n = label(drawn & (distance <= radius), np.ones((3, 3)))
                sizes = np.bincount(parts.ravel(), minlength=n + 1)
                sizes[0] = 0
                if not n:
                    continue
                own = parts == sizes.argmax()
                signature = hashlib.sha256(
                    repr(box).encode() + own.tobytes()
                ).hexdigest()
                if signature in seen:
                    continue
                seen.add(signature)
                loops = _loops(own)
                if (
                    len(loops) != 2
                    or sum(map(len, loops)) > 4096
                    or not np.asarray(binary_fill_holes(own), bool)[cavity].all()
                ):
                    self.diagnostics["topology_exclusions"] += 1
                    continue
                # Prioritize a complete compact primitive before spending graph
                # rebuilds on a broad palette-independent cavity scan. Curved
                # alternatives remain eligible under the same bounded pool.
                primitive_count = 0
                for loop in loops:
                    points = np.array([*loop, loop[0]], float)
                    points += (box.x, box.y)
                    points = points / evidence.scale + evidence.offset
                    primitive_count += (
                        ellipse(points, self.options.boundary_tolerance) is not None
                    )
                scanned += 1
                ranked.append(
                    (-primitive_count, -int(own.sum()), signature, box, own, radius)
                )
                ranked.sort(key=lambda r: r[:3])
                del ranked[MAX_BANDS:]
        self.diagnostics["bands_scanned"] = (
            self.diagnostics.get("bands_scanned", 0) + scanned
        )
        for _primitives, _area, signature, box, own, radius in ranked:
            if work.interrupted:
                return
            self.diagnostics["bands"] += 1
            yield box, own, radius, signature

    def prepare(self, state, box, own, work):
        """An unpublished exact cut; only the composed replacement is scored."""
        partition = state.partition
        if partition is None or work.interrupted:
            return None
        document, graph = state.document, self.graph
        owners = partition.owners
        members = tuple(int(i) for i in np.unique(graph.labels[box.slices][own]))
        if any(i not in owners or graph.regions[i].fixed for i in members):
            self.diagnostics["owner_exclusions"] += 1
            return None
        ids = tuple(sorted({owners[i] for i in members}))
        surfaces = {s.id: s for s in partition.surfaces}
        nodes = 0
        parent, frame = None, None
        for oid in ids:
            if work.interrupted:
                return None
            element = document.element(oid)
            style = path_style(document, element)
            ancestry = document.ancestry(oid)
            current = ancestry[-2].id
            matrix = tuple(root_matrix(document, oid))
            nodes += sum(len(s.nodes) for s in document.geometry_for(oid).subpaths)
            if (
                surfaces[oid].role != "surface"
                or surfaces[oid].covered
                or len(ids) > MAX_PATHS
                or nodes > MAX_NODES
                or style["stroke"] != "none"
                or float(style["opacity"]) != 1
                or element.get("clip-path", "none") != "none"
                or any(a.locks for a in ancestry)
                or any(
                    n.pinned
                    for s in document.geometry_for(oid).subpaths
                    for n in s.nodes
                )
                or (parent is not None and (parent != current or frame != matrix))
            ):
                self.diagnostics["owner_exclusions"] += 1
                return None
            parent, frame = current, matrix
        assert parent is not None
        classification = np.zeros(graph.labels.shape, bool)
        classification[box.slices] = own
        previous = partition.atoms or Atoms.original(graph)
        all_members = tuple(sorted(i for oid in ids for i in surfaces[oid].members))
        try:
            atoms, left, _right = previous.split(
                graph, all_members, classification, work
            )
        except ValueError:
            self.diagnostics["atom_exclusions"] += 1
            return None
        selected = set(left)
        clip = curve_path(
            mask_geometry(own, box, self.evidence, document, ids[0]), "evenodd"
        )
        editor = Editor(document, selection=Selection(whole_document=True))
        replacements, ink_ids = [], []
        start = len(previous.cuts)
        known = {e.id for e in document.elements()}
        with editor.transaction("Cut source ridge ownership") as tx:
            for oid in ids:
                if work.interrupted:
                    return None
                element = document.element(oid)
                descendants = atoms.descendants(surfaces[oid].members, start)
                ink_members = tuple(i for i in descendants if i in selected)
                remainder = tuple(i for i in descendants if i not in selected)
                if remainder:
                    ink, rest = cut_geometry(
                        document,
                        oid,
                        surfaces[oid].members,
                        graph,
                        self.evidence,
                        classification,
                        clip,
                        work,
                    )
                    if not ink.subpaths or not rest.subpaths:
                        self.diagnostics["geometry_exclusions"] += 1
                        return None
                    ink_id = f"{oid}-ridge-{atoms.key[:12]}"
                    if ink_id in known:
                        return None
                    ink = identified(ink, ink_id)
                    rest = identified(rest, oid)
                    tx.replace_geometry(oid, rest)
                    at = next(
                        i
                        for i, c in enumerate(document.element(parent).children)
                        if c.id == oid
                    )
                    tx.insert_object(
                        parent,
                        replace(
                            element,
                            id=ink_id,
                            geometry_id=ink.id,
                        ),
                        index=at + 1,
                        geometries=(ink,),
                    )
                    replacements.append(Surface(oid, remainder))
                else:
                    ink_id = oid
                ink_ids.append(ink_id)
                replacements.append(Surface(ink_id, ink_members))
        if work.interrupted:
            return None
        if (
            sum(
                len(s.nodes)
                for oid in set(ids) | set(ink_ids)
                for s in editor.snapshot.document.geometry_for(oid).subpaths
            )
            > 2 * MAX_NODES
        ):
            self.diagnostics["geometry_exclusions"] += 1
            return None
        refined = partition.split(ids, tuple(replacements), atoms)
        context = None
        if atoms.cuts == previous.cuts:
            # Reassigning complete atoms needs no new namespace or graph copy.
            refined = replace(refined, atoms=partition.atoms)
            branch = graph
        elif self.resolver is not None:
            try:
                context = self.resolver(refined, work)
            except ValueError:
                self.diagnostics["atom_exclusions"] += 1
                return None
            branch = context.graph
        else:
            # Rebuild against the current graph by replaying only this delta;
            # the resulting namespace remains the original immutable lineage.
            delta = Atoms.original(graph)
            delta = replace(delta, cuts=atoms.cuts[start:])
            branch = replace(
                delta.graph(self.evidence, graph, work), source_atoms=atoms.key
            )
        if work.interrupted:
            return None
        from vectrify.refine.cel_plan.proposals import MAX_BRANCH_BYTES, Operators

        if Operators._graph_bytes(branch) > MAX_BRANCH_BYTES:
            self.diagnostics["atom_exclusions"] += 1
            return None
        proposed = replace(state, document=editor.snapshot.document, partition=refined)
        return RidgeCut(
            proposed,
            replace(self.evidence, labels=branch.labels),
            branch,
            tuple(ink_ids),
            left,
            ids,
            context,
        )

    def record(self, factory):
        for key, value in factory.diagnostics.items():
            if isinstance(value, int):
                self.rim_diagnostics[key] = self.rim_diagnostics.get(key, 0) + value
        for source, target in (
            (factory.restoration_rejections, self.restoration_rejections),
            (
                factory.diagnostics.get("rim_underpaint_exclusions", {}),
                self.underpaint_rejections,
            ),
        ):
            for key, value in source.items():
                target[key] = target.get(key, 0) + value

    def proposals(self, state, work):
        if (
            state.partition is None
            or self.options.line_width
            or self.evidence.filled_line_width
        ):
            return
        from vectrify.refine.cel_plan.proposals import bounds

        for box, own, radius, signature in self.bands(work):
            if work.interrupted:
                return
            proof_factory = InkReplacement(self.evidence, self.graph, self.options)
            proof = proof_factory.proof(box, own, work)
            if proof is None:
                self.diagnostics["ridge_exclusions"] += 1
                continue
            try:
                prepared = self.prepare(state, box, own, work)
            except StageInterruptedError:
                return
            if prepared is None or work.interrupted:
                continue
            current, evidence, graph = prepared.state, prepared.evidence, prepared.graph
            ink_ids, members, old_ids = prepared.ids, prepared.members, prepared.before
            factory = InkReplacement(evidence, graph, self.options)
            sampled = factory.samples.samples(members, work)
            if sampled is None:
                return
            xy, rgba, _size = sampled
            paint = fit_samples(xy, rgba, gradients=False)
            parent = current.document.ancestry(ink_ids[0])[-2]
            parts = [p for p in parent.children if p.id in ink_ids]
            original = union_geometry(
                [current.document.geometry_for(p.id) for p in parts],
                [path_style(current.document, p) for p in parts],
            )
            emitted = False
            try:
                for edit in factory.rims(
                    current,
                    ink_ids,
                    members,
                    original,
                    box,
                    own,
                    parts[-1].id,
                    paint,
                    *proof,
                    work,
                ):
                    if work.interrupted:
                        return
                    assert edit.partition is not None
                    cuts = (
                        len(edit.partition.atoms.cuts) if edit.partition.atoms else 0
                    ) - (
                        len(state.partition.atoms.cuts) if state.partition.atoms else 0
                    )
                    existing = {e.id for e in state.document.elements()} | {
                        e.id for e in edit.document.elements()
                    }
                    changed = tuple(sorted(set(old_ids) | (set(edit.ids) & existing)))
                    details = {
                        **(edit.details or {}),
                        "chain_constraints": discard(
                            state.details.get("chain_constraints"), changed
                        ),
                        "source_ridge": {
                            "radius": radius,
                            "mask": signature,
                            "pixels": int(own.sum()),
                            "owners": len(old_ids),
                            "cuts": cuts,
                        },
                    }
                    self.diagnostics["proposals"] += 1
                    self.diagnostics["cuts"] += cuts
                    emitted = True
                    yield Proposal(
                        "source-ridge",
                        changed,
                        (signature, radius, *edit.parameters),
                        state.key,
                        edit.document,
                        bounds(state.document, edit.document, changed),
                        details=details,
                        dependencies=edit.dependencies,
                        partition=edit.partition,
                    )
            finally:
                if not emitted:
                    self.diagnostics["restoration_exclusions"] += 1
                self.record(factory)

    def __call__(self, state, work):
        bounded = Work(
            min(work.deadline, time.monotonic() + work.remaining * 0.25),
            work.stop,
            work.timings,
        )
        try:
            yield from self.proposals(state, bounded)
        finally:
            self.diagnostics["time_bounded"] += int(bounded.interrupted)
