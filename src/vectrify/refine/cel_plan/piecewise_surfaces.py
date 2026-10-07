"""Joint whole-family contour, straight shade partition and side-paint proposals.

A connected family need not first survive a single-paint intermediate edit.
Complete source support competes against two models from the outset. Exact
unions and contained fitted exteriors compete; only native scores admit them.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pathops

from vectrify.document import Editor, Geometry, PathNode, Selection, Subpath
from vectrify.document.hit_test import _flatten
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
from vectrify.refine.cel_plan.atoms import Atoms
from vectrify.refine.cel_plan.constraints import discard, merged
from vectrify.refine.cel_plan.families import _gradient, _opacity
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.local import Box
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.nested import in_core
from vectrify.refine.cel_plan.ownership import Surface
from vectrify.refine.cel_plan.search import Proposal
from vectrify.refine.cel_plan.surface_models import prediction
from vectrify.refine.cel_plan.surface_splits import SurfaceSplits, fit, lines
from vectrify.refine.colour_regions import simplified_indices
from vectrify.refine.crossings import crossings

MAX_SEEDS = 8
MAX_CONTACTS = 65_536
MAX_PATHS = 128
MAX_MEMBERS = 512
MAX_NODES = 6000
MAX_PIXELS = 262_144
MAX_ANALYSIS = 1536**2
MAX_POINTS = 16_384
MAX_PROPOSALS = 16
CHUNK_PIXELS = 65_536
RESIDUAL = 48.0


def corner_vertices(points, lengths, uniform, at, work):
    """Snap arc-supported turns to original vertices using stable tangent rays.

    The nearest vertex to a classified raster turn may be a neighboring ripple.
    Two local tangent fits supply the junction hypothesis; no new anchor is
    inserted and no corner may move during line simplification.
    """
    result = []
    count = len(uniform) - 1
    span = cel.CORNER_SPAN
    for index in cel.run_corners(uniform, True):
        if work.interrupted:
            return None
        upper = min(len(lengths) - 1, int(np.searchsorted(lengths, at[index])))
        lower = max(0, upper - 1)
        vertex = (
            lower if at[index] - lengths[lower] < lengths[upper] - at[index] else upper
        )
        if count >= 6 * span:
            rays = [
                uniform[(index + sign * np.arange(span, 3 * span + 1)) % count]
                for sign in (-1, 1)
            ]
            normals, centers = [], []
            for ray in rays:
                centers.append(ray.mean(axis=0))
                _, vectors = np.linalg.eigh(np.cov(ray.T))
                normals.append(vectors[:, 0])
            if abs(np.linalg.det(normals)) > 0.25:
                junction = np.linalg.solve(
                    np.asarray(normals),
                    np.array([n @ c for n, c in zip(normals, centers, strict=True)]),
                )
                if np.linalg.norm(junction - uniform[index]) <= 2 * span:
                    distance = np.abs(lengths - at[index])
                    distance = np.minimum(distance, lengths[-1] - distance)
                    nearby = np.flatnonzero(distance <= 2 * span)
                    vertex = int(
                        nearby[
                            np.argmin(np.linalg.norm(points[nearby] - junction, axis=1))
                        ]
                    )
        result.append(vertex)
    return result


def prediction_pair(paints, xy, normal, rho):
    selected = xy @ normal < rho
    predicted = np.empty((len(xy), 3))
    for side, paint in zip((selected, ~selected), paints, strict=True):
        predicted[side] = prediction(paint, xy[side], extend=True)
    return predicted


class PiecewiseSurfaces:
    def __init__(self, families, options):
        self.families, self.options = families, options
        self.splitter = SurfaceSplits(families, options)
        self.diagnostics = {
            "seed_pairs": 0,
            "lines": 0,
            "seed_exclusions": 0,
            "screened_pixels": 0,
            "group_limits": 0,
            "refit_exclusions": 0,
            "order_exclusions": 0,
            "core_exclusions": 0,
            "atom_exclusions": 0,
            "compact_fits": 0,
            "compact_limits": 0,
            "compact_area_exclusions": 0,
            "compact_cost_exclusions": 0,
            "proposals": 0,
        }

    def _samples(self, members, work):
        if work.interrupted:
            return None
        boxes = [self.families.boxes[i] for i in members]
        if len(boxes) > MAX_MEMBERS or any(b is None for b in boxes):
            self.diagnostics["group_limits"] += 1
            return None
        box = Box(
            min(b[1].start for b in boxes),
            min(b[0].start for b in boxes),
            max(b[1].stop for b in boxes),
            max(b[0].stop for b in boxes),
        )
        if box.area > MAX_PIXELS:
            self.diagnostics["group_limits"] += 1
            return None
        graph, evidence = self.families.graph, self.families.evidence
        own = np.isin(graph.labels[box.slices], members) & ~evidence.empty[box.slices]
        y, x = np.nonzero(own)
        xy = np.column_stack((x + box.x + 0.5, y + box.y + 0.5))
        rgba = np.column_stack(
            (
                evidence.target[box.slices][own] / 255,
                evidence.opacity[box.slices][own]
                if evidence.opacity is not None
                else np.ones(len(x)),
            )
        )
        if len(x) < 32:
            return None
        return box, own, xy, rgba

    def _screen(self, eligible, paints, normal, rho, work):
        """Whole-owner maximum residual; a distant atom or outlier cannot hide."""
        graph, evidence = self.families.graph, self.families.evidence
        if graph.labels.size > MAX_ANALYSIS:
            self.diagnostics["group_limits"] += 1
            return None
        ids = sorted(eligible)
        lookup = np.full(len(graph.regions), -1, np.int32)
        for index, oid in enumerate(ids):
            lookup[list(eligible[oid][0].members)] = index
        maxima = np.zeros(len(ids))
        counts = np.zeros(len(ids), np.int64)
        h, w = graph.labels.shape
        for box in Box(0, 0, w, h).chunks(CHUNK_PIXELS):
            if work.interrupted:
                return None
            owners = lookup[graph.labels[box.slices]]
            y, x = np.nonzero((owners >= 0) & ~evidence.empty[box.slices])
            if not len(x):
                continue
            xy = np.column_stack((x + box.x + 0.5, y + box.y + 0.5))
            residual = np.max(
                np.abs(
                    prediction_pair(paints, xy, normal, rho) * 255
                    - evidence.target[box.slices][y, x]
                ),
                axis=1,
            )
            np.maximum.at(maxima, owners[y, x], residual)
            counts += np.bincount(owners[y, x], minlength=len(ids))
            self.diagnostics["screened_pixels"] += len(x)
        return {
            oid
            for index, oid in enumerate(ids)
            if maxima[index] <= RESIDUAL
            and counts[index] == eligible[oid][1]
            and counts[index]
        }

    def groups(self, state, work):
        if state.partition is None:
            return
        eligible = self.families.surface_models._eligible(state, work)
        owners = state.partition.owners
        neighbors = {oid: set() for oid in eligible}
        contacts = {}
        for edge in self.families.graph.boundaries:
            if work.interrupted:
                return
            a, b = owners.get(edge.left), owners.get(edge.right)
            if (
                a not in eligible
                or b not in eligible
                or a == b
                or eligible[a][3] != eligible[b][3]
            ):
                continue
            neighbors[a].add(b)
            neighbors[b].add(a)
            pair = tuple(sorted((a, b)))
            if len(contacts) >= MAX_CONTACTS and pair not in contacts:
                self.diagnostics["group_limits"] += 1
                return
            contacts[pair] = contacts.get(pair, 0) + float(
                np.linalg.norm(np.diff(edge.points, axis=0), axis=1).sum()
            )
        pairs = sorted(
            contacts,
            key=lambda pair: (
                -sum(eligible[i][1] for i in pair),
                -contacts[pair],
                pair,
            ),
        )[:MAX_SEEDS]
        seen = set()
        for pair in pairs:
            if work.interrupted:
                return
            members = tuple(sorted(i for oid in pair for i in eligible[oid][0].members))
            samples = self._samples(members, work)
            if samples is None:
                continue
            box, own, xy, rgba = samples
            self.diagnostics["seed_pairs"] += 1
            for normal, rho in lines(
                self.families.evidence.target[box.slices], own, (box.x, box.y), work
            ):
                if work.interrupted:
                    return
                self.diagnostics["lines"] += 1
                side = xy @ normal < rho
                if min(int(side.sum()), int((~side).sum())) < 16:
                    continue
                paints = (
                    fit(xy[side], rgba[side], gradients=self.options.gradients),
                    fit(xy[~side], rgba[~side], gradients=self.options.gradients),
                )
                allowed = self._screen(eligible, paints, normal, rho, work)
                if allowed is None:
                    return
                if not set(pair).issubset(allowed):
                    self.diagnostics["seed_exclusions"] += 1
                    continue
                queue, visited, ids, node_count = deque(pair), set(), [], 0
                while queue:
                    if work.interrupted:
                        return
                    oid = queue.popleft()
                    if oid in visited:
                        continue
                    visited.add(oid)
                    if oid not in allowed:
                        continue
                    if (
                        len(ids) >= MAX_PATHS
                        or node_count + eligible[oid][2] > MAX_NODES
                    ):
                        self.diagnostics["group_limits"] += 1
                        continue
                    ids.append(oid)
                    node_count += eligible[oid][2]
                    queue.extend(sorted(neighbors[oid] - visited))
                ids = tuple(sorted(ids))
                key = (ids, tuple(normal), rho)
                if len(ids) < 3 or key in seen:
                    continue
                seen.add(key)
                members = tuple(
                    sorted(i for oid in ids for i in eligible[oid][0].members)
                )
                samples = self._samples(members, work)
                if samples is None:
                    continue
                _, _, xy, rgba = samples
                side = xy @ normal < rho
                if min(int(side.sum()), int((~side).sum())) < max(16, 0.05 * len(xy)):
                    continue
                left = self.splitter._paints(xy[side], rgba[side], work)
                right = self.splitter._paints(xy[~side], rgba[~side], work)
                if not left or not right:
                    self.diagnostics["refit_exclusions"] += 1
                    continue
                for a, _ in left:
                    for b, _ in right:
                        yield ids, members, normal, rho, (a, b)

    def _compact(self, state, oid, shape, work):
        """Fit union contours with anchored lines/curves; retain exact holes.

        Actual opaque underpaint must already cover the whole family. The
        intersection forbids expansion over another paint or a source hole.
        Unfitted tiny contours are preserved rather than vetoing the family.
        """
        evidence = self.families.evidence
        matrix = root_matrix(state.document, oid)
        sx, sy = evidence.scale
        ox, oy = evidence.offset
        native = transformed_geometry(shape, matrix)
        analysis = transformed_geometry(native, (sx, 0, 0, sy, -ox * sx, -oy * sy))
        contours = {"contained-lines": [], "contained-curves": []}
        total = 0
        for subpath in analysis.subpaths:
            if work.interrupted:
                return []
            if not subpath.closed or len(subpath.nodes) < 3:
                for values in contours.values():
                    values.append(subpath)
                continue
            points = [subpath.nodes[0].endpoint]
            for node in subpath.nodes[1:]:
                if work.interrupted:
                    return []
                if node.command == "C":
                    points.extend(
                        _flatten(
                            (
                                points[-1],
                                (node.values[0], node.values[1]),
                                (node.values[2], node.values[3]),
                                node.endpoint,
                            ),
                            0.25,
                        )
                    )
                else:
                    points.append(node.endpoint)
                if total + len(points) > MAX_POINTS:
                    self.diagnostics["compact_limits"] += 1
                    return []
            if points[-1] != points[0]:
                points.append(points[0])
            total += len(points)
            if total > MAX_POINTS:
                self.diagnostics["compact_limits"] += 1
                return []
            points = np.asarray(points)
            lengths = np.r_[
                0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            ]
            if not np.isfinite(lengths[-1]) or lengths[-1] > MAX_POINTS - total:
                self.diagnostics["compact_limits"] += 1
                return []
            keep_length = np.r_[True, np.diff(lengths) > 1e-8]
            at = np.linspace(0.0, lengths[-1], max(4, int(np.ceil(lengths[-1])) + 1))
            uniform = np.column_stack(
                [
                    np.interp(at, lengths[keep_length], points[keep_length, axis])
                    for axis in (0, 1)
                ]
            )
            total += max(0, len(uniform) - len(points))
            if total > MAX_POINTS:
                self.diagnostics["compact_limits"] += 1
                return []
            # Corner classification uses arc distance, not serialization density.
            # Snap each supported corner back to an ORIGINAL contour vertex.
            corners = corner_vertices(points, lengths, uniform, at, work)
            if corners is None:
                return []
            anchors = sorted({0, len(points) - 1, *corners})
            keep = set(anchors)
            for first, last in pairwise(anchors):
                if work.interrupted:
                    return []
                keep.update(
                    first + i
                    for i in simplified_indices(
                        points[first : last + 1], self.options.boundary_tolerance
                    )
                )
            nodes = tuple(
                PathNode("fit", "M" if i == 0 else "L", tuple(points[k]))
                for i, k in enumerate(sorted(keep)[:-1])
            )
            if len(nodes) < 3:
                contours["contained-lines"].append(subpath)
            else:
                contours["contained-lines"].append(Subpath("fit", nodes, True))
            if not corners:
                curve = fitted(uniform, self.options.boundary_tolerance).contour
            else:
                curve_nodes = [PathNode("fit", "M", tuple(points[0]))]
                for first, last in pairwise(anchors):
                    if work.interrupted:
                        return []
                    inside = (at > lengths[first]) & (at < lengths[last])
                    run = np.vstack((points[first], uniform[inside], points[last]))
                    curve_nodes.extend(
                        fitted(run, self.options.boundary_tolerance).contour.nodes[1:]
                    )
                curve = Subpath("fit", tuple(curve_nodes), True)
            contours["contained-curves"].append(curve)
        original = curve_path(shape)
        original_nodes = sum(len(s.nodes) for s in shape.subpaths)
        results = []
        for kind, values in contours.items():
            if work.interrupted:
                return []
            fitted_shape = transformed_geometry(
                Geometry("piecewise-fit", tuple(values)), (1 / sx, 0, 0, 1 / sy, ox, oy)
            )
            fitted_shape = transformed_geometry(fitted_shape, inverse_matrix(matrix))
            contained = pathops.op(
                curve_path(fitted_shape), original, pathops.PathOp.INTERSECTION
            )
            if work.interrupted:
                return []
            if abs(contained.area) < 0.95 * abs(original.area):
                self.diagnostics["compact_area_exclusions"] += 1
                continue
            result = path_geometry(contained)
            if (
                crossings(result)
                or sum(len(s.nodes) for s in result.subpaths) >= 0.9 * original_nodes
            ):
                self.diagnostics["compact_cost_exclusions"] += 1
                continue
            self.diagnostics["compact_fits"] += 1
            results.append((kind, result))
        return results

    def __call__(self, state, work: Work):
        from vectrify.refine.cel_plan.proposals import bounds

        if state.partition is None:
            return
        emitted = 0
        for ids, members, normal, rho, paints in self.groups(state, work):
            if work.interrupted or emitted >= MAX_PROPOSALS:
                return
            document = state.document
            parent = document.ancestry(ids[0])[-2]
            selected = [e for e in parent.children if e.id in ids]
            survivor = selected[-1].id
            shape = union_geometry(
                [document.geometry_for(e.id) for e in selected],
                [path_style(document, e) for e in selected],
            )
            if work.interrupted:
                return
            if (
                crossings(shape)
                or sum(len(s.nodes) for s in shape.subpaths) > MAX_NODES
            ):
                self.diagnostics["group_limits"] += 1
                continue
            core = in_core(state, members, survivor, shape, work)
            if self.families.evidence.opacity is not None and not core:
                self.diagnostics["core_exclusions"] += 1
                continue
            if not self.families.surface_models._order_safe(state, ids, work):
                self.diagnostics["order_exclusions"] += 1
                continue
            variants = [("exact", shape)]
            compact = self._compact(state, survivor, shape, work) if core else []
            variants = [*compact, *variants]
            yy, xx = np.ogrid[
                : self.families.graph.labels.shape[0],
                : self.families.graph.labels.shape[1],
            ]
            classification = (xx + 0.5) * normal[0] + (yy + 0.5) * normal[1] < rho
            atoms = state.partition.atoms or Atoms.original(self.families.graph)
            try:
                atoms, left, right = atoms.split(
                    self.families.graph, members, classification, work
                )
            except ValueError:
                self.diagnostics["atom_exclusions"] += 1
                continue
            digest = hashlib.sha256(
                repr((ids, tuple(normal), rho)).encode()
            ).hexdigest()[:12]
            new_id = f"{survivor}-piecewise-{digest}"
            if new_id in {e.id for e in document.elements()}:
                continue
            surfaces = (Surface(survivor, left), Surface(new_id, right))
            partition = (
                state.partition.split(ids, surfaces, atoms)
                if atoms.cuts
                else state.partition.replace(ids, surfaces)
            )
            for kind, whole in variants:
                if work.interrupted or emitted >= MAX_PROPOSALS:
                    return
                shapes = self.splitter._geometry(
                    state, survivor, normal, rho, geometry=whole
                )
                if work.interrupted:
                    return
                if (
                    any(not s.subpaths or crossings(s) for s in shapes)
                    or sum(len(p.nodes) for s in shapes for p in s.subpaths) > MAX_NODES
                ):
                    self.diagnostics["group_limits"] += 1
                    continue
                editor = Editor(document, selection=Selection(whole_document=True))
                index = next(
                    i for i, e in enumerate(parent.children) if e.id == survivor
                )
                with editor.transaction("Propose coherent piecewise material") as tx:
                    tx.delete_objects(frozenset(set(ids) - {survivor}))
                    tx.replace_geometry(survivor, shapes[0])
                    tx.insert_object(
                        parent.id,
                        replace(
                            document.element(survivor),
                            id=new_id,
                            geometry_id=shapes[1].id,
                        ),
                        index=index - len(ids) + 2,
                        geometries=(shapes[1],),
                    )
                split_document = editor.snapshot.document
                for child, paint in zip((survivor, new_id), paints, strict=True):
                    opacity = _opacity(split_document, child)
                    if opacity <= 0:
                        break
                    with editor.transaction("Fit coherent side paint") as tx:
                        tx.set_fill(
                            child,
                            _gradient(
                                paint,
                                self.families.evidence,
                                split_document,
                                child,
                                opacity,
                            )
                            if paint.gradient
                            else paint.color,
                        )
                        tx.set_attributes(
                            child,
                            {
                                "fill-rule": "nonzero",
                                "stroke": "none",
                                "fill-opacity": "1"
                                if paint.gradient
                                else repr(min(1.0, paint.opacity / opacity)),
                            },
                        )
                else:
                    proposed = editor.snapshot.document
                    records = discard(state.details.get("chain_constraints"), ids)
                    if kind == "exact":
                        for child in (survivor, new_id):
                            rebound = merged(
                                state.details.get("chain_constraints"),
                                document,
                                proposed,
                                ids,
                                child,
                                work,
                            )
                            if (
                                records is not None
                                and rebound is not None
                                and child in rebound.get("paths", {})
                            ):
                                records["paths"][child] = rebound["paths"][child]
                    holds = set(state.details.get("geometry_constraints", ())) - set(
                        ids
                    )
                    holds.update((survivor, new_id))
                    emitted += 1
                    self.diagnostics["proposals"] += 1
                    yield Proposal(
                        "piecewise-surface",
                        (*ids, new_id),
                        (
                            kind,
                            tuple(float(v) for v in normal),
                            rho,
                            "gradient" if paints[0].gradient else "flat",
                            "gradient" if paints[1].gradient else "flat",
                        ),
                        state.key,
                        proposed,
                        bounds(document, proposed, (*ids, new_id)),
                        details={
                            "regions": sum(
                                s.role != "underlay" for s in partition.surfaces
                            ),
                            "geometry_constraints": sorted(holds),
                            "chain_constraints": records,
                            "piecewise_surface": {
                                "contour_model": kind,
                                "removed_paths": len(ids) - 2,
                                "source_members": members,
                                "source_cut_count": len(atoms.cuts),
                            },
                        },
                        dependencies=(parent.id,),
                        partition=partition,
                    )
