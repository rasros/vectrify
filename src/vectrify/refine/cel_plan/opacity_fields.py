"""Compact connected source-opacity marks without merging physical components.

An exact source-grid contour can replace multiple paint/alpha fragments of one
weak component. Source maximum alpha retains its measured coverage; excessive
flat mass or incomplete ownership leaves all original paths. These are bounded
proposals, subject to the full native policy rather than opacity exemptions.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
from scipy.ndimage import find_objects, label

from vectrify.document import Editor, Element, Selection
from vectrify.document.hit_test import multiply
from vectrify.document.join import path_style
from vectrify.document.model import paint_server
from vectrify.document.paint import hex_colour
from vectrify.document.redraw import root_matrix
from vectrify.document.svg import parse_path
from vectrify.document.topology import inverse_matrix
from vectrify.refine import cel
from vectrify.refine.cel_plan.component_edits import ComponentEdit
from vectrify.refine.cel_plan.constraints import discard
from vectrify.refine.cel_plan.model import StageInterruptedError
from vectrify.refine.cel_plan.ownership import Surface
from vectrify.refine.cel_plan.search import Proposal
from vectrify.refine.crossings import crossings

MAX_PIXELS = 1536**2
MAX_PATHS = 4096
MAX_COMPONENTS = 4096
MAX_FIELDS = 128
MAX_PARENTS = 4
MAX_NODES = 6000
MAX_OPACITY = 0.05


class OpacityFields:
    def __init__(self, evidence, graph):
        self.evidence, self.graph = evidence, graph
        self.diagnostics = dict.fromkeys(
            (
                "bounded",
                "incomplete_components",
                "excess_flat_mass",
                "geometry_exclusions",
                "fields",
                "replaced_paths",
                "proposals",
            ),
            0,
        )

    def __call__(self, state, work):
        from vectrify.refine.cel_plan.proposals import bounds

        def check():
            if work.interrupted:
                raise StageInterruptedError("Source opacity fields interrupted")

        evidence, graph = self.evidence, self.graph
        if state.partition is None or evidence.opacity is None:
            return
        if graph.source_atoms != (
            state.partition.atoms.key if state.partition.atoms else None
        ):
            raise ValueError("Opacity fields require the current source atom graph")
        check()
        if graph.labels.size > MAX_PIXELS or len(state.partition.surfaces) > MAX_PATHS:
            self.diagnostics["bounded"] += 1
            return
        components, count = label(~evidence.empty)
        if count > MAX_COMPONENTS:
            self.diagnostics["bounded"] += 1
            return
        check()
        # Original and active split atoms each have one physical component.
        # Empty retired atoms remain zero and never acquire new ownership.
        component_for = np.zeros(len(graph.regions), np.int32)
        np.maximum.at(component_for, graph.labels.ravel(), components.ravel())
        minimum = np.full(len(graph.regions), count + 1, np.int32)
        np.minimum.at(minimum, graph.labels.ravel(), components.ravel())
        component_for[minimum != component_for] = 0
        check()
        boxes = find_objects(components, max_label=count)
        groups = defaultdict(list)
        document = state.document
        constrained = set(state.details.get("paint_constraints", ()))
        for surface in state.partition.surfaces:
            check()
            if (
                surface.role != "surface"
                or surface.covered
                or surface.id in constrained
                or any(graph.regions[i].fixed for i in surface.members)
                or any(graph.regions[i].area == 0 for i in surface.members)
                or any(
                    graph.regions[i].opacity_range is None
                    or graph.regions[i].opacity_range[1] > MAX_OPACITY
                    for i in surface.members
                )
            ):
                continue
            ids = np.unique(component_for[list(surface.members)])
            if len(ids) != 1 or ids[0] <= 0:
                continue
            element = document.element(surface.id)
            style = path_style(document, element)
            ancestry = document.ancestry(surface.id)
            if (
                style["stroke"] != "none"
                or style["fill"] == "none"
                or paint_server(style["fill"]) is not None
                or float(style["opacity"]) != 1
                or any(
                    a.locks
                    or a.get("filter", "none") != "none"
                    or a.get("clip-path", "none") != "none"
                    for a in ancestry
                )
                or any(float(a.get("opacity", "1")) != 1 for a in ancestry[:-1])
                or any(
                    n.pinned
                    for s in document.geometry_for(surface.id).subpaths
                    for n in s.nodes
                )
            ):
                continue
            parent = ancestry[-2]
            groups[(parent.id, int(ids[0]))].append(surface)
        # Separate parent edits remain independent. Nested siblings are sealed
        # verbatim and cannot be declared, reordered or changed by this edit.
        by_parent = defaultdict(list)
        total_fields = total_nodes = 0
        for (parent, component), owners in sorted(groups.items()):
            check()
            if len(owners) < 2:
                continue
            box = boxes[component - 1]
            if box is None:
                continue
            own = components[box] == component
            members = tuple(sorted(i for s in owners for i in s.members))
            if {int(i) for i in np.unique(graph.labels[box][own])} != set(members):
                self.diagnostics["incomplete_components"] += 1
                continue
            samples = evidence.opacity[box][own]
            opacity = float(samples.max())
            # The flat field has no weaker analysis samples than the source.
            # The existing native mass ceiling still rejects excessive paint;
            # use it only to exclude a futile proposal, never as an exemption.
            if (
                opacity * len(samples)
                > float(samples.sum()) * 1.25 + len(samples) / 255 + 1e-7
            ):
                self.diagnostics["excess_flat_mass"] += 1
                continue
            color = hex_colour(
                tuple(
                    np.average(evidence.target[box][own], axis=0, weights=samples) / 255
                )
            )

            def boundary(points, _bound):
                check()
                return [
                    ("L", tuple(float(v) for v in p))
                    for p in cel.simplify(points, 0)[1:]
                ]

            data = cel.region_outlines(
                own.astype(np.int32), 0, fit_boundary=boundary, check=check
            ).get(1)
            if data is None:
                continue
            geometry = parse_path(data)
            if crossings(geometry):
                self.diagnostics["geometry_exclusions"] += 1
                continue
            total_fields += 1
            total_nodes += sum(len(s.nodes) for s in geometry.subpaths)
            if total_fields > MAX_FIELDS or total_nodes > MAX_NODES:
                self.diagnostics["bounded"] += 1
                return
            sx, sy = evidence.scale
            ox, oy = evidence.offset
            frame = multiply(
                inverse_matrix(root_matrix(document, parent)),
                (1 / sx, 0, 0, 1 / sy, ox + box[1].start / sx, oy + box[0].start / sy),
            )
            digest = hashlib.sha256(repr((parent, members)).encode()).hexdigest()[:16]
            oid = f"cel-opacity-{digest}"
            if any(e.id == oid for e in document.elements()):
                continue
            element = Element(
                oid,
                "path",
                attributes=(
                    ("fill", color),
                    ("fill-opacity", repr(opacity)),
                    ("stroke", "none"),
                    ("fill-rule", "evenodd"),
                    ("transform", "matrix(" + " ".join(map(str, frame)) + ")"),
                ),
                geometry_id=geometry.id,
            )
            by_parent[parent].append((owners, Surface(oid, members), element, geometry))
        check()
        if len(by_parent) > MAX_PARENTS:
            self.diagnostics["bounded"] += 1
            return
        for parent, fields in by_parent.items():
            check()
            removed = {s.id for owners, *_ in fields for s in owners}
            added = tuple(s for _, s, *_ in fields)
            partition = state.partition.replace(tuple(sorted(removed)), added)
            editor = Editor(document, selection=Selection(whole_document=True))
            with editor.transaction("Propose connected source opacity fields") as tx:
                # In-place positions retain order relative to all other paint.
                for owners, _, element, geometry in fields:
                    current = tx.preview.element(parent)
                    old_ids = {s.id for s in owners}
                    position = min(
                        i for i, c in enumerate(current.children) if c.id in old_ids
                    )
                    tx.delete_objects(frozenset(old_ids))
                    tx.insert_object(
                        parent, element, index=position, geometries=(geometry,)
                    )
            proposed = editor.snapshot.document
            partition.validate(proposed)
            if not partition.follows(state.partition):
                raise ValueError("Source opacity field lost primary ownership")
            ids = (*sorted(removed), *(s.id for s in added))
            component = ComponentEdit.bind(document, state.partition, parent, work)
            check()
            self.diagnostics["fields"] += len(added)
            self.diagnostics["replaced_paths"] += len(removed)
            self.diagnostics["proposals"] += 1
            yield Proposal(
                "source-opacity-fields",
                ids,
                (len(added),),
                state.key,
                proposed,
                bounds(document, proposed, ids),
                partition=partition,
                component=component,
                dependencies=(parent,),
                details={
                    "geometry_constraints": sorted(
                        (set(state.details.get("geometry_constraints", ())) - removed)
                        | {s.id for s in added}
                    ),
                    "paint_constraints": sorted(constrained - removed),
                    "chain_constraints": discard(
                        state.details.get("chain_constraints"), tuple(removed)
                    ),
                    "opacity_fields": {
                        "fields": len(added),
                        "replaced_paths": len(removed),
                        "source_components": count,
                        "support": "exact-source-grid",
                        "paint": "source-maximum-alpha-flat",
                    },
                },
            )
