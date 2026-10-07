"""Individual paint, boundary and evidence-supported ink competitors.

These are proposal generators, never acceptance rules. Round-robin operator
slots keep cheap paint edits from consuming every exact evaluation. Geometry
uses existing shared-edge/corner constraints; ink keeps its local width/color.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import replace
from weakref import WeakValueDictionary

import numpy as np
import pathops
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

from vectrify.document import Editor, Element, Geometry, Selection
from vectrify.document.join import path_style
from vectrify.document.model import paint_server
from vectrify.document.paint import hex_colour, mean_colour
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan import constraints as chains
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.ink import measure
from vectrify.refine.cel_plan.ink_replace import InkReplacement
from vectrify.refine.cel_plan.local import Box
from vectrify.refine.cel_plan.model import Evidence, Graph, Options, Work
from vectrify.refine.cel_plan.overlays import ClosedOverlays
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.refine import (
    MAX_PATH_NODES,
    _bounds,
    _geometry_proposal,
    _paths,
    _spatial_order,
)
from vectrify.refine.cel_plan.score import composite
from vectrify.refine.cel_plan.search import Proposal, State
from vectrify.refine.cel_plan.source_ridges import SourceRidges
from vectrify.refine.crossings import crossings

MAX_OPERATOR_ITEMS = 64
MAX_INK_BOUNDARIES = 256
MAX_BRANCHES = 2
MAX_BRANCH_BYTES = 32 * 1024 * 1024


def bounds(before, after, ids) -> Box:
    boxes = []
    for document in (before, after):
        known = {element.id for element in document.elements()}
        boxes.extend(_bounds(document, oid) for oid in ids if oid in known)
    return Box(
        math.floor(min(box[0] for box in boxes)),
        math.floor(min(box[1] for box in boxes)),
        math.ceil(max(box[2] for box in boxes)),
        math.ceil(max(box[3] for box in boxes)),
    )


class Operators:
    def __init__(
        self,
        evidence: Evidence,
        graph: Graph,
        options: Options,
        *,
        root: Operators | None = None,
    ):
        self.evidence, self.graph, self.options = evidence, graph, options
        self._root = self if root is None else root
        self._image: Image.Image | None = None
        self._components: np.ndarray | None = None
        self._typical: float | None = None
        self._ink_models = {}
        self._ink_checked = set()
        self._branches: OrderedDict[str, Operators] = OrderedDict()
        self._live_branches: WeakValueDictionary[str, Operators] = WeakValueDictionary()
        self._namespace: str | None = None
        self.families = Families(evidence, graph, options)
        self.replacements = InkReplacement(evidence, graph, options)
        self.ridges = SourceRidges(evidence, graph, options, resolver=self._root.branch)
        self.overlays = ClosedOverlays(
            evidence,
            graph,
            options,
            families=self.families,
            restoration=self.replacements,
        )
        self.schedule_diagnostics = {
            "version": 2,
            "compaction_parents": 0,
            "visual_parents": 0,
            "family_proposals": 0,
            "reserved_proposals": 0,
            "native_boolean_failures": 0,
            "composition_parents": 0,
            "source_graph_rebuilds": 0,
            "source_graph_views": 0,
            "source_graph_accounting_version": 2,
            "source_graph_original_bytes": self._graph_bytes(graph),
            "source_graph_cache_bytes": 0,
            "source_graph_cache_peak_bytes": 0,
        }

    @staticmethod
    def _graph_allocations(graph):
        """Conservative graph storage keyed by retained allocation identity."""
        allocations = {("graph", id(graph)): 512}

        def array(value):
            owner = value
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            # A view retains its whole underlying allocation, not just its
            # visible slice. Read-only source arrays may be shared safely.
            allocations[("array", id(owner))] = owner.nbytes + 128

        array(graph.labels)
        for values, item_bytes in (
            (graph.regions, 8),
            (graph.boundaries, 8),
            (graph.junctions, 64),
            (graph.hidden, 32),
        ):
            allocations[("container", id(values))] = 64 + len(values) * item_bytes
        for region in graph.regions:
            allocations[("region", id(region))] = 1024
        for boundary in graph.boundaries:
            allocations[("boundary", id(boundary))] = 512
            array(boundary.points)
        return allocations

    @staticmethod
    def _graph_bytes(graph):
        return sum(Operators._graph_allocations(graph).values())

    def _additional_graph_bytes(self, graphs):
        original = self._graph_allocations(self.graph)
        retained = {}
        for graph in graphs:
            retained.update(self._graph_allocations(graph))
        return sum(size for key, size in retained.items() if key not in original)

    def _live_graph_bytes(self, extra=()):
        """Additional live branch storage beyond the existing original graph.

        The original graph is owned by the planner independently of this cache.
        Shared values are counted once, including branches held by active
        generators after cache eviction. Python records remain conservatively
        charged; this is graph-cache accounting, not process peak RSS.
        """
        return self._additional_graph_bytes(
            (*[b.graph for b in self._live_branches.values()], *extra)
        )

    def branch(self, partition: Partition | None, work: Work):
        atoms = partition.atoms if partition is not None else None
        if atoms is None:
            if self._namespace is not None:
                raise ValueError("Source atom namespace cannot be discarded")
            return self
        key = atoms.key
        if key == self._namespace:
            return self
        if self._namespace is not None:
            raise ValueError("Resolve source branches against the original graph")
        if key not in self._branches and key in self._live_branches:
            self._branches[key] = self._live_branches[key]
        if key not in self._branches:
            graph = atoms.graph(self.evidence, self.graph, work)
            size = self._additional_graph_bytes((graph,))
            if size > MAX_BRANCH_BYTES:
                raise ValueError("Source graph cache bounds exceeded")
            while self._branches and (
                len(self._branches) >= MAX_BRANCHES
                or self._live_graph_bytes((graph,)) > MAX_BRANCH_BYTES
            ):
                self._branches.popitem(last=False)
            if self._live_graph_bytes((graph,)) > MAX_BRANCH_BYTES:
                raise ValueError("Active source graphs exceed branch memory bounds")
            branch = Operators(
                replace(self.evidence, labels=graph.labels),
                graph,
                self.options,
                root=self._root,
            )
            branch._namespace = key
            self._branches[key] = branch
            self._live_branches[key] = branch
            self.schedule_diagnostics[
                "source_graph_rebuilds" if atoms.cuts else "source_graph_views"
            ] += 1
        self._branches.move_to_end(key)
        while len(self._branches) > MAX_BRANCHES:
            self._branches.popitem(last=False)
        size = self._live_graph_bytes()
        self.schedule_diagnostics["source_graph_cache_bytes"] = size
        self.schedule_diagnostics["source_graph_cache_peak_bytes"] = max(
            size, self.schedule_diagnostics["source_graph_cache_peak_bytes"]
        )
        return self._branches[key]

    def validate_partition(self, partition: Partition, work: Work):
        graph = self.branch(partition, work).graph
        if set(partition.owners) != set(range(len(graph.regions))) - graph.hidden - {
            r.id for r in graph.regions if r.area == 0
        }:
            raise ValueError("Source ownership does not cover the branch graph")

    def paint(self, state: State, work: Work):
        document = state.document
        constrained = frozenset(state.details.get("paint_constraints", ()))
        eligible = []
        for oid in _paths(document):
            if work.interrupted:
                return
            if oid in constrained:
                continue
            style = path_style(document, document.element(oid))
            server = paint_server(style["fill"])
            if server is not None:
                eligible.append(oid)
        for oid in _spatial_order(document, eligible, MAX_OPERATOR_ITEMS):
            if work.interrupted:
                return
            style = path_style(document, document.element(oid))
            server = paint_server(style["fill"])
            assert server is not None
            color = mean_colour(document.element(server))
            opacity = float(style["fill-opacity"]) * color[3]
            editor = Editor(document, selection=Selection(whole_document=True))
            with editor.transaction("Propose flat planned paint") as transaction:
                transaction.set_fill(oid, hex_colour(color))
                transaction.set_attributes(oid, {"fill-opacity": repr(opacity)})
            proposed = editor.snapshot.document
            yield Proposal(
                "flat-paint",
                (oid,),
                (hex_colour(color), opacity),
                state.key,
                proposed,
                bounds(document, proposed, (oid,)),
                estimate=-12,
            )

    def geometry(self, state: State, work: Work):
        document = state.document
        constrained = frozenset(state.details.get("geometry_constraints", ()))
        chain_constraints = state.details.get("chain_constraints")
        eligible = [
            oid
            for oid in _paths(document)
            if (
                oid not in constrained
                or chains.bind(document, oid, chain_constraints) is not None
            )
            and sum(len(s.nodes) for s in document.geometry_for(oid).subpaths)
            <= MAX_PATH_NODES
        ]
        if self._image is None:
            self._image = Image.fromarray(
                np.rint(composite(self.evidence.rgba) * 255).astype(np.uint8)
            )
        for oid in _spatial_order(document, eligible, MAX_OPERATOR_ITEMS):
            if work.interrupted:
                return
            proposed = _geometry_proposal(
                document,
                oid,
                self._image,
                self.options,
                work,
                constrained,
                move=False,
                chain_constraints=chain_constraints,
            )
            if proposed is None:
                continue
            ids = tuple(
                sorted(
                    other
                    for other in _paths(proposed)
                    if proposed.geometry_for(other) != document.geometry_for(other)
                )
            )
            yield Proposal(
                "boundary-fit",
                ids,
                (self.options.boundary_tolerance,),
                state.key,
                proposed,
                bounds(document, proposed, ids),
                details={
                    "chain_constraints": chains.refresh(
                        chain_constraints, document, proposed, ids
                    )
                },
            )

    def ink(self, state: State, work: Work):
        evidence = self.evidence
        if self._typical is None:
            self._components, _ = label(~evidence.empty)
            depth = np.asarray(distance_transform_edt(evidence.drawn))
            self._typical = (
                max(0.8, float(np.percentile(depth[evidence.drawn], 65)))
                if evidence.drawn.any()
                else 1.0
            )
        eligible = sorted(
            (
                boundary
                for boundary in self.graph.boundaries
                if min(boundary.left, boundary.right) >= 0
                and boundary.left not in self.graph.hidden
                and boundary.right not in self.graph.hidden
            ),
            key=lambda boundary: (
                -boundary.line_support,
                -len(boundary.points),
                boundary.id,
            ),
        )[:MAX_INK_BOUNDARIES]
        known = {element.id for element in state.document.elements()}
        for boundary in eligible:
            if work.interrupted:
                return
            suffix = f"{self._namespace[:12]}-" if self._namespace else ""
            oid = f"cel-local-ink-{suffix}{boundary.id}"
            if oid in known:
                continue
            if boundary.id not in self._ink_checked:
                self._ink_checked.add(boundary.id)
                proof = measure(boundary.points, evidence.target, self._typical)
                if proof is None:
                    continue
                scale = np.array(evidence.scale)
                points = proof.points / scale + evidence.offset
                model = fitted(points, self.options.boundary_tolerance)
                contour = replace(
                    model.contour,
                    id=f"{oid}-contour",
                    nodes=tuple(
                        replace(node, id=f"{oid}-node-{index}")
                        for index, node in enumerate(model.contour.nodes)
                    ),
                )
                geometry = Geometry(f"{oid}-geometry", (contour,))
                if crossings(geometry):
                    continue
                width = self.options.line_width or proof.width / float(
                    np.sqrt(np.prod(scale))
                )
                y = np.clip(
                    boundary.points[:, 1].astype(int), 0, evidence.empty.shape[0] - 1
                )
                x = np.clip(
                    boundary.points[:, 0].astype(int), 0, evidence.empty.shape[1] - 1
                )
                assert self._components is not None
                components = np.unique(self._components[y, x])
                parent = (
                    f"cel-component-{components[0]}"
                    if len(components) == 1 and components[0]
                    else None
                )
                self._ink_models[boundary.id] = (
                    geometry,
                    width,
                    hex_colour(tuple(proof.paint / 255)),
                    parent,
                    model.kind,
                    proof.support,
                    proof.peak_gap,
                )
            if boundary.id not in self._ink_models:
                continue
            geometry, width, color, parent, model, support, gap = self._ink_models[
                boundary.id
            ]
            parent = parent if parent in known else state.document.root.id
            inverse = inverse_matrix(root_matrix(state.document, parent))
            element = Element(
                oid,
                "path",
                (
                    ("fill", "none"),
                    ("stroke", color),
                    ("stroke-width", repr(width)),
                    ("stroke-linecap", "round"),
                    ("stroke-linejoin", "round"),
                    (
                        "transform",
                        f"matrix({' '.join(str(value) for value in inverse)})",
                    ),
                ),
                geometry_id=geometry.id,
            )
            editor = Editor(state.document, selection=Selection(whole_document=True))
            with editor.transaction("Propose continuous planned ink") as transaction:
                transaction.insert_object(parent, element, geometries=(geometry,))
            proposed = editor.snapshot.document
            yield Proposal(
                "boundary-ink",
                (oid,),
                (boundary.id, width, color, support, gap),
                state.key,
                proposed,
                bounds(state.document, proposed, (oid,)),
                details={
                    "geometry_constraints": [
                        *state.details.get("geometry_constraints", ()),
                        oid,
                    ]
                }
                if model != "curve"
                else None,
                dependencies=(parent,),
            )

    def __call__(self, state: State, work: Work):
        branch = self.branch(state.partition, work)
        if branch is not self:
            yield from branch(state, work)
            return
        context = state.details.get("search_budget", {})
        target = context.get("representation_target", state.snapshot.evaluation.cost)
        pressure = state.snapshot.evaluation.cost / max(1, target)
        if context.get("node_target", 0):
            pressure = max(
                pressure,
                state.snapshot.evaluation.structure["nodes"] / context["node_target"],
            )
        compaction = pressure > 1.25
        burst = (
            {"fast": 1, "balanced": 2, "high": 3}[self.options.quality]
            if compaction
            else 1
        )
        self.schedule_diagnostics[
            "compaction_parents" if compaction else "visual_parents"
        ] += 1
        iterators = [
            iter(self.families(state, work)),
            iter(self.overlays(state, work)),
            iter(self.paint(state, work)),
            iter(self.geometry(state, work)),
            iter(self.ink(state, work)),
            iter(self.replacements(state, work)),
            iter(self.ridges(state, work)),
        ]
        alive = set(range(len(iterators)))
        try:
            for cycle in range(MAX_OPERATOR_ITEMS):
                # Fixed shared-frontier anchor; requested complexity never
                # changes which pool these operators try to construct.
                reserved_count = len(iterators) - 1
                rotation = (len(state.edits) + cycle) % reserved_count
                reserved = [
                    1 + (i + rotation) % reserved_count for i in range(reserved_count)
                ]
                slots = [0] * burst + reserved
                if cycle == 0 and state.edits:
                    # Give a complementary layer interpretation an opportunity
                    # before another broad family scan consumes this child's
                    # discovery window. Every other slot remains in the cycle.
                    preferred = {
                        "ink-replacement": 1,
                        "closed-overlay": 5,
                        "closed-material": 5,
                    }.get(state.edits[-1]["operator"])
                    if preferred is not None:
                        slots.remove(preferred)
                        slots.insert(0, preferred)
                        self.schedule_diagnostics["composition_parents"] += 1
                for slot in slots:
                    if work.interrupted or not alive:
                        return
                    if slot not in alive:
                        continue
                    try:
                        proposal = next(iterators[slot], None)
                    except pathops.PathOpsError:
                        # Optional curve booleans can fail even for renderable
                        # SVGs. Reject this cursor, retaining validated states
                        # and allowing the other operator families to compete.
                        self.schedule_diagnostics["native_boolean_failures"] += 1
                        alive.remove(slot)
                        continue
                    if proposal is None:
                        alive.remove(slot)
                        continue
                    self.schedule_diagnostics[
                        "family_proposals" if slot == 0 else "reserved_proposals"
                    ] += 1
                    yield proposal
        finally:
            for iterator in iterators:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
