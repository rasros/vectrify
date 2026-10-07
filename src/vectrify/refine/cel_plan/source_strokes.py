"""Replace source ink without forcing the surrounding materials to merge.

Discovery crosses palette boundaries. Exact source cuts retain unselected
marks; neighboring paints continue only into the removed ink footprint, with
their existing colors, gradients and frames. The native evaluator chooses the
complete stroke/paint edit.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace

import numpy as np

from vectrify.document import Editor, Selection
from vectrify.document.join import (
    curve_path,
    path_geometry,
    path_style,
    transformed_geometry,
    union_geometry,
)
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.component_edits import ComponentEdit
from vectrify.refine.cel_plan.constraints import discard
from vectrify.refine.cel_plan.ink_models import models
from vectrify.refine.cel_plan.ink_replace import MAX_NODES, InkReplacement, identified
from vectrify.refine.cel_plan.layer_order import ordered
from vectrify.refine.cel_plan.local import MAX_CROP_PIXELS, Box
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.nested import opaque_fill
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.search import Proposal
from vectrify.refine.cel_plan.source_ridges import SourceRidges
from vectrify.refine.colour_regions import colour
from vectrify.refine.crossings import crossings

MAX_CORES = 4
MAX_PROPOSALS = 8


class SourceStrokes:
    def __init__(self, evidence, graph, options, *, resolver=None):
        self.evidence, self.graph, self.options = evidence, graph, options
        self.cutter = SourceRidges(evidence, graph, options, resolver=resolver)
        self.diagnostics = dict.fromkeys(
            (
                "cores",
                "models",
                "bounded",
                "owner_exclusions",
                "restoration_exclusions",
                "order_exclusions",
                "proposals",
                "cuts",
                "time_bounded",
            ),
            0,
        )
        self.restoration_rejections = {}

    def carriers(self, state, work):
        document = state.document
        assert state.partition is not None
        bases = [
            s for s in state.partition.surfaces if s.covered or s.role == "underlay"
        ]
        bases.sort(key=lambda s: (-len(s.members) - len(s.covered), s.id))
        for base in bases[:MAX_CORES]:
            if work.interrupted:
                return
            style = path_style(document, document.element(base.id))
            if not opaque_fill(document, style["fill"]) or style["stroke"] != "none":
                continue
            geometry = document.geometry_for(base.id)
            if sum(len(s.nodes) for s in geometry.subpaths) > MAX_NODES:
                self.diagnostics["bounded"] += 1
                continue
            yield (
                (*base.members, *base.covered),
                curve_path(
                    transformed_geometry(geometry, root_matrix(document, base.id)),
                    style["fill-rule"],
                ),
            )
        if bases or self.evidence.opacity is not None:
            return
        # An opaque source needs no isolated alpha carrier. Its existing filled
        # surfaces prove the silhouette instead; preserve their holes and all
        # paint, rather than introducing a whole-canvas backdrop.
        parents = {}
        for surface in state.partition.surfaces:
            if work.interrupted:
                return
            element = document.element(surface.id)
            style = path_style(document, element)
            if (
                surface.role != "surface"
                or style["stroke"] != "none"
                or not opaque_fill(document, style["fill"])
                or float(style["fill-opacity"]) != 1
                or float(style["opacity"]) != 1
                or element.get("clip-path", "none") != "none"
            ):
                continue
            parent = document.ancestry(surface.id)[-2].id
            parents.setdefault(parent, []).append(surface)
        for _parent, surfaces in sorted(parents.items())[:MAX_CORES]:
            if work.interrupted:
                return
            if (
                sum(
                    len(sub.nodes)
                    for surface in surfaces
                    for sub in document.geometry_for(surface.id).subpaths
                )
                > MAX_NODES
            ):
                self.diagnostics["bounded"] += 1
                continue
            styles = [path_style(document, document.element(s.id)) for s in surfaces]
            shape = union_geometry(
                [
                    transformed_geometry(
                        document.geometry_for(s.id), root_matrix(document, s.id)
                    )
                    for s in surfaces
                ],
                styles,
            )
            yield tuple(i for s in surfaces for i in s.members), curve_path(shape)

    def proposals(self, state, work):
        if state.partition is None or work.interrupted:
            return
        # Partial-opacity explicit-width evidence already contains a width
        # interpretation. Do not reinterpret those pixels a second time.
        if self.evidence.filled_line_width:
            return
        from vectrify.refine.cel_plan.proposals import bounds

        document, graph, evidence = state.document, self.graph, self.evidence
        emitted = 0
        seen = set()
        for members, carrier in self.carriers(state, work):
            if work.interrupted:
                return
            support = np.isin(graph.labels, members)
            ink = support & evidence.drawn & ~evidence.empty
            self.diagnostics["cores"] += 1
            found = models(
                ink, evidence, self.options, work, carrier=carrier, prune_spurs=True
            )
            for model in found:
                if work.interrupted or emitted >= MAX_PROPOSALS:
                    return
                signature = hashlib.sha256(model.selected.tobytes()).hexdigest()
                if signature in seen:
                    continue
                seen.add(signature)
                self.diagnostics["models"] += 1
                y, x = np.nonzero(model.selected)
                box = Box(
                    int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1
                )
                box = box.expand(32, graph.labels.shape)
                if box.area > MAX_CROP_PIXELS:
                    self.diagnostics["bounded"] += 1
                    continue
                own = model.selected[box.slices]
                owners = state.partition.owners
                selected = {
                    owners.get(int(i)) for i in np.unique(graph.labels[model.selected])
                }
                if selected.intersection(state.details.get("paint_constraints", ())):
                    self.diagnostics["owner_exclusions"] += 1
                    continue
                try:
                    prepared = self.cutter.prepare(state, box, own, work)
                except StageInterruptedError:
                    return
                if prepared is None or work.interrupted:
                    continue
                current = prepared.state
                assert current.partition is not None
                parent = current.document.ancestry(prepared.ids[0])[-2]
                parts = [p for p in parent.children if p.id in prepared.ids]
                survivor = parts[-1].id
                original = union_geometry(
                    [current.document.geometry_for(p.id) for p in parts],
                    [path_style(current.document, p) for p in parts],
                )
                matrix = root_matrix(current.document, survivor)
                footprint = transformed_geometry(
                    model.footprint, inverse_matrix(matrix)
                )
                factory = InkReplacement(
                    prepared.evidence, prepared.graph, self.options
                )
                restored = factory.restorations(
                    current,
                    prepared.members,
                    original,
                    box,
                    own,
                    survivor,
                    work,
                    coverage=footprint,
                    require_core=evidence.opacity is not None,
                    continue_neighbors=True,
                )
                for reason, count in factory.restoration_rejections.items():
                    self.restoration_rejections[reason] = (
                        self.restoration_rejections.get(reason, 0) + count
                    )
                if restored is None or work.interrupted:
                    self.diagnostics["restoration_exclusions"] += 1
                    continue
                continuations, _neighbors = restored
                normalized = []
                for element, geometry in continuations:
                    if work.interrupted:
                        return
                    if crossings(geometry):
                        # Float boolean endpoints can overshoot the closing
                        # seam by a few ulps. Resolve the actual filled winding,
                        # preserving curves and checking the resulting edit.
                        path = curve_path(geometry)
                        path.simplify()
                        geometry = identified(path_geometry(path), element.id)
                    normalized.append((element, geometry))
                continuations = normalized
                continued = {e.id for e, _ in continuations}
                # Boolean cuts can introduce the same closing-seam overshoot
                # in a retained filled fragment. Resolve only newly crossed
                # remainders; unchanged marks and original paint stay exact.
                remainders = []
                for oid in prepared.before:
                    if work.interrupted:
                        return
                    if oid in prepared.ids or oid in continued:
                        continue
                    element = current.document.element(oid)
                    geometry = current.document.geometry_for(oid)
                    if crossings(geometry) > crossings(document.geometry_for(oid)):
                        rule = path_style(current.document, element)["fill-rule"]
                        path = curve_path(geometry, rule)
                        path.simplify()
                        remainders.append(
                            (element, identified(path_geometry(path), oid))
                        )
                changed = current.partition.replace(
                    prepared.ids, (Surface(survivor, prepared.members, "overlay"),)
                )
                changed = Partition(
                    tuple(
                        replace(
                            s,
                            covered=tuple(
                                sorted(set(s.covered) | set(prepared.members))
                            ),
                        )
                        if s.id in continued
                        else s
                        for s in changed.surfaces
                    ),
                    changed.atoms,
                )
                editor = Editor(
                    current.document, selection=Selection(whole_document=True)
                )
                shape = identified(model.geometry, survivor)
                inverse_parent = inverse_matrix(
                    root_matrix(current.document, parent.id)
                )
                parent_frame = " ".join(str(v) for v in inverse_parent)
                with editor.transaction(
                    "Replace source ink and continue existing paint"
                ) as tx:
                    tx.delete_objects(frozenset(set(prepared.ids) - {survivor}))
                    tx.replace_geometry(survivor, shape)
                    tx.set_fill(survivor, "none")
                    tx.set_attributes(
                        survivor,
                        {
                            "fill-opacity": "1",
                            "stroke": colour(model.paint),
                            "stroke-width": repr(model.details["width"]),
                            "stroke-opacity": "1",
                            "stroke-linecap": "round",
                            "stroke-linejoin": "round",
                            "transform": f"matrix({parent_frame})",
                        },
                    )
                    for element, geometry in [*continuations, *remainders]:
                        tx.replace_geometry(element.id, geometry)
                        tx.set_attributes(element.id, {"fill-rule": "nonzero"})
                native_old = transformed_geometry(original, matrix)
                affected = union_geometry(
                    [native_old, model.footprint], [{"fill-rule": "nonzero"}] * 2
                )
                proposed = ordered(
                    editor.snapshot.document,
                    survivor,
                    continued,
                    (),
                    affected,
                    work,
                    factory.diagnostics,
                )
                if work.interrupted:
                    return
                if proposed is None:
                    self.diagnostics["order_exclusions"] += 1
                    continue
                existing = {e.id for e in document.elements()} | {
                    e.id for e in proposed.elements()
                }
                edited = tuple(
                    sorted(
                        set(prepared.before)
                        | ((set(prepared.ids) | continued) & existing)
                    )
                )
                live = {e.id for e in proposed.elements()}
                holds = (
                    set(state.details.get("geometry_constraints", ())) | set(edited)
                ) & live
                cuts = (len(changed.atoms.cuts) if changed.atoms else 0) - (
                    len(state.partition.atoms.cuts) if state.partition.atoms else 0
                )
                try:
                    component = ComponentEdit.bind(
                        document, state.partition, parent.id, work
                    )
                except StageInterruptedError:
                    return
                if work.interrupted:
                    return
                self.diagnostics["proposals"] += 1
                self.diagnostics["cuts"] += cuts
                emitted += 1
                yield Proposal(
                    "source-strokes",
                    edited,
                    (signature, model.details["width"]),
                    state.key,
                    proposed,
                    bounds(document, proposed, edited),
                    details={
                        "geometry_constraints": sorted(holds),
                        "chain_constraints": discard(
                            state.details.get("chain_constraints"), edited
                        ),
                        "source_strokes": {
                            **model.details,
                            "mask": signature,
                            "pixels": int(model.selected.sum()),
                            "owners": len(prepared.before),
                            "cuts": cuts,
                            "continued_neighbors": len(continued),
                            "material_interpretation": "retain-existing-paint",
                        },
                    },
                    dependencies=(parent.id,),
                    partition=changed,
                    component=component,
                )

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
