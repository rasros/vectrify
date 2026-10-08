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
import pathops

from vectrify.document import Editor, Selection
from vectrify.document.join import (
    curve_path,
    path_style,
    transformed_geometry,
    union_geometry,
)
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.atoms import Atoms
from vectrify.refine.cel_plan.component_edits import ComponentEdit
from vectrify.refine.cel_plan.constraints import discard
from vectrify.refine.cel_plan.fill_winding import resolved
from vectrify.refine.cel_plan.ink_models import (
    carrier_width,
    footprint,
    models,
    owned_model,
)
from vectrify.refine.cel_plan.ink_replace import (
    MAX_COMPONENT_NEIGHBORS,
    MAX_NODES,
    InkReplacement,
    identified,
)
from vectrify.refine.cel_plan.layer_order import ordered_surfaces
from vectrify.refine.cel_plan.local import MAX_CROP_PIXELS, Box
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.nested import opaque_fill
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.search import Proposal
from vectrify.refine.cel_plan.silhouette_ink import SilhouetteInk
from vectrify.refine.cel_plan.source_ridges import SourceRidges
from vectrify.refine.colour_regions import colour
from vectrify.refine.crossings import crossings

MAX_CORES = 4
MAX_PROPOSALS = 8
MAX_CONTACT_PATHS = 256


class SourceStrokes:
    def __init__(
        self,
        evidence,
        graph,
        options,
        *,
        resolver=None,
        boundary_contacts=False,
        outlines=False,
        perimeter_only=False,
    ):
        if perimeter_only and not outlines:
            raise ValueError(
                "Perimeter-only proposals require source outline discovery"
            )
        self.evidence, self.graph, self.options = evidence, graph, options
        self.boundary_contacts = boundary_contacts
        self.perimeter_only = perimeter_only
        self.silhouettes = SilhouetteInk(evidence, options) if outlines else None
        self._outline_models = None
        self.cutter = SourceRidges(
            evidence,
            graph,
            options,
            resolver=resolver,
            max_paths=MAX_CONTACT_PATHS if boundary_contacts else None,
            allow_invisible=boundary_contacts,
        )
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
                "restoration_neighbors_peak",
                "outline_carrier_exclusions",
                "outline_retained_exclusions",
            ),
            0,
        )
        self.restoration_rejections = {}

    def retained_strokes(self, state, work):
        """Prove exterior replacement does not duplicate an existing stroke.

        Its own local width is expanded before applying the actual frame,
        preserving anisotropic/sheared bodies. Unsupported stroke styles or
        excessive existing geometry exclude this optional interpretation.
        """
        bodies = []
        nodes = 0
        for element in state.document.elements():
            if work.interrupted:
                return None
            if element.tag != "path":
                continue
            style = path_style(state.document, element)
            if style["stroke"] == "none":
                continue
            geometry = state.document.geometry_for(element.id)
            nodes += sum(len(s.nodes) for s in geometry.subpaths)
            if nodes > MAX_NODES or len(bodies) >= MAX_CONTACT_PATHS:
                return None
            cap, join = style["stroke-linecap"], style["stroke-linejoin"]
            if cap not in {"round", "butt"} or join != "round":
                return None
            try:
                width = float(style["stroke-width"])
            except ValueError:
                return None
            if not np.isfinite(width) or width <= 0:
                return None
            body = footprint(geometry, width, cap=cap)
            body = transformed_geometry(body, root_matrix(state.document, element.id))
            bodies.append(curve_path(body))
        return None if work.interrupted else tuple(bodies)

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
        if self.silhouettes is not None and self._outline_models is None:
            outline_models = self.silhouettes(work)
            if work.interrupted:
                return
            # Only complete source discovery is cached, including a proved
            # empty pool. Ownership and current carrier proofs remain per state.
            self._outline_models = outline_models
        retained = self.retained_strokes(state, work) if self._outline_models else ()
        for members, carrier in self.carriers(state, work):
            if work.interrupted:
                return
            support = np.isin(graph.labels, members)
            ink = support & evidence.drawn & ~evidence.empty
            self.diagnostics["cores"] += 1
            found = (
                ()
                if self.perimeter_only
                else models(
                    ink,
                    evidence,
                    self.options,
                    work,
                    carrier=carrier,
                    prune_spurs=True,
                    boundary_contacts=self.boundary_contacts,
                )
            )
            # A material candidate must be able to retain every compatible
            # source style together. Cut and restore their union once; publishing
            # a prefix would make later styles compete with its overlay owners.
            groups = [tuple(found)] if len(found) > 1 else []
            for model in self._outline_models or ():
                if retained is None or any(
                    abs(
                        pathops.op(
                            curve_path(model.footprint),
                            body,
                            pathops.PathOp.INTERSECTION,
                        ).area
                    )
                    > 1e-8
                    for body in retained
                ):
                    self.diagnostics["outline_retained_exclusions"] += 1
                    continue
                owned = owned_model(model, support, evidence, work)
                ceiling = (
                    carrier_width(
                        owned.geometry,
                        owned.details["width"],
                        carrier,
                        work,
                        fixed=True,
                        cap=owned.details["linecap"],
                    )
                    if owned is not None
                    else None
                )
                if owned is None or ceiling is None:
                    self.diagnostics["outline_carrier_exclusions"] += 1
                    continue
                groups.append((owned,))
            groups.extend((model,) for model in found)
            for group in groups:
                if work.interrupted or emitted >= MAX_PROPOSALS:
                    return
                selected_mask = np.zeros(graph.labels.shape, dtype=bool)
                classes = np.zeros(graph.labels.shape, dtype=np.uint8)
                overlap = False
                for index, model in enumerate(group):
                    if work.interrupted:
                        return
                    if np.any(selected_mask & model.selected):
                        overlap = True
                        break
                    selected_mask |= model.selected
                    classes[model.selected] = index
                if overlap:
                    self.diagnostics["bounded"] += 1
                    continue
                signature = hashlib.sha256(selected_mask.tobytes()).hexdigest()
                if signature in seen:
                    continue
                seen.add(signature)
                self.diagnostics["models"] += 1
                y, x = np.nonzero(selected_mask)
                box = Box(
                    int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1
                )
                box = box.expand(32, graph.labels.shape)
                if box.area > MAX_CROP_PIXELS:
                    self.diagnostics["bounded"] += 1
                    continue
                own = selected_mask[box.slices]
                owners = state.partition.owners
                selected = {
                    owners.get(int(i)) for i in np.unique(graph.labels[selected_mask])
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
                native_coverage = union_geometry(
                    [model.footprint for model in group],
                    [{"fill-rule": "nonzero"}] * len(group),
                )
                footprint = transformed_geometry(
                    native_coverage, inverse_matrix(matrix)
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
                    continuation_limit=MAX_COMPONENT_NEIGHBORS
                    if self.boundary_contacts
                    else None,
                    allow_carrier_contact=any(
                        m.details["linecap"] == "butt" for m in group
                    ),
                )
                self.diagnostics["restoration_neighbors_peak"] = max(
                    self.diagnostics["restoration_neighbors_peak"],
                    factory.diagnostics["restoration_neighbors_peak"],
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
                unproved = False
                for element, geometry in continuations:
                    if work.interrupted:
                        return
                    if crossings(geometry):
                        # Float boolean endpoints can overshoot the closing
                        # seam by a few ulps. Resolve the actual filled winding,
                        # preserving curves and checking the resulting edit.
                        corrected = resolved(
                            geometry,
                            root_matrix(current.document, element.id),
                            "nonzero",
                            work,
                        )
                        if corrected is None:
                            unproved = True
                            break
                        geometry = identified(corrected, element.id)
                    normalized.append((element, geometry))
                if unproved:
                    self.diagnostics["bounded"] += 1
                    continue
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
                        corrected = resolved(
                            geometry, root_matrix(current.document, oid), rule, work
                        )
                        if corrected is None:
                            unproved = True
                            break
                        remainders.append((element, identified(corrected, oid)))
                if unproved:
                    self.diagnostics["bounded"] += 1
                    continue
                atoms = current.partition.atoms
                style_members = (prepared.members,)
                if len(group) > 1:
                    try:
                        atoms, style_members = (
                            atoms or Atoms.original(prepared.graph)
                        ).partition(
                            prepared.graph, prepared.members, classes, len(group), work
                        )
                    except StageInterruptedError:
                        return
                    except ValueError:
                        self.diagnostics["bounded"] += 1
                        continue
                if not all(style_members):
                    self.diagnostics["bounded"] += 1
                    continue
                glyphs = (
                    survivor,
                    *tuple(
                        f"{survivor}-style-{signature[:12]}-{i}"
                        for i in range(1, len(group))
                    ),
                )
                known = {e.id for e in current.document.elements()}
                if known.intersection(glyphs[1:]):
                    self.diagnostics["bounded"] += 1
                    continue
                changed = current.partition.split(
                    prepared.ids,
                    tuple(
                        Surface(oid, members, "overlay")
                        for oid, members in zip(glyphs, style_members, strict=True)
                    ),
                    atoms or Atoms.original(prepared.graph),
                )
                selected_members = {i for members in style_members for i in members}
                changed = Partition(
                    tuple(
                        replace(
                            s,
                            covered=tuple(sorted(set(s.covered) | selected_members)),
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
                inverse_parent = inverse_matrix(
                    root_matrix(current.document, parent.id)
                )
                parent_frame = " ".join(str(v) for v in inverse_parent)
                with editor.transaction(
                    "Replace source ink and continue existing paint"
                ) as tx:
                    tx.delete_objects(frozenset(set(prepared.ids) - {survivor}))
                    template = current.document.element(survivor)
                    at = next(
                        i for i, c in enumerate(parent.children) if c.id == survivor
                    )
                    # Account for selected paths deleted before this position.
                    at -= sum(
                        c.id in set(prepared.ids) - {survivor}
                        for c in parent.children[:at]
                    )
                    for index, (oid, model) in enumerate(
                        zip(glyphs, group, strict=True)
                    ):
                        shape = identified(model.geometry, oid)
                        if index:
                            tx.insert_object(
                                parent.id,
                                replace(template, id=oid, geometry_id=shape.id),
                                index=at + index,
                                geometries=(shape,),
                            )
                        else:
                            tx.replace_geometry(oid, shape)
                        tx.set_fill(oid, "none")
                        tx.set_attributes(
                            oid,
                            {
                                "fill-opacity": "1",
                                "stroke": colour(model.paint),
                                "stroke-width": repr(model.details["width"]),
                                "stroke-opacity": "1",
                                "stroke-linecap": model.details["linecap"],
                                "stroke-linejoin": "round",
                                "transform": f"matrix({parent_frame})",
                            },
                        )
                    for element, geometry in [*continuations, *remainders]:
                        tx.replace_geometry(element.id, geometry)
                        tx.set_attributes(element.id, {"fill-rule": "nonzero"})
                native_old = transformed_geometry(original, matrix)
                affected = union_geometry(
                    [native_old, group[0].footprint], [{"fill-rule": "nonzero"}] * 2
                )
                proposed = ordered_surfaces(
                    editor.snapshot.document,
                    glyphs,
                    continued,
                    (),
                    (affected, *(m.footprint for m in group[1:])),
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
                        | ((set(prepared.ids) | continued | set(glyphs)) & existing)
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
                model_details = (
                    group[0].details
                    if len(group) == 1
                    else {
                        "model": "source-stroke-bundle",
                        "groups": [m.details for m in group],
                        "runs": sum(m.details["runs"] for m in group),
                    }
                )
                yield Proposal(
                    "source-strokes",
                    edited,
                    (
                        signature,
                        tuple(
                            (m.details["width"], m.details["linecap"]) for m in group
                        ),
                    ),
                    state.key,
                    proposed,
                    bounds(document, proposed, edited),
                    details={
                        "geometry_constraints": sorted(holds),
                        "chain_constraints": discard(
                            state.details.get("chain_constraints"), edited
                        ),
                        "source_strokes": {
                            **model_details,
                            "mask": signature,
                            "pixels": int(selected_mask.sum()),
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
