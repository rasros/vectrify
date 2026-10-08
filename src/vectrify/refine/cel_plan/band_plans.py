"""Co-plan a stroke and exact independent marks before material atom allocation.

The ordinary material proposal and this alternative share one ancestor. Replaying
its source classes from that ancestor preserves the immutable ledger prefix and
its existing limits. This is not a post-hoc rewrite of an accepted namespace.
Unrepresented source pixels retain the ordinary band's owner; a painted native
comparison must still preserve measured source support and individual gaps.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pathops
from scipy.ndimage import distance_transform_edt, label

from vectrify.document import Editor, Selection, export_svg
from vectrify.document.join import curve_path, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.refine.cel_plan.atoms import (
    CHUNK_PIXELS,
    MAX_PARTS,
    AtomLimitError,
    Atoms,
)
from vectrify.refine.cel_plan.constraints import discard
from vectrify.refine.cel_plan.filled_bands import (
    MAX_NATIVE_PIXELS,
    MAX_NODES,
    MAX_PATHS,
    MAX_WIDTH,
    _check,
    invert,
    supported_style,
)
from vectrify.refine.cel_plan.ink_replace import identified
from vectrify.refine.cel_plan.local import MAX_CROP_PIXELS, Box
from vectrify.refine.cel_plan.model import StageInterruptedError
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.score import render

TOLERANCES = (0.25, 0.15, 0.1)
MAX_SUBPATHS = 16


def separate(geometry, rule, work):
    """Retain every other contour exactly; holes and interacting fills exclude."""
    _check(work)
    if (
        not 2 <= len(geometry.subpaths) <= MAX_SUBPATHS
        or any(not s.closed for s in geometry.subpaths)
        or sum(len(s.nodes) for s in geometry.subpaths) > MAX_NODES
    ):
        return None
    index = max(
        range(len(geometry.subpaths)), key=lambda i: len(geometry.subpaths[i].nodes)
    )
    main = replace(geometry, subpaths=(geometry.subpaths[index],))
    marks = replace(
        geometry,
        subpaths=tuple(s for i, s in enumerate(geometry.subpaths) if i != index),
    )
    a, b = curve_path(main, rule), curve_path(marks, rule)
    if pathops.op(a, b, pathops.PathOp.INTERSECTION).area > 1e-8:
        return None
    combined = pathops.op(a, b, pathops.PathOp.UNION)
    if pathops.op(combined, curve_path(geometry, rule), pathops.PathOp.XOR).area > 1e-8:
        return None
    _check(work)
    return main, marks


def mark_components(own, main, marks, scale, work):
    """Split only source components with unambiguous nearby mark support.

    Source pixels absent from both vector fills keep their existing main owner.
    That inheritance is not a geometric support proof or permission to fill them.
    """
    _check(work)
    da = distance_transform_edt(main <= 1 / 255, sampling=(1 / scale[1], 1 / scale[0]))
    db = distance_transform_edt(marks <= 1 / 255, sampling=(1 / scale[1], 1 / scale[0]))
    components, count = label(own, np.ones((3, 3)))
    _check(work)
    ma = np.bincount(components[own & (da <= 2) & (da < db)], minlength=count + 1) > 0
    mb = np.bincount(components[own & (db <= 2) & (db < da)], minlength=count + 1) > 0
    if (ma & mb).any() or not ma.any() or not mb.any():
        return None
    selected = mb[components] & own
    inherited = int((own & ~ma[components] & ~mb[components]).sum())
    _check(work)
    return selected, inherited


class BandPlans:
    """One compound band with bounded width alternatives per material proposal."""

    def __init__(self, evidence, graph, original, *, guard, width_fixed=False):
        self.evidence, self.graph, self.original, self.guard = (
            evidence,
            graph,
            original,
            guard,
        )
        self.width_fixed = width_fixed
        self.diagnostics = dict.fromkeys(
            ("eligible", "source_exclusions", "bounded", "ambiguous", "proposals"), 0
        )

    def __call__(self, state, proposal, work):
        try:
            yield from self.proposals(state, proposal, work)
        except StageInterruptedError:
            return
        except (AtomLimitError, pathops.PathOpsError):
            self.diagnostics["bounded"] += 1

    def proposals(self, state, proposal, work):
        _check(work)
        old, part = state.partition, proposal.partition
        if (
            old is None
            or part is None
            or part.atoms is None
            or proposal.component is None
        ):
            return
        if proposal.parent != state.key or not part.follows(old):
            raise ValueError(
                "Band co-planning requires an ordinary proposal from the same ancestor"
            )
        if self.graph.source_atoms != (
            old.atoms.key if old.atoms is not None else None
        ):
            raise ValueError("Band co-planning requires the ancestor source graph")
        if np.prod(self.evidence.source_size) > MAX_NATIVE_PIXELS:
            self.diagnostics["bounded"] += 1
            return
        scoped = [
            s for s in part.surfaces if s.id in proposal.ids and s.role != "underlay"
        ]
        if not 0 < len(scoped) < MAX_PARTS:
            self.diagnostics["bounded"] += 1
            return
        labels = part.atoms.labels(self.original, work)
        eligible = []
        for surface in scoped:
            _check(work)
            style = supported_style(proposal.document, surface.id)
            if style is None:
                continue
            own = np.isin(labels, surface.members)
            if not own.any() or self.evidence.drawn[own].sum() < own.sum() * 0.6:
                continue
            geometry = proposal.document.geometry_for(surface.id)
            nodes = sum(len(s.nodes) for s in geometry.subpaths)
            if 2 <= len(geometry.subpaths) <= MAX_SUBPATHS and nodes <= MAX_NODES:
                eligible.append((nodes, surface, style))
        eligible.sort(key=lambda v: (-v[0], v[1].id))
        before, guard = None, None
        for old_nodes, surface, style in eligible[:MAX_PATHS]:
            _check(work)
            self.diagnostics["eligible"] += 1
            geometry = proposal.document.geometry_for(surface.id)
            parts = separate(geometry, style["fill-rule"], work)
            if parts is None:
                continue
            main, marks = parts
            own = np.isin(labels, surface.members)
            coverage = self.coverage(
                proposal.document, surface.id, geometry, own, style["fill-rule"], work
            )
            if coverage is None:
                self.diagnostics["bounded"] += 1
                continue
            box, raster = coverage
            classified = mark_components(
                own[box.slices], raster(main), raster(marks), self.evidence.scale, work
            )
            if classified is None:
                self.diagnostics["ambiguous"] += 1
                continue
            localmarks, inherited = classified
            if guard is None:
                try:
                    guard = self.guard(work)
                except ValueError:
                    self.diagnostics["bounded"] += 1
                    return
                if guard is None:
                    return
            if before is None:
                before = guard.observe(
                    render(export_svg(proposal.document), self.evidence.source_size),
                    work=work,
                )
            cached = None
            published = 0
            for tolerance in TOLERANCES:
                _check(work)
                model = invert(main, style["fill-rule"], work, tolerance=tolerance)
                if model is None:
                    break
                if len(model.geometry.subpaths[0].nodes) >= len(main.subpaths[0].nodes):
                    continue
                newid = surface.id + "-band-marks"
                if any(el.id == newid for el in proposal.document.elements()):
                    continue
                delta = min(0.25, model.width * 0.1)
                widths = (
                    (model.width,)
                    if self.width_fixed or self.evidence.filled_line_width > 0
                    else tuple(
                        model.width + delta * amount
                        for amount in (0, -1, -0.5, 0.5, 1)
                        if 0.8 <= model.width + delta * amount <= MAX_WIDTH
                    )
                )
                for width in widths:
                    _check(work)
                    variant = replace(model, width=width)
                    document = self.document(
                        proposal.document, surface.id, newid, variant, marks, style
                    )
                    comparison = guard.compare_observed(
                        before,
                        render(export_svg(document), self.evidence.source_size),
                        work=work,
                    )
                    if not comparison["qualified_samples"] or comparison["rejections"]:
                        self.diagnostics["source_exclusions"] += 1
                        continue
                    bodyalpha = raster(variant.geometry, variant.width)
                    if cached is None:
                        cached = self.partition(
                            state,
                            proposal,
                            scoped,
                            labels,
                            surface,
                            newid,
                            box,
                            localmarks,
                            bodyalpha,
                            raster(marks),
                            work,
                        )
                        if cached is None:
                            return
                    partition, newlabels = cached
                    active = set(map(int, np.unique(newlabels))) - self.original.hidden
                    surfaces = []
                    for record in partition.surfaces:
                        if record.id == surface.id:
                            covered = tuple(
                                sorted(
                                    (
                                        set(
                                            map(
                                                int,
                                                np.unique(
                                                    newlabels[box.slices][
                                                        bodyalpha > 1 / 255
                                                    ]
                                                ),
                                            )
                                        )
                                        & active
                                    )
                                    - set(record.members)
                                )
                            )
                            record = replace(record, covered=covered)
                        surfaces.append(record)
                    partition = replace(partition, surfaces=tuple(surfaces))
                    # Import locally: proposals imports this optional generator.
                    from vectrify.refine.cel_plan.proposals import bounds

                    ids = (*proposal.ids, newid)
                    bb = bounds(state.document, document, ids)
                    proposal.component.validate(
                        state.document, document, old, partition, ids, bb, work
                    )
                    details = proposal.details or {}
                    self.diagnostics["proposals"] += 1
                    yield replace(
                        proposal,
                        document=document,
                        partition=partition,
                        ids=ids,
                        bounds=bb,
                        parameters=(
                            *proposal.parameters,
                            ("band-stroke", surface.id, tolerance, variant.width),
                        ),
                        estimate=proposal.estimate
                        + len(model.geometry.subpaths[0].nodes)
                        + sum(len(s.nodes) for s in marks.subpaths)
                        - old_nodes,
                        details={
                            **details,
                            "geometry_constraints": sorted(
                                set(details.get("geometry_constraints", ()))
                                | {surface.id, newid}
                            ),
                            "chain_constraints": discard(
                                details.get("chain_constraints"), (surface.id,)
                            ),
                            "planned_band_stroke": {
                                "id": surface.id,
                                "marks": newid,
                                "tolerance": tolerance,
                                "width": variant.width,
                                "intrinsic_width": model.width,
                                "stroke_nodes": len(model.geometry.subpaths[0].nodes),
                                "exact_mark_contours": len(marks.subpaths),
                                "inherited_unrepresented_source_pixels": inherited,
                                "source_line_comparison": comparison,
                                "ownership": "co-planned-from-material-ancestor",
                            },
                        },
                    )
                    published += 1
                if published:
                    return

    def coverage(self, document, oid, geometry, own, rule, work):
        e = self.evidence
        matrix = root_matrix(document, oid)
        nb = np.asarray(curve_path(transformed_geometry(geometry, matrix), rule).bounds)
        if not np.isfinite(nb).all() or np.max(np.abs(nb)) > 1_000_000:
            return None
        ys, xs = np.nonzero(own)
        margin = 16 + int(
            np.ceil(MAX_WIDTH * np.linalg.norm(matrix[:4]) * max(e.scale) / 2)
        )
        lo = np.minimum(
            np.floor((nb[:2] - e.offset) * e.scale) - margin, (xs.min(), ys.min())
        )
        hi = np.maximum(
            np.ceil((nb[2:] - e.offset) * e.scale) + margin,
            (xs.max() + 1, ys.max() + 1),
        )
        box = Box(int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])).expand(
            0, self.graph.labels.shape
        )
        if not 0 < box.area <= MAX_CROP_PIXELS:
            return None
        sx, sy = e.scale
        ox, oy = e.offset
        frame = " ".join(str(v) for v in matrix)

        def raster(shape, width=None):
            _check(work)
            style = (
                f'fill="white" fill-rule="{rule}"'
                if width is None
                else f'fill="none" stroke="white" stroke-width="{width}" '
                'stroke-linecap="butt" stroke-linejoin="round"'
            )
            svg = (
                f'<svg width="{box.right - box.x}" height="{box.bottom - box.y}" '
                f'viewBox="{ox + box.x / sx} {oy + box.y / sy} '
                f'{(box.right - box.x) / sx} {(box.bottom - box.y) / sy}">'
                f'<path d="{shape.path_data()}" transform="matrix({frame})" '
                f"{style}/></svg>"
            )
            result = render(svg, (box.right - box.x, box.bottom - box.y))[..., 3]
            _check(work)
            return result

        return box, raster

    def partition(
        self,
        state,
        proposal,
        scoped,
        labels,
        surface,
        newid,
        box,
        marks,
        bodyalpha,
        markalpha,
        work,
    ):
        _check(work)
        indices = {s.id: i for i, s in enumerate(scoped)}
        lookup = np.full(proposal.partition.atoms.namespace_count, -1, np.int32)
        for s in scoped:
            lookup[list(s.members)] = indices[s.id]
        classes = np.maximum(lookup[labels], 0)
        crop = classes[box.slices]
        crop[marks] = len(scoped)
        members = tuple(
            sorted(
                i
                for s in state.partition.surfaces
                if s.id in proposal.ids and s.role != "underlay"
                for i in s.members
            )
        )
        if not np.array_equal(np.isin(self.graph.labels, members), lookup[labels] >= 0):
            raise ValueError(
                "Material scope changed source ownership outside its ancestor"
            )
        atoms, groups = (state.partition.atoms or Atoms.original(self.graph)).partition(
            self.graph,
            members,
            classes,
            len(scoped) + 1,
            work,
            compact=True,
            retain_parent=True,
        )
        newlabels = atoms.labels(self.original, work)
        mapping = {}
        for start in range(0, labels.size, CHUNK_PIXELS):
            _check(work)
            pairs = np.unique(
                np.column_stack(
                    (
                        labels.ravel()[start : start + CHUNK_PIXELS],
                        newlabels.ravel()[start : start + CHUNK_PIXELS],
                    )
                ),
                axis=0,
            )
            for old, new in pairs:
                mapping.setdefault(int(old), set()).add(int(new))

        def remap(values):
            return tuple(sorted({i for old in values for i in mapping[old]}))

        surfaces = [
            Surface(
                s.id,
                groups[indices[s.id]] if s.id in indices else remap(s.members),
                s.role,
                remap(s.covered),
            )
            for s in proposal.partition.surfaces
            if s.id != surface.id
        ]
        active = set(map(int, np.unique(newlabels))) - self.original.hidden
        for oid, owned, alpha, role in (
            (surface.id, groups[indices[surface.id]], bodyalpha, "overlay"),
            (newid, groups[-1], markalpha, "surface"),
        ):
            if not owned:
                return None
            covered = tuple(
                sorted(
                    (
                        set(map(int, np.unique(newlabels[box.slices][alpha > 1 / 255])))
                        & active
                    )
                    - set(owned)
                )
            )
            surfaces.append(Surface(oid, owned, role, covered))
        partition = Partition(tuple(surfaces), atoms)
        if not partition.follows(state.partition):
            raise ValueError("Co-planned stroke lost the original source lineage")
        _check(work)
        return partition, newlabels

    @staticmethod
    def document(document, oid, newid, model, marks, style):
        editor = Editor(document, selection=Selection(whole_document=True))
        element = document.element(oid)
        parent = document.ancestry(oid)[-2]
        remaining = identified(marks, newid)
        with editor.transaction(
            "Co-plan editable ink stroke and independent marks"
        ) as tx:
            tx.replace_geometry(oid, identified(model.geometry, oid))
            tx.set_fill(oid, "none")
            tx.set_attributes(
                oid,
                {
                    "stroke": style["fill"],
                    "stroke-width": repr(model.width),
                    "stroke-opacity": style["fill-opacity"],
                    "stroke-linecap": "butt",
                    "stroke-linejoin": "round",
                    "fill-opacity": "1",
                },
            )
            tx.insert_object(
                parent.id,
                replace(element, id=newid, geometry_id=remaining.id),
                index=next(i for i, c in enumerate(parent.children) if c.id == oid) + 1,
                geometries=(remaining,),
            )
        return editor.snapshot.document
