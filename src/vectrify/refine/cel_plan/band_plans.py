"""Co-plan a stroke and exact independent marks before material atom allocation.

The ordinary material proposal and this alternative share one ancestor. Replaying
its source classes from that ancestor preserves the immutable ledger prefix and
its existing limits. This is not a post-hoc rewrite of an accepted namespace.
Unrepresented source pixels retain the ordinary band's owner; a painted native
comparison must still preserve measured source support and individual gaps.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pathops
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from vectrify.document import Editor, Selection, export_svg
from vectrify.document.join import curve_path, path_style, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.atoms import (
    CHUNK_PIXELS,
    MAX_PARTS,
    AtomLimitError,
    Atoms,
)
from vectrify.refine.cel_plan.band_fit import BandFit, opaque_core
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
from vectrify.refine.cel_plan.local import MAX_CROP_PIXELS, Box, _native_raster
from vectrify.refine.cel_plan.model import StageInterruptedError
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.paint_continuation import PaintContinuation
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.source_bands import SourceBands
from vectrify.refine.cel_plan.source_caps import SourceCaps

TOLERANCES = (0.25, 0.15, 0.1)
MAX_SUBPATHS = 16
MAX_COMPOUND_NODES = 1024
MAX_COMPOUND_SUBPATHS = 64
MAX_EXTENSIONS = 5


def separate(geometry, rule, work, *, index=None):
    """Retain every other contour exactly; holes and interacting fills exclude."""
    _check(work)
    if (
        not 2
        <= len(geometry.subpaths)
        <= (MAX_SUBPATHS if index is None else MAX_COMPOUND_SUBPATHS)
        or any(not s.closed for s in geometry.subpaths)
        or sum(len(s.nodes) for s in geometry.subpaths)
        > (MAX_NODES if index is None else MAX_COMPOUND_NODES)
    ):
        return None
    if index is None:
        index = max(
            range(len(geometry.subpaths)), key=lambda i: len(geometry.subpaths[i].nodes)
        )
    elif type(index) is not int or not 0 <= index < len(geometry.subpaths):
        raise ValueError("A separated band requires an existing contour index")
    if len(geometry.subpaths[index].nodes) > MAX_NODES:
        return None
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

    def __init__(
        self,
        evidence,
        graph,
        original,
        *,
        guard,
        width_fixed=False,
        width=0,
        tolerance=0,
    ):
        self.evidence, self.graph, self.original, self.guard = (
            evidence,
            graph,
            original,
            guard,
        )
        self.width_fixed = width_fixed
        self.width, self.tolerance = width, tolerance
        self._source_fitter = None
        self.diagnostics = dict.fromkeys(
            ("eligible", "source_exclusions", "bounded", "ambiguous", "proposals"), 0
        )

    def __call__(self, state, proposal, work):
        try:
            parents = []
            for candidate in self.proposals(state, proposal, work):
                yield candidate
                if len(parents) < MAX_EXTENSIONS:
                    parents.append(candidate)
            # Preserve every existing width alternative before offering a
            # complete sibling with another isolated band. Replaying from the
            # same ancestor avoids splitting a finished, saturated ledger.
            extended = []
            for candidate in parents:
                for sibling in self.proposals(state, candidate, work, isolated=True):
                    yield sibling
                    if len(extended) < MAX_EXTENSIONS:
                        extended.append(sibling)
            # Keep the entire preceding intrinsic-band prefix before complete
            # source-fitted siblings. Every ledger replays the same ancestor.
            for candidate in extended:
                yield from self.proposals(
                    state, candidate, work, isolated=True, source_fit=True
                )
        except StageInterruptedError:
            return
        except (AtomLimitError, pathops.PathOpsError):
            self.diagnostics["bounded"] += 1

    def proposals(self, state, proposal, work, *, isolated=False, source_fit=False):
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
        eligible: list[tuple[int, Surface, dict[str, str], int | None]] = []
        for surface in scoped:
            _check(work)
            style = supported_style(proposal.document, surface.id)
            if style is None:
                continue
            own = np.isin(labels, surface.members)
            if not own.any() or (
                not isolated and self.evidence.drawn[own].sum() < own.sum() * 0.6
            ):
                continue
            geometry = proposal.document.geometry_for(surface.id)
            nodes = sum(len(s.nodes) for s in geometry.subpaths)
            if isolated:
                if (
                    not 2 <= len(geometry.subpaths) <= MAX_COMPOUND_SUBPATHS
                    or nodes > MAX_COMPOUND_NODES
                    or not self.evidence.drawn[own].any()
                ):
                    continue
                for index, sub in enumerate(geometry.subpaths):
                    _check(work)
                    if not sub.closed or not 3 <= len(sub.nodes) <= MAX_NODES:
                        continue
                    extent = curve_path(replace(geometry, subpaths=(sub,))).bounds
                    if max(extent[2] - extent[0], extent[3] - extent[1]) < 8:
                        continue
                    eligible.append((len(sub.nodes), surface, style, index))
            elif 2 <= len(geometry.subpaths) <= MAX_SUBPATHS and nodes <= MAX_NODES:
                eligible.append((nodes, surface, style, None))
        eligible.sort(key=lambda v: (-v[0], v[1].id))
        before, guard = None, None
        for _priority, surface, style, index in eligible[:MAX_PATHS]:
            _check(work)
            self.diagnostics["eligible"] += 1
            geometry = proposal.document.geometry_for(surface.id)
            old_nodes = sum(len(s.nodes) for s in geometry.subpaths)
            parts = separate(geometry, style["fill-rule"], work, index=index)
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
            mainalpha = raster(main)
            if isolated:
                sampled = own[box.slices] & (mainalpha > 0.05)
                if (
                    not sampled.any()
                    or self.evidence.drawn[box.slices][sampled].mean() < 0.6
                ):
                    continue
            classified = mark_components(
                own[box.slices], mainalpha, raster(marks), self.evidence.scale, work
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
            source_seed = None
            if source_fit:
                source_seed = SourceBands(self.evidence, guard).seed(
                    proposal.document,
                    surface.id,
                    main,
                    style["fill-rule"],
                    work,
                    tolerance=self.tolerance or 0.75,
                    width=self.evidence.filled_line_width or self.width,
                )
                if source_seed is None:
                    continue
            for tolerance in (self.tolerance or 0.75,) if source_fit else TOLERANCES:
                _check(work)
                model = (
                    replace(
                        source_seed.band,
                        geometry=transformed_geometry(
                            source_seed.band.geometry,
                            inverse_matrix(root_matrix(proposal.document, surface.id)),
                        ),
                    )
                    if source_seed is not None
                    else invert(main, style["fill-rule"], work, tolerance=tolerance)
                )
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
                    if source_fit
                    or self.width_fixed
                    or self.evidence.filled_line_width > 0
                    else tuple(
                        model.width + delta * amount
                        for amount in (0, -1, -0.5, 0.5, 1)
                        if 0.8 <= model.width + delta * amount <= MAX_WIDTH
                    )
                )
                for width in widths:
                    _check(work)
                    variant = replace(model, width=width)
                    extra_ids, source_details = (), None
                    if source_seed is not None:
                        source_own = np.zeros_like(own)
                        source_own[box.slices] = own[box.slices] & (mainalpha > 0.05)
                        fitted = self.source_document(
                            proposal,
                            surface.id,
                            newid,
                            main,
                            marks,
                            style,
                            source_seed,
                            source_own,
                            labels,
                            guard,
                            work,
                        )
                        if fitted is None:
                            self.diagnostics["source_exclusions"] += 1
                            continue
                        document, extra_ids, source_details = fitted
                        variant = replace(
                            model,
                            geometry=document.geometry_for(surface.id),
                            width=float(
                                path_style(document, document.element(surface.id))[
                                    "stroke-width"
                                ]
                            ),
                        )
                    else:
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
                        if record.id == surface.id or record.id in extra_ids:
                            view, painted = box, bodyalpha
                            if record.id != surface.id:
                                record_style = path_style(
                                    document, document.element(record.id)
                                )
                                coverage = self.coverage(
                                    document,
                                    record.id,
                                    document.geometry_for(record.id),
                                    np.isin(newlabels, record.members),
                                    record_style["fill-rule"],
                                    work,
                                )
                                if coverage is None:
                                    self.diagnostics["bounded"] += 1
                                    return
                                view, rasterize = coverage
                                painted = rasterize(
                                    document.geometry_for(record.id),
                                    float(record_style["stroke-width"])
                                    if record_style["stroke"] != "none"
                                    else None,
                                    cap=record_style["stroke-linecap"],
                                )
                            covered = tuple(
                                sorted(
                                    (
                                        set(
                                            map(
                                                int,
                                                np.unique(
                                                    newlabels[view.slices][
                                                        painted > 1 / 255
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

                    ids = tuple(dict.fromkeys((*proposal.ids, newid, *extra_ids)))
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
                        - old_nodes
                        + sum(
                            sum(
                                len(s.nodes)
                                for s in document.geometry_for(oid).subpaths
                            )
                            - sum(
                                len(s.nodes)
                                for s in proposal.document.geometry_for(oid).subpaths
                            )
                            for oid in extra_ids
                        ),
                        details={
                            **details,
                            "geometry_constraints": sorted(
                                set(details.get("geometry_constraints", ()))
                                | {surface.id, newid, *extra_ids}
                            ),
                            "chain_constraints": discard(
                                details.get("chain_constraints"),
                                (surface.id, *extra_ids),
                            ),
                            "planned_band_stroke": {
                                "id": surface.id,
                                "marks": newid,
                                **({"contour": index} if isolated else {}),
                                "tolerance": tolerance,
                                "width": variant.width,
                                **(
                                    {"intrinsic_width": model.width}
                                    if source_seed is None
                                    else {"source_width": source_seed.band.width}
                                ),
                                "stroke_nodes": len(model.geometry.subpaths[0].nodes),
                                "exact_mark_contours": len(marks.subpaths),
                                "inherited_unrepresented_source_pixels": inherited,
                                "source_line_comparison": comparison,
                                "ownership": "co-planned-from-material-ancestor",
                                **(
                                    {"source_fit": source_details}
                                    if source_details is not None
                                    else {}
                                ),
                            },
                            **(
                                {
                                    "planned_band_strokes": [
                                        *details.get(
                                            "planned_band_strokes",
                                            [details.get("planned_band_stroke", {})],
                                        ),
                                        {
                                            "id": surface.id,
                                            "marks": newid,
                                            "contour": index,
                                            "width": variant.width,
                                            "stroke_nodes": len(
                                                model.geometry.subpaths[0].nodes
                                            ),
                                        },
                                    ]
                                }
                                if isolated
                                else {}
                            ),
                        },
                    )
                    published += 1
                    if isolated:
                        return
                if published:
                    return

    def source_document(
        self, proposal, oid, newid, main, marks, style, seed, own, labels, guard, work
    ):
        """Assemble source caps and adjacent material before bounded fitting."""
        frame = root_matrix(proposal.document, oid)
        linear = np.asarray(frame[:4]).reshape(2, 2).T
        scale = float(np.linalg.norm(linear[:, 0]))
        if scale <= 1e-12 or not np.allclose(
            linear.T @ linear, np.eye(2) * scale**2, rtol=1e-8, atol=1e-10
        ):
            return None
        model = replace(
            seed.band,
            geometry=transformed_geometry(seed.band.geometry, inverse_matrix(frame)),
            width=seed.band.width / scale,
        )
        document = self.document(
            proposal.document, oid, newid, model, marks, {**style, "fill": seed.paint}
        )
        nb = curve_path(transformed_geometry(main, frame)).bounds
        parent = document.ancestry(oid)[-2].id
        result = SourceCaps(guard).extend(
            document,
            (
                s.id
                for s in proposal.partition.surfaces
                if s.id != oid and document.ancestry(s.id)[-2].id == parent
            ),
            nb,
            work,
        )
        witnesses = ()
        if result is not None:
            document, witnesses = result
        ids = [w["id"] for w in witnesses]
        eligible = {
            s.id: s
            for s in proposal.partition.surfaces
            if s.id != oid
            and s.role != "underlay"
            and document.ancestry(s.id)[-2].id == parent
            and path_style(document, document.element(s.id))["stroke"] == "none"
        }
        lookup = {i: s.id for s in eligible.values() for i in s.members}
        neighborhood = (
            binary_dilation(own, iterations=4)
            & ~own
            & ~self.evidence.drawn
            & ~self.evidence.empty
        )
        values, counts = np.unique(labels[neighborhood], return_counts=True)
        votes = {}
        for value, count in zip(values, counts, strict=True):
            if int(value) in lookup:
                target = lookup[int(value)]
                votes[target] = votes.get(target, 0) + int(count)
        continuation: dict[str, Any] | None = None
        if votes:
            target = max(votes, key=lambda k: (votes[k], k))
            core = opaque_core(document, oid, main, self.evidence.source_size, work)
            if core is None:
                return None
            result = PaintContinuation().extend(
                document, oid, main, target, work, core=core, rule=style["fill-rule"]
            )
            if result is not None:
                document, continuation = result
                ids.append(target)
        if self._source_fitter is None:
            self._source_fitter = BandFit(self.evidence, guard)
        result = self._source_fitter.fit(
            proposal.document,
            document,
            oid,
            seed,
            work,
            width_fixed=self.width_fixed or self.evidence.filled_line_width > 0,
        )
        if result is None:
            return None
        document, fit = result
        if continuation is not None:
            target = continuation["material"]
            assert isinstance(target, str)
            order = continuation["stroke_order"]
            assert isinstance(order, tuple)
            assert isinstance(order[0], int)
            # Independently remove only the proposed paint/order change while
            # retaining the fitted ink and recovered caps. Added material must
            # neither change alpha nor paint outside old fill / actual body.
            editor = Editor(document, selection=Selection(whole_document=True))
            with editor.transaction("Check exact continuation complement") as tx:
                tx.replace_geometry(target, proposal.document.geometry_for(target))
                tx.reorder_object(oid, order[0])
            _check(work)
            native = _native_raster(
                ET.fromstring(export_svg(document)), self.evidence.source_size
            ).root
            complement = _native_raster(
                ET.fromstring(export_svg(editor.snapshot.document)),
                self.evidence.source_size,
            ).root
            if not np.array_equal(native[..., 3], complement[..., 3]):
                return None
            root = ET.Element(
                "{http://www.w3.org/2000/svg}svg",
                {
                    "width": str(self.evidence.source_size[0]),
                    "height": str(self.evidence.source_size[1]),
                },
            )
            oldfill = ET.SubElement(
                root,
                "{http://www.w3.org/2000/svg}path",
                {
                    "d": main.path_data(),
                    "transform": "matrix(" + " ".join(map(str, frame)) + ")",
                    "fill": "white",
                    "fill-rule": style["fill-rule"],
                },
            )
            allowed = _native_raster(root, self.evidence.source_size).root[..., 3] > 0
            root.remove(oldfill)
            ink_style = path_style(document, document.element(oid))
            ET.SubElement(
                root,
                "{http://www.w3.org/2000/svg}path",
                {
                    "d": document.geometry_for(oid).path_data(),
                    "transform": "matrix(" + " ".join(map(str, frame)) + ")",
                    "fill": "none",
                    "stroke": "white",
                    "stroke-width": ink_style["stroke-width"],
                    "stroke-linecap": "butt",
                    "stroke-linejoin": "round",
                },
            )
            allowed |= _native_raster(root, self.evidence.source_size).root[..., 3] > 0
            _check(work)
            if (np.any(native != complement, axis=-1) & ~allowed).any():
                return None
            continuation = {
                **continuation,
                "native_alpha_exact": True,
                "native_footprint_exact": True,
            }
        return (
            document,
            tuple(dict.fromkeys(ids)),
            {
                "fit": fit,
                "caps": witnesses,
                "continuation": continuation,
                "material_votes": votes,
            },
        )

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

        def raster(shape, width=None, *, cap="butt"):
            _check(work)
            style = (
                f'fill="white" fill-rule="{rule}"'
                if width is None
                else f'fill="none" stroke="white" stroke-width="{width}" '
                f'stroke-linecap="{cap}" stroke-linejoin="round"'
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
