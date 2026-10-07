"""Source-supported straight shade cuts with exact atom and contour ownership.

Bounded oriented edge voting proposes lines independently of the coarse label
partition. Both sides must fit their complete source support. Current contour
booleans retain the exterior and holes; native validation decides admission.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pathops

from vectrify.document import Editor, Selection
from vectrify.document.join import curve_path, path_geometry, path_style
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.atoms import Atoms
from vectrify.refine.cel_plan.constraints import discard, merged
from vectrify.refine.cel_plan.families import _gradient, _opacity
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.nested import in_core
from vectrify.refine.cel_plan.opacity import fit_samples
from vectrify.refine.cel_plan.ownership import Surface
from vectrify.refine.cel_plan.search import Proposal
from vectrify.refine.cel_plan.surface_models import prediction
from vectrify.refine.crossings import crossings

MAX_OWNERS = 8
MAX_PIXELS = 262_144
MAX_EDGES = 4096
MAX_LINES = 4
MAX_PROPOSALS = 16
MAX_NODES = 6000
MAX_MEMBERS = 128
MIN_JUMP = 16.0
MAX_RESIDUAL = 48.0


def fit(xy, rgba, *, gradients):
    step = max(1, (len(xy) + 4095) // 4096)
    return fit_samples(xy[::step], rgba[::step], gradients=gradients)


def lines(target, own, offset, work):
    """Vote in bounded normal/rho bins using complete local color differences.

    Voting can sample edge observations; the paint screen below cannot sample
    away a disagreeing pixel. Quantized orientation is a proposal restriction,
    not a claim of general fitted canonical geometry.
    """
    observations = []
    for axis in (0, 1):
        if work.interrupted:
            return
        difference = np.max(np.abs(np.diff(target, axis=axis)), axis=2)
        adjacent = own[:-1] & own[1:] if axis == 0 else own[:, :-1] & own[:, 1:]
        y, x = np.nonzero(adjacent & (difference >= MIN_JUMP))
        if not len(x):
            continue
        step = max(1, (len(x) + MAX_EDGES // 2 - 1) // (MAX_EDGES // 2))
        x, y = x[::step], y[::step]
        xy = np.column_stack((x + offset[0] + 0.5, y + offset[1] + 0.5))
        xy[:, 1 - axis] += 0.5
        observations.append(xy)
    if not observations:
        return
    points = np.concatenate(observations)
    candidates = []
    for theta in np.arange(32) * np.pi / 32:
        if work.interrupted:
            return
        normal = np.array((np.cos(theta), np.sin(theta)))
        rho = np.rint(points @ normal).astype(int)
        values, counts = np.unique(rho, return_counts=True)
        for index in np.argsort(-counts, kind="stable")[:2]:
            value = int(values[index])
            supported = points[rho == value]
            tangent = supported @ np.array((-normal[1], normal[0]))
            if len(supported) < 8 or np.ptp(tangent) < 8:
                continue
            candidates.append((-int(counts[index]), float(theta), value))
    seen = []
    for _support, theta, rho in sorted(candidates):
        if work.interrupted:
            return
        normal = np.array((np.cos(theta), np.sin(theta)))
        # Neighboring orientation/rho bins represent one candidate boundary.
        if any(abs(theta - a) < np.pi / 16 and abs(rho - b) < 2 for a, b in seen):
            continue
        seen.append((theta, rho))
        yield normal, rho
        if len(seen) >= MAX_LINES:
            return


class SurfaceSplits:
    def __init__(self, families, options):
        self.families, self.options = families, options
        self.diagnostics = {
            "owners": 0,
            "bounded": 0,
            "lines": 0,
            "paint_exclusions": 0,
            "core_exclusions": 0,
            "atom_exclusions": 0,
            "proposals": 0,
        }

    def _paints(self, xy, rgba, work):
        paint = fit(xy, rgba, gradients=self.options.gradients)
        variants = [paint]
        if paint.gradient is not None:
            variants.append(fit(xy, rgba, gradients=False))
        result = []
        for paint in variants:
            if work.interrupted:
                return []
            residual = prediction(paint, xy, extend=False) * 255 - rgba[:, :3] * 255
            if np.max(np.abs(residual)) <= MAX_RESIDUAL:
                result.append((paint, float(np.square(residual).mean())))
        return result

    def _geometry(self, state, oid, normal, rho, *, geometry=None):
        evidence = self.families.evidence
        size = 2 * float(np.hypot(*self.families.graph.labels.shape)) + 4
        tangent = np.array((-normal[1], normal[0]))
        center = normal * rho
        points = np.array(
            [
                center + tangent * size,
                center - tangent * size,
                center - tangent * size - normal * size,
                center + tangent * size - normal * size,
            ]
        )
        points = points / evidence.scale + evidence.offset
        a, b, c, d, e, f = inverse_matrix(root_matrix(state.document, oid))
        points = points @ np.array(((a, b), (c, d))) + (e, f)
        clip = pathops.Path()
        clip.moveTo(*points[0])
        for point in points[1:]:
            clip.lineTo(*point)
        clip.close()
        original = curve_path(
            state.document.geometry_for(oid) if geometry is None else geometry,
            path_style(state.document, state.document.element(oid))["fill-rule"]
            if geometry is None
            else "nonzero",
        )
        left = pathops.op(original, clip, pathops.PathOp.INTERSECTION)
        right = pathops.op(original, clip, pathops.PathOp.DIFFERENCE)
        # The two children retain precisely the current filled contour set.
        return path_geometry(left), path_geometry(right)

    def __call__(self, state, work: Work):
        from vectrify.refine.cel_plan.proposals import bounds

        if state.partition is None:
            return
        graph, evidence = self.families.graph, self.families.evidence
        eligible = self.families.surface_models._eligible(state, work)
        emitted = 0
        for oid in sorted(eligible, key=lambda i: (-eligible[i][1], i))[:MAX_OWNERS]:
            if work.interrupted or emitted >= MAX_PROPOSALS:
                return
            surface = eligible[oid][0]
            boxes = [self.families.boxes[i] for i in surface.members]
            if len(boxes) > MAX_MEMBERS or any(b is None for b in boxes):
                self.diagnostics["bounded"] += 1
                continue
            y0, y1 = min(b[0].start for b in boxes), max(b[0].stop for b in boxes)
            x0, x1 = min(b[1].start for b in boxes), max(b[1].stop for b in boxes)
            if (y1 - y0) * (x1 - x0) > MAX_PIXELS:
                self.diagnostics["bounded"] += 1
                continue
            roi = (slice(y0, y1), slice(x0, x1))
            own = np.isin(graph.labels[roi], surface.members) & ~evidence.empty[roi]
            if int(own.sum()) < 32:
                continue
            self.diagnostics["owners"] += 1
            y, x = np.nonzero(own)
            xy = np.column_stack((x + x0 + 0.5, y + y0 + 0.5))
            rgba = np.column_stack(
                (
                    evidence.target[roi][own] / 255,
                    evidence.opacity[roi][own]
                    if evidence.opacity is not None
                    else np.ones(len(x)),
                )
            )
            baseline = fit(xy, rgba, gradients=self.options.gradients)
            baseline_error = float(
                np.square(
                    prediction(baseline, xy, extend=False) * 255 - rgba[:, :3] * 255
                ).mean()
            )
            for normal, rho in lines(evidence.target[roi], own, (x0, y0), work):
                if work.interrupted or emitted >= MAX_PROPOSALS:
                    return
                self.diagnostics["lines"] += 1
                selected = xy @ normal < rho
                if min(int(selected.sum()), int((~selected).sum())) < max(
                    16, 0.05 * len(x)
                ):
                    continue
                left_paints = self._paints(xy[selected], rgba[selected], work)
                right_paints = self._paints(xy[~selected], rgba[~selected], work)
                if not left_paints or not right_paints:
                    self.diagnostics["paint_exclusions"] += 1
                    continue
                shapes = self._geometry(state, oid, normal, rho)
                if work.interrupted:
                    return
                if (
                    any(not s.subpaths or crossings(s) for s in shapes)
                    or sum(len(p.nodes) for s in shapes for p in s.subpaths) > MAX_NODES
                ):
                    self.diagnostics["bounded"] += 1
                    continue
                if evidence.opacity is not None and any(
                    not in_core(state, surface.members, oid, shape, work)
                    for shape in shapes
                ):
                    self.diagnostics["core_exclusions"] += 1
                    continue
                # Classification records ALL source support, including pixels
                # of other owners, before selecting only this surface's atoms.
                yy, xx = np.ogrid[: graph.labels.shape[0], : graph.labels.shape[1]]
                classification = (xx + 0.5) * normal[0] + (yy + 0.5) * normal[1] < rho
                atoms = state.partition.atoms or Atoms.original(graph)
                try:
                    atoms, left_members, right_members = atoms.split(
                        graph, surface.members, classification, work
                    )
                except ValueError:
                    self.diagnostics["atom_exclusions"] += 1
                    continue
                new_id = f"{oid}-split-{atoms.key[:12]}-{rho}"
                known = {e.id for e in state.document.elements()}
                if new_id in known:
                    continue
                partition = state.partition.split(
                    (oid,),
                    (Surface(oid, left_members), Surface(new_id, right_members)),
                    atoms,
                )
                element = state.document.element(oid)
                parent = state.document.ancestry(oid)[-2]
                index = next(
                    i for i, child in enumerate(parent.children) if child.id == oid
                )
                for left_paint, left_error in left_paints:
                    for right_paint, right_error in right_paints:
                        if work.interrupted or emitted >= MAX_PROPOSALS:
                            return
                        split_error = (
                            left_error * selected.sum()
                            + right_error * (~selected).sum()
                        ) / len(x)
                        if split_error >= baseline_error * 0.9:
                            self.diagnostics["paint_exclusions"] += 1
                            continue
                        editor = Editor(
                            state.document, selection=Selection(whole_document=True)
                        )
                        with editor.transaction(
                            "Propose source-supported shade cut"
                        ) as tx:
                            tx.replace_geometry(oid, shapes[0])
                            tx.insert_object(
                                parent.id,
                                replace(element, id=new_id, geometry_id=shapes[1].id),
                                index=index + 1,
                                geometries=(shapes[1],),
                            )
                        split_document = editor.snapshot.document
                        for child, paint in ((oid, left_paint), (new_id, right_paint)):
                            opacity = _opacity(split_document, child)
                            if opacity <= 0:
                                break
                            with editor.transaction("Fit shade-side paint") as tx:
                                tx.set_fill(
                                    child,
                                    _gradient(
                                        paint, evidence, split_document, child, opacity
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
                            records = discard(
                                state.details.get("chain_constraints"), (oid,)
                            )
                            for child in (oid, new_id):
                                rebound = merged(
                                    state.details.get("chain_constraints"),
                                    state.document,
                                    proposed,
                                    (oid,),
                                    child,
                                    work,
                                )
                                if (
                                    records is not None
                                    and rebound is not None
                                    and child in rebound.get("paths", {})
                                ):
                                    records["paths"][child] = rebound["paths"][child]
                            holds = set(state.details.get("geometry_constraints", ()))
                            # Both children retain the parent's protected exterior;
                            # The new cut stays protected pending fit permissions.
                            holds.update((oid, new_id))
                            emitted += 1
                            self.diagnostics["proposals"] += 1
                            yield Proposal(
                                "surface-split",
                                (oid, new_id),
                                (
                                    tuple(float(v) for v in normal),
                                    rho,
                                    "gradient" if left_paint.gradient else "flat",
                                    "gradient" if right_paint.gradient else "flat",
                                ),
                                state.key,
                                proposed,
                                bounds(state.document, proposed, (oid, new_id)),
                                details={
                                    "regions": sum(
                                        s.role != "underlay" for s in partition.surfaces
                                    ),
                                    "geometry_constraints": sorted(holds),
                                    "chain_constraints": records,
                                    "source_split": {
                                        "namespace": atoms.key,
                                        "new_atoms": len(atoms.cuts)
                                        - (
                                            len(state.partition.atoms.cuts)
                                            if state.partition.atoms
                                            else 0
                                        ),
                                        "paint_error_before": baseline_error,
                                        "paint_error_after": float(split_error),
                                    },
                                },
                                dependencies=(parent.id,),
                                partition=partition,
                            )
                        if work.interrupted:
                            return
