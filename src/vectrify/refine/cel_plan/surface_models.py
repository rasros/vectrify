"""Whole-owner material models across fragmented shade/edge evidence.

A seed supplies a bounded linear paint hypothesis. Complete source pixels,
connectivity, current paint frames and exact local/full checks determine whether
it can replace a family. Fixed atoms and chosen ink overlays remain independent;
no majority assignment or relaxed coverage policy is used.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pathops
from cairosvg.colors import color

from vectrify.document import Editor, Selection
from vectrify.document.hit_test import multiply
from vectrify.document.join import (
    curve_path,
    path_style,
    transformed_geometry,
    union_geometry,
)
from vectrify.document.model import paint_server
from vectrify.document.paint import gradient_stops
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.constraints import discard
from vectrify.refine.cel_plan.families import Families, _gradient, _opacity
from vectrify.refine.cel_plan.local import Box
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.nested import in_core
from vectrify.refine.cel_plan.opacity import Paint, fit_samples
from vectrify.refine.cel_plan.ownership import Surface
from vectrify.refine.cel_plan.search import Proposal, State
from vectrify.refine.crossings import crossings

MAX_PIXELS = 1536**2
CHUNK_PIXELS = 65_536
MAX_OWNERS = 4096
MAX_SEEDS = 8
MAX_PATHS = 128
MAX_NODES = 6000
MAX_PROPOSALS = 16
MAX_ORDER_PROOFS = 128
RESIDUALS = (24.0, 48.0)


def prediction(paint: Paint, xy: np.ndarray, *, extend: bool = True) -> np.ndarray:
    """Extend a seed's linear hypothesis; exported paint refits the whole family.

    A seed's observed spatial extent is not the extent of the new surface. SVG
    gradient clamping would prevent a small seed from proposing a broad ramp.
    The final whole-family paint is independently screened and rendered.
    """
    if paint.gradient is None:
        return np.tile(color(paint.color)[:3], (len(xy), 1))
    ramp = paint.gradient
    axis = np.asarray(ramp.end) - ramp.start
    u = ((xy - ramp.start) @ axis / float(axis @ axis))[:, None]
    if not extend:
        u = np.clip(u, 0, 1)
    ends = np.asarray([color(s.colour)[:3] for s in ramp.stops])
    return np.clip(ends[0] * (1 - u) + ends[1] * u, 0, 1)


class MaterialSurfaces:
    def __init__(self, families: Families, options: Options):
        self.families, self.options = families, options
        self.ink_pixels = np.bincount(
            families.graph.labels.ravel(),
            weights=families.evidence.drawn.ravel(),
            minlength=len(families.graph.regions),
        )
        self.diagnostics = {
            "eligible_peak": 0,
            "seed_models": 0,
            "inspected_pixels": 0,
            "seed_residual_exclusions": 0,
            "refit_residual_exclusions": 0,
            "bounded_groups": 0,
            "core_exclusions": 0,
            "order_exclusions": 0,
            "order_proofs": 0,
            "order_limits": 0,
            "proposals": 0,
        }

    def _eligible(self, state: State, work: Work):
        if state.partition is None:
            return {}
        graph = self.families.graph
        constrained = set(state.details.get("paint_constraints", ()))
        eligible = {}
        for surface in state.partition.surfaces:
            if work.interrupted:
                return {}
            if (
                surface.role != "surface"
                or surface.covered
                or surface.id in constrained
                or any(graph.regions[i].fixed for i in surface.members)
            ):
                continue
            element = state.document.element(surface.id)
            style = path_style(state.document, element)
            if (
                style["stroke"] != "none"
                or style["fill"] == "none"
                or float(style["opacity"]) != 1
                or float(style["fill-opacity"]) != 1
                or element.get("clip-path", "none") != "none"
            ):
                continue
            server = paint_server(style["fill"])
            if server is None:
                if color(style["fill"])[3] != 1:
                    continue
            else:
                stops = gradient_stops(state.document.element(server))
                if not stops or any(stop[1][3] != 1 for stop in stops):
                    continue
            regions = [graph.regions[i] for i in surface.members]
            components = {r.component for r in regions}
            nodes = sum(
                len(s.nodes) for s in state.document.geometry_for(surface.id).subpaths
            )
            if len(components) != 1 or nodes > MAX_NODES:
                continue
            eligible[surface.id] = (
                surface,
                sum(r.area for r in regions),
                nodes,
                (
                    state.document.ancestry(surface.id)[-2].id,
                    element.get("transform"),
                    next(iter(components)),
                ),
            )
            if len(eligible) > MAX_OWNERS:
                self.diagnostics["bounded_groups"] += 1
                return {}
        self.diagnostics["eligible_peak"] = max(
            self.diagnostics["eligible_peak"], len(eligible)
        )
        return eligible

    def _screen(self, eligible, paint, work, *, extend=True):
        evidence, graph = self.families.evidence, self.families.graph
        if graph.labels.size > MAX_PIXELS:
            self.diagnostics["bounded_groups"] += 1
            return None
        ids = sorted(eligible)
        owner = np.full(len(graph.regions), -1, dtype=np.int32)
        for index, oid in enumerate(ids):
            owner[list(eligible[oid][0].members)] = index
        maximum = np.zeros(len(ids))
        count = np.zeros(len(ids), dtype=np.int64)
        height, width = graph.labels.shape
        for box in Box(0, 0, width, height).chunks(CHUNK_PIXELS):
            if work.interrupted:
                return None
            labels = owner[graph.labels[box.slices]]
            y, x = np.nonzero((labels >= 0) & ~evidence.empty[box.slices])
            if not len(x):
                continue
            xy = np.column_stack((x + box.x + 0.5, y + box.y + 0.5))
            residual = (
                np.max(
                    np.abs(
                        prediction(paint, xy, extend=extend)
                        - evidence.target[box.slices][y, x] / 255
                    ),
                    axis=1,
                )
                * 255
            )
            np.maximum.at(maximum, labels[y, x], residual)
            count += np.bincount(labels[y, x], minlength=len(ids))
            self.diagnostics["inspected_pixels"] += len(x)
        if work.interrupted:
            return None
        # Complete source support is required, including an owner's distant atom.
        return {
            oid: float(maximum[index])
            for index, oid in enumerate(ids)
            if count[index] == eligible[oid][1] and count[index]
        }

    def groups(self, state: State, work: Work):
        eligible = self._eligible(state, work)
        if not eligible or state.partition is None:
            return
        neighbors = {oid: set() for oid in eligible}
        owners = state.partition.owners
        for edge in self.families.graph.boundaries:
            if work.interrupted:
                return
            a, b = owners.get(edge.left), owners.get(edge.right)
            if (
                a in eligible
                and b in eligible
                and a != b
                and eligible[a][3] == eligible[b][3]
            ):
                neighbors[a].add(b)
                neighbors[b].add(a)
        seen = set()
        # Coarse ink may be a shade fragment, so it is not a candidate veto.
        # Start material hypotheses in surfaces; every candidate owner, including
        # a coarse-ink owner, must satisfy the complete source-paint screen.
        seeds = sorted(
            (
                oid
                for oid in eligible
                if sum(self.ink_pixels[list(eligible[oid][0].members)])
                < 0.6 * eligible[oid][1]
            ),
            key=lambda oid: (-eligible[oid][1], oid),
        )[:MAX_SEEDS]
        for seed in seeds:
            if work.interrupted:
                return
            samples = self.families.samples(eligible[seed][0].members, work)
            if samples is None or not samples[2]:
                return
            paint = fit_samples(
                samples[0], samples[1], gradients=self.options.gradients
            )
            self.diagnostics["seed_models"] += 1
            maximum = self._screen(eligible, paint, work)
            if maximum is None:
                return
            for limit in RESIDUALS:
                if maximum.get(seed, float("inf")) > limit:
                    self.diagnostics["seed_residual_exclusions"] += 1
                    continue
                queue, visited, ids, nodes = deque([seed]), set(), [], 0
                while queue:
                    if work.interrupted:
                        return
                    oid = queue.popleft()
                    if oid in visited:
                        continue
                    visited.add(oid)
                    if maximum.get(oid, float("inf")) > limit:
                        continue
                    if len(ids) >= MAX_PATHS or nodes + eligible[oid][2] > MAX_NODES:
                        self.diagnostics["bounded_groups"] += 1
                        continue
                    ids.append(oid)
                    nodes += eligible[oid][2]
                    queue.extend(sorted(neighbors[oid] - visited))
                ids = tuple(sorted(ids))
                if len(ids) < 2:
                    continue
                members = tuple(
                    sorted(i for oid in ids for i in eligible[oid][0].members)
                )
                samples = self.families.samples(members, work)
                if samples is None or not samples[2]:
                    return
                fitted = fit_samples(
                    samples[0], samples[1], gradients=self.options.gradients
                )
                paints = [fitted]
                if fitted.gradient is not None:
                    paints.append(fit_samples(samples[0], samples[1], gradients=False))
                for variant in paints:
                    key = (ids, variant)
                    if key in seen:
                        continue
                    checked = self._screen(
                        {oid: eligible[oid] for oid in ids}, variant, work, extend=False
                    )
                    if checked is None:
                        return
                    if len(checked) != len(ids) or max(checked.values()) > limit:
                        self.diagnostics["refit_residual_exclusions"] += 1
                        continue
                    seen.add(key)
                    yield ids, members, variant, limit, max(checked.values())

    def _order_safe(self, state, ids, work):
        from vectrify.refine.cel_plan.proposals import bounds

        document = state.document
        parent = document.ancestry(ids[0])[-2]
        selected = set(ids)
        indices = [i for i, child in enumerate(parent.children) if child.id in selected]
        # Merging into the last fragment moves only earlier fragment geometry
        # above intervening siblings. Prove precisely that changing prefix.
        accumulated = None
        intervening = None
        group_nodes = proofs = 0
        native = bounds(document, document, ids)
        inverse = inverse_matrix(root_matrix(document, ids[0]))

        def disjoint():
            nonlocal proofs
            if intervening is None:
                return True
            if work.interrupted or proofs >= MAX_ORDER_PROOFS:
                self.diagnostics["order_limits"] += 1
                return False
            overlap = pathops.op(accumulated, intervening, pathops.PathOp.INTERSECTION)
            proofs += 1
            self.diagnostics["order_proofs"] += 1
            return abs(overlap.area) <= 1e-8 and not work.interrupted

        for child in parent.children[min(indices) : max(indices)]:
            if work.interrupted:
                return False
            if child.id in selected:
                # Every sibling since the preceding selected fragment crosses
                # precisely the same accumulated prefix. Their actual union
                # needs one proof, rather than one proof per tiny raster cell.
                if not disjoint():
                    return False
                intervening = None
                group_nodes = 0
                shape = curve_path(
                    document.geometry_for(child.id),
                    path_style(document, child)["fill-rule"],
                )
                accumulated = (
                    shape
                    if accumulated is None
                    else pathops.op(accumulated, shape, pathops.PathOp.UNION)
                )
                if sum(1 for _ in accumulated) > MAX_NODES:
                    self.diagnostics["order_limits"] += 1
                    return False
                continue
            if accumulated is None:
                continue
            if child.tag != "path":
                return False
            if not bounds(document, document, (child.id,)).intersection(native).area:
                continue
            style = path_style(document, child)
            geometry = document.geometry_for(child.id)
            group_nodes += sum(len(s.nodes) for s in geometry.subpaths)
            if (
                style["stroke"] != "none"
                or child.get("clip-path", "none") != "none"
                or group_nodes > MAX_NODES
            ):
                self.diagnostics["order_limits"] += 1
                return False
            geometry = transformed_geometry(
                geometry, multiply(inverse, root_matrix(document, child.id))
            )
            shape = curve_path(geometry, style["fill-rule"])
            intervening = (
                shape
                if intervening is None
                else pathops.op(intervening, shape, pathops.PathOp.UNION)
            )
            if sum(1 for _ in intervening) > MAX_NODES:
                self.diagnostics["order_limits"] += 1
                return False
        return disjoint() and not work.interrupted

    def __call__(self, state: State, work: Work):
        from vectrify.refine.cel_plan.proposals import bounds

        if state.partition is None:
            return
        emitted = 0
        for ids, members, paint, limit, residual in self.groups(state, work):
            if work.interrupted or emitted >= MAX_PROPOSALS:
                return
            document = state.document
            parent = document.ancestry(ids[0])[-2]
            selected = [c for c in parent.children if c.id in ids]
            survivor = selected[-1].id
            shape = union_geometry(
                [document.geometry_for(c.id) for c in selected],
                [path_style(document, c) for c in selected],
            )
            if work.interrupted:
                return
            if sum(len(s.nodes) for s in shape.subpaths) > MAX_NODES or crossings(
                shape
            ):
                self.diagnostics["bounded_groups"] += 1
                continue
            if self.families.evidence.opacity is not None and not in_core(
                state, members, survivor, shape, work
            ):
                self.diagnostics["core_exclusions"] += 1
                continue
            if not self._order_safe(state, ids, work):
                self.diagnostics["order_exclusions"] += int(not work.interrupted)
                continue
            opacity = _opacity(document, survivor)
            if opacity <= 0:
                continue
            partition = state.partition.replace(ids, (Surface(survivor, members),))
            editor = Editor(document, selection=Selection(whole_document=True))
            with editor.transaction("Propose broad coherent material") as transaction:
                transaction.delete_objects(frozenset(set(ids) - {survivor}))
                transaction.replace_geometry(survivor, shape)
                transaction.set_fill(
                    survivor,
                    _gradient(
                        paint, self.families.evidence, document, survivor, opacity
                    )
                    if paint.gradient
                    else paint.color,
                )
                transaction.set_attributes(
                    survivor,
                    {
                        "fill-rule": "nonzero",
                        "stroke": "none",
                        "fill-opacity": "1"
                        if paint.gradient
                        else repr(min(1.0, paint.opacity / opacity)),
                    },
                )
            proposed = editor.snapshot.document
            holds = set(state.details.get("geometry_constraints", ()))
            preserve = bool(holds.intersection(ids))
            holds.difference_update(ids)
            if preserve:
                holds.add(survivor)
            self.diagnostics["proposals"] += 1
            emitted += 1
            yield Proposal(
                "family-surface",
                ids,
                ("gradient" if paint.gradient else "flat", limit, members),
                state.key,
                proposed,
                bounds(document, proposed, ids),
                details={
                    "regions": sum(s.role != "underlay" for s in partition.surfaces),
                    "geometry_constraints": sorted(holds),
                    "chain_constraints": discard(
                        state.details.get("chain_constraints"), ids
                    ),
                    "material_surface": {
                        "paint_model": "gradient" if paint.gradient else "flat",
                        "source_members": members,
                        "removed_paths": len(ids) - 1,
                        "maximum_paint_residual": residual,
                        "paint_residual_limit": limit,
                    },
                },
                dependencies=(parent.id,),
                partition=partition,
            )
