"""Whole-core source paint cells, independent of the existing fill fragments.

The actual coverage carrier retains its contour, holes and intrinsic opacity.
Source-driven binary material cells replace primary fill paths together; ink
and protected owners remain above them. All primary atoms are retained or
split exactly, and native scoring chooses between this and the detailed state.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace

import numpy as np
import pathops
from cairosvg.colors import color
from scipy.ndimage import (
    binary_propagation,
    distance_transform_edt,
    find_objects,
    gaussian_filter,
    label,
)

from vectrify.document import Editor, Geometry, Selection
from vectrify.document.join import (
    curve_path,
    path_geometry,
    path_style,
    transformed_geometry,
)
from vectrify.document.lines import open_path
from vectrify.document.model import paint_server
from vectrify.document.paint import gradient_stops
from vectrify.document.redraw import root_matrix
from vectrify.document.svg import parse_path
from vectrify.document.topology import inverse_matrix
from vectrify.refine import cel
from vectrify.refine.cel_plan.atoms import Atoms
from vectrify.refine.cel_plan.component_edits import ComponentEdit
from vectrify.refine.cel_plan.constraints import discard
from vectrify.refine.cel_plan.facet_lines import FacetLines
from vectrify.refine.cel_plan.families import _gradient, _opacity
from vectrify.refine.cel_plan.fill_winding import resolved
from vectrify.refine.cel_plan.geometry import Boundaries, InkBoundaries, ink_limits
from vectrify.refine.cel_plan.ink_models import models as ink_models
from vectrify.refine.cel_plan.ink_models import owned_model
from vectrify.refine.cel_plan.ink_replace import identified
from vectrify.refine.cel_plan.local import Box
from vectrify.refine.cel_plan.material_groups import grouped, ink_paint_links
from vectrify.refine.cel_plan.materials import MAX_REGIONS, moments
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.nested import in_core, opaque_fill
from vectrify.refine.cel_plan.opacity import Paint
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.refine import _bounds
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import Proposal
from vectrify.refine.cel_plan.source_roles import moments as role_moments
from vectrify.refine.cel_plan.source_roles import observations
from vectrify.refine.cel_plan.surface_models import prediction
from vectrify.refine.cel_plan.surface_splits import (
    MAX_RESIDUAL,
    SurfaceSplits,
    fit,
    lines,
)
from vectrify.refine.colour_regions import colour
from vectrify.refine.crossings import crossings

MAX_PIXELS = 1536**2
MAX_NATIVE_PIXELS = 4 * 1024**2
MAX_PATHS = 4096
MAX_INPUT_NODES = 16_384
MAX_OUTPUT_NODES = 6000
MAX_CORES = 4
MAX_CELLS = 8
MAX_PLANES = 32
PLANE_PREFIXES = (1, 2, 3, 4, 6, 8, 12, 16, 24, MAX_PLANES)
MAX_REGION_CELLS = 64
MAX_REGION_EDGES = 16_384
CHUNK = 65_536


def _filled(geometry, rule):
    path = curve_path(geometry, rule)
    path.simplify()
    return path_geometry(path)


def _supported_outlines(labels, boundary, check):
    """Retry collapsed materials on both sides of their shared source chains."""
    outlines = cel.region_outlines(labels, 0, fit_boundary=boundary, check=check)
    collapsed = set()
    for i, data in outlines.items():
        check()
        if i > 0 and not _filled(parse_path(data), "evenodd").subpaths:
            collapsed.add(i)
    if not collapsed:
        return outlines, 0
    padded = np.pad(labels, 1, constant_values=-1)

    def supported_boundary(points, tolerance):
        check()
        middle = (points[0] + points[1]) / 2 + 1
        direction = points[1] - points[0]
        normal = np.array([-direction[1], direction[0]]) * 0.25
        x, y = np.floor(middle + normal).astype(int)
        left = int(padded[y, x])
        x, y = np.floor(middle - normal).astype(int)
        right = int(padded[y, x])
        if collapsed.intersection((left, right)):
            fitted = cel.simplify(points, 0)
            return [("L", tuple(map(float, p))) for p in fitted[1:]]
        return boundary(points, tolerance)

    return (
        cel.region_outlines(labels, 0, fit_boundary=supported_boundary, check=check),
        len(collapsed),
    )


@dataclass(eq=False)
class Cell:
    key: str
    indices: np.ndarray
    region: Geometry
    draw: Geometry
    paint: Paint
    error: float
    residual: float
    support: int | None = None
    stroke: dict | None = None
    footprint: Geometry | None = None
    covered_classes: tuple[int, ...] = ()
    ink: bool = False


class CoreCells:
    """Owned material replacements with bounded joint ink/material hypotheses.

    Joint mode can consume paint owners that overlap inferred ink. Dynamic
    paint costs and shared fitting compete in native search for substantially
    fragmented components. Static budget grouping remains an offline ablation.
    Neither mode claims to complete continuous ink and facet reconstruction.
    """

    def __init__(
        self,
        families,
        options,
        *,
        minimum_paths=3,
        joint=False,
        grouping="static",
        boundary_fit="polygon",
        ink_support="paired",
        layout="regions",
        ink_roles="connected",
        ink_coverage="visible",
        source_profiles=None,
    ):
        if layout not in {"regions", "planes", "ink-planes"} or (
            layout != "regions" and not joint
        ):
            raise ValueError("Planar layout requires a joint component proposal")
        if layout == "ink-planes" and (
            ink_support != "connected" or grouping not in {"ward", "paint-fit"}
        ):
            raise ValueError("Ink planes require connected dynamic source ink")
        if grouping not in {"static", "ward", "paint-fit"} or (
            grouping != "static" and not joint
        ):
            raise ValueError("Dynamic grouping requires a joint material proposal")
        if boundary_fit not in {"polygon", "curve", "anchored"} or (
            boundary_fit != "polygon" and not joint
        ):
            raise ValueError("Boundary fitting requires a joint material proposal")
        if ink_support not in {"paired", "connected"} or (
            ink_support != "paired"
            and (not joint or grouping not in {"ward", "paint-fit"})
        ):
            raise ValueError("Connected ink requires a dynamic joint material proposal")
        if ink_roles not in {"connected", "fitted"} or (
            ink_roles == "fitted"
            and (ink_support != "connected" or layout != "regions")
        ):
            raise ValueError(
                "Fitted ink roles require connected joint material regions"
            )
        self.families, self.options = families, options
        self.minimum_paths = minimum_paths
        self.joint = joint
        self.grouping = grouping
        self.boundary_fit = boundary_fit
        self.ink_support = ink_support
        self.ink_roles = ink_roles
        if ink_coverage not in {"visible", "fractional"} or (
            ink_coverage == "fractional"
            and (ink_support != "connected" or layout != "regions")
        ):
            raise ValueError(
                "Fractional ink coverage requires connected material regions"
            )
        self.ink_coverage = ink_coverage
        if source_profiles is not None and (
            ink_support != "connected" or layout != "regions"
        ):
            raise ValueError("Source line profiles require connected material regions")
        self.source_profiles = source_profiles
        self.layout = layout
        self.splitter = SurfaceSplits(families, options)
        self._ink = None
        self._ridge_support = None
        self._moments = None
        self._facet_lines = None
        self.diagnostics: dict = dict.fromkeys(
            (
                "cores",
                "selected_paths",
                "retained_paths",
                "source_pixels",
                "votes",
                "cells",
                "atom_exclusions",
                "geometry_exclusions",
                "alpha_exclusions",
                "style_exclusions",
                "bounded",
                "proposals",
                "ridge_pixels",
                "ridge_owners",
                "region_exclusions",
                "region_proposals",
                "region_candidates",
                "region_retained_owners",
                "planar_paint_exclusions",
                "joint_proposals",
                "joint_forced_merges",
                "empty_cell_exclusions",
                "degenerate_region_retries",
                "hierarchy_components_peak",
                "hierarchy_retained_owners_peak",
                "hierarchy_queue_peak",
                "hierarchy_queue_rebuilds",
                "hierarchy_ink_cells_peak",
                "hierarchy_ink_paint_links_peak",
                "hierarchy_unmet_budgets",
                "source_role_observations_peak",
                "source_role_mixed_owners_peak",
                "source_stroke_attempts",
                "source_stroke_models",
                "source_stroke_carrier_exclusions",
                "source_stroke_owner_exclusions",
                "source_chain_small_roots",
                "source_class_roots_peak",
                "source_final_cells_peak",
                "source_fitted_role_pixels",
                "source_unmodeled_material_pixels",
                "hierarchy_model_evaluations",
                "hierarchy_model_limits",
                "hierarchy_alpha_exclusions",
                "plane_prefixes",
                "plane_cells_peak",
                "facet_edge_points",
                "ink_plane_seeds",
                "ink_plane_cells",
                "ink_plane_pixels",
                "ink_plane_duplicate_seeds",
                "anchored_ink_straights",
                "anchored_ink_curves",
                "anchored_ink_ellipses",
                "anchored_ink_precise_marks",
            ),
            0,
        )

    def _ink_support(self, work):
        """Broad dark material is not ink without a source-supported trough."""
        if self._ridge_support is None:
            evidence = self.families.evidence
            visible = ~evidence.empty
            light = gaussian_filter(cel.lightness(evidence.target) * visible, 0.5)
            light /= np.maximum(gaussian_filter(visible.astype(float), 0.5), 1e-12)
            paired = np.zeros(light.shape, bool)
            for reach in (3, 6, 12):
                for dy, dx in ((0, reach), (reach, 0), (reach, reach), (reach, -reach)):
                    if work.interrupted:
                        return None
                    plus = np.roll(light, (dy, dx), axis=(0, 1))
                    minus = np.roll(light, (-dy, -dx), axis=(0, 1))
                    pv = np.roll(visible, (dy, dx), axis=(0, 1))
                    mv = np.roll(visible, (-dy, -dx), axis=(0, 1))
                    if self.ink_support == "connected":
                        # Exterior ink needs a lighter painted interior, not
                        # white or arbitrary RGB outside the silhouette.
                        surface = np.minimum(
                            np.where(pv, plus, np.inf), np.where(mv, minus, np.inf)
                        )
                        supported = np.isfinite(surface) & (surface - light >= 12)
                    else:
                        supported = pv & mv & (np.minimum(plus, minus) - light >= 12)
                    supported &= light <= 150
                    if evidence.opacity is not None:
                        alpha = evidence.opacity
                        pa = np.roll(alpha, (dy, dx), axis=(0, 1))
                        ma = np.roll(alpha, (-dy, -dx), axis=(0, 1))
                        if self.ink_support == "connected":
                            pa = np.where(pv, pa, alpha)
                            ma = np.where(mv, ma, alpha)
                        low = np.minimum(np.minimum(alpha, pa), ma)
                        high = np.maximum(np.maximum(alpha, pa), ma)
                        supported &= high - low <= 0.05 * high
                    margin = max(abs(dy), abs(dx))
                    supported[:margin] = supported[-margin:] = False
                    supported[:, :margin] = supported[:, -margin:] = False
                    paired |= supported
            paired &= evidence.drawn & ~evidence.empty
            if self.ink_support == "connected":
                # Trough pixels establish the role of an existing drawn chain.
                # Follow that support through junctions and faint edge pixels;
                # do not fabricate bridges, close gaps or promote isolated
                # dark surfaces which have no painted-side source contrast.
                paired = binary_propagation(
                    paired,
                    structure=np.ones((3, 3)),
                    mask=evidence.drawn & ~evidence.empty,
                )
            paired.flags.writeable = False
            self._ridge_support = paired
        return self._ridge_support

    def _ridge_owners(self, state, work):
        """Long supported troughs retain owners in ordinary material proposals.

        Small uncertain specks remain finite native score data. Explicitly fixed
        and paint-constrained owners are protected independently by eligibility.
        """
        if self._ink is None:
            paired = self._ink_support(work)
            if paired is None:
                return None
            components, count = label(paired, np.ones((3, 3)))
            sizes = np.bincount(components.ravel(), minlength=count + 1)
            keep = np.zeros(count + 1, bool)
            for index, box in enumerate(find_objects(components), start=1):
                if work.interrupted:
                    return None
                if (
                    box is not None
                    and sizes[index] >= 8
                    and max(box[0].stop - box[0].start, box[1].stop - box[1].start) >= 8
                ):
                    keep[index] = True
            self._ink = keep[components]
            self.diagnostics["ridge_pixels"] = int(self._ink.sum())
        owner_map = state.partition.owners
        owners = {
            owner_map[int(i)]
            for i in np.unique(self.families.graph.labels[self._ink])
            if int(i) in owner_map
        }
        self.diagnostics["ridge_owners"] = len(owners)
        return owners

    def _paint(self, xy, rgb, indices, work):
        if work.interrupted or len(indices) < (1 if self.joint else 16):
            return None
        selected = indices[:: max(1, (len(indices) + 4095) // 4096)]
        rgba = np.column_stack((rgb[selected] / 255, np.ones(len(selected))))
        paint = fit(xy[selected], rgba, gradients=self.options.gradients)
        error = 0.0
        residual = 0.0
        for start in range(0, len(indices), CHUNK):
            if work.interrupted:
                return None
            part = indices[start : start + CHUNK]
            difference = prediction(paint, xy[part], extend=False) * 255 - rgb[part]
            error += float(np.square(difference).sum())
            residual = max(residual, float(np.max(np.abs(difference))))
        return paint, error, residual

    def _split(self, state, base, cell, xy, rgb, own, work):
        evidence = self.families.evidence
        mask = np.zeros(own.shape, bool)
        points = np.floor(xy[cell.indices]).astype(int)
        mask[points[:, 1], points[:, 0]] = True
        best = None
        if self.layout != "regions":
            if self._facet_lines is None:
                self._facet_lines = FacetLines(evidence.target, ~evidence.empty)
            votes = self._facet_lines(mask, work, scope=xy[cell.indices])
        else:
            votes = lines(evidence.target, mask, (0, 0), work)
        for normal, rho in votes:
            if work.interrupted:
                return None
            self.diagnostics["votes"] += 1
            side = xy[cell.indices] @ normal < rho
            left, right = cell.indices[side], cell.indices[~side]
            if min(len(left), len(right)) < max(16, 0.02 * len(cell.indices)):
                continue
            a, b = self._paint(xy, rgb, left, work), self._paint(xy, rgb, right, work)
            if a is None or b is None:
                continue
            gain = cell.error - a[1] - b[1]
            if gain <= 0 or (best is not None and gain <= best[0]):
                continue
            shapes = self.splitter._geometry(
                state, base.id, normal, rho, geometry=cell.region
            )
            if any(not s.subpaths for s in shapes):
                continue
            best = (
                gain,
                normal,
                rho,
                Cell(cell.key + "0", left, shapes[0], cell.draw, *a),
                Cell(cell.key + "1", right, shapes[1], shapes[1], *b),
            )
        if self._facet_lines is not None:
            self.diagnostics["facet_edge_points"] = self._facet_lines.diagnostics[
                "points"
            ]
        return best

    def _regions(self, state, base, selected, whole, xy, rgb, own, work):
        """Group source observations before partitioning complete owners once.

        Nearest material continues underneath the independently retained marks.
        Shared source contours are simplified together. Ordinary proposals
        require opaque-interior proof; joint ablations score carrier fringes
        through the unchanged native policy instead.
        """
        evidence, graph = self.families.evidence, self.families.graph
        indices = {s.id: i for i, s in enumerate(selected)}
        lookup = np.full(len(graph.regions), -1, np.int32)
        for surface in selected:
            lookup[list(surface.members)] = indices[surface.id]
        source = lookup[graph.labels]
        ink_pixels = evidence.drawn
        if self.grouping != "static":
            ink_pixels = self._ink_support(work)
            if ink_pixels is None:
                return
        roles = None
        support = (
            np.isin(
                graph.labels, base.members if base.role == "underlay" else base.covered
            )
            & ~evidence.empty
        )
        source_models = ()
        carrier = None

        def discover():
            # Role alternatives must share physical source extraction. Never
            # rediscover on the smaller fitted-role mask or material palette.
            carrier = curve_path(
                transformed_geometry(whole, root_matrix(state.document, base.id))
            )
            self.diagnostics["source_stroke_attempts"] += 1
            return carrier, ink_models(
                ink_pixels & support,
                evidence,
                self.options,
                work,
                carrier=carrier,
                prune_spurs=True,
                boundary_contacts=True,
                **(
                    {"fractional_coverage": True}
                    if self.ink_coverage == "fractional"
                    else {}
                ),
                **(
                    {"source_profiles": self.source_profiles}
                    if self.source_profiles is not None
                    else {}
                ),
            )

        role_ink = ink_pixels
        if self.ink_roles == "fitted":
            # Explicit offline competitor: uncertain dark source may be paint
            # or shade. The connected interpretation remains independently
            # available; this does not claim to protect every unfitted line.
            carrier, source_models = discover()
            if work.interrupted:
                return
            role_ink = np.zeros(own.shape, bool)
            for model in source_models:
                if work.interrupted:
                    return
                role_ink |= model.selected
            self.diagnostics["source_fitted_role_pixels"] += int(
                np.count_nonzero(role_ink & own)
            )
            self.diagnostics["source_unmodeled_material_pixels"] += int(
                np.count_nonzero(ink_pixels & own & ~role_ink)
            )
        observation_owners = np.arange(len(selected))
        if self.ink_support == "connected":
            roles = observations(source, own, role_ink, work)
            if roles is None:
                self.diagnostics["bounded"] += not work.interrupted
                return
            source, observation_owners = roles.source, roles.owners
            self.diagnostics["source_role_observations_peak"] = max(
                self.diagnostics["source_role_observations_peak"], len(roles.owners)
            )
            self.diagnostics["source_role_mixed_owners_peak"] = max(
                self.diagnostics["source_role_mixed_owners_peak"],
                int(np.sum(np.bincount(roles.owners, minlength=len(selected)) == 2)),
            )
        observation_count = len(observation_owners)
        pixels = source[own]
        sizes = np.bincount(pixels, minlength=observation_count)
        colors = (
            np.column_stack(
                [
                    np.bincount(pixels, weights=rgb[:, k], minlength=observation_count)
                    for k in range(3)
                ]
            )
            / sizes[:, None]
        )
        ink_kinds = (
            roles.ink
            if roles is not None
            else np.bincount(
                pixels, weights=ink_pixels[own], minlength=observation_count
            )
            >= sizes * 0.6
        ) & (cel.lightness(colors) <= 160)
        if work.interrupted:
            return
        nearest = distance_transform_edt(
            ~own, return_distances=False, return_indices=True
        )
        extension = source[tuple(nearest)]
        edges = set()
        for axis in (0, 1):
            a = extension[:-1] if axis == 0 else extension[:, :-1]
            b = extension[1:] if axis == 0 else extension[:, 1:]
            visible = (
                support[:-1] & support[1:]
                if axis == 0
                else support[:, :-1] & support[:, 1:]
            )
            differing = visible & (a != b)
            aa, bb = a.ravel(), b.ravel()
            differing = differing.ravel()
            for start in range(0, len(aa), CHUNK):
                if work.interrupted:
                    return
                mask = differing[start : start + CHUNK]
                pairs = np.column_stack(
                    (aa[start : start + CHUNK][mask], bb[start : start + CHUNK][mask])
                )
                pairs.sort(axis=1)
                edges.update(map(tuple, np.unique(pairs, axis=0).tolist()))
                if len(edges) > MAX_REGION_EDGES:
                    self.diagnostics["bounded"] += 1
                    return
        edges = sorted(edges)
        paint_edges = ()
        statistics = alpha_ranges = None
        if self.grouping != "static":
            paint_edges = ink_paint_links(colors, ink_kinds, work)
            if paint_edges is None:
                return
            if len(edges) + len(paint_edges) > MAX_REGION_EDGES:
                self.diagnostics["bounded"] += 1
                return
        if self.grouping == "paint-fit":
            if len(graph.regions) > MAX_REGIONS:
                self.diagnostics["bounded"] += 1
                return
            if roles is not None:
                statistics = role_moments(roles, evidence, graph, self.options, work)
            else:
                if self._moments is None:
                    # The unchanged carrier supplies source alpha; fit the
                    # intrinsic paint and score complete native RGBA later.
                    self._moments = moments(
                        replace(evidence, opacity=None), graph, self.options, work
                    )
                statistics = np.array(
                    [self._moments[list(s.members)].sum(axis=0) for s in selected]
                )
            alpha_ranges = np.ones((observation_count, 2))
        edges.sort(
            key=lambda e: (float(np.linalg.norm(colors[e[0]] - colors[e[1]])), e)
        )
        matrix = root_matrix(state.document, base.id)
        inverse = inverse_matrix(matrix)
        from vectrify.document.hit_test import multiply

        frame = multiply(
            inverse,
            (1 / evidence.scale[0], 0, 0, 1 / evidence.scale[1], *evidence.offset),
        )
        if self.ink_support == "connected" and self.ink_roles == "connected":
            # Discover physical chains once on complete source carrier support,
            # before a material budget selects owners. Palette eligibility must
            # not change source junctions, endpoints or gaps. Contact and spur
            # handling use the same extractor as the source-ink factory.
            carrier, source_models = discover()
            if work.interrupted:
                return
        chain_observations = np.zeros(observation_count, bool)
        if source_models:
            chain_pixels = np.zeros(own.shape, bool)
            for model in source_models:
                if work.interrupted:
                    return
                chain_pixels |= model.selected
            chain_observations = (
                np.bincount(
                    pixels, weights=chain_pixels[own], minlength=observation_count
                )
                > 0
            )
        seen = set()
        # Offer a substantial structural alternative before spending the shared
        # search window on small savings. Native scoring still decides retention.
        budgets = tuple(
            dict.fromkeys(
                (
                    max(2, min(32, observation_count // 4)),
                    max(2, min(64, observation_count // 2)),
                )
            )
        )
        for threshold in budgets if self.joint else (56, 28, 12):
            eligible = None
            if self.grouping != "static":
                result = grouped(
                    colors,
                    sizes,
                    edges,
                    threshold,
                    work,
                    kinds=ink_kinds,
                    paint_edges=paint_edges,
                    statistics=statistics,
                    alpha_ranges=alpha_ranges,
                    gradients=self.options.gradients,
                )
                if result is None:
                    return
                merged, eligible, hierarchy = result
                areas = np.bincount(merged, weights=sizes, minlength=observation_count)
                self.diagnostics["joint_forced_merges"] += hierarchy["merges"]
                for key, field in (
                    ("hierarchy_components_peak", "substantial_components"),
                    ("hierarchy_retained_owners_peak", "retained_disconnected_owners"),
                    ("hierarchy_queue_peak", "queue_peak"),
                    ("hierarchy_ink_cells_peak", "ink_materials"),
                    ("hierarchy_ink_paint_links_peak", "ink_paint_links"),
                ):
                    self.diagnostics[key] = max(self.diagnostics[key], hierarchy[field])
                self.diagnostics["hierarchy_queue_rebuilds"] += hierarchy[
                    "queue_rebuilds"
                ]
                self.diagnostics["hierarchy_unmet_budgets"] += hierarchy["budget_unmet"]
                self.diagnostics["hierarchy_model_evaluations"] += hierarchy[
                    "model_evaluations"
                ]
                self.diagnostics["hierarchy_model_limits"] += hierarchy[
                    "model_limit_hit"
                ]
                self.diagnostics["hierarchy_alpha_exclusions"] += hierarchy[
                    "alpha_exclusions"
                ]
            else:
                parents = list(range(observation_count))
                remaining = (
                    int(np.sum(sizes >= 16)) if self.joint else observation_count
                )
                means, areas = colors.copy(), sizes.copy()

                def root(i, parents=parents):
                    while parents[i] != i:
                        parents[i] = parents[parents[i]]
                        i = parents[i]
                    return i

                for a, b in edges:
                    if work.interrupted:
                        return
                    a, b = root(a), root(b)
                    if a == b or (
                        not self.joint
                        and np.linalg.norm(means[a] - means[b]) > threshold
                    ):
                        continue
                    if self.joint and remaining <= threshold:
                        break
                    if b < a:
                        a, b = b, a
                    if self.joint:
                        remaining -= int(areas[a] >= 16) + int(areas[b] >= 16)
                    means[a] = (means[a] * areas[a] + means[b] * areas[b]) / (
                        areas[a] + areas[b]
                    )
                    areas[a] += areas[b]
                    parents[b] = a
                    if self.joint:
                        remaining += int(areas[a] >= 16)
                    else:
                        remaining -= 1
                    if self.joint:
                        self.diagnostics["joint_forced_merges"] += 1
                merged = np.array([root(i) for i in range(observation_count)])
            roots, counts = np.unique(merged[pixels], return_counts=True)
            roots = roots[np.argsort(-counts, kind="stable")]
            signature = tuple(int(i) for i in merged)
            if signature in seen:
                continue
            seen.add(signature)
            owner_counts = np.bincount(merged, minlength=observation_count)
            if not self.joint:
                roots = np.array(
                    [r for r in roots if owner_counts[r] >= 2 and areas[r] >= 16]
                )[:MAX_REGION_CELLS]
            else:
                # Isolated tiny groups keep their original independently owned
                # geometry. They cannot consume the whole material-cell budget
                # or disappear through a degenerate fitted contour.
                # A complete source-fitted chain can cross a tiny endpoint
                # atom. Its body proof supplies specific support to that ink
                # role, so a paint budget cannot trim the endpoint. Unmodeled
                # tiny marks and disconnected unsupported roots stay retained.
                chain_roots = (
                    np.bincount(
                        merged, weights=chain_observations, minlength=observation_count
                    )
                    > 0
                )
                roots = np.array(
                    [
                        r
                        for r in roots
                        if (areas[r] >= 16 or (ink_kinds[r] and chain_roots[r]))
                        and (eligible is None or eligible[r])
                    ],
                    dtype=np.int32,
                )
                self.diagnostics["source_chain_small_roots"] += int(
                    np.count_nonzero(areas[roots] < 16)
                )
                if len(roots) > MAX_REGION_CELLS and self.ink_support != "connected":
                    self.diagnostics["region_exclusions"] += 1
                    continue
            if self.ink_support == "connected":
                # The coverage carrier is a material. Supported ink is drawn
                # above its neighboring materials, never used as their base.
                roots = np.r_[roots[~ink_kinds[roots]], roots[ink_kinds[roots]]]
                if not len(roots) or ink_kinds[roots[0]]:
                    self.diagnostics["region_exclusions"] += 1
                    continue
            if not len(roots):
                continue
            # A virtual observation is not an independently editable owner.
            # Delete an existing owner only if every one of its role pieces
            # has a final class. Otherwise retain its complete original paint
            # and geometry; never lose its small or disconnected remainder.
            retained_owners = np.unique(observation_owners[~np.isin(merged, roots)])
            eligible_owners = ~np.isin(np.arange(len(selected)), retained_owners)
            accepted_observations = eligible_owners[observation_owners]
            active = np.unique(merged[pixels[accepted_observations[pixels]]])
            roots = roots[np.isin(roots, active)]
            if not len(roots):
                continue
            if self.ink_support == "connected" and ink_kinds[roots[0]]:
                self.diagnostics["region_exclusions"] += 1
                continue
            self.diagnostics["region_candidates"] += len(roots)
            region_selected = tuple(
                s for i, s in enumerate(selected) if eligible_owners[i]
            )
            if len(region_selected) < 3:
                continue
            self.diagnostics["region_retained_owners"] += len(selected) - len(
                region_selected
            )
            root_cells = {int(r): i for i, r in enumerate(roots)}
            palette = np.array([root_cells.get(int(r), -1) for r in merged], np.int32)
            accepted = source >= 0
            accepted &= accepted_observations[np.maximum(source, 0)]
            nearest = distance_transform_edt(
                ~accepted, return_distances=False, return_indices=True
            )
            # Virtual roots can outnumber exported cells. Up to 4,096 source
            # observations plus eight styles fit uint16 without aliasing; the
            # final combined classification must still meet the 64-cell cap.
            classes = palette[source[tuple(nearest)]].astype(
                np.uint16 if self.ink_support == "connected" else np.uint8
            )
            outline_labels = np.where(support, classes.astype(np.int32) + 1, 0)
            ink_cells = np.r_[False, ink_kinds[roots]]
            primary_pixels = accepted[own]
            stroke_models = {}
            if self.ink_support == "connected":
                assert carrier is not None
                changed = classes.copy()
                chosen = {}
                for complete in source_models:
                    # Keep fully owned physical chains, even when another
                    # independent run of this paint touches a retained owner.
                    # Native geometry is never cropped to a material budget.
                    model = owned_model(complete, own & accepted, evidence, work)
                    if work.interrupted:
                        return
                    if model is None:
                        self.diagnostics["source_stroke_owner_exclusions"] += len(
                            complete.geometry.subpaths
                        )
                        continue
                    self.diagnostics["source_stroke_owner_exclusions"] += (
                        model.details.get("owner_excluded_runs", 0)
                    )
                    outside = abs(
                        pathops.op(
                            curve_path(model.footprint),
                            carrier,
                            pathops.PathOp.DIFFERENCE,
                        ).area
                    )
                    if outside > 1e-8:
                        self.diagnostics["source_stroke_carrier_exclusions"] += 1
                        continue
                    if not model.selected[own & accepted].any():
                        continue
                    index = len(roots) + len(chosen)
                    changed[model.selected] = index
                    chosen[index] = model
                # Judge all compatible source styles together. One style can
                # temporarily add a class while a later style retires several
                # virtual roots. No geometry, source cut or partial style set
                # is published before the complete final count is known.
                active = np.unique(changed[own & accepted])
                kinds = np.r_[ink_kinds[roots], np.ones(len(chosen), bool)]
                self.diagnostics["source_class_roots_peak"] = max(
                    self.diagnostics["source_class_roots_peak"], len(roots)
                )
                self.diagnostics["source_final_cells_peak"] = max(
                    self.diagnostics["source_final_cells_peak"], len(active)
                )
                if len(active) > MAX_REGION_CELLS or kinds[active[0]]:
                    self.diagnostics["region_exclusions"] += 1
                    continue
                remap = np.zeros(len(roots) + len(chosen), np.uint8)
                remap[active] = np.arange(len(active), dtype=np.uint8)
                classes = remap[changed]
                ink_cells = np.r_[False, kinds[active]]
                stroke_models = {int(remap[i]): model for i, model in chosen.items()}
                self.diagnostics["source_stroke_models"] += len(stroke_models)
                outline_labels = np.where(support, classes.astype(np.int32) + 1, 0)

            def check():
                if work.interrupted:
                    from vectrify.refine.cel_plan.model import StageInterruptedError

                    raise StageInterruptedError("Source material contours interrupted")

            models = Boundaries()
            ink_models_boundary = InkBoundaries()
            precise_ink = np.zeros(graph.labels.shape, bool)
            ink_mask = support & ink_cells[outline_labels]
            if self.boundary_fit == "anchored":
                components, component_count = label(ink_mask, np.ones((3, 3)))
                component_sizes = np.bincount(
                    components.ravel(), minlength=component_count + 1
                )
                precise = component_sizes <= 64
                precise[0] = False
                precise_ink = precise[components]

            def boundary(
                points: np.ndarray,
                _tolerance: float,
                models=models,
                outline_labels=outline_labels,
                ink_cells=ink_cells,
                ink_mask=ink_mask,
                source_labels=outline_labels,
                ink_models_boundary=ink_models_boundary,
                precise_ink=precise_ink,
            ) -> list[tuple[str, tuple[float, ...]]]:
                check()
                if self.grouping != "static" and len(points) > 8:
                    middle = (points[0] + points[1]) / 2
                    direction = points[1] - points[0]
                    normal = np.array([-direction[1], direction[0]]) * 0.25
                    x, y = np.floor(middle + normal).astype(int)
                    left = int(outline_labels[y, x])
                    left_precise = bool(precise_ink[y, x])
                    x, y = np.floor(middle - normal).astype(int)
                    right = int(outline_labels[y, x])
                    if ink_cells[left] or ink_cells[right]:
                        if self.boundary_fit == "anchored":
                            # A tiny disconnected mark can have its contour
                            # split into open chains by neighboring material
                            # junctions. Preserve the whole source mark, not
                            # just callbacks that happen to be closed loops.
                            if left_precise or precise_ink[y, x]:
                                self.diagnostics["anchored_ink_precise_marks"] += 1
                                return cel.curve_nodes(
                                    points,
                                    min(self.options.boundary_tolerance, 0.25)
                                    * min(evidence.scale),
                                    smooth=0,
                                    fit=cel.FILL_FIT,
                                )
                            limits = (
                                ink_limits(points, ink_mask, materials=source_labels)
                                * min(evidence.scale)
                                if len(points) <= 4096
                                else None
                            )
                            nodes = ink_models_boundary(
                                points,
                                min(self.options.boundary_tolerance, 0.75)
                                * min(evidence.scale),
                                limits=limits,
                            )
                            for decision in ink_models_boundary.decisions:
                                field = {
                                    "straight": "anchored_ink_straights",
                                    "raw-curve": "anchored_ink_curves",
                                    "ellipse": "anchored_ink_ellipses",
                                    "precise-mark": "anchored_ink_precise_marks",
                                }[decision["model"]]
                                self.diagnostics[field] += 1
                            ink_models_boundary.decisions.clear()
                            return nodes
                        # Gaussian fill smoothing can erase short lettering or
                        # hatching. Fit the raw paired ink/material boundary.
                        return cel.curve_nodes(
                            points,
                            0.25 * min(evidence.scale),
                            smooth=0,
                            fit=cel.FILL_FIT,
                        )
                if self.joint and 8 < len(points) <= 4096:
                    nodes = models(
                        points, min(evidence.scale) * (self.options.tolerance or 1.5)
                    )
                    if models.decisions[-1][
                        "model"
                    ] != "curve" or self.boundary_fit in {"curve", "anchored"}:
                        return nodes
                    # Generic smoothing can collapse a one-pixel-wide closed
                    # material strip. Keep canonical polygons as the fallback;
                    # general curve fitting follows a viable joint structure.
                fitted = cel.simplify(
                    points,
                    0
                    if self.joint and len(points) <= 8
                    else 0.75 * min(evidence.scale),
                )
                return [("L", (float(x), float(y))) for x, y in fitted[1:]]

            material_classes = classes
            if self.ink_support == "connected":
                # Extend materials underneath source ink before changing its
                # width. This is a complete coupled proposal; gaps exposed by
                # narrower strokes receive the corresponding adjacent paint.
                material = support & ~ink_cells[classes.astype(int) + 1]
                nearest = distance_transform_edt(
                    ~material, return_distances=False, return_indices=True
                )
                material_classes = classes[tuple(nearest)]
                material_labels = np.where(support, material_classes.astype(int) + 1, 0)
                ink_labels = np.where(
                    support
                    & ink_cells[classes.astype(int) + 1]
                    & ~np.isin(classes, tuple(stroke_models)),
                    classes.astype(int) + 1,
                    0,
                )
                outlines = {}
                for labels in (material_labels, ink_labels):

                    def fit_boundary(points, tolerance, labels=labels):
                        return boundary(points, tolerance, outline_labels=labels)

                    found, retried = _supported_outlines(labels, fit_boundary, check)
                    self.diagnostics["degenerate_region_retries"] += retried
                    outlines.update((i, data) for i, data in found.items() if i > 0)
            elif self.joint:
                # Two independently supported straight sides can collapse a
                # thin material to the same chord. Restore its canonical chains
                # on BOTH neighboring fills before constructing the edit.
                outlines, retried = _supported_outlines(outline_labels, boundary, check)
                self.diagnostics["degenerate_region_retries"] += retried
            else:
                outlines = cel.region_outlines(
                    outline_labels, 0, fit_boundary=boundary, check=check
                )
            cells = []
            for i in range(len(ink_cells) - 1):
                if work.interrupted:
                    return
                region_indices = np.flatnonzero(primary_pixels & (classes[own] == i))
                fitted = self._paint(xy, rgb, region_indices, work)
                if fitted is None or (i not in stroke_models and i + 1 not in outlines):
                    break
                stroke, footprint, covered = None, None, ()
                if i in stroke_models:
                    model = stroke_models[i]
                    shape, footprint, stroke = (
                        model.geometry,
                        model.footprint,
                        model.details,
                    )
                    paint = Paint(colour(model.paint), 1.0)
                    difference = model.paint - rgb[region_indices]
                    fitted = (
                        paint,
                        float(np.square(difference).sum()),
                        float(np.abs(difference).max()),
                    )
                else:
                    shape = transformed_geometry(parse_path(outlines[i + 1]), frame)
                    shape = _filled(shape, "evenodd")
                    if self.ink_support == "connected" and not ink_cells[i + 1]:
                        hidden = (
                            support
                            & (material_classes == i)
                            & ink_cells[classes.astype(int) + 1]
                        )
                        covered = tuple(int(k) for k in np.unique(classes[hidden]))
                cells.append(
                    Cell(
                        f"r{i}",
                        region_indices,
                        shape,
                        whole if i == 0 else shape,
                        *fitted,
                        i,
                        stroke,
                        footprint,
                        covered,
                        bool(ink_cells[i + 1]),
                    )
                )
            if len(cells) != len(ink_cells) - 1:
                self.diagnostics["region_exclusions"] += 1
                continue
            yield cells, classes, threshold, region_selected

    @staticmethod
    def _mask(geometry, matrix, box, *, rule="nonzero"):
        width, height = box.right - box.x, box.bottom - box.y
        transform = " ".join(str(v) for v in matrix)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="{box.x} {box.y} {width} {height}">'
            f'<path fill="black" fill-rule="{rule}" '
            f'transform="matrix({transform})" d="{geometry.path_data()}"/></svg>'
        )
        return render(svg, ((box.right - box.x), (box.bottom - box.y)))[..., 3]

    def _interior(self, state, base, whole, work):
        matrix = root_matrix(state.document, base.id)
        native = transformed_geometry(whole, matrix)
        original = curve_path(native)
        x0, y0, x1, y1 = original.bounds
        box = Box(
            math.floor(x0) - 2, math.floor(y0) - 2, math.ceil(x1) + 2, math.ceil(y1) + 2
        )
        if box.area > MAX_NATIVE_PIXELS:
            self.diagnostics["bounded"] += 1
            return None
        # Overlay coverage must never accumulate alpha on the coverage edge.
        # Repaint the original carrier there; other cells stay in its rendered
        # opaque interior, proved again against native pixel coverage below.
        border = curve_path(native)
        border.stroke(3.0, pathops.LineCap.ROUND_CAP, pathops.LineJoin.ROUND_JOIN, 4)
        border.convertConicsToQuads(0.05)
        interior = pathops.op(original, border, pathops.PathOp.DIFFERENCE)
        geometry = transformed_geometry(path_geometry(interior), inverse_matrix(matrix))
        if not geometry.subpaths or work.interrupted:
            return None
        if sum(len(s.nodes) for s in geometry.subpaths) > MAX_OUTPUT_NODES:
            self.diagnostics["geometry_exclusions"] += 1
            return None
        return (
            geometry,
            box,
            self._mask(
                state.document.geometry_for(base.id),
                matrix,
                box,
                rule=path_style(state.document, state.document.element(base.id))[
                    "fill-rule"
                ],
            ),
        )

    @staticmethod
    def _supported_fill(document, element, style):
        if (
            style["fill"] == "none"
            or style["stroke"] != "none"
            or element.get("clip-path", "none") != "none"
            or element.get("filter", "none") != "none"
        ):
            return False
        if any(not 0 <= float(style[k]) <= 1 for k in ("opacity", "fill-opacity")):
            return False
        server = paint_server(style["fill"])
        if server is None:
            return 0 <= color(style["fill"])[3] <= 1
        gradient = document.element(server)
        stops = gradient_stops(gradient)
        return (
            gradient.tag == "linearGradient"
            and bool(stops)
            and all(0 <= s[1][3] <= 1 for s in stops)
        )

    def _owners(self, state, base, box, opaque, work):
        support = set(base.members if base.role == "underlay" else base.covered)
        parent = state.document.ancestry(base.id)[-2]
        selected = []
        ink = set() if self.joint else self._ridge_owners(state, work)
        if ink is None:
            return None
        nodes = 0
        constrained = set(state.details.get("paint_constraints", ()))
        surfaces = {s.id: s for s in state.partition.surfaces}
        ink_support = {
            i
            for s in state.partition.surfaces
            if s.role == "overlay"
            for i in s.members
        }
        for child in parent.children:
            if work.interrupted:
                return None
            surface = surfaces.get(child.id)
            if (
                surface is None
                or surface.id == base.id
                or surface.role != "surface"
                or (surface.covered and not set(surface.covered).issubset(ink_support))
                or surface.id in constrained
                or any(self.families.graph.regions[i].fixed for i in surface.members)
                or any(a.locks for a in state.document.ancestry(child.id))
                or any(
                    n.pinned
                    for s in state.document.geometry_for(child.id).subpaths
                    for n in s.nodes
                )
            ):
                continue
            if not set(surface.members).issubset(support) or child.id in ink:
                continue
            style = path_style(state.document, child)
            if not self._supported_fill(state.document, child, style):
                continue
            shape = _filled(state.document.geometry_for(child.id), style["fill-rule"])
            nodes += sum(len(s.nodes) for s in shape.subpaths)
            if len(selected) >= MAX_PATHS or nodes > MAX_INPUT_NODES:
                self.diagnostics["bounded"] += 1
                return None
            if not self.joint and not in_core(
                state, surface.members, child.id, shape, work
            ):
                continue
            matrix = root_matrix(state.document, child.id)
            a, b, c, d = _bounds(state.document, child.id)
            if self.joint:
                a, b, c, d = curve_path(
                    transformed_geometry(state.document.geometry_for(child.id), matrix),
                    style["fill-rule"],
                ).bounds
            if self.joint and (
                math.floor(a) < box.x
                or math.floor(b) < box.y
                or math.ceil(c) > box.right
                or math.ceil(d) > box.bottom
            ):
                self.diagnostics["alpha_exclusions"] += 1
                continue
            crop = Box(
                max(box.x, math.floor(a) - 2),
                max(box.y, math.floor(b) - 2),
                min(box.right, math.ceil(c) + 2),
                min(box.bottom, math.ceil(d) + 2),
            )
            if crop.area and not self.joint:
                coverage = self._mask(
                    state.document.geometry_for(child.id),
                    matrix,
                    crop,
                    rule=style["fill-rule"],
                )
                original = opaque[
                    crop.y - box.y : crop.bottom - box.y,
                    crop.x - box.x : crop.right - box.x,
                ]
                if ((coverage > 0) & (original != 1)).any():
                    self.diagnostics["alpha_exclusions"] += 1
                    continue
            selected.append(surface)
        if len(selected) < self.minimum_paths:
            return None
        return tuple(selected)

    def _retained(self, state, base, selected, whole, work):
        """Only known supported owners may keep paint above a changed material.

        Their geometry, source ownership and mutual order remain untouched.
        The carrier and retained paints keep their intrinsic opacity. Ordinary
        proposals prove removed/new paint stays in the opaque interior. Joint
        ablations allow a changed fringe, with complete native policy checks.
        """
        document = state.document
        known = {s.id for s in state.partition.surfaces}
        overlays = {s.id for s in state.partition.surfaces if s.role == "overlay"}
        removed = {s.id for s in selected} | {base.id}
        matrix = root_matrix(document, base.id)
        native = transformed_geometry(whole, matrix)
        filled = curve_path(native)
        a, b, c, d = filled.bounds
        for child in document.ancestry(base.id)[-2].children:
            if work.interrupted:
                return False
            if child.id in removed:
                continue
            p, q, r, s = _bounds(document, child.id)
            if max(a, p) >= min(c, r) or max(b, q) >= min(d, s):
                continue
            if child.tag != "path" or child.id not in known:
                self.diagnostics["style_exclusions"] += 1
                return False
            style = path_style(document, child)
            if (
                child.id in overlays
                and style["fill"] == "none"
                and style["stroke"] != "none"
                and float(style["opacity"]) == 1
                and float(style["stroke-opacity"]) == 1
                and opaque_fill(document, style["stroke"])
                and child.get("clip-path", "none") == "none"
                and child.get("filter", "none") == "none"
            ):
                # A supported opaque stroke stays above the repainted material.
                # Prove its complete native footprint inside the unchanged
                # carrier; membership or centerline bounds alone are not enough.
                width = float(style["stroke-width"])
                caps = {
                    "round": pathops.LineCap.ROUND_CAP,
                    "butt": pathops.LineCap.BUTT_CAP,
                    "square": pathops.LineCap.SQUARE_CAP,
                }
                joins = {
                    "round": pathops.LineJoin.ROUND_JOIN,
                    "bevel": pathops.LineJoin.BEVEL_JOIN,
                    "miter": pathops.LineJoin.MITER_JOIN,
                }
                miter = float(style["stroke-miterlimit"])
                geometry = document.geometry_for(child.id)
                if (
                    not math.isfinite(width)
                    or width <= 0
                    or not math.isfinite(miter)
                    or miter <= 0
                    or style["stroke-linecap"] not in caps
                    or style["stroke-linejoin"] not in joins
                    or sum(len(s.nodes) for s in geometry.subpaths) > MAX_OUTPUT_NODES
                ):
                    self.diagnostics["style_exclusions"] += 1
                    return False
                shape = open_path(geometry)
                shape.stroke(
                    width,
                    caps[style["stroke-linecap"]],
                    joins[style["stroke-linejoin"]],
                    miter,
                )
                shape.convertConicsToQuads(0.05)
                footprint = transformed_geometry(
                    path_geometry(shape), root_matrix(document, child.id)
                )
                if (
                    sum(len(s.nodes) for s in footprint.subpaths) > MAX_OUTPUT_NODES
                    or abs(
                        pathops.op(
                            curve_path(footprint), filled, pathops.PathOp.DIFFERENCE
                        ).area
                    )
                    > 1e-8
                ):
                    self.diagnostics["style_exclusions"] += 1
                    return False
                self.diagnostics["retained_paths"] += 1
                continue
            # Current owned opaque fills are explicit retained marks. Only
            # directly supported opaque gradients share that interpretation.
            supported = self._supported_fill(document, child, style)
            if not supported:
                if (
                    style["stroke"] != "none"
                    or child.get("clip-path", "none") != "none"
                    or child.get("filter", "none") != "none"
                ):
                    self.diagnostics["style_exclusions"] += 1
                    return False
                shape = transformed_geometry(
                    document.geometry_for(child.id), root_matrix(document, child.id)
                )
                overlap = pathops.op(
                    filled,
                    curve_path(shape, style["fill-rule"]),
                    pathops.PathOp.INTERSECTION,
                )
                if abs(overlap.area) > 1e-8:
                    self.diagnostics["style_exclusions"] += 1
                    return False
            self.diagnostics["retained_paths"] += 1
        return not work.interrupted

    def __call__(self, state, work: Work):

        if state.partition is None or work.interrupted:
            return
        graph, evidence = self.families.graph, self.families.evidence
        ink_plane_seen = set()
        if graph.labels.size > MAX_PIXELS:
            self.diagnostics["bounded"] += 1
            return
        bases = [
            s for s in state.partition.surfaces if s.role == "underlay" or s.covered
        ]
        bases.sort(key=lambda s: -len(s.members if s.role == "underlay" else s.covered))
        for base in bases[:MAX_CORES]:
            document = state.document
            element = document.element(base.id)
            style = path_style(document, element)
            if (
                style["stroke"] != "none"
                or float(style["opacity"]) != 1
                or float(style["fill-opacity"]) != 1
                or not opaque_fill(document, style["fill"])
                or element.get("clip-path", "none") != "none"
                or any(a.locks for a in document.ancestry(base.id))
                or any(
                    n.pinned
                    for s in document.geometry_for(base.id).subpaths
                    for n in s.nodes
                )
            ):
                continue
            if any(
                a.get("clip-path", "none") != "none"
                or a.get("filter", "none") != "none"
                for a in document.ancestry(base.id)
            ):
                self.diagnostics["style_exclusions"] += 1
                continue
            whole = _filled(document.geometry_for(base.id), style["fill-rule"])
            interior = self._interior(state, base, whole, work)
            if interior is None:
                continue
            inner, box, opaque = interior
            selected = self._owners(state, base, box, opaque, work)
            if selected is None:
                continue
            if not self._retained(state, base, selected, whole, work):
                continue
            members = tuple(sorted(i for s in selected for i in s.members))
            own = np.isin(graph.labels, members) & ~evidence.empty
            y, x = np.nonzero(own)
            xy = np.column_stack((x + 0.5, y + 0.5))
            rgb = evidence.target[own]
            fitted = self._paint(xy, rgb, np.arange(len(x)), work)
            if fitted is None:
                continue
            self.diagnostics["cores"] += 1
            self.diagnostics["selected_paths"] += len(selected)
            self.diagnostics["source_pixels"] += len(x)
            for regions, classes, threshold, region_selected in (
                self._regions(state, base, selected, whole, xy, rgb, own, work)
                if self.layout != "planes"
                else ()
            ):
                if self.layout == "ink-planes":
                    yield from self._ink_planes(
                        state,
                        base,
                        region_selected,
                        regions,
                        classes,
                        threshold,
                        whole,
                        xy,
                        rgb,
                        inner,
                        box,
                        opaque,
                        work,
                        ink_plane_seen,
                    )
                    continue
                proposal = self._proposal(
                    state,
                    base,
                    region_selected,
                    regions,
                    [],
                    inner,
                    box,
                    opaque,
                    work,
                    source_classes=classes,
                    region_threshold=threshold,
                )
                if proposal is not None:
                    self.diagnostics["region_proposals"] += 1
                    yield proposal
            if self.joint and self.layout != "planes":
                continue
            initial = Cell("", np.arange(len(x)), whole, whole, *fitted)
            for cells, cuts in self._planes(state, base, initial, xy, rgb, own, work):
                proposal = self._proposal(
                    state, base, selected, cells, cuts, inner, box, opaque, work
                )
                if proposal is not None:
                    yield proposal

    def _planes(self, state, base, initial, xy, rgb, own, work):
        """Complete bounded binary prefixes; canceled work publishes none."""
        planar = self.layout != "regions"
        cell_limit = MAX_PLANES if planar else MAX_CELLS
        cells, cuts, fitted_splits = [initial], [], {}
        for _step in range(cell_limit):
            if work.interrupted:
                return
            if not planar or len(cells) in PLANE_PREFIXES:
                self.diagnostics["plane_prefixes"] += planar
                yield cells, cuts
                if work.interrupted:
                    return
            self.diagnostics["plane_cells_peak"] = max(
                self.diagnostics["plane_cells_peak"], len(cells) if planar else 0
            )
            if len(cells) >= cell_limit:
                return
            best = None
            for cell in cells:
                if cell.key not in fitted_splits:
                    fitted_splits[cell.key] = self._split(
                        state, base, cell, xy, rgb, own, work
                    )
                candidate = fitted_splits[cell.key]
                if candidate is not None and (
                    best is None or candidate[0] > best[1][0]
                ):
                    best = cell, candidate
            if work.interrupted:
                return
            if best is None:
                if planar and len(cells) not in PLANE_PREFIXES:
                    self.diagnostics["plane_prefixes"] += 1
                    yield cells, cuts
                return
            old, (_gain, normal, rho, left, right) = best
            cells.remove(old)
            cells.extend((left, right))
            cells.sort(key=lambda c: c.key)
            cuts.append((old.key, tuple(float(v) for v in normal), float(rho)))

    @staticmethod
    def _classes(cells, cuts, shape):
        classes = np.zeros(shape, np.uint8)
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        for index, cell in enumerate(cells):
            scope = np.ones(shape, bool)
            for prefix, normal, rho in cuts:
                if cell.key.startswith(prefix) and len(cell.key) > len(prefix):
                    side = (xx + 0.5) * normal[0] + (yy + 0.5) * normal[1] < rho
                    scope &= side if cell.key[len(prefix)] == "0" else ~side
            classes[scope] = index
        return classes

    def _ink_planes(
        self,
        state,
        base,
        selected,
        regions,
        source_classes,
        threshold,
        whole,
        xy,
        rgb,
        inner,
        box,
        opaque,
        work,
        seen,
    ):
        """Separate observed ink from plane paint before fitting either layer.

        Region grammar supplies source-supported ink, including unchanged
        filled interpretations where a stroke model is unsupported. Material
        cuts see only visible material observations; both layers share one
        complete source-atom namespace and one atomic component proposal.
        """
        graph, evidence = self.families.graph, self.families.evidence
        ink = [cell for cell in regions if cell.ink]
        material = [cell.indices for cell in regions if not cell.ink]
        if not material or len(ink) >= MAX_REGION_CELLS or work.interrupted:
            return
        indices = np.sort(np.concatenate(material))
        own = np.zeros(graph.labels.shape, bool)
        x, y = np.floor(xy[indices]).astype(int).T
        own[y, x] = True
        ink_classes = tuple(cell.support for cell in ink)
        support = (
            np.isin(
                graph.labels, base.members if base.role == "underlay" else base.covered
            )
            & ~evidence.empty
        )
        source_ink = np.isin(source_classes, ink_classes) & support
        signature = hashlib.sha256(
            repr((base.id, tuple(s.id for s in selected))).encode()
        )
        signature.update(indices.tobytes())
        signature.update(source_ink.tobytes())
        for cell in ink:
            signature.update(cell.indices.tobytes())
            signature.update(
                repr((cell.paint, cell.stroke, cell.error, cell.residual)).encode()
            )
            signature.update(cell.draw.path_data().encode())
            if cell.footprint is not None:
                signature.update(cell.footprint.path_data().encode())
        key = signature.hexdigest()
        if key in seen:
            self.diagnostics["ink_plane_duplicate_seeds"] += 1
            return
        seen.add(key)
        fitted = self._paint(xy, rgb, indices, work)
        if fitted is None:
            return
        retained = tuple(
            i
            for surface in state.partition.surfaces
            if surface.role == "overlay"
            for i in surface.members
        )
        visible_material = (
            ~evidence.empty & ~source_ink & ~np.isin(graph.labels, retained)
        )
        self._facet_lines = FacetLines(evidence.target, visible_material)
        self.diagnostics["ink_plane_seeds"] += 1
        self.diagnostics["ink_plane_cells"] += len(ink)
        self.diagnostics["ink_plane_pixels"] += int(source_ink.sum())
        initial = Cell("", indices, whole, whole, *fitted)
        for planes, cuts in self._planes(state, base, initial, xy, rgb, own, work):
            if len(planes) + len(ink) > MAX_REGION_CELLS:
                self.diagnostics["region_exclusions"] += 1
                continue
            underpaint = self._classes(planes, cuts, graph.labels.shape)
            classes = underpaint.copy()
            ink_indices = tuple(range(len(planes), len(planes) + len(ink)))
            cells = [replace(cell, covered_classes=ink_indices) for cell in planes]
            for old, index in zip(ink, ink_indices, strict=True):
                classes[source_classes == old.support] = index
                cells.append(
                    replace(old, key=f"ink{index}", support=index, covered_classes=())
                )
            proposal = self._proposal(
                state,
                base,
                selected,
                cells,
                cuts,
                inner,
                box,
                opaque,
                work,
                source_classes=classes,
                region_threshold=threshold,
                underpaint_classes=underpaint,
            )
            if proposal is not None:
                yield proposal

    def _proposal(
        self,
        state,
        base,
        selected,
        cells,
        cuts,
        inner,
        box,
        opaque,
        work,
        *,
        source_classes=None,
        region_threshold=None,
        underpaint_classes=None,
    ):
        from vectrify.refine.cel_plan.proposals import bounds

        document, graph, evidence = (
            state.document,
            self.families.graph,
            self.families.evidence,
        )
        if (
            not self.joint
            and source_classes is None
            and any(c.residual > MAX_RESIDUAL for c in cells)
        ):
            self.diagnostics["planar_paint_exclusions"] += 1
            return None
        classes = (
            self._classes(cells, cuts, graph.labels.shape)
            if source_classes is None
            else source_classes
        )
        members = tuple(sorted(i for s in selected for i in s.members))
        atoms = state.partition.atoms or Atoms.original(graph)
        try:
            atoms, groups = atoms.partition(graph, members, classes, len(cells), work)
        except ValueError:
            self.diagnostics["atom_exclusions"] += 1
            return None
        matrix = root_matrix(document, base.id)
        shapes = [document.geometry_for(base.id)]
        clip_geometry = (
            _filled(
                shapes[0], path_style(document, document.element(base.id))["fill-rule"]
            )
            if self.joint
            else inner
        )
        for cell in cells[1:]:
            if cell.stroke is not None:
                assert cell.footprint is not None
                native_clip = transformed_geometry(clip_geometry, matrix)
                if (
                    abs(
                        pathops.op(
                            curve_path(cell.footprint),
                            curve_path(native_clip),
                            pathops.PathOp.DIFFERENCE,
                        ).area
                    )
                    > 1e-8
                ):
                    self.diagnostics["alpha_exclusions"] += 1
                    return None
                shapes.append(cell.draw)
                continue
            shape = path_geometry(
                pathops.op(
                    curve_path(cell.draw),
                    curve_path(clip_geometry),
                    pathops.PathOp.INTERSECTION,
                )
            )
            if self.layout != "regions" and crossings(shape):
                shape = resolved(shape, matrix, "nonzero", work)
                if shape is None:
                    self.diagnostics["geometry_exclusions"] += 1
                    return None
            if not shape.subpaths:
                self.diagnostics["empty_cell_exclusions"] += 1
                return None
            coverage = None if self.joint else self._mask(shape, matrix, box)
            if coverage is not None and ((coverage > 0) & (opaque != 1)).any():
                self.diagnostics["alpha_exclusions"] += 1
                return None
            shapes.append(shape)
        if work.interrupted:
            return None
        if sum(len(s.nodes) for g in shapes for s in g.subpaths) > MAX_OUTPUT_NODES:
            self.diagnostics["geometry_exclusions"] += 1
            return None
        digest = hashlib.sha256(repr((base.id, cuts, groups)).encode()).hexdigest()[:12]
        ids = tuple(f"{base.id}-material-{digest}-{i}" for i in range(1, len(cells)))
        # The coverage carrier takes the first cell's primary material atoms;
        # its fringe ownership stays intact. Other cells have exact primary
        # support, with explicit hidden support for their default subtrees.
        start = len(state.partition.atoms.cuts) if state.partition.atoms else 0
        expanded = [
            replace(
                s,
                members=atoms.descendants(s.members, start),
                covered=atoms.descendants(s.covered, start),
            )
            for s in state.partition.surfaces
        ]
        remove = {s.id for s in selected} | {base.id}
        expanded_base = next(s for s in expanded if s.id == base.id)
        old_support = set(
            expanded_base.members if base.role == "underlay" else expanded_base.covered
        )
        first = tuple(
            sorted(
                (
                    *(() if base.role == "underlay" else expanded_base.members),
                    *groups[0],
                )
            )
        )
        new_surfaces = [
            Surface(base.id, first, covered=tuple(sorted(old_support - set(first))))
        ]
        for index, (oid, cell) in enumerate(zip(ids, cells[1:], strict=True), start=1):
            scope = cell.key[: cell.key.rfind("1") + 1]
            covered = {
                i
                for c, group in zip(cells, groups, strict=True)
                if cell.support is None
                and c.key != cell.key
                and c.key.startswith(scope)
                for i in group
            }
            covered.update(i for c in cell.covered_classes for i in groups[c])
            support = set(
                base.members if base.role == "underlay" else base.covered
            ) - set(members)
            active_cells = [i for i, c in enumerate(cells) if c.key.startswith(scope)]
            paint_classes = (
                underpaint_classes
                if underpaint_classes is not None and not cell.ink
                else classes
            )
            hidden = (
                np.isin(paint_classes, active_cells)
                if cell.support is None
                else paint_classes == cell.support
            ) & np.isin(graph.labels, tuple(support))
            covered.update(
                atoms.descendants(
                    tuple(int(i) for i in np.unique(graph.labels[hidden])), start
                )
            )
            new_surfaces.append(
                Surface(
                    oid,
                    groups[index],
                    "overlay" if cell.stroke else "surface",
                    covered=tuple(sorted(covered)),
                )
            )
        partition = Partition(
            tuple(s for s in expanded if s.id not in remove) + tuple(new_surfaces),
            atoms,
        )
        if not partition.follows(state.partition):
            raise ValueError("Whole-core material edit lost source ownership")
        editor = Editor(document, selection=Selection(whole_document=True))
        parent = document.ancestry(base.id)[-2]
        position = next(i for i, c in enumerate(parent.children) if c.id == base.id)
        known = {e.id for e in document.elements()}
        if any(oid in known for oid in ids):
            return None
        with editor.transaction("Propose whole-core source materials") as tx:
            tx.delete_objects(frozenset(s.id for s in selected))
            for index, (oid, shape) in enumerate(
                zip(ids, shapes[1:], strict=True), start=1
            ):
                if cells[index].stroke:
                    shape = identified(shape, oid)
                tx.insert_object(
                    parent.id,
                    replace(document.element(base.id), id=oid, geometry_id=shape.id),
                    index=position + index,
                    geometries=(shape,),
                )
        for oid, cell in zip((base.id, *ids), cells, strict=True):
            snapshot = editor.snapshot.document
            opacity = _opacity(snapshot, oid)
            with editor.transaction("Fit source material cell") as tx:
                tx.set_fill(
                    oid,
                    "none"
                    if cell.stroke
                    else _gradient(cell.paint, evidence, snapshot, oid, opacity)
                    if cell.paint.gradient
                    else cell.paint.color,
                )
                tx.set_attributes(
                    oid,
                    {
                        "fill-rule": path_style(document, document.element(base.id))[
                            "fill-rule"
                        ]
                        if oid == base.id
                        else "nonzero",
                        "fill-opacity": "1",
                        "stroke": "none",
                    },
                )
                if cell.stroke:
                    inverse_parent = inverse_matrix(root_matrix(snapshot, parent.id))
                    matrix_text = " ".join(str(v) for v in inverse_parent)
                    tx.set_attributes(
                        oid,
                        {
                            "stroke": cell.paint.color,
                            "stroke-width": repr(cell.stroke["width"]),
                            "stroke-opacity": "1",
                            "stroke-linecap": cell.stroke.get("linecap", "round"),
                            "stroke-linejoin": "round",
                            "transform": f"matrix({matrix_text})",
                        },
                    )
        proposed = editor.snapshot.document
        try:
            component = ComponentEdit.bind(document, state.partition, parent.id, work)
        except StageInterruptedError:
            return None
        self.diagnostics["cells"] += len(cells)
        self.diagnostics["proposals"] += 1
        self.diagnostics["joint_proposals"] += self.joint
        old_ids = tuple(s.id for s in selected)
        holds = set(state.details.get("geometry_constraints", ())) - set(old_ids)
        holds.update((base.id, *ids))
        return Proposal(
            "joint-core-cells" if self.joint else "core-material-cells",
            (*old_ids, base.id, *ids),
            (len(cells), tuple(cuts), region_threshold),
            state.key,
            proposed,
            bounds(document, proposed, (*old_ids, base.id, *ids)),
            details={
                "regions": sum(s.role != "underlay" for s in partition.surfaces),
                "geometry_constraints": sorted(holds),
                "chain_constraints": discard(
                    state.details.get("chain_constraints"), old_ids
                ),
                "paint_constraints": sorted(
                    set(state.details.get("paint_constraints", ())) - {base.id}
                ),
                "core_material_cells": {
                    "cells": len(cells),
                    "removed_paths": len(selected) - len(ids),
                    "source_cut_count": len(atoms.cuts),
                    "source_squared_error": sum(c.error for c in cells),
                    "source_maximum_rgb_residual": max(c.residual for c in cells),
                    "region_threshold": None if self.joint else region_threshold,
                    "joint": self.joint,
                    "layout": self.layout,
                    "facet_edge_model": "multiscale-color-normal"
                    if self.layout != "regions"
                    else None,
                    "grouping": self.grouping,
                    "paint_fit_alpha": "fixed-coverage-carrier"
                    if self.grouping == "paint-fit"
                    else None,
                    "boundary_fit": self.boundary_fit,
                    "ink_support": self.ink_support,
                    "ink_roles": self.ink_roles,
                    "ink_coverage": self.ink_coverage,
                    "source_chain_discovery": "complete-carrier-before-material-budget"
                    if self.ink_support == "connected"
                    else None,
                    "source_roles": (
                        "stroke-body-and-material"
                        if self.ink_roles == "fitted"
                        else "pixel-ink-and-material"
                    )
                    if self.ink_support == "connected"
                    else "owner-majority",
                    "stroke_models": [c.stroke for c in cells if c.stroke],
                    "ink_cells": sum(c.ink for c in cells),
                    "material_planes": len(cells) - sum(c.ink for c in cells)
                    if self.layout == "ink-planes"
                    else None,
                    "model_stage": "source-ink-and-facet-planes"
                    if self.layout == "ink-planes"
                    else "source-facet-planes"
                    if self.layout == "planes"
                    else "source-ink-and-materials"
                    if self.ink_support == "connected"
                    else "dynamic-materials"
                    if self.grouping != "static"
                    else "paint-budget-ablation"
                    if self.joint
                    else "material",
                    "cell_budget": region_threshold if self.joint else None,
                    "coverage_interpretation": "carrier-fringe"
                    if self.joint
                    else "exact-alpha",
                },
            },
            dependencies=(parent.id,),
            partition=partition,
            component=component,
        )
