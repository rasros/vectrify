"""Invert complete filled ink bands into actual editable centerline strokes.

This explicit competitor keeps one original primary owner, paint, frame and
order. It adds no carrier or restored material. Independent marks, branches,
variable-width shapes and unsupported styles exclude a whole-path conversion;
they are never silently dropped to make a stroke. Native painted source-line
comparison precedes publication; the ordinary evaluator still decides quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter1d,
    label,
    map_coordinates,
)

from vectrify.document import Editor, Geometry, Selection, export_svg
from vectrify.document.join import curve_path, path_style
from vectrify.document.model import paint_server
from vectrify.refine import cel
from vectrify.refine.cel_plan.component_edits import ComponentEdit
from vectrify.refine.cel_plan.constraints import discard
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.ink_replace import identified
from vectrify.refine.cel_plan.local import MAX_CROP_PIXELS, Box
from vectrify.refine.cel_plan.model import StageInterruptedError
from vectrify.refine.cel_plan.refine import _bounds
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import Proposal
from vectrify.refine.crossings import crossings

MAX_POINTS = 16_384
MAX_NODES = 256
MAX_PATHS = 16
MAX_WIDTH = 12
MAX_NATIVE_PIXELS = 1536**2
PROFILE_CHUNK = 1024


def supported_style(document, oid):
    """Flat ink can change representation without borrowing dormant effects."""
    element = document.element(oid)
    geometry = document.geometry_for(oid)
    style = path_style(document, element)
    if (
        style["fill"] == "none"
        or paint_server(style["fill"]) is not None
        or style["stroke"] != "none"
        or any(
            a.get("clip-path", "none") != "none" or a.locks
            for a in document.ancestry(oid)
        )
        or any(
            a.get(key, "none") != "none"
            for a in document.ancestry(oid)
            for key in (
                "filter",
                "mask",
                "marker-start",
                "marker-mid",
                "marker-end",
                "stroke-dasharray",
                "vector-effect",
            )
        )
        or any(n.pinned for s in geometry.subpaths for n in s.nodes)
    ):
        return None
    return style


def _check(work):
    if work.interrupted:
        raise StageInterruptedError("Filled band inversion interrupted")


@dataclass(frozen=True)
class Band:
    geometry: Geometry
    width: float
    points: int


def invert(geometry, rule, work, *, tolerance=0.25):
    """One complete physical band, including a hole, without spur pruning.

    Intrinsic vector coverage recentres the skeleton at subpixel precision.
    Only the contiguous transverse band containing each skeleton sample lends
    weight. This creates a proposal, not a source-support or alpha proof.
    """
    _check(work)
    if rule not in {"nonzero", "evenodd"} or not 0 < tolerance <= 1:
        raise ValueError("Band inversion requires a supported rule and tolerance")
    if (
        not geometry.subpaths
        or any(not sub.closed for sub in geometry.subpaths)
        or sum(len(sub.nodes) for sub in geometry.subpaths) > MAX_NODES
    ):
        return None
    bounds = np.asarray(curve_path(geometry).bounds, float)
    if not np.isfinite(bounds).all() or np.max(np.abs(bounds)) > 1_000_000:
        return None
    origin = np.floor(bounds[:2] - 4).astype(int)
    size = np.ceil(bounds[2:] + 4).astype(int) - origin
    if min(size) <= 0 or int(np.prod(size)) > MAX_CROP_PIXELS:
        return None
    svg = (
        f'<svg width="{size[0]}" height="{size[1]}" '
        f'viewBox="{origin[0]} {origin[1]} {size[0]} {size[1]}">'
        f'<path d="{geometry.path_data()}" fill="white" fill-rule="{rule}"/></svg>'
    )
    alpha = render(svg, (int(size[0]), int(size[1])))[..., 3]
    _check(work)
    mask = alpha > 0.05
    # A compound ink path can contain independent marks. Do not erase them or
    # assign their source ownership to the surviving longest chain.
    _components, count = label(mask, np.ones((3, 3)))
    if count != 1:
        return None
    depth = np.asarray(distance_transform_edt(mask))
    if float(np.percentile(depth[mask], 95)) > MAX_WIDTH / 2:
        return None
    skeleton = cel.thin(mask)
    _check(work)
    runs = cel.line_runs(skeleton, spur=0, depth=depth)
    _check(work)
    if len(runs) != 1 or not 2 <= len(runs[0]) <= MAX_POINTS:
        return None
    raw = runs[0] + origin
    length = float(np.linalg.norm(np.diff(raw, axis=0), axis=1).sum())
    width = float(alpha.sum() / max(length, 1e-12))
    if not 0.8 <= width <= MAX_WIDTH or length < max(8, 8 * width):
        return None
    closed = np.array_equal(raw[0], raw[-1])
    source = raw[:-1] if closed else raw
    mode = "wrap" if closed else "nearest"
    smooth = gaussian_filter1d(source, 2, axis=0, mode=mode)
    tangent = (
        np.roll(smooth, -1, axis=0) - np.roll(smooth, 1, axis=0)
        if closed
        else np.gradient(smooth, axis=0)
    )
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
    offsets = np.arange(-MAX_WIDTH, MAX_WIDTH + 0.125, 0.25)
    widths = np.empty(len(source))
    shift = np.empty(len(source))
    for start in range(0, len(source), PROFILE_CHUNK):
        _check(work)
        end = min(start + PROFILE_CHUNK, len(source))
        positions = (
            source[start:end, None, :]
            + normal[start:end, None, :] * offsets[None, :, None]
        )
        coverage = map_coordinates(
            alpha,
            [positions[..., 1] - 0.5 - origin[1], positions[..., 0] - 0.5 - origin[0]],
            order=1,
            mode="constant",
            cval=0,
        )
        active = coverage > 1e-6
        components = np.cumsum(~active, axis=1)
        middle = len(offsets) // 2
        if not active[:, middle].all():
            return None
        coverage *= active & (components == components[:, middle, None])
        mass = coverage.sum(axis=1)
        widths[start:end] = mass * 0.25
        shift[start:end] = (coverage * offsets).sum(axis=1) / np.maximum(mass, 1e-12)
    # A constant-width competitor must describe the complete band. Wide
    # terminals remain a separate unresolved interpretation, not new caps.
    if np.percentile(widths, 90) / max(np.percentile(widths, 10), 0.25) > 2:
        return None
    width = float(np.median(widths))
    if not 0.8 <= width <= MAX_WIDTH:
        return None
    centered = source + shift[:, None] * normal
    points = gaussian_filter1d(centered, 1, axis=0, mode=mode)
    if closed:
        points = np.vstack((points, points[0]))
    else:
        points[[0, -1]] = centered[[0, -1]]
        # Locate each existing band's physical cap along its own tangent.
        # The first half-coverage exit stops the query; another band cannot
        # supply an endpoint. A painted native comparison still tests the
        # resulting complete body, and never authorizes a missing connection.
        for end, inside in ((0, min(4, len(points) - 1)), (-1, max(-5, -len(points)))):
            direction = points[end] - points[inside]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                return None
            direction /= norm
            distances = np.arange(0, 2 * MAX_WIDTH + 0.125, 0.125)
            queries = points[end] + distances[:, None] * direction
            opacity = map_coordinates(
                alpha,
                [queries[:, 1] - 0.5 - origin[1], queries[:, 0] - 0.5 - origin[0]],
                order=1,
                mode="constant",
                cval=0,
            )
            exits = np.flatnonzero(opacity < 0.5)
            if not len(exits) or exits[0] == 0:
                return None
            i = int(exits[0])
            fraction = (opacity[i - 1] - 0.5) / (opacity[i - 1] - opacity[i])
            distance = distances[i - 1] + 0.125 * fraction
            points[end] += direction * distance
    _check(work)
    contour = fitted(points, tolerance).contour
    result = Geometry("band-stroke", (contour,))
    if len(contour.nodes) > MAX_NODES or crossings(result):
        return None
    _check(work)
    return Band(result, width, len(raw))


class FilledBands:
    """Opt-in whole-owner proposals; compound splitting is deliberately absent."""

    def __init__(self, evidence, graph, options, *, guard):
        self.evidence, self.graph, self.options, self.guard = (
            evidence,
            graph,
            options,
            guard,
        )
        self.ink = np.bincount(
            graph.labels.ravel(),
            weights=evidence.drawn.ravel(),
            minlength=len(graph.regions),
        )
        self.diagnostics = dict.fromkeys(
            ("eligible", "no_band", "source_exclusions", "proposals", "bounded"), 0
        )

    def __call__(self, state, work):
        try:
            yield from self.proposals(state, work)
        except StageInterruptedError:
            return

    def proposals(self, state, work):
        _check(work)
        partition = state.partition
        if partition is None:
            return
        namespace = partition.atoms.key if partition.atoms is not None else None
        if namespace != self.graph.source_atoms:
            self.diagnostics["bounded"] += 1
            return
        document = state.document
        width, height = self.evidence.source_size
        if width * height > MAX_NATIVE_PIXELS:
            self.diagnostics["bounded"] += 1
            return
        before = None
        eligible = []
        for surface in partition.surfaces:
            _check(work)
            if surface.role == "underlay":
                continue
            geometry = document.geometry_for(surface.id)
            style = supported_style(document, surface.id)
            area = sum(self.graph.regions[i].area for i in surface.members)
            if (
                not area
                or sum(self.ink[i] for i in surface.members) < area * 0.6
                or style is None
            ):
                continue
            eligible.append(
                (sum(len(s.nodes) for s in geometry.subpaths), surface, style)
            )
        eligible.sort(key=lambda v: (-v[0], v[1].id))
        for old_nodes, surface, style in eligible[:MAX_PATHS]:
            _check(work)
            self.diagnostics["eligible"] += 1
            model = invert(document.geometry_for(surface.id), style["fill-rule"], work)
            if model is None or len(model.geometry.subpaths[0].nodes) >= old_nodes:
                self.diagnostics["no_band"] += 1
                continue
            try:
                guard = self.guard(work)
            except ValueError:
                self.diagnostics["bounded"] += 1
                return
            if guard is None:
                self.diagnostics["source_exclusions"] += 1
                continue
            if before is None:
                before = render(export_svg(document), self.evidence.source_size)
                _check(work)
            parent = document.ancestry(surface.id)[-2].id
            component = ComponentEdit.bind(document, partition, parent, work)
            for cap in ("butt", "round"):
                _check(work)
                editor = Editor(document, selection=Selection(whole_document=True))
                with editor.transaction(
                    "Replace filled ink band with editable stroke"
                ) as tx:
                    tx.replace_geometry(
                        surface.id, identified(model.geometry, surface.id)
                    )
                    tx.set_fill(surface.id, "none")
                    tx.set_attributes(
                        surface.id,
                        {
                            "stroke": style["fill"],
                            "stroke-width": repr(model.width),
                            "stroke-opacity": style["fill-opacity"],
                            "fill-opacity": "1",
                            "stroke-linecap": cap,
                            "stroke-linejoin": "round",
                        },
                    )
                proposed = editor.snapshot.document
                actual = render(export_svg(proposed), self.evidence.source_size)
                comparison = guard.compare(before, actual, work=work)
                if not comparison["qualified_samples"] or comparison["rejections"]:
                    self.diagnostics["source_exclusions"] += 1
                    continue
                boxes = [_bounds(d, surface.id) for d in (document, proposed)]
                box = Box(
                    math.floor(min(b[0] for b in boxes)),
                    math.floor(min(b[1] for b in boxes)),
                    math.ceil(max(b[2] for b in boxes)),
                    math.ceil(max(b[3] for b in boxes)),
                )
                self.diagnostics["proposals"] += 1
                yield Proposal(
                    "filled-band-stroke",
                    (surface.id,),
                    (model.width, cap),
                    state.key,
                    proposed,
                    box,
                    estimate=len(model.geometry.subpaths[0].nodes) - old_nodes,
                    details={
                        "geometry_constraints": sorted(
                            set(state.details.get("geometry_constraints", ()))
                            | {surface.id}
                        ),
                        "chain_constraints": discard(
                            state.details.get("chain_constraints"), (surface.id,)
                        ),
                        "filled_band_stroke": {
                            "width": model.width,
                            "cap": cap,
                            "source_points": model.points,
                            "source_line_comparison": comparison,
                            "ownership": "complete-original-owner",
                        },
                    },
                    dependencies=(parent,),
                    partition=partition,
                    component=component,
                )
