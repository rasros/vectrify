"""Share an independently observed source junction between editable strokes.

Two distinct qualified source chains must already share the exact physical
point, and an existing complete stroke must retain it. A nearby continuation
can use that same point only after local raw-source ink and native body-gap
proofs. This constructor never closes a source gap or chooses a new junction.
Complete painted, alpha, locality and ownership validation remain mandatory.
"""

from __future__ import annotations

from dataclasses import replace
from xml.etree import ElementTree as ET

import numpy as np
from cairosvg.colors import color
from scipy.ndimage import map_coordinates

from vectrify.document import Editor, Geometry, Selection
from vectrify.document.join import path_style, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.filled_bands import _check
from vectrify.refine.cel_plan.ink import measure_link
from vectrify.refine.cel_plan.local import _native_raster
from vectrify.refine.cel_plan.paint_continuation import _supported
from vectrify.refine.cel_plan.source_absence import ALPHA_TOLERANCE
from vectrify.refine.cel_plan.source_caps import _native

MAX_PROFILES = 128
MAX_PATHS = 32
MAX_NODES = 1024
MAX_PORTS = 128
MAX_MOVEMENT = 1.5
MAX_WIDTH = 12
MATCH = 1e-6


class SourceJunctions:
    def __init__(self, evidence, guard):
        self.evidence, self.guard = evidence, guard

    def connect(self, document, ids, bounds, work):
        """Normalize a near endpoint to its unique already observed junction.

        Only the endpoint values change. Width, paint, controls, node identities,
        physical source junctions and all other paths remain exact. Ambiguity,
        source absence, pins/effects and bounds discard the whole competitor.
        """
        _check(work)
        if (
            self.guard.shape != self.evidence.rgba.shape
            or self.evidence.source_size != self.evidence.rgba.shape[1::-1]
        ):
            raise ValueError("Source junctions must use the original native frame")
        bounds = np.asarray(bounds, float)
        if (
            bounds.shape != (4,)
            or not np.isfinite(bounds).all()
            or np.any(bounds[2:] < bounds[:2])
        ):
            raise ValueError("Source junctions require finite native bounds")
        profiles = self.guard.original_profiles(work=work)
        if len(profiles) > MAX_PROFILES:
            return None
        groups, terminals = {}, []
        for index, profile in enumerate(profiles):
            _check(work)
            observed = self.guard.source_breaks(profile, work=work)
            if (
                profile.component is None
                or observed is None
                or observed.anchors is None
                or np.array_equal(observed.anchors[0], observed.anchors[-1])
            ):
                continue
            terminals.extend((observed.points[0], observed.points[-1]))
            signature = min(
                observed.anchors.tobytes(), observed.anchors[::-1].tobytes()
            )
            for end in (0, -1):
                point = observed.points[end]
                if not ((point >= bounds[:2]) & (point <= bounds[2:])).all():
                    continue
                near = slice(0, 3) if end == 0 else slice(-3, None)
                if (
                    len(observed.points) < 3
                    or not observed.qualified[near].all()
                    or observed.gaps[near].any()
                ):
                    continue
                key = (profile.component, tuple(point))
                groups.setdefault(key, {})[signature] = (index, observed, end)
        groups = {
            key: tuple(value.values())
            for key, value in groups.items()
            if len(value) >= 2
        }
        if not groups:
            return None
        ports = []
        seen_nodes = paths = 0
        for oid in dict.fromkeys(ids):
            _check(work)
            element = document.element(oid)
            if element.tag != "path" or not _supported(document, oid):
                continue
            style = path_style(document, element)
            paint = color(style["stroke"])
            if (
                style["fill"] != "none"
                or paint is None
                or paint[3] != 1
                or style["stroke-linecap"] not in {"round", "butt"}
                or style["stroke-linejoin"] != "round"
            ):
                continue
            frame = root_matrix(document, oid)
            linear = np.asarray(frame[:4]).reshape(2, 2).T
            scale = float(np.linalg.norm(linear[:, 0]))
            width = float(style["stroke-width"]) * scale
            if (
                scale <= 1e-12
                or not 0.8 <= width <= MAX_WIDTH
                or not np.allclose(
                    linear.T @ linear, np.eye(2) * scale**2, rtol=1e-8, atol=1e-10
                )
            ):
                continue
            geometry = transformed_geometry(document.geometry_for(oid), frame)
            paths += 1
            seen_nodes += sum(len(s.nodes) for s in geometry.subpaths)
            if paths > MAX_PATHS or seen_nodes > MAX_NODES:
                return None
            for i, sub in enumerate(geometry.subpaths):
                if sub.closed or len(sub.nodes) < 2:
                    continue
                for end in (0, -1):
                    point = np.asarray(sub.nodes[end].endpoint)
                    if ((point >= bounds[:2]) & (point <= bounds[2:])).all():
                        ports.append(
                            (
                                oid,
                                i,
                                end,
                                point,
                                sub,
                                style,
                                frame,
                                width,
                                np.asarray(paint[:3]) * 255,
                            )
                        )
                        if len(ports) > MAX_PORTS:
                            return None
        changes: dict[str, Geometry] = {}
        witnesses = []
        for oid, i, end, point, sub, _style, frame, width, paint in ports:
            _check(work)
            # A retained physical source terminal/junction is never moved to
            # some other nearby source point, even if the paint is connected.
            if any(np.linalg.norm(point - terminal) <= MATCH for terminal in terminals):
                continue
            candidates = []
            for (_component, coordinate), incident in groups.items():
                junction = np.asarray(coordinate)
                distance = float(np.linalg.norm(point - junction))
                if not MATCH < distance <= MAX_MOVEMENT:
                    continue
                holders = []
                for (
                    host,
                    _i,
                    side,
                    at,
                    host_sub,
                    _style,
                    _frame,
                    host_width,
                    host_paint,
                ) in ports:
                    if (
                        host == oid
                        or np.linalg.norm(at - junction) > MATCH
                        or document.ancestry(host)[-2].id
                        != document.ancestry(oid)[-2].id
                        or max(width, host_width) > 1.6 * min(width, host_width)
                        or np.max(np.abs(paint - host_paint)) > 24
                    ):
                        continue
                    other = np.asarray(host_sub.nodes[-1 if side == 0 else 0].endpoint)
                    if any(
                        np.linalg.norm(
                            other - observed.points[-1 if source_end == 0 else 0]
                        )
                        <= MATCH
                        for _index, observed, source_end in incident
                    ):
                        holders.append(host)
                if not holders:
                    continue
                # Identify the continuation using a near outgoing source span,
                # not just whichever black endpoint happens to be closest.
                node = sub.nodes[1] if end == 0 else sub.nodes[-1]
                target = np.asarray(
                    node.values[:2]
                    if end == 0 and node.command == "C"
                    else node.values[2:4]
                    if end == -1 and node.command == "C"
                    else sub.nodes[1 if end == 0 else -2].endpoint
                )
                tangent = target - point
                length = float(np.linalg.norm(tangent))
                if length < 1e-8:
                    continue
                direction = tangent / length
                continuations = []
                for index, observed, _source_end in incident:
                    limit = np.linalg.norm(observed.points - junction, axis=1) <= 5
                    delta = observed.points - point
                    along = delta @ direction
                    perpendicular = np.linalg.norm(
                        delta - along[:, None] * direction, axis=1
                    )
                    # A junction can obscure a particular side probe. Require
                    # a consecutive qualified outgoing source interval within
                    # the actual stroke corridor, not one arbitrary lookahead.
                    support = (
                        limit
                        & (along >= 0.25)
                        & (along <= 3)
                        & (perpendicular <= max(0.75, width / 2))
                        & observed.qualified
                        & ~observed.gaps
                    )
                    if np.any(support[:-1] & support[1:]):
                        continuations.append(index)
                if len(continuations) != 1:
                    continue
                candidates.append(
                    (junction, incident, continuations[0], tuple(sorted(set(holders))))
                )
            if len(candidates) != 1:
                continue
            junction, incident, continuation, holders = candidates[0]
            reach = int(np.ceil(max(3, 2.5 * width))) + 2
            lo = np.maximum(
                0, np.floor(np.minimum(point, junction)).astype(int) - reach
            )
            hi = np.minimum(
                self.evidence.source_size,
                np.ceil(np.maximum(point, junction)).astype(int) + reach + 1,
            )
            source = self.evidence.rgba[lo[1] : hi[1], lo[0] : hi[0]]
            proof = measure_link(
                np.asarray((point, junction)) - lo,
                source[..., :3] * 255,
                width,
                visible=source[..., 3] > 1 / 255,
                opacity=source[..., 3],
                junctions=lambda samples, j=junction - lo: (
                    np.linalg.norm(samples - j, axis=1) <= MAX_MOVEMENT
                ),
                paint=paint,
            )
            _check(work)
            if proof is None or np.max(np.abs(proof.paint - paint)) > 24:
                continue
            geometry = changes.get(oid, document.geometry_for(oid))
            subs = list(geometry.subpaths)
            nodes = list(subs[i].nodes)
            local = _native(junction, inverse_matrix(frame))
            nodes[end] = replace(
                nodes[end], values=(*nodes[end].values[:-2], *map(float, local))
            )
            subs[i] = replace(subs[i], nodes=tuple(nodes))
            changes[oid] = replace(geometry, subpaths=tuple(subs))
            witnesses.append(
                {
                    "id": oid,
                    "contour": i,
                    "end": "start" if end == 0 else "end",
                    "from": point.tolist(),
                    "to": junction.tolist(),
                    "profiles": [v[0] for v in incident],
                    "continuation_profile": continuation,
                    "holders": holders,
                    "raw_source_ink": True,
                }
            )
        if not changes:
            return None
        gaps = self.guard.gap_centres(limit=4096, work=work)
        for oid, geometry in changes.items():
            _check(work)
            frame = root_matrix(document, oid)
            style = path_style(document, document.element(oid))
            width = float(style["stroke-width"]) * float(
                np.linalg.norm(np.asarray(frame[:4]).reshape(2, 2).T[:, 0])
            )
            samples = []
            for shape in (document.geometry_for(oid), geometry):
                root = ET.Element(
                    "svg",
                    {
                        "width": str(self.evidence.source_size[0]),
                        "height": str(self.evidence.source_size[1]),
                    },
                )
                ET.SubElement(
                    root,
                    "path",
                    {
                        "d": transformed_geometry(shape, frame).path_data(),
                        "fill": "none",
                        "stroke": "white",
                        "stroke-width": repr(width),
                        "stroke-linecap": style["stroke-linecap"],
                        "stroke-linejoin": "round",
                    },
                )
                native = _native_raster(root, self.evidence.source_size).root
                samples.append(
                    map_coordinates(
                        native[..., 3],
                        [gaps[:, 1] - 0.5, gaps[:, 0] - 0.5],
                        order=1,
                        mode="constant",
                        cval=0,
                        output=np.float64,
                    )
                    / 255
                )
                _check(work)
            if np.any(samples[1] > np.maximum(samples[0], ALPHA_TOLERANCE) + 1e-7):
                return None
        editor = Editor(document, selection=Selection(whole_document=True))
        with editor.transaction("Share original source junctions") as tx:
            for oid, geometry in changes.items():
                _check(work)
                tx.replace_geometry(oid, geometry)
        _check(work)
        return editor.snapshot.document, tuple(
            {**w, "native_body_gaps_preserved": True} for w in witnesses
        )
