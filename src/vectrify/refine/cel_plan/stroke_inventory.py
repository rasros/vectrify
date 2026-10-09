"""Source-only inventory of exported editable stroke bodies and physical ports.

Raster likeness cannot tell a filled band from a centerline stroke. This bounded
diagnostic queries the frozen original ink observations against actual exported
stroke bodies. Coverage is a representation observation, not proof of complete
chains, correct paint, ownership, removed old fills or valid junctions.
"""

from __future__ import annotations

import math
from collections import defaultdict
from xml.etree import ElementTree as ET

import numpy as np
from cairosvg.colors import color
from scipy.ndimage import map_coordinates

from vectrify.document.join import path_style, transformed_geometry
from vectrify.document.model import paint_server
from vectrify.document.redraw import root_matrix
from vectrify.refine.cel_plan.filled_bands import MAX_NATIVE_PIXELS, _check
from vectrify.refine.cel_plan.line_fidelity import MAX_PROFILES, MAX_SAMPLES
from vectrify.refine.cel_plan.local import _native_raster

MAX_PATHS = 256
MAX_NODES = 16_384
SUPPORT_ALPHA = 0.05
EFFECTS = (
    "clip-path",
    "filter",
    "mask",
    "marker-start",
    "marker-mid",
    "marker-end",
    "stroke-dasharray",
    "vector-effect",
)


def _style(document, element):
    style = path_style(document, element)
    ancestors = document.ancestry(element.id)
    if (
        style["fill"] != "none"
        or style["stroke"] == "none"
        or paint_server(style["stroke"]) is not None
        or any(a.tag in {"defs", "clipPath", "mask"} or a.locks for a in ancestors)
        or any(a.get(k, "none") != "none" for a in ancestors for k in EFFECTS)
        or any(a.get("display") == "none" for a in ancestors)
        or any(
            n.pinned
            for s in document.geometry_for(element.id).subpaths
            for n in s.nodes
        )
    ):
        return None
    visibility = "visible"
    opacity = 1.0
    try:
        for ancestor in ancestors:
            visibility = ancestor.get("visibility", visibility)
            opacity *= float(ancestor.get("opacity", "1"))
        opacity *= float(style["stroke-opacity"])
        opacity *= color(style["stroke"])[3]
        width = float(style["stroke-width"])
    except (TypeError, ValueError):
        return None
    if (
        visibility != "visible"
        or not math.isfinite(opacity)
        or not 0 < opacity <= 1
        or not math.isfinite(width)
        or width <= 0
    ):
        return None
    return style, opacity


class StrokeInventory:
    def __init__(self, guard):
        self.guard = guard

    def observe(self, document, work):
        """Count original qualified positions supported by actual stroke bodies.

        Use the complete native viewport and actual local width/frame/caps/joins.
        Filled paths, dormant definitions and unsupported effects contribute no
        stroke support. Literal ports are reported without authorizing a snap.
        """
        _check(work)
        h, w = self.guard.shape[:2]
        if h * w > MAX_NATIVE_PIXELS:
            raise ValueError("Stroke inventory exceeds the native pixel bound")
        profiles = self.guard.original_profiles(work=work)
        if len(profiles) > MAX_PROFILES:
            raise ValueError("Stroke inventory exceeds the source profile bound")
        observations = [self.guard.source_breaks(p, work=work) for p in profiles]
        if sum(len(o.points) for o in observations) > MAX_SAMPLES:
            raise ValueError("Stroke inventory exceeds the source sample bound")
        root = ET.Element("svg", {"width": str(w), "height": str(h)})
        groups = {}
        included, excluded = [], []
        ports = defaultdict(list)
        count, contours = 0, 0
        for element in document.elements():
            _check(work)
            if element.tag != "path":
                continue
            style = path_style(document, element)
            if style["fill"] != "none" or style["stroke"] == "none":
                continue
            supported = _style(document, element)
            if supported is None:
                excluded.append(element.id)
                continue
            geometry = document.geometry_for(element.id)
            count += sum(len(s.nodes) for s in geometry.subpaths)
            if len(included) >= MAX_PATHS or count > MAX_NODES:
                raise ValueError("Stroke inventory exceeds the editable geometry bound")
            style, _opacity = supported
            frame = root_matrix(document, element.id)
            if not np.isfinite(frame).all():
                raise ValueError("Stroke inventory requires a finite native frame")
            parent = root
            for ancestor in document.ancestry(element.id)[:-1]:
                if ancestor.id not in groups:
                    groups[ancestor.id] = ET.SubElement(
                        parent, "g", {"opacity": ancestor.get("opacity", "1")}
                    )
                parent = groups[ancestor.id]
            ET.SubElement(
                parent,
                "path",
                {
                    "d": geometry.path_data(),
                    "transform": "matrix(" + " ".join(map(str, frame)) + ")",
                    "fill": "none",
                    "stroke": "white",
                    "stroke-width": style["stroke-width"],
                    "stroke-linecap": style["stroke-linecap"],
                    "stroke-linejoin": style["stroke-linejoin"],
                    "stroke-miterlimit": style["stroke-miterlimit"],
                    "stroke-opacity": repr(
                        float(style["stroke-opacity"]) * color(style["stroke"])[3]
                    ),
                    "opacity": element.get("opacity", "1"),
                },
            )
            included.append(element.id)
            contours += len(geometry.subpaths)
            for index, sub in enumerate(transformed_geometry(geometry, frame).subpaths):
                if sub.closed or len(sub.nodes) < 2:
                    continue
                for end, node in (("start", sub.nodes[0]), ("end", sub.nodes[-1])):
                    ports[node.endpoint].append(
                        {"id": element.id, "contour": index, "end": end}
                    )
        _check(work)
        alpha = _native_raster(root, (w, h)).root[..., 3].astype(np.float64) / 255
        _check(work)
        rows = []
        qualified_total, missing_total = 0, 0
        source_ports = defaultdict(list)
        for index, (profile, observed) in enumerate(
            zip(profiles, observations, strict=True)
        ):
            _check(work)
            points = observed.points
            ink = observed.qualified & ~observed.gaps
            actual = map_coordinates(
                alpha,
                [points[:, 1] - 0.5, points[:, 0] - 0.5],
                order=1,
                mode="constant",
                cval=0,
            )
            missing = ink & (actual < SUPPORT_ALPHA)
            qualified_total += int(ink.sum())
            missing_total += int(missing.sum())
            rows.append(
                {
                    "profile": index,
                    "component": list(profile.component)
                    if profile.component is not None
                    else None,
                    "qualified_samples": int(ink.sum()),
                    "stroke_supported_samples": int((ink & ~missing).sum()),
                    "missing_samples": int(missing.sum()),
                    "missing_indices": np.flatnonzero(missing).tolist(),
                    "source_gap_samples": int(observed.gaps.sum()),
                    "native_bounds": [
                        *points.min(axis=0).tolist(),
                        *points.max(axis=0).tolist(),
                    ],
                }
            )
            if observed.anchors is None or np.array_equal(points[0], points[-1]):
                continue
            for at in (0, -1):
                if observed.qualified[at] and not observed.gaps[at]:
                    source_ports[tuple(points[at])].append(index)
        shared = []
        for point, ends in sorted(ports.items()):
            if len({(end["id"], end["contour"]) for end in ends}) < 2:
                continue
            shared.append(
                {
                    "point": list(point),
                    "ports": ends,
                    "source_profiles": source_ports[point],
                }
            )
        _check(work)
        return {
            "version": 1,
            "scope": (
                "original qualified source samples versus editable stroke bodies"
            ),
            "outline_completion_proved": False,
            "support_alpha": SUPPORT_ALPHA,
            "stroke_objects": included,
            "stroke_contours": contours,
            "stroke_nodes": count,
            "excluded_stroke_objects": excluded,
            "qualified_samples": qualified_total,
            "missing_samples": missing_total,
            "supported_fraction": (qualified_total - missing_total) / qualified_total
            if qualified_total
            else 0.0,
            "literal_shared_ports": shared,
            "profiles": rows,
        }
