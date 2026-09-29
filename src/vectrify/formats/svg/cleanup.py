"""Conservative final geometry cleanup for generated, static SVGs.

Polyline geometry is kept exact. Compatible opaque siblings become compound
paths without changing paint order or crossing clip/group boundaries. This is
not a general Boolean union: overlapping even-odd fills, referenced objects,
transparency, and unsupported styling are deliberately preserved.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import TypedDict
from xml.etree import ElementTree as ET

_SVG = "{http://www.w3.org/2000/svg}"
_TOKEN = re.compile(r"[MLZ]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_SOLID = re.compile(r"(?:#[0-9a-fA-F]{3}|#[0-9a-fA-F]{6}|[a-zA-Z]+)\Z")
_PAINT = {
    "fill",
    "stroke",
    "stroke-width",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-miterlimit",
    "fill-rule",
    "clip-rule",
    "opacity",
    "fill-opacity",
    "stroke-opacity",
}
_EFFECTS = {
    "filter",
    "mask",
    "marker-start",
    "marker-mid",
    "marker-end",
    "stroke-dasharray",
    "stroke-dashoffset",
    "pathLength",
}


class CleanupStats(TypedDict, total=False):
    paths_before: int
    paths_after: int
    line_vertices_before: int
    line_vertices_after: int
    vertices_removed: int
    paths_merged: int
    duplicate_paths_removed: int
    empty_paths_removed: int
    skipped_reason: str


def _parse(data: str):
    if _TOKEN.sub("", data).strip(" \t\r\n,"):
        return None
    tokens = _TOKEN.findall(data)
    paths, points, command = [], [], None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"M", "L", "Z"}:
            command = token
            index += 1
            if command == "Z":
                if not points:
                    return None
                paths.append((points, True))
                points, command = [], None
                continue
            if command == "M" and points:
                paths.append((points, False))
                points = []
        if command not in {"M", "L"} or index + 1 >= len(tokens):
            return None
        if tokens[index] in {"M", "L", "Z"} or tokens[index + 1] in {"M", "L", "Z"}:
            return None
        if command == "L" and not points:
            return None
        x, y = tokens[index : index + 2]
        points.append((Decimal(x), Decimal(y), x, y))
        index += 2
        command = "L"
    if points:
        paths.append((points, False))
    return paths


def _between(a, b, c):
    ux, uy, vx, vy = b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]
    return ux * vy == uy * vx and ux * vx + uy * vy >= 0


def _simplify(points, closed):
    selected = []
    for point in points:
        if selected and point[:2] == selected[-1][:2]:
            continue
        while len(selected) >= 2 and _between(selected[-2], selected[-1], point):
            selected.pop()
        selected.append(point)
    if closed and len(selected) > 3 and selected[0][:2] == selected[-1][:2]:
        selected.pop()
    if closed:
        while len(selected) > 3 and _between(selected[-1], selected[0], selected[1]):
            selected.pop(0)
        while len(selected) > 3 and _between(selected[-2], selected[-1], selected[0]):
            selected.pop()
    return selected if len(selected) >= (3 if closed else 2) else points


def _box(parsed):
    points = [p for chain, _ in parsed for p in chain]
    if not points:
        return None
    return (
        min(p[0] for p in points),
        min(p[1] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
    )


def _disjoint(first, second):
    return (
        first[2] < second[0]
        or second[2] < first[0]
        or first[3] < second[1]
        or second[3] < first[1]
    )


def cleanup_svg_geometry(
    svg: str, *, unreferenced: frozenset[str] = frozenset()
) -> tuple[str, CleanupStats]:
    """Simplify redundant vertices and merge compatible consecutive paths.

    Shape coordinates are never rounded or approximated. Merging opaque
    strokes can change antialiasing slightly at their shared pixels.

    A path with an ``id`` is left alone, since something may reference it,
    unless the caller lists that id in *unreferenced*; a merge keeps the first
    path's id.
    """
    root = ET.fromstring(svg)
    paths = list(root.iter(_SVG + "path"))
    parsed = {id(p): _parse(p.get("d", "")) for p in paths}

    def count(values):
        return sum(
            len(points) for item in values if item is not None for points, _ in item
        )

    stats: CleanupStats = {
        "paths_before": len(paths),
        "paths_after": len(paths),
        "line_vertices_before": count(parsed.values()),
        "vertices_removed": 0,
        "paths_merged": 0,
        "duplicate_paths_removed": 0,
        "empty_paths_removed": 0,
    }
    # CSS and animation can observe identities and styles outside this tree.
    if any(
        el.tag.rsplit("}", 1)[-1]
        in {"style", "script", "animate", "animateTransform", "set"}
        or "style" in el.attrib
        or "class" in el.attrib
        for el in root.iter()
    ):
        stats["skipped_reason"] = "CSS, scripts, or animation"
        stats["line_vertices_after"] = stats["line_vertices_before"]
        return svg, stats
    # References may inherit markers/dashes from a use element. In that case
    # keep vertices globally, since removing a vertex can move a marker.
    point_effects = any(any(key in el.attrib for key in _EFFECTS) for el in root.iter())
    if not point_effects:
        for path in paths:
            chains = parsed[id(path)]
            if chains is None:
                continue
            reduced = [(_simplify(points, closed), closed) for points, closed in chains]
            removed = count([chains]) - count([reduced])
            if removed:
                path.set(
                    "d",
                    " ".join(
                        "M"
                        + " L".join(f"{p[2]} {p[3]}" for p in points)
                        + (" Z" if closed else "")
                        for points, closed in reduced
                    ),
                )
                parsed[id(path)] = reduced
                stats["vertices_removed"] += removed

    def visit(parent, inherited, protected=False):
        paint = {
            **inherited,
            **{
                key: value
                for key, value in parent.attrib.items()
                if key in _PAINT | _EFFECTS
            },
        }
        protected = protected or parent.tag.rsplit("}", 1)[-1] in {
            "defs",
            "clipPath",
            "mask",
            "pattern",
        }
        # Group opacity/effects compose with descendants; a child cannot undo
        # them by setting its own opacity to one or filter to none.
        try:
            protected = protected or float(parent.get("opacity", "1")) != 1
        except ValueError:
            protected = True
        protected = protected or any(
            parent.get(key, "none") != "none" for key in _EFFECTS
        )
        previous = None
        previous_box = None
        previous_attrs = None
        for child in list(parent):
            if child.tag != _SVG + "path":
                visit(child, paint, protected)
                previous = None
                continue
            attrs = {
                key: value
                for key, value in child.attrib.items()
                if key != "d" and not (key == "id" and value in unreferenced)
            }
            effective = {**paint, **attrs}
            chains = parsed[id(child)]
            safe = not protected and not (set(attrs) - _PAINT) and chains is not None
            safe = safe and not any(
                key in effective and effective[key] != "none" for key in _EFFECTS
            )
            try:
                safe = safe and all(
                    float(effective.get(key, "1")) == 1
                    for key in ("opacity", "fill-opacity", "stroke-opacity")
                )
            except ValueError:
                safe = False
            safe = safe and all(
                _SOLID.fullmatch(effective.get(key, "none"))
                and effective.get(key, "none").lower()
                not in {
                    "inherit",
                    "currentcolor",
                    "context-fill",
                    "context-stroke",
                    "transparent",
                    "initial",
                    "unset",
                    "revert",
                }
                for key in ("fill", "stroke")
            )
            if not safe or chains is None:
                previous = None
                continue
            box = _box(chains)
            if box is None:
                parent.remove(child)
                stats["empty_paths_removed"] += 1
                continue
            if previous is not None and attrs == previous_attrs:
                assert previous_box is not None
                duplicate = child.get("d") == previous.get("d")
                stroke_only = effective.get("fill", "black") == "none"
                if duplicate:
                    parent.remove(child)
                    stats["duplicate_paths_removed"] += 1
                    continue
                solid_fill = effective.get("stroke", "none") in {
                    "none",
                    effective.get("fill", "black"),
                }
                if stroke_only or (solid_fill and _disjoint(previous_box, box)):
                    previous.set("d", previous.get("d") + " " + child.get("d"))
                    previous_chains = parsed[id(previous)]
                    assert previous_chains is not None
                    previous_chains.extend(chains)
                    parent.remove(child)
                    stats["paths_merged"] += 1
                    previous_box = (
                        min(previous_box[0], box[0]),
                        min(previous_box[1], box[1]),
                        max(previous_box[2], box[2]),
                        max(previous_box[3], box[3]),
                    )
                    continue
            previous, previous_box, previous_attrs = child, box, attrs

    visit(root, {})
    remaining = list(root.iter(_SVG + "path"))
    stats["paths_after"] = len(remaining)
    stats["line_vertices_after"] = count(parsed[id(path)] for path in remaining)
    return ET.tostring(root, encoding="unicode"), stats
