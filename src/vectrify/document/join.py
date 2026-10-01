"""Area-weighted paint and curved boolean union for region joins."""

from __future__ import annotations

import math
from dataclasses import replace

import pathops
from cairosvg.colors import color

from vectrify.document.hit_test import (
    IDENTITY,
    HitIndex,
    Matrix,
    _filled,
    _flatten,
    multiply,
    transform,
)
from vectrify.document.model import Document, DocumentError, Element, Geometry
from vectrify.document.paint import solid_paint
from vectrify.document.svg import PAINT, parse_path


def path_style(document: Document, element: Element) -> dict[str, str]:
    style = {
        "fill": "black",
        "stroke": "none",
        "fill-opacity": "1",
        "stroke-opacity": "1",
        "stroke-width": "1",
        "stroke-miterlimit": "4",
        "fill-rule": "nonzero",
        "stroke-linejoin": "miter",
        "stroke-linecap": "butt",
    }
    for ancestor in document.ancestry(element.id):
        style.update(
            (k, v) for k, v in ancestor.attributes if k in PAINT and k != "opacity"
        )
    style["opacity"] = element.get("opacity", "1") or "1"
    return style


def average_paint(
    styles: list[dict[str, str]],
    areas: list[float],
    document: Document | None = None,
) -> dict[str, str]:
    """Average channels by painted area; missing paint contributes zero alpha.

    With *document*, a gradient counts as its mean colour.
    """
    weights = areas if sum(areas) > 0 else [1.0] * len(styles)
    total = sum(weights)
    result = dict(styles[-1])
    for key in ("opacity", "stroke-width", "stroke-miterlimit"):
        if len({s[key] for s in styles}) > 1:
            result[key] = str(
                sum(float(s[key]) * w for s, w in zip(styles, weights, strict=True))
                / total
            )
    for kind in ("fill", "stroke"):
        opacity = kind + "-opacity"
        if len({(s[kind], s[opacity]) for s in styles}) == 1:
            continue
        paints = [
            solid_paint(document, s[kind]) if document is not None else s[kind]
            for s in styles
        ]
        rgba = [color(p) if p != "none" else (0, 0, 0, 0) for p in paints]
        alpha_weights = [
            w * c[3] * float(s[opacity])
            for w, c, s in zip(weights, rgba, styles, strict=True)
        ]
        alpha = sum(alpha_weights)
        if alpha == 0:
            result[kind], result[opacity] = "none", "1"
        else:
            rgb = [
                round(
                    255
                    * sum(c[i] * w for c, w in zip(rgba, alpha_weights, strict=True))
                    / alpha
                )
                for i in range(3)
            ]
            result[kind] = "#" + "".join(f"{v:02x}" for v in rgb)
            result[opacity] = str(alpha / total)
    return result


def join_paint(
    styles: list[dict[str, str]],
    areas: list[float],
    color_index: int | None = None,
    document: Document | None = None,
) -> dict[str, str]:
    """Keep numeric paint area-weighted, optionally taking one source's colors."""
    paint = average_paint(styles, areas, document)
    if color_index is not None:
        source = styles[color_index]
        for key in ("fill", "stroke", "fill-opacity", "stroke-opacity"):
            paint[key] = source[key]
    return paint


def painted_weights(document: Document, paths: list[Element]) -> list[float]:
    # Retain definitions and ancestry, but avoid flattening unrelated artwork.
    keep = {a.id for p in paths for a in document.ancestry(p.id)}
    from dataclasses import replace

    def prune(element: Element) -> Element:
        if element.tag == "defs":
            return element
        return replace(
            element,
            children=tuple(
                prune(c) for c in element.children if c.id in keep or c.tag == "defs"
            ),
        )

    subset = replace(document, root=prune(document.root))
    # References can target drawing objects outside this subset.
    try:
        subset.validate()
    except DocumentError:
        subset = document
    index = HitIndex(subset, tolerance=0.05)
    return [index.painted_area(p.id) for p in paths]


def union_geometry(
    geometries: list[Geometry], styles: list[dict[str, str]]
) -> Geometry:
    result = None
    try:
        for geometry, style in zip(geometries, styles, strict=True):
            path = curve_path(geometry, style["fill-rule"])
            result = (
                path
                if result is None
                else pathops.op(result, path, pathops.PathOp.UNION)
            )
        if result is None:
            raise DocumentError("No regions to join")
        return path_geometry(result)
    except pathops.PathOpsError as exc:
        raise DocumentError("Could not resolve the selected region contours") from exc


def transformed_geometry(geometry: Geometry, matrix: Matrix) -> Geometry:
    a, b, c, d, e, f = matrix
    return replace(
        geometry,
        subpaths=tuple(
            replace(
                sub,
                nodes=tuple(
                    replace(
                        node,
                        values=tuple(
                            value
                            for x, y in zip(
                                node.values[::2], node.values[1::2], strict=True
                            )
                            for value in (a * x + c * y + e, b * x + d * y + f)
                        ),
                    )
                    for node in sub.nodes
                ),
            )
            for sub in geometry.subpaths
        ),
    )


def curve_path(geometry: Geometry, rule: str = "nonzero") -> pathops.Path:
    path = pathops.Path(
        fillType=pathops.FillType.EVEN_ODD
        if rule == "evenodd"
        else pathops.FillType.WINDING
    )
    for sub in geometry.subpaths:
        for node in sub.nodes:
            if node.command == "M":
                path.moveTo(*node.values)
            elif node.command == "L":
                path.lineTo(*node.values)
            else:
                path.cubicTo(*node.values)
        path.close()
    return path


def path_geometry(path: pathops.Path) -> Geometry:
    # Conics arise only from ellipse/rounded-rectangle clip primitives.
    path.convertConicsToQuads(0.01)
    commands = []
    verbs = {
        pathops.PathVerb.MOVE: "M",
        pathops.PathVerb.LINE: "L",
        pathops.PathVerb.QUAD: "Q",
        pathops.PathVerb.CUBIC: "C",
        pathops.PathVerb.CLOSE: "Z",
    }
    for verb, points in path:
        if verb not in verbs:
            raise DocumentError("The clipping result contains an unsupported curve")
        commands.append(
            verbs[verb] + " ".join(str(v) for point in points for v in point)
        )
    return parse_path(" ".join(commands))


def clip_intersection(path: pathops.Path, clip: pathops.Path) -> pathops.Path:
    try:
        return pathops.op(path, clip, pathops.PathOp.INTERSECTION)
    except pathops.PathOpsError:
        # Degenerate/self-touching traced polygons can defeat Skia's float
        # intersection. Resolve directed winding with double-precision noding;
        # lines stay exact and curves are flattened to 0.01 document units.
        from shapely.geometry.polygon import orient

        def filled(value: pathops.Path):
            geometry = path_geometry(value)
            contours = []
            for sub in geometry.subpaths:
                points = [sub.nodes[0].endpoint]
                for node in sub.nodes[1:]:
                    if node.command == "C":
                        v = node.values
                        points.extend(
                            _flatten(
                                (points[-1], (v[0], v[1]), (v[2], v[3]), node.endpoint),
                                0.01,
                            )
                        )
                    else:
                        points.append(node.endpoint)
                contours.append((tuple(points), True))
            return _filled(
                tuple(contours),
                "evenodd" if value.fillType == pathops.FillType.EVEN_ODD else "nonzero",
            )

        result = filled(path).intersection(filled(clip))
        output = pathops.Path()

        def append(shape):
            if shape.geom_type == "Polygon":
                polygon = orient(shape, sign=1)
                for ring in [polygon.exterior, *polygon.interiors]:
                    points = list(ring.coords)
                    output.moveTo(*points[0])
                    for point in points[1:-1]:
                        output.lineTo(*point)
                    output.close()
            elif hasattr(shape, "geoms"):
                for part in shape.geoms:
                    append(part)

        append(result)
        return output


def clip_outline(
    document: Document,
    element: Element,
    parent: Matrix = IDENTITY,
    rule: str = "nonzero",
) -> pathops.Path:
    """Resolve a clip's shapes/use instances in the destination coordinate space."""
    matrix = multiply(parent, transform(element.get("transform")))
    rule = element.get("clip-rule", rule) or rule

    def number(name: str, default: float = 0) -> float:
        return float(element.get(name, str(default)) or default)

    if element.tag == "path":
        path = curve_path(document.geometry_for(element.id), rule).transform(*matrix)
    elif element.tag == "use":
        offset = (1, 0, 0, 1, number("x"), number("y"))
        path = clip_outline(
            document,
            document.element((element.get("href") or "")[1:]),
            multiply(matrix, offset),
            rule,
        )
    elif element.tag in {"g", "clipPath", "svg"}:
        path = pathops.Path()
        for child in element.children:
            if child.tag == "defs":
                continue
            path = pathops.op(
                path, clip_outline(document, child, matrix, rule), pathops.PathOp.UNION
            )
    else:
        path = pathops.Path(
            fillType=pathops.FillType.EVEN_ODD
            if rule == "evenodd"
            else pathops.FillType.WINDING
        )
        if element.tag in {"circle", "ellipse", "rect"}:
            if element.tag == "rect":
                x, y, w, h = number("x"), number("y"), number("width"), number("height")
                rx = min(number("rx", number("ry")), w / 2)
                ry = min(number("ry", number("rx")), h / 2)
            else:
                rx = number("r") if element.tag == "circle" else number("rx")
                ry = number("r") if element.tag == "circle" else number("ry")
                x, y, w, h = number("cx") - rx, number("cy") - ry, 2 * rx, 2 * ry
            if w > 0 and h > 0:
                path.moveTo(x + rx, y)
                path.lineTo(x + w - rx, y)
                if rx and ry:
                    path.conicTo(x + w, y, x + w, y + ry, math.sqrt(0.5))
                else:
                    path.lineTo(x + w, y)
                path.lineTo(x + w, y + h - ry)
                if rx and ry:
                    path.conicTo(x + w, y + h, x + w - rx, y + h, math.sqrt(0.5))
                else:
                    path.lineTo(x + w, y + h)
                path.lineTo(x + rx, y + h)
                if rx and ry:
                    path.conicTo(x, y + h, x, y + h - ry, math.sqrt(0.5))
                else:
                    path.lineTo(x, y + h)
                path.lineTo(x, y + ry)
                if rx and ry:
                    path.conicTo(x, y, x + rx, y, math.sqrt(0.5))
                else:
                    path.lineTo(x, y)
                path.close()
        elif element.tag in {"polygon", "polyline"}:
            values = [
                float(v)
                for v in (element.get("points") or "").replace(",", " ").split()
            ]
            for i in range(0, len(values), 2):
                if i == 0:
                    path.moveTo(*values[i : i + 2])
                else:
                    path.lineTo(*values[i : i + 2])
            if values:
                path.close()
        path = path.transform(*matrix)
    clipping = element.get("clip-path", "none") or "none"
    if clipping != "none":
        definition = document.element(clipping[5:-1])
        if definition.get("clipPathUnits") == "objectBoundingBox":
            raise DocumentError(
                "Resolve objectBoundingBox clipping before joining across groups"
            )
        path = clip_intersection(path, clip_outline(document, definition, matrix))
    return path


def bake_group_path(
    document: Document, element: Element, container_id: str
) -> tuple[Geometry, dict[str, str], bool]:
    """Resolve the groups below the common container before reparenting a path."""
    ancestry = document.ancestry(element.id)
    start = next(i for i, a in enumerate(ancestry) if a.id == container_id) + 1
    matrix = IDENTITY
    clips = []
    style = path_style(document, element)
    for ancestor in ancestry[start:]:
        matrix = multiply(matrix, transform(ancestor.get("transform")))
        if ancestor.id != element.id:
            style["opacity"] = str(
                float(style["opacity"]) * float(ancestor.get("opacity", "1") or 1)
            )
        clipping = ancestor.get("clip-path", "none") or "none"
        if clipping != "none":
            definition = document.element(clipping[5:-1])
            if definition.get("clipPathUnits") == "objectBoundingBox":
                raise DocumentError(
                    "Resolve objectBoundingBox clipping before joining across groups"
                )
            clips.append(clip_outline(document, definition, matrix))
    if style["stroke"] != "none":
        a, b, c, d, _, _ = matrix
        scale = math.hypot(a, b)
        if not math.isclose(
            scale, math.hypot(c, d), rel_tol=1e-9, abs_tol=1e-9
        ) or not math.isclose(a * c + b * d, 0, abs_tol=1e-9):
            raise DocumentError(
                "Convert non-uniformly scaled strokes to outlines "
                "before joining across groups"
            )
        style["stroke-width"] = str(float(style["stroke-width"]) * scale)
    geometry = transformed_geometry(document.geometry_for(element.id), matrix)
    if clips:
        if style["fill"] == "none":
            raise DocumentError(
                "Resolve clipping on stroke-only paths before joining across groups"
            )
        path = curve_path(geometry, style["fill-rule"])
        for clip in clips:
            path = clip_intersection(path, clip)
        geometry = replace(path_geometry(path), id=geometry.id)
        style["fill-rule"] = "nonzero"
    return geometry, style, bool(clips) or matrix != IDENTITY
