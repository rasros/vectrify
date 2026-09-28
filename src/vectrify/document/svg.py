"""SVG adapter for the editor's explicit, static subset.

Unsupported input is reported, never silently stripped. This adapter is
separate from the existing CLI importer and does not change its behavior.
"""

from __future__ import annotations

import math
import re
from xml.etree import ElementTree as ET

from vectrify.document.model import (
    Document,
    DocumentError,
    Element,
    Geometry,
    PathNode,
    Subpath,
    new_id,
)

SVG = "http://www.w3.org/2000/svg"
XLINK = "{http://www.w3.org/1999/xlink}href"
NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
TOKEN = re.compile(rf"[A-Za-z]|{NUMBER}")
PAINT = frozenset(
    {
        "fill",
        "stroke",
        "fill-rule",
        "clip-rule",
        "opacity",
        "fill-opacity",
        "stroke-opacity",
        "stroke-width",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-miterlimit",
    }
)
GEOMETRY = {
    "svg": {"width", "height", "viewBox", "preserveAspectRatio"},
    "g": set(),
    "defs": set(),
    "clipPath": {"clipPathUnits"},
    "path": set(),
    "rect": {"x", "y", "width", "height", "rx", "ry"},
    "circle": {"cx", "cy", "r"},
    "ellipse": {"cx", "cy", "rx", "ry"},
    "line": {"x1", "y1", "x2", "y2"},
    "use": {"x", "y", "href"},
}
CONTAINERS = {"svg", "g", "defs", "clipPath"}
NUMERIC = {
    "x",
    "y",
    "width",
    "height",
    "rx",
    "ry",
    "r",
    "cx",
    "cy",
    "x1",
    "y1",
    "x2",
    "y2",
    "opacity",
    "fill-opacity",
    "stroke-opacity",
    "stroke-width",
    "stroke-miterlimit",
}


class UnsupportedSvgError(DocumentError):
    def __init__(self, issues: list[str]):
        self.issues = tuple(issues)
        super().__init__("Unsupported SVG: " + "; ".join(issues))


def validate_attributes(tag: str, attributes: dict[str, str]) -> None:
    if tag not in GEOMETRY:
        raise DocumentError(f"Unsupported element: {tag}")
    allowed = GEOMETRY[tag] | PAINT | {"transform", "clip-path"}
    for name, value in attributes.items():
        if name not in allowed:
            raise DocumentError(f"Unsupported attribute: {name}")
        if name in NUMERIC:
            if not re.fullmatch(NUMBER, value) or not math.isfinite(float(value)):
                raise DocumentError(f"{name} must be a finite number in document units")
            if (
                name
                in {
                    "width",
                    "height",
                    "rx",
                    "ry",
                    "r",
                    "stroke-width",
                    "stroke-miterlimit",
                }
                and float(value) < 0
            ):
                raise DocumentError(f"{name} cannot be negative")
            if "opacity" in name and not 0 <= float(value) <= 1:
                raise DocumentError(f"{name} must be between zero and one")
        if name in {"fill", "stroke"}:
            if not re.fullmatch(
                r"none|#[0-9a-fA-F]{3,8}|[a-zA-Z]+|rgba?\([0-9.,%\s+-]+\)", value
            ):
                raise DocumentError("Only solid paint is supported")
            if value.lower() in {
                "inherit",
                "currentcolor",
                "context-fill",
                "context-stroke",
                "unset",
                "revert",
            }:
                raise DocumentError("Context-dependent paint is not supported")
        if name == "viewBox":
            numbers = value.replace(",", " ").split()
            if len(numbers) != 4 or not all(
                re.fullmatch(NUMBER, n) and math.isfinite(float(n)) for n in numbers
            ):
                raise DocumentError("viewBox must contain four finite numbers")
            if float(numbers[2]) <= 0 or float(numbers[3]) <= 0:
                raise DocumentError("viewBox dimensions must be positive")
        if name == "transform":
            matches = list(
                re.finditer(
                    r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^()]*)\)", value
                )
            )
            remainder = re.sub(
                r"(matrix|translate|scale|rotate|skewX|skewY)\s*\([^()]*\)", "", value
            )
            if not matches or remainder.strip(" ,\t\r\n"):
                raise DocumentError("Invalid SVG transform")
            for match in matches:
                args = match[2].replace(",", " ").split()
                arities = {
                    "matrix": {6},
                    "translate": {1, 2},
                    "scale": {1, 2},
                    "rotate": {1, 3},
                    "skewX": {1},
                    "skewY": {1},
                }
                if len(args) not in arities[match[1]] or not all(
                    re.fullmatch(NUMBER, n) and math.isfinite(float(n)) for n in args
                ):
                    raise DocumentError("Invalid SVG transform arguments")
        enums = {
            "fill-rule": {"nonzero", "evenodd"},
            "clip-rule": {"nonzero", "evenodd"},
            "stroke-linecap": {"butt", "round", "square"},
            "stroke-linejoin": {"miter", "round", "bevel"},
            "clipPathUnits": {"userSpaceOnUse", "objectBoundingBox"},
        }
        if name in enums and value not in enums[name]:
            raise DocumentError(f"Unsupported {name}: {value}")


def parse_path(data: str) -> Geometry:
    """Normalize lines and Beziers to absolute M/L/C without rounding.

    Arc conversion is intentionally not guessed; callers get an import issue.
    """
    if TOKEN.sub("", data).strip(" ,\t\r\n"):
        raise DocumentError("Invalid path data")
    tokens = TOKEN.findall(data)
    subpaths = []
    nodes = []
    current = start = (0.0, 0.0)
    cubic_control = quadratic_control = None
    command = None
    previous = None
    i = 0

    def finish(closed=False):
        if nodes:
            subpaths.append(Subpath(new_id("subpath"), tuple(nodes), closed))
            nodes.clear()

    while i < len(tokens):
        if tokens[i].isalpha():
            command = tokens[i]
            i += 1
        if command is None or command.upper() not in "MLHVCSQTZ":
            raise DocumentError(f"Unsupported path command: {command}")
        upper = command.upper()
        relative = command.islower()
        if upper == "Z":
            if not nodes:
                raise DocumentError("Closepath needs a subpath")
            finish(True)
            current = start
            cubic_control = quadratic_control = None
            previous, command = "Z", None
            continue
        arity = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2}[upper]
        if i + arity > len(tokens) or any(t.isalpha() for t in tokens[i : i + arity]):
            raise DocumentError("Incomplete path command")
        values = tuple(float(t) for t in tokens[i : i + arity])
        i += arity
        if not all(math.isfinite(v) for v in values):
            raise DocumentError("Path coordinates must be finite")

        def point(x, y, current=current, relative=relative):
            return (x + current[0], y + current[1]) if relative else (x, y)

        def reflected(control, current=current):
            return (
                (2 * current[0] - control[0], 2 * current[1] - control[1])
                if control
                else current
            )

        if upper == "M":
            finish()
            endpoint = point(*values)
            start = endpoint
            nodes.append(PathNode(new_id("node"), "M", endpoint))
            command = "l" if relative else "L"
        else:
            if not nodes:
                if previous != "Z":
                    raise DocumentError("Path must begin with moveto")
                nodes.append(PathNode(new_id("node"), "M", current))
                start = current
            if upper in "LHV":
                endpoint = (
                    point(*values)
                    if upper == "L"
                    else (
                        (
                            (current[0] + values[0] if relative else values[0]),
                            current[1],
                        )
                        if upper == "H"
                        else (
                            current[0],
                            (current[1] + values[0] if relative else values[0]),
                        )
                    )
                )
                nodes.append(PathNode(new_id("node"), "L", endpoint))
            else:
                endpoint = point(*values[-2:])
                if upper in "CS":
                    first = (
                        point(*values[:2])
                        if upper == "C"
                        else reflected(
                            cubic_control if previous in {"C", "S"} else None
                        )
                    )
                    second = point(*values[-4:-2])
                    cubic_control = second
                else:
                    control = (
                        point(*values[:2])
                        if upper == "Q"
                        else reflected(
                            quadratic_control if previous in {"Q", "T"} else None
                        )
                    )
                    first = tuple(
                        a + 2 * (b - a) / 3
                        for a, b in zip(current, control, strict=True)
                    )
                    second = tuple(
                        a + 2 * (b - a) / 3
                        for a, b in zip(endpoint, control, strict=True)
                    )
                    quadratic_control = control
                nodes.append(
                    PathNode(new_id("node"), "C", (*first, *second, *endpoint))
                )
        if upper not in {"C", "S"}:
            cubic_control = None
        if upper not in {"Q", "T"}:
            quadratic_control = None
        current, previous = endpoint, upper
    finish()
    return Geometry(new_id("geometry"), tuple(subpaths))


def import_svg(svg: str) -> Document:
    if "<!DOCTYPE" in svg.upper() or "<!ENTITY" in svg.upper():
        raise UnsupportedSvgError(["XML document types and entities"])
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise UnsupportedSvgError([f"Invalid XML: {exc}"]) from exc
    issues: list[str] = []
    geometries: list[Geometry] = []

    def read(node: ET.Element) -> Element:
        if node.tag.startswith("{") and not node.tag.startswith("{" + SVG + "}"):
            issues.append(f"Foreign element {node.tag}")
        tag = node.tag.rsplit("}", 1)[-1]
        attrs = dict(node.attrib)
        object_id = attrs.pop("id", None) or new_id("object")
        name = attrs.pop("data-vectrify-name", "")
        if XLINK in attrs:
            href = attrs.pop(XLINK)
            if "href" in attrs and attrs["href"] != href:
                issues.append(f"{object_id}: conflicting href attributes")
            attrs["href"] = href
        if tag in {"polygon", "polyline"}:
            points = attrs.pop("points", "")
            coordinates = re.findall(NUMBER, points)
            if re.sub(NUMBER, "", points).strip(" ,\t\r\n") or len(coordinates) % 2:
                issues.append(f"{object_id}: invalid polygon/polyline points")
            elif coordinates:
                attrs["d"] = (
                    "M"
                    + " L".join(
                        " ".join(coordinates[i : i + 2])
                        for i in range(0, len(coordinates), 2)
                    )
                    + (" Z" if tag == "polygon" else "")
                )
            tag = "path"
        data = attrs.pop("d", "") if tag == "path" else ""
        if (node.text and node.text.strip()) or (node.tail and node.tail.strip()):
            issues.append(f"{object_id}: text content is unsupported")
        try:
            validate_attributes(tag, attrs)
        except DocumentError as exc:
            issues.append(f"{object_id}: {exc}")
        geometry_id = None
        if tag == "path":
            try:
                geometry = parse_path(data)
                geometry_id = geometry.id
                geometries.append(geometry)
            except DocumentError as exc:
                issues.append(f"{object_id}: {exc}")
        children = tuple(read(child) for child in node)
        if children and tag not in CONTAINERS:
            issues.append(f"{object_id}: {tag} cannot contain child elements")
        if tag == "svg" and node is not root:
            issues.append(f"{object_id}: nested SVG viewports are unsupported")
        if tag == "use" and "href" not in attrs:
            issues.append(f"{object_id}: use requires a local reference")
        return Element(
            object_id, tag, tuple(attrs.items()), children, geometry_id, name=name
        )

    document = Document(read(root), tuple(geometries))
    if issues:
        raise UnsupportedSvgError(issues)
    document.validate()
    return document


def export_svg(document: Document) -> str:
    document.validate()

    def write(element: Element) -> ET.Element:
        attrs = dict(element.attributes)
        validate_attributes(element.tag, attrs)
        attrs["id"] = element.id
        if element.name:
            attrs["data-vectrify-name"] = element.name
        if element.geometry_id:
            attrs["d"] = document.geometry(element.geometry_id).path_data()
        if "href" in attrs:
            attrs["xlink:href"] = attrs.pop("href")
        node = ET.Element(element.tag, attrs)
        node.extend(write(child) for child in element.children)
        return node

    root = write(document.root)
    root.set("xmlns", SVG)
    if any(element.get("href") for element in document.elements()):
        root.set("xmlns:xlink", "http://www.w3.org/1999/xlink")
    return ET.tostring(root, encoding="unicode")
