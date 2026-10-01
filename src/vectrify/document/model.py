"""Immutable SVG editing state, independent of rendering and search workers."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from uuid import uuid4


class DocumentError(ValueError):
    """The document or edit cannot be represented safely."""


class EditKind(str, Enum):
    GEOMETRY = "geometry"
    PAINT = "paint"
    TRANSFORM = "transform"
    STRUCTURE = "structure"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass(frozen=True)
class PathNode:
    """A segment endpoint and, for a cubic, its two control points.

    Pinning fixes the endpoint; cubic handles can still be adjusted. IDs are
    independent of array position and survive coordinate changes.
    """

    id: str
    command: str
    values: tuple[float, ...]
    pinned: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))
        arity = {"M": 2, "L": 2, "C": 6}.get(self.command)
        if arity is None or len(self.values) != arity:
            raise DocumentError("Expected an M/L/C node with complete coordinates")
        if not all(math.isfinite(value) for value in self.values):
            raise DocumentError("Path coordinates must be finite")

    @property
    def endpoint(self) -> tuple[float, float]:
        return self.values[-2], self.values[-1]


@dataclass(frozen=True)
class Subpath:
    id: str
    nodes: tuple[PathNode, ...]
    closed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        if not self.nodes or self.nodes[0].command != "M":
            raise DocumentError("A subpath must start with a moveto")
        if any(node.command == "M" for node in self.nodes[1:]):
            raise DocumentError("A subpath cannot contain another moveto")


@dataclass(frozen=True)
class Geometry:
    id: str
    subpaths: tuple[Subpath, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "subpaths", tuple(self.subpaths))

    def path_data(self) -> str:
        parts = []
        for subpath in self.subpaths:
            for node in subpath.nodes:
                # Round-trip doubles; no display/export precision loss in state.
                parts.append(node.command + " ".join(repr(v) for v in node.values))
            if subpath.closed:
                parts.append("Z")
        return " ".join(parts)

    def node(self, node_id: str) -> PathNode:
        for subpath in self.subpaths:
            for node in subpath.nodes:
                if node.id == node_id:
                    return node
        raise DocumentError(f"Unknown path node: {node_id}")

    def replace_node(self, updated: PathNode) -> Geometry:
        self.node(updated.id)
        return replace(
            self,
            subpaths=tuple(
                replace(
                    s,
                    nodes=tuple(updated if n.id == updated.id else n for n in s.nodes),
                )
                for s in self.subpaths
            ),
        )

    def detached(self) -> Geometry:
        return Geometry(
            new_id("geometry"),
            tuple(
                Subpath(
                    new_id("subpath"),
                    tuple(replace(n, id=new_id("node")) for n in s.nodes),
                    s.closed,
                )
                for s in self.subpaths
            ),
        )


@dataclass(frozen=True)
class Element:
    id: str
    tag: str
    attributes: tuple[tuple[str, str], ...] = ()
    children: tuple[Element, ...] = ()
    geometry_id: str | None = None
    locks: frozenset[str] = frozenset()
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "attributes", tuple(tuple(pair) for pair in self.attributes)
        )
        object.__setattr__(self, "children", tuple(self.children))
        object.__setattr__(self, "locks", frozenset(self.locks))
        if (
            not isinstance(self.name, str)
            or len(self.name) > 200
            or any(
                ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF or ord(c) in {0xFFFE, 0xFFFF}
                for c in self.name
            )
        ):
            raise DocumentError(
                "Object names must be at most 200 characters without control characters"
            )

    def get(self, name: str, default: str | None = None) -> str | None:
        return dict(self.attributes).get(name, default)


@dataclass(frozen=True)
class Selection:
    object_ids: frozenset[str] = frozenset()
    node_ids: frozenset[str] = frozenset()
    whole_document: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_ids", frozenset(self.object_ids))
        object.__setattr__(self, "node_ids", frozenset(self.node_ids))

    @classmethod
    def all(cls) -> Selection:
        return cls(whole_document=True)


@dataclass(frozen=True)
class Document:
    root: Element
    geometries: tuple[Geometry, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometries", tuple(self.geometries))

    def artboard(self) -> tuple[float, float, float, float]:
        """The root's viewBox as x, y, width, height, or its size at the origin.

        This is the drawing's own coordinate frame: the editor shows the
        artboard in it and stretches the reference image over it.
        """
        viewbox = self.root.get("viewBox")
        if viewbox:
            x, y, w, h = (float(v) for v in viewbox.replace(",", " ").split())
            return x, y, w, h
        return (
            0.0,
            0.0,
            float(self.root.get("width") or 1024),
            float(self.root.get("height") or 768),
        )

    def elements(self) -> tuple[Element, ...]:
        def walk(element):
            yield element
            for child in element.children:
                yield from walk(child)

        return tuple(walk(self.root))

    def element(self, element_id: str) -> Element:
        for element in self.elements():
            if element.id == element_id:
                return element
        raise DocumentError(f"Unknown object: {element_id}")

    def geometry(self, geometry_id: str) -> Geometry:
        geometry = self._index()[0].get(geometry_id)
        if geometry is None:
            raise DocumentError(f"Unknown geometry: {geometry_id}")
        return geometry

    def node_position(self, geometry_id: str, node_id: str):
        """(subpath, index) of a geometry's node, or None if it has none."""
        self.geometry(geometry_id)
        return self._index()[1].get((geometry_id, node_id))

    def _index(self):
        # A document never changes, so its lookups are built once, when first
        # needed: edges are looked up thousands of times per edit.
        index = self.__dict__.get("_lookup")
        if index is None:
            geometries = {g.id: g for g in self.geometries}
            nodes = {
                (g.id, node.id): (subpath, i)
                for g in self.geometries
                for subpath in g.subpaths
                for i, node in enumerate(subpath.nodes)
            }
            index = geometries, nodes
            object.__setattr__(self, "_lookup", index)
        return index

    def geometry_for(self, element_id: str) -> Geometry:
        element = self.element(element_id)
        if element.geometry_id is not None:
            return self.geometry(element.geometry_id)
        if element.tag == "use":
            reference = element.get("href", "") or ""
            return self.geometry_for(reference.removeprefix("#"))
        raise DocumentError(f"Object {element_id} does not reference a path")

    def ancestry(self, element_id: str) -> tuple[Element, ...]:
        def walk(element, ancestors):
            if element.id == element_id:
                return (*ancestors, element)
            for child in element.children:
                result = walk(child, (*ancestors, element))
                if result:
                    return result
            return ()

        result = walk(self.root, ())
        if not result:
            raise DocumentError(f"Unknown object: {element_id}")
        return result

    def replace_element(self, updated: Element) -> Document:
        self.element(updated.id)

        def walk(element):
            if element.id == updated.id:
                return updated
            children = tuple(walk(child) for child in element.children)
            return (
                replace(element, children=children)
                if children != element.children
                else element
            )

        return replace(self, root=walk(self.root))

    def replace_geometry(self, updated: Geometry) -> Document:
        self.geometry(updated.id)
        return replace(
            self,
            geometries=tuple(
                updated if g.id == updated.id else g for g in self.geometries
            ),
        )

    def selection_ids(self, selection: Selection) -> frozenset[str]:
        if selection.whole_document and selection.object_ids:
            raise DocumentError("Choose explicit objects or the whole document")
        selected = set()
        for element_id in selection.object_ids:
            selected.update(e.id for e in Document(self.element(element_id)).elements())
        ids = frozenset(selected)
        if selection.whole_document:
            ids = frozenset(e.id for e in self.elements())
        available_nodes = set()
        for element_id in ids:
            element = self.element(element_id)
            if element.tag in {"path", "use"}:
                try:
                    geometry = self.geometry_for(element_id)
                except DocumentError:
                    continue
                available_nodes.update(n.id for s in geometry.subpaths for n in s.nodes)
        if not selection.node_ids <= available_nodes:
            raise DocumentError("Selected nodes must belong to selected objects")
        return ids

    def dependents(self, object_ids: set[str]) -> frozenset[str]:
        """Objects changed through group inheritance, use, or clipping."""
        affected = set(object_ids)
        elements = self.elements()
        changed = True
        while changed:
            changed = False
            for element in elements:
                refs = references(element)
                if element.id in affected:
                    additions = {child.id for child in element.children}
                elif any(ref in affected for ref in refs):
                    additions = {element.id}
                else:
                    # A changed child changes a referenced group's silhouette.
                    additions = set()
                    for ref in refs:
                        target = self.element(ref)
                        if any(e.id in affected for e in Document(target).elements()):
                            additions.add(element.id)
                if not additions <= affected:
                    affected.update(additions)
                    changed = True
        return frozenset(affected)

    def geometry_users(self, geometry_id: str) -> frozenset[str]:
        return self.dependents(
            {e.id for e in self.elements() if e.geometry_id == geometry_id}
        )

    def validate(self) -> None:
        """Check the document, once: it never changes."""
        if self.__dict__.get("_valid"):
            return
        self._validate()
        object.__setattr__(self, "_valid", True)

    def _validate(self) -> None:
        from vectrify.document.svg import (
            CONTAINERS,
            GEOMETRY,
            PAINT,
            gradient_placement,
            validate_attributes,
        )

        elements = self.elements()
        object_ids = [e.id for e in elements]
        geometry_ids = [g.id for g in self.geometries]
        path_ids = [s.id for g in self.geometries for s in g.subpaths]
        node_ids = [n.id for g in self.geometries for s in g.subpaths for n in s.nodes]
        identifiers = [
            *object_ids,
            *geometry_ids,
            *path_ids,
            *node_ids,
        ]
        if any(not value for value in identifiers) or len(identifiers) != len(
            set(identifiers)
        ):
            raise DocumentError("Document identities must be unique and nonempty")
        if self.root.tag != "svg":
            raise DocumentError("Document root must be svg")
        edges = {}
        parents = {c.id: e.tag for e in elements for c in e.children}
        for element in elements:
            validate_attributes(element.tag, dict(element.attributes))
            if element.children and element.tag not in CONTAINERS | {"linearGradient"}:
                raise DocumentError(f"{element.tag} cannot contain children")
            problem = gradient_placement(
                element.tag, parents.get(element.id), [c.tag for c in element.children]
            )
            if problem:
                raise DocumentError(problem.capitalize())
            if element.tag == "svg" and element is not self.root:
                raise DocumentError("Nested SVG viewports are unsupported")
            if not element.locks <= set(EditKind) | PAINT | GEOMETRY[element.tag] | {
                "transform"
            }:
                raise DocumentError("Unknown property lock")
            keys = [key for key, _ in element.attributes]
            if len(keys) != len(set(keys)) or "id" in keys or "d" in keys:
                raise DocumentError(
                    "Identity and path geometry are not ordinary attributes"
                )
            if element.tag == "path":
                if element.geometry_id not in geometry_ids:
                    raise DocumentError("Path must reference existing geometry")
            elif element.geometry_id is not None:
                raise DocumentError("Only paths may own geometry")
            refs = references(element)
            if any(ref not in object_ids for ref in refs):
                raise DocumentError(f"Dangling reference on {element.id}")
            clip = element.get("clip-path")
            if clip and clip != "none" and self.element(clip[5:-1]).tag != "clipPath":
                raise DocumentError("Clipping must reference a clipPath")
            for kind in ("fill", "stroke"):
                server = paint_server(element.get(kind))
                if server and self.element(server).tag != "linearGradient":
                    raise DocumentError(
                        f"{kind.capitalize()} must reference a gradient"
                    )
            if element.tag == "use":
                href = element.get("href")
                if not href or self.element(href[1:]).tag not in {
                    "g",
                    "path",
                    "use",
                    "rect",
                    "circle",
                    "ellipse",
                    "line",
                }:
                    raise DocumentError("Use must reference a supported graphic")
            edges[element.id] = [*(c.id for c in element.children), *refs]
        complete = set()

        def visit(element_id, visiting):
            if element_id in visiting:
                raise DocumentError("Cyclic SVG reference")
            if element_id not in complete:
                for target in edges[element_id]:
                    visit(target, visiting | {element_id})
                complete.add(element_id)

        for element_id in object_ids:
            visit(element_id, set())


def references(element: Element) -> tuple[str, ...]:
    refs = []
    href = element.get("href")
    if href:
        if not href.startswith("#") or len(href) == 1:
            raise DocumentError("Only local SVG references are supported")
        refs.append(href[1:])
    clip = element.get("clip-path")
    if clip and clip != "none":
        if not clip.startswith("url(#") or not clip.endswith(")"):
            raise DocumentError("Only local clip references are supported")
        refs.append(clip[5:-1])
    for kind in ("fill", "stroke"):
        server = paint_server(element.get(kind))
        if server:
            refs.append(server)
    return tuple(refs)


def paint_server(value: str | None) -> str | None:
    """The ID a fill or stroke of ``url(#id)`` references, or None."""
    if value and value.startswith("url("):
        if not value.startswith("url(#") or not value.endswith(")"):
            raise DocumentError("Only local paint references are supported")
        return value[5:-1]
    return None
