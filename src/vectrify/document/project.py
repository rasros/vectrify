"""Versioned project persistence; plain SVG does not store editor node IDs."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from vectrify.document.model import (
    Document,
    DocumentError,
    Element,
    Geometry,
    PathNode,
    Selection,
    Subpath,
)
from vectrify.document.svg import export_svg


def save_project(document: Document, selection: Selection | None = None) -> str:
    selection = selection or Selection()
    export_svg(document)  # Check both graph invariants and the supported SVG subset.
    document.selection_ids(selection)

    def element_data(element: Element) -> dict:
        return {
            "id": element.id,
            "tag": element.tag,
            "name": element.name,
            "attributes": dict(element.attributes),
            "geometry_id": element.geometry_id,
            "locks": sorted(element.locks),
            "children": [element_data(child) for child in element.children],
        }

    return json.dumps(
        {
            "version": 3,
            "root": element_data(document.root),
            "geometries": [asdict(geometry) for geometry in document.geometries],
            "selection": {
                "object_ids": sorted(selection.object_ids),
                "node_ids": sorted(selection.node_ids),
                "whole_document": selection.whole_document,
            },
        },
        allow_nan=False,
        indent=2,
    )


def load_project(source: str) -> tuple[Document, Selection]:
    try:
        data = json.loads(source)
        # Version 2 also stored shared boundary links between edges. Editing
        # no longer links paths, so those are dropped: the contours they
        # joined already meet exactly.
        if data["version"] not in {1, 2, 3}:
            raise DocumentError("Unsupported project version")

        def element_data(item: Any) -> Element:
            return Element(
                id=item["id"],
                name=item.get("name", ""),
                tag=item["tag"],
                attributes=tuple(item["attributes"].items()),
                children=tuple(element_data(child) for child in item["children"]),
                geometry_id=item["geometry_id"],
                locks=frozenset(item["locks"]),
            )

        geometries = tuple(
            Geometry(
                item["id"],
                tuple(
                    Subpath(
                        subpath["id"],
                        tuple(
                            PathNode(
                                node["id"],
                                node["command"],
                                tuple(node["values"]),
                                node["pinned"],
                            )
                            for node in subpath["nodes"]
                        ),
                        subpath["closed"],
                    )
                    for subpath in item["subpaths"]
                ),
            )
            for item in data["geometries"]
        )
        document = Document(element_data(data["root"]), geometries)
        selection_data = data["selection"]
        selection = Selection(
            object_ids=frozenset(selection_data["object_ids"]),
            node_ids=frozenset(selection_data["node_ids"]),
            whole_document=selection_data["whole_document"],
        )
        export_svg(document)
        document.selection_ids(selection)
        return document, selection
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise DocumentError(f"Invalid project: {exc}") from exc
