"""A stroke-only geometry consumer placed in a named outline layer."""

from __future__ import annotations

from vectrify.document import DocumentError, Selection
from vectrify.document.hit_test import multiply
from vectrify.document.model import Element, new_id
from vectrify.document.regions import object_matrix
from vectrify.document.topology import inverse_matrix

SOURCE = "data-vectrify-outline-source"


def linked_targets(document, ids):
    targets = set(ids)
    for oid in ids:
        element = document.element(oid)
        if element.geometry_id is None:
            continue
        users = document.geometry_users(element.geometry_id)
        if any(document.element(u).get(SOURCE) for u in users):
            targets.update(users)
    return sorted(targets)


def create(agent, source: str, colour: str, width: float, layer: str) -> dict:
    editor = agent.session.editor
    document = editor.snapshot.document
    element = document.element(source)
    if element.tag != "path" or element.geometry_id is None:
        raise DocumentError("Linked outlines require a path; convert the shape first")
    if element.get(SOURCE):
        raise DocumentError("Choose the filled source path, not its outline")
    if any(a.get("clip-path", "none") != "none" for a in document.ancestry(source)):
        raise DocumentError("Cannot move a clipped outline into another layer")
    root = document.root
    parent = next((c for c in root.children if c.tag == "g" and c.name == layer), None)
    if parent is None:
        parent = Element(new_id("outlines"), "g", name=layer)
    if parent.get("opacity", "1") != "1" or parent.get("clip-path", "none") != "none":
        raise DocumentError(
            "The outline layer has opacity or clipping; choose another layer"
        )
    frame = (
        object_matrix(document, parent.id)
        if any(c.id == parent.id for c in root.children)
        else object_matrix(document, root.id)
    )
    matrix = multiply(
        inverse_matrix(frame),
        object_matrix(document, source),
    )
    transform = "matrix(" + " ".join(str(v) for v in matrix) + ")"
    existing = [e for e in document.elements() if e.get(SOURCE) == source]
    tx = editor.transaction("Linked outline", selection=Selection(whole_document=True))
    if not any(c.id == parent.id for c in root.children):
        tx.insert_object(root.id, parent)
    # The layer is deliberately above every root-level fill and shading group.
    tx.reorder_object(parent.id, len(tx.preview.root.children) - 1)
    attrs = {
        "fill": "none",
        "stroke": colour,
        "stroke-width": str(width),
        "stroke-linejoin": "miter",
        "stroke-linecap": "round",
        "opacity": "1",
        "stroke-opacity": "1",
        "transform": transform,
    }
    if existing:
        if len(existing) > 1:
            raise DocumentError("Multiple linked outlines exist for this source")
        outline = existing[0]
        tx.set_attributes(outline.id, attrs)
        if tx.preview.ancestry(outline.id)[-2].id != parent.id:
            tx.move_objects(frozenset({outline.id}), parent.id, len(parent.children))
        if outline.geometry_id != element.geometry_id:
            tx.share_geometry(outline.id, source)
    else:
        outline = Element(
            new_id("outline"),
            "path",
            attributes=tuple({**attrs, SOURCE: source}.items()),
            geometry_id=element.geometry_id,
            name=f"{element.name or source} outline",
        )
        tx.insert_object(parent.id, outline)
    tx.commit()
    editor.select(Selection(frozenset({outline.id})))
    return {
        "outline": outline.id,
        "source": source,
        "geometry": element.geometry_id,
        "layer": parent.id,
        "colour": colour,
        "width": width,
        "width_units": "source local units",
        "geometry_users": sorted(tx.preview.geometry_users(element.geometry_id)),
    }
