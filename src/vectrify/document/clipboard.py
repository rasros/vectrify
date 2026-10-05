"""Self-contained object snapshots for the editor's local clipboard."""

from dataclasses import replace

from vectrify.document.editor import INITIAL_PAINT
from vectrify.document.model import (
    Document,
    DocumentError,
    Element,
    new_id,
    references,
)


def copy_objects(document: Document, object_ids: frozenset[str]) -> Document:
    """Keep selected subtrees, their drawing context and referenced assets."""
    if not object_ids:
        raise DocumentError("Select objects to copy")
    for oid in object_ids:
        if (
            any(
                a.tag in {"svg", "defs", "clipPath", "linearGradient", "stop"}
                for a in document.ancestry(oid)[1:]
            )
            or oid == document.root.id
        ):
            raise DocumentError("Copy drawing objects, not definitions")

    def picked(element: Element) -> tuple[Element, ...]:
        if element.id in object_ids:
            return (element,)
        children = tuple(c for child in element.children for c in picked(child))
        if not children:
            return ()
        # Plain ancestry groups add no appearance or editing constraints.
        if element.tag == "g" and not element.attributes and not element.locks:
            return children
        return (replace(element, children=children),)

    paint = {**INITIAL_PAINT, **dict(document.root.attributes)}
    paint = {k: v for k, v in paint.items() if k in INITIAL_PAINT}
    branches = tuple(
        replace(e, attributes=tuple({**paint, **dict(e.attributes)}.items()))
        for child in document.root.children
        for e in picked(child)
    )
    # References need complete original subtrees. An ancestry group pruned to
    # selected children cannot stand in for that group's referenced silhouette.
    kept = {
        e.id
        for branch in branches
        for e in Document(branch).elements()
        if e.children == document.element(e.id).children
    }
    resources: set[str] = set()
    definitions: list[Element] = []
    pending = [
        document.root,
        *[e for branch in branches for e in Document(branch).elements()],
    ]
    while pending:
        element = pending.pop()
        for ref in references(element):
            if ref in kept or ref in resources:
                continue
            definition = document.element(ref)
            # A referenced group takes its complete subtree with it.
            members = Document(definition).elements()
            resources.update(e.id for e in members)
            definitions.append(definition)
            pending.extend(members)
    # A reference to both a group and its child needs only the group definition.
    definition_ids = {d.id for d in definitions}
    definitions = [
        e
        for e in definitions
        if not any(a.id in definition_ids for a in document.ancestry(e.id)[:-1])
    ]
    # Give resources their own namespace, even if a referenced group contains
    # a selected object that also appears in the drawing fragment.
    resource_ids = {oid: new_id("object") for oid in resources}
    branch_refs = {oid: value for oid, value in resource_ids.items() if oid not in kept}

    def retarget(element: Element, *, resource: bool = False) -> Element:
        return replace(
            element,
            id=resource_ids[element.id] if resource else element.id,
            attributes=_attributes(element, resource_ids if resource else branch_refs),
            children=tuple(retarget(c, resource=resource) for c in element.children),
            paint_owner=None,
        )

    definitions = [retarget(e, resource=True) for e in definitions]
    branches = tuple(retarget(e) for e in branches)
    defs = (
        (Element(new_id("object"), "defs", children=tuple(definitions)),)
        if definitions
        else ()
    )
    root = replace(
        document.root,
        attributes=_attributes(document.root, branch_refs),
        children=(*defs, *branches),
        paint_owner=None,
    )
    gids = {e.geometry_id for e in Document(root).elements()}
    result = Document(root, tuple(g for g in document.geometries if g.id in gids))
    result.validate()
    return result


def _attributes(element: Element, ids: dict[str, str]) -> tuple[tuple[str, str], ...]:
    attrs = dict(element.attributes)
    for key, value in attrs.items():
        if key == "href":
            attrs[key] = f"#{ids.get(value[1:], value[1:])}"
        elif key in {"fill", "stroke", "clip-path"} and value.startswith("url(#"):
            attrs[key] = f"url(#{ids.get(value[5:-1], value[5:-1])})"
    return tuple(attrs.items())


def paste_objects(clipboard: Document) -> Document:
    """Allocate fresh object, geometry and point IDs, retargeting references."""
    ids = {e.id: new_id("object") for e in clipboard.elements()}
    geometries = tuple(g.detached() for g in clipboard.geometries)
    gids = {
        old.id: new.id
        for old, new in zip(clipboard.geometries, geometries, strict=True)
    }

    def clone(element: Element) -> Element:
        return replace(
            element,
            id=ids[element.id],
            attributes=_attributes(element, ids),
            children=tuple(clone(c) for c in element.children),
            geometry_id=gids[element.geometry_id] if element.geometry_id else None,
        )

    return Document(clone(clipboard.root), geometries)
