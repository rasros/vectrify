"""Explicit SVG clipping for arbitrary drawn targets, independent of paint."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from vectrify.document import DocumentError, Selection
from vectrify.document.clipping import INSET, SOURCE, inset_geometry
from vectrify.document.hit_test import multiply
from vectrify.document.join import path_style
from vectrify.document.model import Element, new_id
from vectrify.document.regions import object_matrix
from vectrify.document.topology import inverse_matrix


def apply(
    agent,
    targets: list[str],
    sources: list[str],
    action: str,
    linked: bool,
    inset: float,
):
    editor = agent.session.editor
    document = editor.snapshot.document
    for oid in targets:
        if any(a.id in targets for a in document.ancestry(oid)[:-1]):
            raise DocumentError(
                "Choose targets without selecting both parent and child"
            )
    boundaries = []
    for oid in sources:
        element = document.element(oid)
        if element.tag != "path" or element.geometry_id is None:
            raise DocumentError("Boundaries must be paths; convert shapes first")
        geometry = document.geometry_for(oid)
        if not geometry.subpaths or any(not s.closed for s in geometry.subpaths):
            raise DocumentError("Boundary paths must have closed contours")
        if inset:
            inset_geometry(geometry, inset, path_style(document, element)["fill-rule"])
        boundaries.append((element, geometry, object_matrix(document, oid)))
    result: list[dict[str, Any]] = []
    # Scope is explicit; the transaction's structural command checks target
    # dependents and locks. Sources are read, never moved or repainted.
    tx = editor.transaction("Clipping", selection=Selection(frozenset(targets)))
    with tx:
        for oid in targets:
            if action == "remove":
                tx.set_clip(oid, None)
                result.append({"target": oid, "clip": None})
                continue
            frame = inverse_matrix(object_matrix(document, oid))
            children = []
            geometries = []
            for element, geometry, matrix in boundaries:
                if not linked:
                    geometry = replace(
                        geometry,
                        id=new_id("geometry"),
                        subpaths=tuple(
                            replace(
                                subpath,
                                id=new_id("subpath"),
                                nodes=tuple(
                                    replace(node, id=new_id("node"))
                                    for node in subpath.nodes
                                ),
                            )
                            for subpath in geometry.subpaths
                        ),
                    )
                    geometries.append(geometry)
                attrs = {
                    "fill": "#ffffff",
                    "stroke": "none",
                    "clip-rule": path_style(document, element)["fill-rule"],
                    "fill-rule": path_style(document, element)["fill-rule"],
                    "transform": "matrix("
                    + " ".join(str(v) for v in multiply(frame, matrix))
                    + ")",
                }
                if linked:
                    attrs[SOURCE] = element.id
                if inset:
                    attrs[INSET] = str(inset)
                children.append(
                    Element(
                        new_id("boundary"),
                        "path",
                        attributes=tuple(attrs.items()),
                        geometry_id=geometry.id,
                        name="Clipping shape",
                    )
                )
            clip = Element(
                new_id("clip"),
                "clipPath",
                attributes=(("clipPathUnits", "userSpaceOnUse"),),
                children=tuple(children),
                name="Clipping boundary",
            )
            tx.set_clip(oid, clip, tuple(geometries))
            result.append(
                {
                    "target": oid,
                    "clip": clip.id,
                    "sources": sources,
                    "linked": linked,
                    "inset": inset,
                }
            )
    editor.select(Selection(frozenset(targets)))
    return result
