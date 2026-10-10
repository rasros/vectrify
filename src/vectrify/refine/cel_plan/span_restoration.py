"""Construct a complete material floor before planning a positive outline stroke.

The complete original field receives the original base material. An adjoining
face can overlay it only inside native cells proved opaque in the actual parent.
Exact complete alpha and literal field locality are mandatory. This intermediate
has no outline stroke and cannot be accepted; source/body/gap, physical-family
ownership and final native verification remain separate. Search does not enable
this constructor yet.
"""

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from xml.etree import ElementTree as ET

import numpy as np
import pathops

from vectrify.document import Document, Editor, Element, Geometry, Selection, export_svg
from vectrify.document.hit_test import multiply
from vectrify.document.join import (
    curve_path,
    path_geometry,
    path_style,
    transformed_geometry,
)
from vectrify.document.model import paint_server
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.band_fit import opaque_core
from vectrify.refine.cel_plan.filled_bands import MAX_NATIVE_PIXELS, MAX_NODES, _check
from vectrify.refine.cel_plan.ink_replace import identified
from vectrify.refine.cel_plan.local import _native_raster
from vectrify.refine.cel_plan.paint_continuation import _paint, _supported
from vectrify.refine.cel_plan.source_bands import MAX_EXTENT
from vectrify.refine.cel_plan.source_faces import SourceFace, _edges
from vectrify.refine.cel_plan.source_family import FamilyPart, _commands
from vectrify.refine.cel_plan.source_slices import MAX_COMPOUND_NODES
from vectrify.refine.cel_plan.source_spans import SourceSpan


@dataclass(frozen=True)
class SpanRestoration:
    document: Document
    parts: tuple[FamilyPart, ...]
    field: Geometry
    face: Geometry | None
    proof: dict


def _frame(document, oid):
    frame = root_matrix(document, oid)
    determinant = frame[0] * frame[3] - frame[1] * frame[2]
    return (
        np.isfinite(frame).all() and isfinite(determinant) and abs(determinant) >= 1e-12
    )


def _material(document, owner, material):
    return (
        _supported(document, material)
        and _frame(document, material)
        and _paint(document, path_style(document, document.element(material)))
        and document.ancestry(material)[-2].id == document.ancestry(owner)[-2].id
        and all(s.closed for s in document.geometry_for(material).subpaths)
    )


def construct(
    document,
    oid,
    span: SourceSpan,
    base_material,
    size,
    work,
    *,
    face: SourceFace | None = None,
) -> SpanRestoration | None:
    """Keep all old material contours exact and restore only the complete field."""
    _check(work)
    if len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
        raise ValueError("Span restoration requires a positive native viewport")
    original = document.geometry_for(oid)
    style = path_style(document, document.element(oid))
    if (
        np.prod(size) > MAX_NATIVE_PIXELS
        or not _supported(document, oid)
        or not _frame(document, oid)
        or not _paint(document, style)
        or paint_server(style["fill"]) is not None
        or style["fill-rule"] != "nonzero"
        or not _material(document, oid, base_material)
        or not span.field.subpaths
        or any(
            not s.closed
            for s in (*original.subpaths, *span.field.subpaths, *span.retained.subpaths)
        )
        or sum(len(s.nodes) for s in span.field.subpaths) > MAX_NODES
        or sum(len(s.nodes) for s in span.retained.subpaths) > MAX_COMPOUND_NODES
        or _commands(span.retained.subpaths[: len(original.subpaths)])
        != _commands(original.subpaths)
    ):
        return None
    frame = root_matrix(document, oid)
    full, old, retained = (
        curve_path(span.field),
        curve_path(original),
        curve_path(span.retained),
    )
    bounds = np.asarray(curve_path(transformed_geometry(span.field, frame)).bounds)
    if (
        not full.area
        or not np.isfinite(bounds).all()
        or np.max(np.abs(bounds)) > 1_000_000
        or (bounds[2:] - bounds[:2]).max() > MAX_EXTENT
        or pathops.op(full, old, pathops.PathOp.DIFFERENCE).area
        or pathops.op(retained, full, pathops.PathOp.INTERSECTION).area
        or pathops.op(
            retained,
            pathops.op(old, full, pathops.PathOp.DIFFERENCE),
            pathops.PathOp.XOR,
        ).area
    ):
        return None
    shade = None
    fields = [(base_material, span.field)]
    if face is not None:
        if (
            face.material in {oid, base_material}
            or not _material(document, oid, face.material)
            or any(not s.closed for s in face.field.subpaths)
            or sum(len(s.nodes) for s in face.field.subpaths) > MAX_NODES
            or pathops.op(curve_path(face.field), full, pathops.PathOp.DIFFERENCE).area
            or not (
                _edges(transformed_geometry(face.field, frame))
                & _edges(
                    transformed_geometry(
                        document.geometry_for(face.material),
                        root_matrix(document, face.material),
                    )
                )
            )
        ):
            return None
        core = opaque_core(document, oid, span.field, size, work, footprint_cells=True)
        if core is None:
            return None
        shade = path_geometry(
            pathops.op(
                curve_path(face.field), curve_path(core), pathops.PathOp.INTERSECTION
            )
        )
        if not shade.subpaths or sum(len(s.nodes) for s in shade.subpaths) > MAX_NODES:
            return None
        fields.append((face.material, shade))
    _check(work)
    key = sha256(
        repr((oid, base_material, span.field.path_data(), face)).encode()
    ).hexdigest()[:12]
    residual_id = f"{oid}-span-residual-{key}"
    parts = [FamilyPart(oid, "stroke"), FamilyPart(residual_id, "residual")]
    for material, _ in fields:
        parts.append(
            FamilyPart(
                f"{oid}-span-material-{len(parts) - 2}-{key}", "restoration", material
            )
        )
    if {p.id for p in parts[1:]} & {e.id for e in document.elements()}:
        return None
    parent = document.ancestry(oid)[-2]
    index = parent.children.index(document.element(oid))
    restore_index = (
        max(
            index,
            *(
                parent.children.index(document.element(material))
                for material, _ in fields
            ),
        )
        + 2
    )
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Construct complete source-span material floor") as tx:
        tx.set_attributes(oid, {"fill": "none", "stroke": "none"})
        residual = identified(span.retained, residual_id)
        tx.insert_object(
            parent.id,
            Element(
                residual_id,
                "path",
                document.element(oid).attributes,
                geometry_id=residual.id,
            ),
            index=index,
            geometries=(residual,),
        )
        for offset, ((material, field), part) in enumerate(
            zip(fields, parts[2:], strict=True)
        ):
            shape = identified(
                transformed_geometry(
                    field,
                    multiply(inverse_matrix(root_matrix(document, material)), frame),
                ),
                part.id,
            )
            tx.insert_object(
                parent.id,
                Element(
                    part.id,
                    "path",
                    document.element(material).attributes,
                    geometry_id=shape.id,
                ),
                index=restore_index + offset,
                geometries=(shape,),
            )
        tx.reorder_object(oid, restore_index + len(fields) - 1)
    candidate = editor.snapshot.document
    before = _native_raster(ET.fromstring(export_svg(document)), size).root
    after = _native_raster(ET.fromstring(export_svg(candidate)), size).root
    _check(work)
    mask_root = ET.Element("svg", {"width": str(size[0]), "height": str(size[1])})
    ET.SubElement(
        mask_root,
        "path",
        {
            "d": span.field.path_data(),
            "transform": "matrix(" + " ".join(map(str, frame)) + ")",
            "fill": "white",
        },
    )
    mask = _native_raster(mask_root, size).root[..., 3] > 0
    if not np.array_equal(before[..., 3], after[..., 3]) or not np.array_equal(
        before[~mask], after[~mask]
    ):
        return None
    _check(work)
    return SpanRestoration(
        candidate,
        tuple(parts),
        span.field,
        shade,
        {
            "scope": "complete-material-floor-without-outline-stroke",
            "native_alpha_exact": True,
            "outside_removed_field_rgba_exact": True,
            "field_pixels": int(mask.sum()),
            "physical_parts": len(parts),
            "face_nodes": sum(len(s.nodes) for s in shade.subpaths)
            if shade is not None
            else 0,
            "positive_body_proved": False,
            "accepted": False,
        },
    )
