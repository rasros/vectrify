"""Explicit physical parts of one unchanged original source owner.

A raw region can contain both a shadow and its attached ink. Rendering those as
different paths does not require inventing another source-region subdivision.
This declaration preserves that source family, not an assertion that its stroke
alone paints the whole region. Native/body/source acceptance remains separate.
"""

import hashlib
from dataclasses import dataclass, replace
from typing import cast
from xml.etree import ElementTree as ET

import numpy as np
import pathops
from cairosvg.colors import color

from vectrify.document import Document, export_svg
from vectrify.document.hit_test import Matrix, multiply
from vectrify.document.join import curve_path, path_style, transformed_geometry
from vectrify.document.model import paint_server
from vectrify.document.redraw import root_matrix
from vectrify.document.svg import parse_path
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.filled_bands import MAX_NATIVE_PIXELS, MAX_NODES, _check
from vectrify.refine.cel_plan.local import _native_raster
from vectrify.refine.cel_plan.paint_continuation import _paint, _supported
from vectrify.refine.cel_plan.source_bands import MAX_EXTENT
from vectrify.refine.cel_plan.stroke_inventory import _style

MAX_PARTS = 8
MAX_FAMILIES = 32
MAX_PATH_BYTES = 256 * 1024


@dataclass(frozen=True)
class FamilyPart:
    id: str
    role: str
    material: str | None = None


def revision(document, oid):
    """Canonical path/ancestry/paint revision, stable across native reload."""
    records: list[object] = [
        (
            e.id,
            e.tag,
            e.attributes,
            e.geometry_id,
            tuple(sorted(e.locks)),
            e.paint_owner,
        )
        for e in document.ancestry(oid)
    ]
    records.append(
        tuple(
            (
                s.id,
                s.closed,
                tuple(
                    (
                        n.id,
                        n.command,
                        tuple(float(v) for v in n.values),
                        n.pinned,
                        n.handles_aligned,
                    )
                    for n in s.nodes
                ),
            )
            for s in document.geometry_for(oid).subpaths
        )
    )
    style = path_style(document, document.element(oid))
    for attr in ("fill", "stroke"):
        server = paint_server(style[attr])
        if server:
            records.append(document.element(server))
    return hashlib.sha256(repr(records).encode()).hexdigest()


def _data(geometry):
    return replace(
        geometry,
        subpaths=tuple(
            replace(
                s,
                nodes=tuple(
                    replace(n, values=tuple(float(v) for v in n.values))
                    for n in s.nodes
                ),
            )
            for s in geometry.subpaths
        ),
    ).path_data()


def _paint_key(document, oid):
    style = path_style(document, document.element(oid))
    server = paint_server(style["fill"])
    if server:

        def resource(element):
            return (
                element.tag,
                element.attributes,
                tuple(resource(child) for child in element.children),
            )

        fill = resource(document.element(server))
    else:
        fill = color(style["fill"])
    return fill, (
        float(style["opacity"]),
        float(style["fill-opacity"]),
        style["fill-rule"],
    )


def _commands(subpaths):
    return tuple(
        (s.closed, tuple((n.command, n.values) for n in s.nodes)) for s in subpaths
    )


@dataclass(frozen=True)
class SourceFamily:
    owner: str
    parts: tuple[FamilyPart, ...]
    original: str
    removed: str
    frame: Matrix
    attributes: tuple[tuple[str, str], ...]
    revisions: tuple[tuple[str, str], ...]

    def __post_init__(self):
        roles = [p.role for p in self.parts]
        ids = [p.id for p in self.parts]
        if (
            not isinstance(self.owner, str)
            or not self.owner
            or any(not isinstance(p.id, str) or not p.id for p in self.parts)
            or not 3 <= len(ids) <= MAX_PARTS
            or len(ids) != len(set(ids))
            or roles.count("stroke") != 1
            or roles.count("residual") != 1
            or any(r not in {"stroke", "residual", "restoration"} for r in roles)
            or next(p.id for p in self.parts if p.role == "stroke") != self.owner
            or any((p.role == "restoration") != bool(p.material) for p in self.parts)
            or any(p.material in ids for p in self.parts if p.material)
            or len(self.frame) != 6
            or not np.isfinite(self.frame).all()
            or len(self.original.encode()) > MAX_PATH_BYTES
            or len(self.removed.encode()) > MAX_PATH_BYTES
            or self.attributes != tuple(sorted(self.attributes))
            or len(dict(self.attributes)) != len(self.attributes)
            or dict(self.attributes).get("fill-rule", "nonzero") != "nonzero"
            or dict(self.attributes).get("fill", "none") == "none"
        ):
            raise ValueError("Invalid physical source family declaration")
        expected = set(ids) | {p.material for p in self.parts if p.material}
        if (
            {oid for oid, _ in self.revisions} != expected
            or len(self.revisions) != len(expected)
            or any(
                len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                for _, digest in self.revisions
            )
        ):
            raise ValueError("Source family needs every part and material revision")

    @property
    def paths(self):
        return tuple(p.id for p in self.parts)

    @property
    def dependencies(self):
        return frozenset(oid for oid, _ in self.revisions)

    def validate(self, document: Document):
        try:
            if any(revision(document, oid) != digest for oid, digest in self.revisions):
                raise ValueError("Source family physical parts or materials changed")
            parent = document.ancestry(self.owner)[-2].id
            if any(
                document.element(p.id).tag != "path"
                or document.ancestry(p.id)[-2].id != parent
                for p in self.parts
            ):
                raise ValueError("Source family parts require one declared component")
            order = [c.id for c in document.element(parent).children]
            if any(
                order.index(p.id) >= order.index(self.owner)
                for p in self.parts
                if p.id != self.owner
            ):
                raise ValueError(
                    "Source family stroke must follow its residual and restorations"
                )
            self._geometry(document)
        except (KeyError, IndexError) as exc:
            raise ValueError("Source family references missing physical paths") from exc

    def validate_before(self, before: Document, after: Document):
        """Bind saved source snapshots to the sealed actual parent document."""
        if (
            not _supported(before, self.owner)
            or not _paint(before, path_style(before, before.element(self.owner)))
            or _data(before.geometry_for(self.owner)) != self.original
            or tuple(root_matrix(before, self.owner)) != self.frame
            or tuple(sorted(before.element(self.owner).attributes)) != self.attributes
        ):
            raise ValueError("Source family does not describe its original owner")
        originals = {e.id for e in before.elements() if e.tag == "path"}
        if any(p.id in originals for p in self.parts if p.id != self.owner):
            raise ValueError("Source family companions must be newly declared paths")
        if any(
            revision(before, oid) != revision(after, oid)
            for oid in originals - {self.owner}
        ):
            raise ValueError("Source family changed an original material or sibling")

    def _geometry(self, document):
        original, removed = parse_path(self.original), parse_path(self.removed)
        if (
            not original.subpaths
            or not removed.subpaths
            or not curve_path(removed).area
            or any(not s.closed for s in (*original.subpaths, *removed.subpaths))
            or sum(len(s.nodes) for s in original.subpaths) > 1024
            or sum(len(s.nodes) for s in removed.subpaths) > MAX_NODES
            or pathops.op(
                curve_path(removed), curve_path(original), pathops.PathOp.DIFFERENCE
            ).area
        ):
            raise ValueError("Source family needs a complete contained removal field")
        bounds = np.array(curve_path(transformed_geometry(removed, self.frame)).bounds)
        if (
            not np.isfinite(bounds).all()
            or (bounds[2:] - bounds[:2]).max() > MAX_EXTENT
        ):
            raise ValueError("Source family removal exceeds native bounds")
        stroke = document.geometry_for(self.owner)
        if (
            not _supported(document, self.owner)
            or _style(document, document.element(self.owner)) is None
            or len(stroke.subpaths) != 1
            or stroke.subpaths[0].closed
            or tuple(root_matrix(document, self.owner)) != self.frame
        ):
            raise ValueError("Source family needs a genuine open editable stroke")
        residual = next(p.id for p in self.parts if p.role == "residual")
        retained = document.geometry_for(residual)
        if (
            not _supported(document, residual)
            or tuple(root_matrix(document, residual)) != self.frame
            or tuple(sorted(document.element(residual).attributes)) != self.attributes
            or _commands(retained.subpaths[: len(original.subpaths)])
            != _commands(original.subpaths)
            or pathops.op(
                curve_path(retained), curve_path(removed), pathops.PathOp.INTERSECTION
            ).area
            or pathops.op(
                curve_path(retained),
                pathops.op(
                    curve_path(original), curve_path(removed), pathops.PathOp.DIFFERENCE
                ),
                pathops.PathOp.XOR,
            ).area
        ):
            raise ValueError(
                "Source family retained shadow does not exactly remove old ink"
            )
        union = pathops.Path()
        for part in self.parts:
            if part.role != "restoration":
                continue
            material = part.material
            assert material is not None
            if (
                not _supported(document, part.id)
                or not _supported(document, material)
                or not _paint(document, path_style(document, document.element(part.id)))
                or _paint_key(document, part.id) != _paint_key(document, material)
                or tuple(root_matrix(document, part.id))
                != tuple(root_matrix(document, material))
                or document.ancestry(material)[-2].id
                != document.ancestry(self.owner)[-2].id
                or _paint_key(document, part.id) == _paint_key(document, residual)
            ):
                raise ValueError(
                    "Source family restoration needs original material paint/frame"
                )
            geometry = transformed_geometry(
                document.geometry_for(part.id),
                multiply(inverse_matrix(self.frame), root_matrix(document, part.id)),
            )
            shape = curve_path(geometry)
            if (
                any(not s.closed for s in geometry.subpaths)
                or not shape.area
                or pathops.op(
                    shape, curve_path(removed), pathops.PathOp.DIFFERENCE
                ).area
            ):
                raise ValueError("Source family restoration escapes its complete field")
            union = pathops.op(union, shape, pathops.PathOp.UNION)
        if pathops.op(union, curve_path(removed), pathops.PathOp.XOR).area:
            raise ValueError(
                "Source family restoration does not cover the complete field"
            )

    @classmethod
    def bind(cls, before, after, owner, removed, parts, size, work):
        """Construct and verify an atomic physical interpretation, not source fit."""
        _check(work)
        if (
            len(size) != 2
            or any(type(n) is not int or n <= 0 for n in size)
            or np.prod(size) > MAX_NATIVE_PIXELS
            or not _supported(before, owner)
            or not _paint(before, path_style(before, before.element(owner)))
        ):
            raise ValueError("Source family exceeds supported construction bounds")
        ids = {p.id for p in parts} | {p.material for p in parts if p.material}
        family = cls(
            owner,
            tuple(parts),
            _data(before.geometry_for(owner)),
            _data(removed),
            cast(Matrix, tuple(root_matrix(before, owner))),
            tuple(sorted(before.element(owner).attributes)),
            tuple(sorted((oid, revision(after, oid)) for oid in ids)),
        )
        family.validate_before(before, after)
        for part in parts:
            if part.material and revision(before, part.material) != revision(
                after, part.material
            ):
                raise ValueError("Source family changed original material dependencies")
        family.validate(after)
        _check(work)
        a = _native_raster(ET.fromstring(export_svg(before)), size).root
        b = _native_raster(ET.fromstring(export_svg(after)), size).root
        if not np.array_equal(a[..., 3], b[..., 3]):
            raise ValueError("Source family complete native alpha changed")
        _check(work)
        return family

    def metadata(self):
        return {
            "owner": self.owner,
            "parts": [
                {
                    "id": p.id,
                    "role": p.role,
                    **({"material": p.material} if p.material else {}),
                }
                for p in self.parts
            ],
            "original": self.original,
            "removed": self.removed,
            "frame": list(self.frame),
            "attributes": [list(a) for a in self.attributes],
            "revisions": [list(r) for r in self.revisions],
        }

    @classmethod
    def from_metadata(cls, value):
        try:
            return cls(
                value["owner"],
                tuple(
                    FamilyPart(p["id"], p["role"], p.get("material"))
                    for p in value["parts"]
                ),
                value["original"],
                value["removed"],
                tuple(value["frame"]),
                tuple(tuple(a) for a in value["attributes"]),
                tuple(tuple(r) for r in value["revisions"]),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Invalid physical source family metadata") from exc
