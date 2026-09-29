"""Reviewable simplification proposals; applying commits the captured transaction."""

from __future__ import annotations

import base64
import math
import xml.etree.ElementTree as ET
from uuid import uuid4

import cairosvg

from vectrify.document import DocumentError, StaleRevisionError, export_svg
from vectrify.document.simplify import SimplifyOptions


class SimplifyPreview:
    def __init__(self, session, payload: dict):
        session.check_revision(payload)
        self.epoch = session.epoch
        self.id = uuid4().hex
        self.transaction = session.editor.transaction("Smooth / simplify shapes")
        before = session.editor.snapshot.document
        self.transaction.simplify_shapes(SimplifyOptions(**payload.get("options", {})))
        after = self.transaction.preview
        geometries = {g.id for g in before.geometries if g != after.geometry(g.id)}
        selected = before.selection_ids(session.editor.snapshot.selection)
        all_geometries = {
            before.geometry_for(oid).id
            for oid in selected
            if before.element(oid).tag in {"path", "use"}
        }

        def stats(document):
            assets = [document.geometry(gid) for gid in all_geometries]
            return {
                "nodes": sum(len(s.nodes) for g in assets for s in g.subpaths),
                "coordinates": sum(
                    len(n.values) for g in assets for s in g.subpaths for n in s.nodes
                ),
                "bytes": sum(len(g.path_data().encode()) for g in assets),
            }

        self.result: dict = {
            "id": self.id,
            "changed": bool(geometries),
            "before": stats(before),
            "after": stats(after),
        }
        self.result["previews"] = render_previews(
            before, after, payload.get("bounds", session.state(svg=False)["bounds"])
        )

    def apply(self, session):
        if session.epoch != self.epoch:
            raise StaleRevisionError(
                "The drawing changed. Preview simplification again."
            )
        self.transaction.commit()
        return session.state()


def render_previews(before, after, bounds, *, highlight=False):
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise DocumentError("Expected preview bounds")
    x, y, width, height = (float(v) for v in bounds)
    if (
        not all(math.isfinite(v) for v in (x, y, width, height))
        or width <= 0
        or height <= 0
    ):
        raise DocumentError("Preview bounds must be finite with positive dimensions")
    scale = 900 / max(width, height)

    def render(document):
        root = ET.fromstring(export_svg(document))
        root.set("viewBox", f"{x} {y} {width} {height}")
        root.set("width", str(max(1, round(width * scale))))
        root.set("height", str(max(1, round(height * scale))))
        root.set("preserveAspectRatio", "none")
        if highlight and document is after:
            from vectrify.document.hit_test import IDENTITY, multiply, transform
            from vectrify.document.topology import edge, inverse_matrix

            for boundary in document.boundaries:
                ref = boundary.members[0]
                owner = next(iter(document.geometry_users(ref.geometry_id)))
                matrix = IDENTITY
                for ancestor in document.ancestry(owner):
                    matrix = multiply(matrix, transform(ancestor.get("transform")))
                points = edge(document, ref).points
                matrix = multiply(matrix, inverse_matrix(ref.matrix))
                data = f"M{points[0][0]} {points[0][1]} "
                data += "C" if len(points) == 4 else "L"
                data += " ".join(str(v) for point in points[1:] for v in point)
                ET.SubElement(
                    root,
                    "{http://www.w3.org/2000/svg}path",
                    {
                        "d": data,
                        "transform": "matrix(" + " ".join(map(str, matrix)) + ")",
                        "fill": "none",
                        "stroke": "#00c8ff",
                        "stroke-width": str(2 / scale),
                    },
                )
        png = cairosvg.svg2png(bytestring=ET.tostring(root), background_color="white")
        assert png is not None
        return "data:image/png;base64," + base64.b64encode(png).decode()

    return {"before": render(before), "after": render(after)}
