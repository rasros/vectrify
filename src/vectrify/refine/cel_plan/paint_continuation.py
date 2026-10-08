"""Continue an existing material inside the removed outline's exact footprint.

Original material contours, paint servers and frames remain exact. The added
contours exclude existing material coverage, avoiding double opacity and boolean
rewrites of unrelated curves. This constructs a competitor, not a source owner
or acceptance proof: complete painted, alpha, gap, locality and ownership checks
are required before publication, including the changed stroke paint order.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pathops
from cairosvg.colors import color

from vectrify.document import Editor, Selection
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
from vectrify.refine.cel_plan.ink_replace import identified
from vectrify.refine.cel_plan.model import StageInterruptedError

MAX_NODES = 1024
MAX_PATCH_NODES = 256
MAX_PATCH_CONTOURS = 16
MAX_AREA = 4096
MAX_EXTENT = 256


def _check(work):
    if work.interrupted:
        raise StageInterruptedError("Outline paint continuation interrupted")


def _supported(document, oid):
    element = document.element(oid)
    if element.tag != "path" or any(
        a.locks
        or any(
            a.get(key, "none") != "none"
            for key in (
                "clip-path",
                "mask",
                "filter",
                "marker-start",
                "marker-mid",
                "marker-end",
                "stroke-dasharray",
                "vector-effect",
            )
        )
        for a in document.ancestry(oid)
    ):
        return False
    geometry = document.geometry_for(oid)
    return (
        document.geometry_users(geometry.id) == frozenset((oid,))
        and document.dependents({oid}) == frozenset((oid,))
        and sum(len(s.nodes) for s in geometry.subpaths) <= MAX_NODES
        and not any(n.pinned for s in geometry.subpaths for n in s.nodes)
    )


def _paint(document, style):
    if (
        style["fill"] == "none"
        or style["stroke"] != "none"
        or float(style["opacity"]) != 1
        or float(style["fill-opacity"]) != 1
    ):
        return False
    server = paint_server(style["fill"])
    if server is None:
        return color(style["fill"])[3] == 1
    gradient = document.element(server)
    # A bounding-box gradient would change on the original contours as soon as
    # the appended patch changes its owner's bounds. Do not silently reframe it.
    return (
        gradient.tag in {"linearGradient", "radialGradient"}
        and gradient.get("gradientUnits") == "userSpaceOnUse"
        and gradient.get("href") is None
        and all(
            float(stop.get("stop-opacity", "1")) == 1
            and color(stop.get("stop-color", "black"))[3] == 1
            for stop in gradient.children
        )
    )


class PaintContinuation:
    """One bounded, explicitly chosen material interpretation per competitor."""

    def extend(
        self,
        document,
        ink_id,
        footprint,
        material_id,
        work,
        *,
        rule="nonzero",
        core=None,
    ):
        """Append exclusive paint in the old filled ink frame; return a witness.

        The caller chooses the material using source ownership, and supplies the
        complete removed fill, not a dilated stroke. A common parent retains its
        group compositing. The actual ink is moved above the continued material;
        every other sibling keeps its relative order. An optional opaque core,
        also in the ink frame, restricts added paint to an independently proved
        interior. The caller still proves actual composite alpha. No black
        filled outline is retained. Interruption and bounds exclude the edit.
        """
        _check(work)
        if rule not in {"nonzero", "evenodd"}:
            raise ValueError("Paint continuation requires a supported fill rule")
        if (
            ink_id == material_id
            or not footprint.subpaths
            or any(not sub.closed for sub in footprint.subpaths)
            or sum(len(s.nodes) for s in footprint.subpaths) > MAX_PATCH_NODES
            or any(n.pinned for s in footprint.subpaths for n in s.nodes)
            or (
                core is not None
                and (
                    not core.subpaths
                    or any(not sub.closed for sub in core.subpaths)
                    or sum(len(s.nodes) for s in core.subpaths) > MAX_NODES
                    or any(n.pinned for s in core.subpaths for n in s.nodes)
                )
            )
            or not _supported(document, ink_id)
            or not _supported(document, material_id)
        ):
            return None
        parent = document.ancestry(ink_id)[-2]
        if document.ancestry(material_id)[-2].id != parent.id:
            return None
        ink = path_style(document, document.element(ink_id))
        material = path_style(document, document.element(material_id))
        if (
            ink["fill"] != "none"
            or ink["stroke"] == "none"
            or not _paint(document, material)
        ):
            return None
        old = document.geometry_for(material_id)
        if any(not sub.closed for sub in old.subpaths):
            return None
        ink_frame, material_frame = (
            root_matrix(document, ink_id),
            root_matrix(document, material_id),
        )
        if any(
            not np.isfinite(frame).all()
            or abs(frame[0] * frame[3] - frame[1] * frame[2]) < 1e-12
            for frame in (ink_frame, material_frame)
        ):
            return None
        native = curve_path(transformed_geometry(footprint, ink_frame), rule)
        extent = np.asarray(native.bounds)
        if (
            not np.isfinite(extent).all()
            or np.max(np.abs(extent)) > 1_000_000
            or np.max(extent[2:] - extent[:2]) > MAX_EXTENT
            or not 0 < native.area <= MAX_AREA
        ):
            return None
        frame = multiply(inverse_matrix(material_frame), ink_frame)
        domain = curve_path(transformed_geometry(footprint, frame), rule)
        existing = curve_path(old, material["fill-rule"])
        try:
            if core is not None:
                domain = pathops.op(
                    domain,
                    curve_path(transformed_geometry(core, frame)),
                    pathops.PathOp.INTERSECTION,
                )
            patch = pathops.op(domain, existing, pathops.PathOp.DIFFERENCE)
            _check(work)
            if (
                patch.area <= 1e-8
                or pathops.op(patch, domain, pathops.PathOp.DIFFERENCE).area > 1e-8
                or pathops.op(patch, existing, pathops.PathOp.INTERSECTION).area > 1e-8
            ):
                return None
            patch_id = material_id + "-band-restoration"
            shape = identified(path_geometry(patch), patch_id)
        except pathops.PathOpsError:
            return None
        nodes = sum(len(s.nodes) for s in shape.subpaths)
        if (
            nodes > MAX_PATCH_NODES
            or len(shape.subpaths) > MAX_PATCH_CONTOURS
            or nodes + sum(len(s.nodes) for s in old.subpaths) > MAX_NODES
            or {n.id for s in shape.subpaths for n in s.nodes}
            & {n.id for g in document.geometries for s in g.subpaths for n in s.nodes}
        ):
            return None
        geometry = replace(old, subpaths=(*old.subpaths, *shape.subpaths))
        before = next(
            i for i, child in enumerate(parent.children) if child.id == ink_id
        )
        at = (
            next(
                i for i, child in enumerate(parent.children) if child.id == material_id
            )
            + 1
        )
        at -= int(before < at)
        # Keep an already higher stroke in place; no needless stacking changes.
        after = before if before > at else at
        editor = Editor(document, selection=Selection(whole_document=True))
        with editor.transaction("Continue material under an editable outline") as tx:
            _check(work)
            tx.replace_geometry(material_id, geometry)
            if after != before:
                tx.reorder_object(ink_id, after)
        _check(work)
        return editor.snapshot.document, {
            "ink": ink_id,
            "material": material_id,
            "patch_nodes": nodes,
            "patch_contours": len(shape.subpaths),
            "native_bounds": extent.tolist(),
            "stroke_order": (before, after),
            "scope": "removed-fill-footprint",
            "opaque_core": core is not None,
        }
