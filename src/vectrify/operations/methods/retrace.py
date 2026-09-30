"""Improve: Retrace shape, which gives each selected path its object's outline.

The paths keep their IDs, paint and stacking; only their contours change,
as one edit. SAM finds the object where it can run, on a CUDA GPU; otherwise,
or when asked, the object is grown from the path's inside by colour. See
``vectrify.refine.retrace``.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import ClassVar

import numpy as np

from vectrify.document import Document, DocumentError, Geometry
from vectrify.document.join import path_style
from vectrify.image_utils import on_white
from vectrify.operations.contract import (
    OperationRequest,
    OperationResult,
    Proposal,
    RunContext,
    register,
)
from vectrify.operations.generate import error, render_region, target_region
from vectrify.operations.settings import Setting, read_settings
from vectrify.refine.samvg import SAMVG_MODEL

LABEL = "Retrace shape"
SETTINGS = {
    # SAM where it can run, else by colour; or by colour alone.
    "mode": Setting(str, "sam", choices=("sam", "colour")),
    "model": Setting(str, SAMVG_MODEL, choices=(SAMVG_MODEL, "facebook/sam-vit-base")),
    # How close, as an RGB distance (0-1), a pixel must be to the path's
    # inside for the colour fill to take it.
    "tolerance": Setting(float, 0.1, minimum=0.01, maximum=1.0, label="tolerance"),
}


def _paths(request: OperationRequest) -> list[str]:
    """The selected paths, refusing anything a retrace cannot replace."""
    document = request.snapshot.document
    selection = request.snapshot.selection
    if selection.whole_document or not selection.object_ids:
        raise DocumentError("Select the paths to retrace")
    oids = sorted(selection.object_ids)
    for oid in oids:
        element = document.element(oid)
        ancestry = document.ancestry(oid)
        name = element.name or oid
        if element.tag != "path" or any(
            a.tag in {"defs", "clipPath"} for a in ancestry
        ):
            raise DocumentError(f"{name}: only visible paths can be retraced")
        if path_style(document, element)["fill"] == "none":
            raise DocumentError(f"{name}: only filled paths can be retraced")
        if any({"geometry", "structure"} & a.locks for a in ancestry):
            raise DocumentError(f"{name}: its geometry is locked")
        geometry = document.geometry_for(oid)
        if document.geometry_users(geometry.id) != {oid}:
            raise DocumentError(
                f"{name}: its geometry is shared; detach it before retracing"
            )
        if any(n.pinned for s in geometry.subpaths for n in s.nodes):
            raise DocumentError(
                f"{name}: unpin its points first; a new outline cannot keep them"
            )
    return oids


def _reuse_ids(original: Geometry, traced: Geometry) -> Geometry:
    """*traced* with the IDs of *original*'s contours and nodes, in order,
    where it has as many contours: the edit then moves those nodes."""
    if len(original.subpaths) != len(traced.subpaths):
        return traced
    return replace(
        traced,
        id=original.id,
        subpaths=tuple(
            replace(
                new,
                id=old.id,
                nodes=tuple(
                    replace(node, id=old.nodes[i].id) if i < len(old.nodes) else node
                    for i, node in enumerate(new.nodes)
                ),
            )
            for old, new in zip(original.subpaths, traced.subpaths, strict=True)
        ),
    )


def _nodes(document: Document, oids) -> int:
    return sum(
        len(s.nodes) for oid in oids for s in document.geometry_for(oid).subpaths
    )


class Retrace:
    action: ClassVar[str] = "improve"
    name: ClassVar[str] = "retrace"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset({"gpu"})

    def validate(self, request: OperationRequest) -> None:
        read_settings(request.settings, SETTINGS, "retrace")
        if request.reference is None:
            raise DocumentError("Add a reference image before retracing a shape")
        if not (request.permissions.geometry and request.permissions.structure):
            raise DocumentError("Allow geometry and structure changes to retrace")
        _paths(request)

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        from vectrify.refine import retrace

        assert request.reference is not None
        settings = read_settings(request.settings, SETTINGS, "retrace")
        document = request.snapshot.document
        oids = _paths(request)
        image = on_white(request.reference)
        pixels = np.asarray(image, dtype=np.float64) / 255
        context.progress(0, "Locating the paths…", total=len(oids) + 1)
        items, skipped = [], {}
        for oid in oids:
            style = path_style(document, document.element(oid))
            item = retrace.target(document, oid, image, style["fill-rule"])
            if item is None:
                skipped[oid] = "it covers none of the reference"
            else:
                items.append(item)
        mode = settings["mode"]
        if mode == "sam" and retrace.sam_problem():
            mode = "colour"
        started = time.perf_counter()
        if mode == "sam" and items:
            results = retrace.retrace_sam(
                items,
                pixels,
                image,
                model=settings["model"],
                progress=lambda message: context.progress(0, message),
            )
        else:
            results = []
            for index, item in enumerate(items):
                context.progress(index, "Growing the region by colour…")
                results.append(
                    retrace.retrace_colour(item, pixels, image, settings["tolerance"])
                )
        tx = request.transaction(LABEL)
        retraced = []
        for item, result in zip(items, results, strict=True):
            if result.geometry is None:
                skipped[item.oid] = result.reason or "nothing was found"
                continue
            geometry = _reuse_ids(item.geometry, result.geometry)
            if len(geometry.subpaths) == len(item.geometry.subpaths):
                tx.reshape_path(item.oid, geometry)
            else:
                tx.replace_geometry(item.oid, geometry)
            retraced.append(item.oid)
        seconds = time.perf_counter() - started
        context.progress(len(oids), "Measuring…")
        region = target_region(request)
        before = error(render_region(document, region), region.image)
        after = error(render_region(tx.preview, region), region.image)
        metrics = {
            "mode": mode,
            "paths": len(retraced),
            "skipped": skipped,
            "before": {"error": before, "nodes": _nodes(document, retraced)},
            "after": {"error": after, "nodes": _nodes(tx.preview, retraced)},
            "seconds": round(seconds, 3),
        }
        changed = tx.preview != document
        message = None
        if not changed:
            reasons = sorted(set(skipped.values()))
            message = "Nothing was retraced: " + "; ".join(reasons or ["no change"])
        return OperationResult(Proposal(tx, changed, metrics=metrics), message=message)


register(Retrace())
