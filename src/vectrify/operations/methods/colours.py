"""Improve: fit flat fill colours of the selected objects, geometry locked.

Compositing is linear in an object's fill colour: every pixel of the rendered
region equals ``backdrop + coverage * fill``, where coverage already includes
the object's antialiasing, opacity, clipping and everything painted in front.
Rendering the region twice, once with the fill black and once white, measures
both terms exactly, so the fill that best matches the reference is a closed-form
least-squares solution per channel. Objects are fitted back to front, each
against the drawing as already refitted. No GPU and no search are involved.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import numpy as np

from vectrify.document import Document, DocumentError
from vectrify.document.join import path_style
from vectrify.operations.contract import (
    OperationRequest,
    OperationResult,
    Proposal,
    RunContext,
    register,
)
from vectrify.operations.generate import Region, error, render_region, target_region
from vectrify.operations.settings import Setting, read_settings
from vectrify.refine.selected import png_url

DRAWABLE = {"path", "rect", "circle", "ellipse", "use"}
SETTINGS = {
    "passes": Setting(int, 1, minimum=1, maximum=5),
    "resolution": Setting(int, 256, minimum=32, maximum=1024, label="resolution"),
}


def targets(document: Document, request: OperationRequest) -> list[str]:
    """Selected drawables with a solid fill, in paint order (back to front)."""
    selected = document.selection_ids(request.snapshot.selection)
    found = []
    for element in document.elements():
        if element.id not in selected or element.tag not in DRAWABLE:
            continue
        if any(a.tag in {"defs", "clipPath"} for a in document.ancestry(element.id)):
            continue
        fill = path_style(document, element)["fill"]
        if fill != "none" and not fill.startswith("url("):
            found.append(element.id)
    return found


def _with_fill(document: Document, oid: str, fill: str) -> Document:
    element = document.element(oid)
    attributes = dict(element.attributes)
    attributes["fill"] = fill
    return document.replace_element(
        replace(element, attributes=tuple(attributes.items()))
    )


def _array(document: Document, region: Region) -> np.ndarray:
    return np.asarray(render_region(document, region), dtype=np.float64) / 255


def best_fill(
    document: Document, oid: str, region: Region, target: np.ndarray
) -> str | None:
    """The flat fill minimizing squared error in the region, or None if hidden."""
    dark = _array(_with_fill(document, oid, "#000000"), region)
    light = _array(_with_fill(document, oid, "#ffffff"), region)
    coverage = light - dark
    weight = float(np.sum(coverage * coverage))
    if weight < 1e-6:
        return None
    channels = np.sum(coverage * (target - dark), axis=(0, 1)) / np.sum(
        coverage * coverage, axis=(0, 1)
    ).clip(1e-12)
    r, g, b = (round(float(c) * 255) for c in np.clip(channels, 0, 1))
    return f"#{r:02x}{g:02x}{b:02x}"


class ColourFit:
    action: ClassVar[str] = "improve"
    name: ClassVar[str] = "colours"
    background: ClassVar[bool] = True
    needs_reference: ClassVar[bool] = True
    resources: ClassVar[frozenset[str]] = frozenset()

    def validate(self, request: OperationRequest) -> None:
        read_settings(request.settings, SETTINGS, "colour-fit")
        if not request.permissions.paint:
            raise DocumentError("Allow paint changes to fit colours")
        if not targets(request.snapshot.document, request):
            raise DocumentError("Select objects with a solid fill")
        target_region(request)

    def run(self, request: OperationRequest, context: RunContext) -> OperationResult:
        settings = read_settings(request.settings, SETTINGS, "colour-fit")
        full = target_region(request)
        scale = min(1, settings["resolution"] / max(full.image.size))
        size = (
            max(1, round(full.image.width * scale)),
            max(1, round(full.image.height * scale)),
        )
        region = replace(full, image=full.image.convert("RGB").resize(size))
        target = np.asarray(region.image, dtype=np.float64) / 255
        document = request.snapshot.document
        ids = targets(document, request)
        total = len(ids) * settings["passes"]
        tx = request.transaction("Fit colours")
        fitted: dict[str, str] = {}
        step = 0
        for _ in range(settings["passes"]):
            for oid in ids:
                if context.stop.is_set():
                    break
                context.progress(step, f"Fitting {step + 1} of {total}…", total=total)
                fill = best_fill(tx.preview, oid, region, target)
                step += 1
                if fill is None:
                    continue
                element = tx.preview.element(oid)
                style = path_style(tx.preview, element)
                changes: dict[str, str | None] = {"fill": fill}
                # An outline painted in the fill colour belongs to the shape.
                if style["stroke"] != "none" and style["stroke"] == style["fill"]:
                    changes["stroke"] = fill
                if any(element.get(k) != v for k, v in changes.items()):
                    tx.set_attributes(oid, changes)
                    fitted[oid] = fill
        before_image = render_region(document, full)
        after_image = render_region(tx.preview, full)
        reference = full.image.convert("RGB")
        return OperationResult(
            Proposal(
                tx,
                bool(fitted),
                metrics={
                    "before": {"error": error(before_image, reference)},
                    "after": {"error": error(after_image, reference)},
                    "objects": len(fitted),
                    "considered": len(ids),
                },
                previews={
                    "reference": png_url(full.image),
                    "before": png_url(before_image),
                    "after": png_url(after_image),
                },
            ),
            message=None if fitted else "The colours already fit the reference",
        )


register(ColourFit())
