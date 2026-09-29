"""Shared plumbing for Generate methods: target region, insertion and scoring.

A Generate method turns (part of) the reference image into SVG in the image's
own pixel coordinates. The editor stretches the reference over the artboard,
so one affine transform places that SVG in the document. The result goes into a
new group, inserted at the front of the chosen container in one transaction.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import cairosvg
import numpy as np
from PIL import Image

from vectrify.document import Document, DocumentError, Element, export_svg, import_svg
from vectrify.document.model import new_id
from vectrify.operations.contract import OperationRequest, OperationResult, Proposal
from vectrify.refine.selected import png_url


def artboard(document: Document) -> tuple[float, float, float, float]:
    root = document.root
    viewbox = root.get("viewBox")
    if viewbox:
        x, y, w, h = (float(v) for v in viewbox.replace(",", " ").split())
        return x, y, w, h
    return 0, 0, float(root.get("width") or 1024), float(root.get("height") or 768)


@dataclass(frozen=True)
class Region:
    """A part of the artboard and the matching crop of the reference."""

    x: float
    y: float
    width: float
    height: float
    image: Image.Image

    @property
    def transform(self) -> str:
        """Maps the crop's pixel coordinates onto the region in the document."""
        sx = self.width / self.image.width
        sy = self.height / self.image.height
        return f"matrix({sx!r} 0 0 {sy!r} {self.x!r} {self.y!r})"


def target_region(request: OperationRequest) -> Region:
    """The focus rectangle if one is set, otherwise the whole artboard."""
    if request.reference is None:
        raise DocumentError("Add a reference image to generate from")
    reference = Image.alpha_composite(
        Image.new("RGBA", request.reference.size, "white"),
        request.reference.convert("RGBA"),
    ).convert("RGB")
    vx, vy, vw, vh = artboard(request.snapshot.document)
    focus = request.snapshot.selection.focus
    if focus is None:
        return Region(vx, vy, vw, vh, reference)
    left, top = max(focus.x, vx), max(focus.y, vy)
    right = min(focus.x + focus.width, vx + vw)
    bottom = min(focus.y + focus.height, vy + vh)
    if right <= left or bottom <= top:
        raise DocumentError("The focus region is outside the artboard")
    sx, sy = reference.width / vw, reference.height / vh
    box = (
        round((left - vx) * sx),
        round((top - vy) * sy),
        round((right - vx) * sx),
        round((bottom - vy) * sy),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise DocumentError("The focus region is smaller than one reference pixel")
    # Snap the region to the crop's whole pixels so the transform is exact.
    return Region(
        vx + box[0] / sx,
        vy + box[1] / sy,
        (box[2] - box[0]) / sx,
        (box[3] - box[1]) / sy,
        reference.crop(box),
    )


def container(request: OperationRequest) -> str:
    """Where generated shapes go: the drawing, or one explicitly selected group."""
    document = request.snapshot.document
    selection = request.snapshot.selection
    if selection.whole_document:
        return document.root.id
    if len(selection.object_ids) == 1:
        oid = next(iter(selection.object_ids))
        if document.element(oid).tag == "g":
            return oid
    raise DocumentError("Select the whole drawing or one group to generate into")


def insert_svg(tx, request: OperationRequest, svg: str, region: Region, name: str):
    """Add *svg* (in the region's pixel space) as a new named group."""
    generated = import_svg(svg)
    if not generated.root.children:
        return None, 0
    group = Element(
        new_id("object"),
        "g",
        (("transform", region.transform),),
        generated.root.children,
        name=name,
    )
    tx.insert_object(container(request), group, geometries=generated.geometries)
    shapes = sum(e.tag != "g" for e in Document(group).elements())
    return group.id, shapes


def render_region(document: Document, region: Region) -> Image.Image:
    root = ET.fromstring(export_svg(document))
    root.set("viewBox", f"{region.x} {region.y} {region.width} {region.height}")
    root.set("width", str(region.image.width))
    root.set("height", str(region.image.height))
    root.set("preserveAspectRatio", "none")
    png = cairosvg.svg2png(bytestring=ET.tostring(root), background_color="white")
    assert png is not None
    with Image.open(io.BytesIO(png)) as image:
        return image.convert("RGB")


def error(image: Image.Image, reference: Image.Image) -> float:
    a = np.asarray(image, dtype=np.float64) / 255
    b = np.asarray(reference, dtype=np.float64) / 255
    return float(np.mean((a - b) ** 2))


def generated_result(
    request: OperationRequest,
    svg: str,
    region: Region,
    *,
    label: str,
    name: str,
    metrics: dict[str, Any] | None = None,
) -> OperationResult:
    """Insert *svg* and measure the region against the reference, before and after."""
    tx = request.transaction(label)
    group, shapes = insert_svg(tx, request, svg, region, name)
    before = render_region(request.snapshot.document, region)
    after = render_region(tx.preview, region) if group else before
    return OperationResult(
        Proposal(
            tx,
            group is not None,
            metrics={
                "before": {"error": error(before, region.image)},
                "after": {"error": error(after, region.image)},
                "shapes": shapes,
                **(metrics or {}),
            },
            previews={
                "reference": png_url(region.image),
                "before": png_url(before),
                "after": png_url(after),
            },
        ),
        message=None if group else "Nothing was generated for this region",
    )
