"""Shared plumbing for Generate methods: target region, insertion and scoring.

A Generate method turns (part of) the reference image into SVG in the image's
own pixel coordinates. The editor stretches the reference over the artboard,
so one affine transform places that SVG in the document. The result goes into a
new group, inserted at the front of the chosen container in one transaction.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import cairosvg
import numpy as np
from PIL import Image

from vectrify.document import (
    Document,
    DocumentError,
    Element,
    HitIndex,
    export_svg,
    import_svg,
)
from vectrify.document.model import new_id
from vectrify.image_utils import on_white, preview_urls
from vectrify.operations.contract import OperationRequest, OperationResult, Proposal


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


# Room around the selection for the search to move into, as a share of the
# selection's larger side, and never less than a few reference pixels.
REGION_MARGIN = 0.1
REGION_MARGIN_PIXELS = 3


def selected_bounds(request: OperationRequest) -> tuple[float, float, float, float]:
    """The painted bounds of the selected objects, or the artboard for a
    whole-drawing request or a selection that paints nothing (an empty group)."""
    document = request.snapshot.document
    vx, vy, vw, vh = document.artboard()
    selection = request.snapshot.selection
    painted = (
        None
        if selection.whole_document or not selection.object_ids
        else HitIndex(document).bounds(selection.object_ids)
    )
    if painted is None:
        return vx, vy, vx + vw, vy + vh
    left, top, right, bottom = painted
    assert request.reference is not None
    pixel = max(vw / request.reference.width, vh / request.reference.height)
    pad = max(
        REGION_MARGIN * max(right - left, bottom - top), REGION_MARGIN_PIXELS * pixel
    )
    return left - pad, top - pad, right + pad, bottom + pad


def target_region(request: OperationRequest) -> Region:
    """The selected objects' surroundings, or the artboard for the whole drawing.

    Scoring a small edit against the whole picture drowns it: a path that
    covers 2% of the artboard moves the error of the whole by almost nothing.
    """
    if request.reference is None:
        raise DocumentError("Add a reference image to generate from")
    reference = on_white(request.reference)
    vx, vy, vw, vh = request.snapshot.document.artboard()
    left, top, right, bottom = selected_bounds(request)
    left, top = max(left, vx), max(top, vy)
    right, bottom = min(right, vx + vw), min(bottom, vy + vh)
    if right <= left or bottom <= top:
        raise DocumentError("The selection is outside the artboard")
    sx, sy = reference.width / vw, reference.height / vh
    box = (
        round((left - vx) * sx),
        round((top - vy) * sy),
        round((right - vx) * sx),
        round((bottom - vy) * sy),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise DocumentError("The selection is smaller than one reference pixel")
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


def validate_generate(request: OperationRequest) -> None:
    """What every Generate method needs: leave to add shapes, a place, a region."""
    if not request.permissions.structure:
        raise DocumentError("Allow structure changes to add generated shapes")
    container(request)
    target_region(request)


def fresh_ids(svg: str) -> str:
    """Rename every id, and each local reference to it, so repeats never collide."""
    root = ET.fromstring(svg)
    renamed = {
        element.get("id"): new_id("generated")
        for element in root.iter()
        if element.get("id")
    }
    if not renamed:
        return svg
    pattern = re.compile(r"url\(\s*#([^)\s]+)\s*\)")
    for element in root.iter():
        for key, value in element.attrib.items():
            if key == "id":
                element.set(key, renamed[value])
            elif key in HREF and value.startswith("#") and value[1:] in renamed:
                element.set(key, "#" + renamed[value[1:]])
            elif "url(" in value:
                element.set(
                    key,
                    pattern.sub(
                        lambda m: f"url(#{renamed.get(m.group(1), m.group(1))})",
                        value,
                    ),
                )
    return ET.tostring(root, encoding="unicode")


HREF = {"href", "{http://www.w3.org/1999/xlink}href"}


def insert_svg(tx, request: OperationRequest, svg: str, region: Region, name: str):
    """Add *svg* (in the region's pixel space) as a new named group."""
    generated = import_svg(fresh_ids(svg))
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


def frame(
    root: ET.Element,
    box: tuple[float, float, float, float],
    size: tuple[int, int],
) -> None:
    """Aim *root*'s viewBox at *box*, stretched over *size* pixels."""
    root.set("viewBox", " ".join(str(v) for v in box))
    root.set("width", str(size[0]))
    root.set("height", str(size[1]))
    root.set("preserveAspectRatio", "none")


def render_region(document: Document, region: Region) -> Image.Image:
    root = ET.fromstring(export_svg(document))
    frame(root, (region.x, region.y, region.width, region.height), region.image.size)
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
            previews=preview_urls(region.image, before, after),
        ),
        message=None if group else "Nothing was generated for this region",
    )
