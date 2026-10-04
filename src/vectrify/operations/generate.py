"""Shared plumbing for Generate methods: target region, insertion and scoring.

A Generate method turns (part of) the reference image into SVG in the image's
own pixel coordinates. The editor stretches the reference over the artboard,
so one affine transform places that SVG in the document. The result goes into a
new group, inserted at the front of the chosen container in one transaction.
"""

from __future__ import annotations

import io
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from typing import Any

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
from vectrify.svg_render import frame as frame
from vectrify.svg_render import render_png

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Region:
    """A part of the artboard and the matching crop of the reference."""

    x: float
    y: float
    width: float
    height: float
    image: Image.Image
    # The crop's opacity, 0-1 by pixel, when the reference has transparent
    # pixels in it; *image* is the crop over white either way.
    alpha: np.ndarray | None = None

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


def selected_bounds(
    request: OperationRequest, margin: float = REGION_MARGIN
) -> tuple[float, float, float, float]:
    """The painted bounds of the selected objects, or the artboard for a
    whole-drawing request or a selection that paints nothing (an empty group)."""
    document = request.snapshot.document
    vx, vy, vw, vh = document.artboard()
    selection = request.snapshot.selection
    painted = (
        None
        if selection.whole_document or not selection.object_ids
        else HitIndex(_holding(document, selection.object_ids)).bounds(
            selection.object_ids
        )
    )
    if painted is None:
        return vx, vy, vx + vw, vy + vh
    left, top, right, bottom = painted
    reference = request.reference
    pixel = max(vw / reference.width, vh / reference.height) if reference else 0.0
    pad = max(margin * max(right - left, bottom - top), REGION_MARGIN_PIXELS * pixel)
    return left - pad, top - pad, right + pad, bottom + pad


def _holding(document: Document, object_ids) -> Document:
    """*document* with only *object_ids*, what they sit in and what is in them.

    Their painted bounds are the same, and indexing a large drawing for a
    few objects' bounds takes seconds.
    """
    keep = {a.id for oid in object_ids for a in document.ancestry(oid)}

    def prune(element: Element, inside: bool) -> Element:
        if element.tag == "defs" or inside:
            return element
        return replace(
            element,
            children=tuple(
                prune(c, c.id in object_ids)
                for c in element.children
                if c.id in keep or c.tag == "defs"
            ),
        )

    subset = replace(document, root=prune(document.root, False))
    # References can target drawing objects outside the subset.
    try:
        subset.validate()
    except DocumentError:
        return document
    return subset


def _on_artboard(
    request: OperationRequest, margin: float = REGION_MARGIN
) -> tuple[float, float, float, float]:
    vx, vy, vw, vh = request.snapshot.document.artboard()
    left, top, right, bottom = selected_bounds(request, margin)
    left, top = max(left, vx), max(top, vy)
    right, bottom = min(right, vx + vw), min(bottom, vy + vh)
    if right <= left or bottom <= top:
        raise DocumentError("The selection is outside the artboard")
    return left, top, right, bottom


def drawing_region(
    request: OperationRequest, long_side: int, margin: float = REGION_MARGIN
) -> Region:
    """The selection's surroundings with the drawing itself as the image.

    What an operation compares against when there is no reference: the
    drawing as it stands, so a change is judged by how far it moves from it.
    """
    left, top, right, bottom = _on_artboard(request, margin)
    width, height = right - left, bottom - top
    scale = long_side / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    blank = Region(left, top, width, height, Image.new("RGB", size))
    return replace(blank, image=render_region(request.snapshot.document, blank))


def target_region(request: OperationRequest, margin: float = REGION_MARGIN) -> Region:
    """The selected objects' surroundings, or the artboard for the whole drawing.

    Scoring a small edit against the whole picture drowns it: a path that
    covers 2% of the artboard moves the error of the whole by almost nothing.
    """
    if request.reference is None:
        raise DocumentError("Add a reference image to generate from")
    reference = on_white(request.reference)
    vx, vy, vw, vh = request.snapshot.document.artboard()
    left, top, right, bottom = _on_artboard(request, margin)
    sx, sy = reference.width / vw, reference.height / vh
    box = (
        round((left - vx) * sx),
        round((top - vy) * sy),
        round((right - vx) * sx),
        round((bottom - vy) * sy),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise DocumentError("The selection is smaller than one reference pixel")
    alpha = None
    if request.reference.has_transparency_data:
        rgba = request.reference.convert("RGBA")
        opacity = np.asarray(rgba.getchannel("A").crop(box))
        if opacity.min() < 255:
            alpha = opacity.astype(np.float32) / 255
    # Snap the region to the crop's whole pixels so the transform is exact.
    return Region(
        vx + box[0] / sx,
        vy + box[1] / sy,
        (box[2] - box[0]) / sx,
        (box[3] - box[1]) / sy,
        reference.crop(box),
        alpha,
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
    """Add *svg* (in the region's pixel space) as a new named group.

    Returns the group's ID and the IDs of the shapes in it, front last.
    """
    generated = import_svg(fresh_ids(svg))
    if not generated.root.children:
        return None, ()
    group = Element(
        new_id("object"),
        "g",
        (("transform", region.transform),),
        generated.root.children,
        name=name,
    )
    tx.insert_object(container(request), group, geometries=generated.geometries)
    shapes = tuple(e.id for e in Document(group).elements() if e.tag != "g")
    return group.id, shapes


# Largest contact distance edge snapping accepts, in document units.
SEAM_LIMIT = 20.0


def snap_seams(tx, paths: frozenset[str], distance: float) -> int:
    """Snap the touching edges of freshly inserted *paths*; the spans matched.

    A trace is worth keeping with its seams as traced, so a refusal only logs.
    """
    if len(paths) < 2:
        return 0
    try:
        return len(tx.snap_edges(min(max(distance, 1e-3), SEAM_LIMIT), paths))
    except DocumentError as exc:
        log.info("Kept the trace without snapping its seams: %s", exc)
        return 0


def render_region(document: Document, region: Region) -> Image.Image:
    png = render_png(
        export_svg(document),
        (region.x, region.y, region.width, region.height),
        region.image.size,
    )
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
    seams: float | None = None,
) -> OperationResult:
    """Insert *svg* and measure the region against the reference, before and after.

    *svg* is in the pixels of *region*'s image. With *seams*, a contact
    distance in those pixels, the touching edges of the inserted paths are
    snapped together so neighbouring regions meet exactly.
    """
    tx = request.transaction(label)
    group, shapes = insert_svg(tx, request, svg, region, name)
    if seams is not None:
        paths = frozenset(i for i in shapes if tx.preview.element(i).tag == "path")
        scale = region.width / region.image.width
        metrics = {
            **(metrics or {}),
            "snapped": snap_seams(tx, paths, seams * scale),
        }
    before = render_region(request.snapshot.document, region)
    after = render_region(tx.preview, region) if group else before
    return OperationResult(
        Proposal(
            tx,
            group is not None,
            metrics={
                "before": {"error": error(before, region.image)},
                "after": {"error": error(after, region.image)},
                "shapes": len(shapes),
                **(metrics or {}),
            },
            previews=preview_urls(region.image, before, after),
        ),
        message=None if group else "Nothing was generated for this region",
    )
