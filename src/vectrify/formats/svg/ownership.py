"""Which element owns which pixel, once everything has been drawn.

An element's contribution is not its bounding box: whatever is painted after it
covers it up, so a ring is an annulus rather than a disc and a fully hidden
element contributes nothing at all. Rendering the drawing once with every
element in its own flat colour answers the question exactly -- the composite
names the winner of every pixel.

The codes are spread across the colour cube rather than numbered 1, 2, 3. With
consecutive codes an antialiased edge between elements 1 and 3 lands on exactly
the code for element 2, and the misattribution is undetectable; spread out, a
blend belongs to nobody and is discarded instead. Shape antialiasing is turned
off, which leaves only glyph edges blending -- on the bench corpus, under 2% of
the canvas.
"""

import copy
import io
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from vectrify.image_utils import rasterize_svg_to_png_bytes

# Small enough that the extra render stays cheap, large enough that the
# thinnest elements the corpus draws still own a few pixels.
MASK_SIZE = 128

UNOWNED = -1

# How close in size two elements must be to be candidates for the same feature.
SIZE_RATIO_LIMIT = 0.5

_STYLE_PAINT_RE = re.compile(r"(fill|stroke|opacity)\s*:[^;]*;?")


def _code(index: int) -> tuple[int, int, int]:
    """A colour for *index*, far from every other index's colour."""
    return (
        (index * 67 + 29) % 251,
        (index * 151 + 71) % 251,
        (index * 211 + 113) % 251,
    )


def drawable_elements(root: ET.Element) -> list[tuple[tuple, ET.Element]]:
    """Every drawable element, paired with the group chain it hangs under.

    Groups are structure rather than content: a seed keeping all 23 dots in one
    <g id="dots"> has 23 elements, not one.
    """
    units: list[tuple[tuple, ET.Element]] = []

    def walk(node: ET.Element, chain: tuple) -> None:
        for child in node:
            if child.tag.split("}")[-1] == "g":
                key = (child.tag, tuple(sorted(child.attrib.items())))
                walk(child, (*chain, key))
            else:
                units.append((chain, child))

    walk(root, ())
    return units


def owner_labels(root: ET.Element, size: int = MASK_SIZE) -> np.ndarray:
    """Index of the element owning each pixel, or UNOWNED where it is unclear.

    *root* is left untouched; the paint pass runs on a copy.
    """
    painted = copy.deepcopy(root)
    units = drawable_elements(painted)

    for index, (_chain, element) in enumerate(units):
        red, green, blue = _code(index)
        colour = f"#{red:02x}{green:02x}{blue:02x}"
        for node in element.iter():
            node.set("fill", colour)
            if node.get("stroke") and node.get("stroke") != "none":
                node.set("stroke", colour)
            # Transparency blends whatever is behind, which would make the
            # pixel belong to two elements at once. clip-path and transform are
            # deliberately kept: they are part of where the element paints.
            for key in ("opacity", "fill-opacity", "stroke-opacity", "filter", "mask"):
                node.attrib.pop(key, None)
            style = node.get("style")
            if style:
                node.set("style", _STYLE_PAINT_RE.sub("", style))
            node.set("shape-rendering", "crispEdges")

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    source = ET.tostring(painted, encoding="unicode", method="xml")
    try:
        png = rasterize_svg_to_png_bytes(source, out_w=size, out_h=size)
    except Exception:
        return np.full((size, size), UNOWNED, dtype=np.int32)

    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int64)
    codes = (pixels[:, :, 0] << 16) | (pixels[:, :, 1] << 8) | pixels[:, :, 2]

    lookup: dict[int, int] = {}
    for index in range(len(units)):
        red, green, blue = _code(index)
        lookup[(red << 16) | (green << 8) | blue] = index

    labels = np.full(codes.shape, UNOWNED, dtype=np.int32)
    for code, index in lookup.items():
        labels[codes == code] = index
    return labels


def overlaps(
    labels_a: np.ndarray, labels_b: np.ndarray, count_a: int, count_b: int
) -> np.ndarray:
    """Intersection-over-union of every element of A with every element of B.

    Counted in one pass over the pixels rather than per pair: at 59 elements
    against 46 that is 2714 comparisons of the whole canvas, which is far too
    slow to run on every crossover.
    """
    if count_a == 0 or count_b == 0:
        return np.zeros((count_a, count_b))

    both = (labels_a != UNOWNED) & (labels_b != UNOWNED)
    pairs = labels_a[both].astype(np.int64) * count_b + labels_b[both].astype(np.int64)
    intersection = np.bincount(pairs, minlength=count_a * count_b).reshape(
        count_a, count_b
    )

    area_a = np.bincount(
        labels_a[labels_a != UNOWNED].ravel(), minlength=count_a
    ).astype(np.float64)
    area_b = np.bincount(
        labels_b[labels_b != UNOWNED].ravel(), minlength=count_b
    ).astype(np.float64)

    union = area_a[:, None] + area_b[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)

    # Two elements of very different size are not the same feature however much
    # they overlap: a leaf blade sits inside a background rect at an IoU of
    # 0.43, and swapping the blade for the rect paints over the whole drawing.
    # Sizes alone cannot say it either -- this rules that pair out while still
    # matching one feature drawn larger in one lineage than the other, which is
    # the difference the seeds are built on.
    larger = np.maximum(area_a[:, None], area_b[None, :])
    smaller = np.minimum(area_a[:, None], area_b[None, :])
    similar = np.divide(smaller, larger, out=np.zeros_like(larger), where=larger > 0)
    return np.where(similar >= SIZE_RATIO_LIMIT, iou, 0.0)


def element_error(
    root: ET.Element,
    reference: np.ndarray,
    render: np.ndarray,
    size: int = MASK_SIZE,
) -> list[float]:
    """Error each element is responsible for, over the pixels it still owns.

    An element that was drawn over answers for nothing, and a ring answers for
    its visible annulus rather than its whole disc -- which is the point of
    resolving ownership rather than using bounding boxes.

    *reference* and *render* are the target and the candidate at *size*.
    """
    labels = owner_labels(root, size=size)
    difference = np.abs(reference.astype(np.float32) - render.astype(np.float32))
    if difference.ndim == 3:
        difference = difference.mean(axis=2)

    count = len(drawable_elements(root))
    if count == 0:
        return []

    owned = labels != UNOWNED
    flat = labels[owned].ravel()
    totals = np.bincount(flat, weights=difference[owned].ravel(), minlength=count)
    return [float(value) for value in totals]
