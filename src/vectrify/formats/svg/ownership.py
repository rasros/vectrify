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

_XMLNS_RE = re.compile(r'\s+xmlns(:\w+)?="[^"]*"')


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


# An element owning more than this share of the canvas is scenery rather than a
# part: a background rect touches everything, and linking through it would make
# the whole drawing one part.
BACKDROP_SHARE = 0.25


def adjacent_parts(labels: np.ndarray, count: int, reach: int = 2) -> list[list[int]]:
    """Elements grouped by whether the regions they own run into each other.

    Crossover needs a unit of inheritance. Deciding each matched element on its
    own coin is uniform crossover, which breaks up exactly the groups that were
    worth keeping: one drawing's wing arrived as two elements, a sweep and the
    feathers at its tip, and flipping them separately grafts half a wing from
    each parent. A part is the unit that should travel together.

    Adjacency of owned pixels answers it without any new geometry or any new
    render -- the labels are already computed to match elements across parents.
    Elements that own most of the canvas are left out of the linking, since a
    backdrop touches everything.
    """
    if count == 0:
        return []

    areas = np.bincount(labels[labels != UNOWNED].ravel(), minlength=count)
    scenery = {
        index for index in range(count) if areas[index] > BACKDROP_SHARE * labels.size
    }

    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for step in range(1, reach + 1):
        for dy, dx in ((0, step), (step, 0), (step, step), (step, -step)):
            here = labels[
                max(0, -dy) : labels.shape[0] - max(0, dy),
                max(0, -dx) : labels.shape[1] - max(0, dx),
            ]
            there = labels[
                max(0, dy) : labels.shape[0] - max(0, -dy),
                max(0, dx) : labels.shape[1] - max(0, -dx),
            ]
            touching = (here != there) & (here != UNOWNED) & (there != UNOWNED)
            for left, right in set(
                zip(here[touching].ravel(), there[touching].ravel(), strict=True)
            ):
                if left in scenery or right in scenery:
                    continue
                parent[find(int(left))] = find(int(right))

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


# The size the check renders at when confirming a nominee. The model is shown
# the drawing at 512 (--resolution-llm), so that is what "paints nothing"
# should mean: an element too small to survive that render is not something to
# report as missing, it is something the search will resolve by growing it.
VERIFY_SIZE = 512


def _drawable_parents(root: ET.Element) -> list[tuple[ET.Element, ET.Element]]:
    """Every drawable element with the node it hangs off, in `drawable_elements`
    order, so an index means the same thing to both."""
    pairs: list[tuple[ET.Element, ET.Element]] = []

    def walk(node: ET.Element) -> None:
        for child in node:
            if child.tag.split("}")[-1] == "g":
                walk(child)
            else:
                pairs.append((node, child))

    walk(root)
    return pairs


def _render(root: ET.Element, size: int) -> np.ndarray | None:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    source = ET.tostring(root, encoding="unicode", method="xml")
    try:
        png = rasterize_svg_to_png_bytes(source, out_w=size, out_h=size)
    except Exception:
        return None
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8)


def invisible_elements(
    root: ET.Element, size: int = VERIFY_SIZE, nominate_at: int = MASK_SIZE
) -> list[int]:
    """Indices of drawable elements whose removal changes nothing on screen.

    Markup can carry an element that paints no pixel at all: behind an opaque
    fill drawn after it, or moved off the shape it belongs to and onto matching
    background. Measured on three separate drawings -- a nostril under a beak's
    white fill, and two eye highlights, one behind its own pupil and one adrift
    outside the eye -- and in every case the element cost nothing, so nothing in
    the search had any reason to correct it and it drifted freely.

    Removal is the test, rather than the ownership pass next door, because
    ownership cannot see this. It repaints every element in a flat colour, which
    destroys exactly the property in question: an eye highlight sitting white on
    white background owns hundreds of pixels once it is recoloured, and reads as
    perfectly visible. Only the drawing as actually painted answers it.

    Nominating at MASK_SIZE first keeps the cost down -- an element invisible at
    the size the model is shown is invisible in a thumbnail of it too -- and
    each nominee is then confirmed at *size*. The confirmation matters: a small
    element can vanish from a thumbnail while being plainly there at full size,
    and a false report tells the model to delete or move a feature that is
    actually present, which is worse than saying nothing.
    """
    pairs = _drawable_parents(root)
    if not pairs:
        return []

    def gone(index: int, at: int, baseline: np.ndarray) -> bool:
        trial = copy.deepcopy(root)
        trial_pairs = _drawable_parents(trial)
        if index >= len(trial_pairs):
            return False
        parent, element = trial_pairs[index]
        parent.remove(element)
        without = _render(trial, at)
        return without is not None and np.array_equal(baseline, without)

    thumb = _render(root, nominate_at)
    if thumb is None:
        return []
    nominees = [index for index in range(len(pairs)) if gone(index, nominate_at, thumb)]
    if not nominees:
        return []

    baseline = _render(root, size)
    if baseline is None:
        return []
    return [index for index in nominees if gone(index, size, baseline)]


def describe_invisible(root: ET.Element, indices: list[int]) -> list[str]:
    """One line per invisible element, enough for the model to find it.

    Names the enclosing group, since that is what the model wrote and what it
    will look for, and quotes the element itself.
    """
    lines: list[str] = []
    units = drawable_elements(root)
    for index in indices:
        if index >= len(units):
            continue
        chain, element = units[index]
        group = ""
        for _tag, attrs in reversed(chain):
            found = dict(attrs).get("id")
            if found:
                group = f' inside <g id="{found}">'
                break
        # Serialising a subtree re-declares the namespace, and that attribute
        # is not in the file. Left in, a model copying this line into a SEARCH
        # block produces text that matches nothing -- the report would cause
        # the failure it is meant to help with.
        text = _XMLNS_RE.sub("", ET.tostring(element, encoding="unicode")).strip()
        if len(text) > 160:
            text = text[:157] + "..."
        lines.append(f"{text}{group}")
    return lines
