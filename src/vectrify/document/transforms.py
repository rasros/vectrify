"""Coordinate frames for document objects and displayed instance geometry."""

from __future__ import annotations

import math
from collections.abc import Iterable

from vectrify.document.hit_test import IDENTITY, Matrix, multiply, transform
from vectrify.document.model import Document, Element

# The linear part (a, b, c, d) of an affine map, as in SVG's matrix(a b c d e f).
Linear = tuple[float, float, float, float]

# The reflections Flip horizontal and Flip vertical make: across a vertical
# line, mirroring left and right, and across a horizontal one.
REFLECTIONS: dict[str, Linear] = {
    "horizontal": (-1.0, 0.0, 0.0, 1.0),
    "vertical": (1.0, 0.0, 0.0, -1.0),
}


def ancestry_matrix(ancestry: Iterable[Element]) -> Matrix:
    """Compose element transforms in order, from outermost to innermost."""
    matrix = IDENTITY
    for element in ancestry:
        matrix = multiply(matrix, transform(element.get("transform")))
    return matrix


def root_matrix(document: Document, object_id: str) -> Matrix:
    """Map an object's own local coordinates into root SVG user space."""
    return ancestry_matrix(document.ancestry(object_id))


def object_matrix(document: Document, object_id: str) -> Matrix:
    """Map displayed geometry into root user space, resolving nested instances.

    An instance adds its x/y offset and the referenced element's transform.
    The referenced element's definition ancestors are not part of its frame.
    """
    matrix = root_matrix(document, object_id)
    element = document.element(object_id)
    while element.tag == "use":
        x = float(element.get("x", "0") or 0)
        y = float(element.get("y", "0") or 0)
        element = document.element((element.get("href") or "#")[1:])
        matrix = multiply(matrix, (1, 0, 0, 1, x, y))
        matrix = multiply(matrix, transform(element.get("transform")))
    return matrix


def about(pivot: tuple[float, float], linear: Linear) -> Matrix:
    """The affine map that applies *linear* and keeps *pivot* where it is."""
    a, b, c, d = linear
    x, y = pivot
    return (a, b, c, d, x - a * x - c * y, y - b * x - d * y)


def rotation(degrees: float) -> Linear:
    """A turn by *degrees*, clockwise on screen as SVG's rotate() is (y down).

    Quarter turns are exact, so turning four times by 90 gives back the
    original transform rather than one a rounding error away.
    """
    quarters = degrees / 90
    if quarters == round(quarters):
        cos, sin = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[
            round(quarters) % 4
        ]
    else:
        radians = math.radians(degrees)
        cos, sin = math.cos(radians), math.sin(radians)
    return (cos, sin, -sin, cos)
