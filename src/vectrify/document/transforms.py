"""Coordinate frames for document objects and displayed instance geometry."""

from __future__ import annotations

from collections.abc import Iterable

from vectrify.document.hit_test import IDENTITY, Matrix, multiply, transform
from vectrify.document.model import Document, Element


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
