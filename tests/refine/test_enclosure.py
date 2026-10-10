"""Interior fitting respects enclosing ink, including transformed outlines and holes."""

import numpy as np
import pytest

from vectrify.document import import_svg
from vectrify.refine.enclosure import protected_pixels
from vectrify.refine.snap import _Frame


def mask(*, attributes="", after=False, hole=False, transform=""):
    outer = (
        '<path id="outline" fill="green" stroke="black" stroke-width="2" '
        + attributes
        + ' d="M4 4 L60 4 L60 60 L4 60 Z'
        + (" M20 20 L44 20 L44 44 L20 44 Z" if hole else "")
        + '" fill-rule="evenodd"/>'
    )
    shade = '<path id="shade" fill="#68826d" d="M10 10 L54 10 L54 16 L10 16 Z"/>'
    svg = (
        '<svg width="64" height="64"><defs><clipPath id="clip">'
        '<rect width="64" height="64"/></clipPath></defs><g '
        + transform
        + ">"
        + (shade + outer if after else outer + shade)
        + "</g></svg>"
    )
    document = import_svg(svg)
    return protected_pixels(
        document,
        "shade",
        document.geometry_for("shade"),
        _Frame(np.eye(2), np.zeros(2)),
        (64, 64),
    )


def test_enclosure_protects_outline_and_exterior_but_leaves_interior_free():
    protected = mask()
    assert protected is not None
    assert protected[0, 0]
    assert protected[4, 30]
    assert not protected[12, 30]
    assert not protected[32, 32]


def test_enclosure_respects_holes_and_common_object_transforms():
    protected = mask(hole=True, transform='transform="translate(100 200) scale(2)"')
    assert protected is not None
    assert protected[32, 32]
    assert not protected[12, 32]


@pytest.mark.parametrize(
    "attributes",
    [
        'opacity="0.5"',
        'fill-opacity="0.5"',
        'stroke-opacity="0.5"',
        'clip-path="url(#clip)"',
    ],
)
def test_ambiguous_enclosures_are_not_assumed(attributes):
    assert mask(attributes=attributes) is None


def test_later_sibling_is_not_assumed_to_be_an_enclosing_background():
    assert mask(after=True) is None
