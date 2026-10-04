"""Document coordinate frames preserve SVG hierarchy and instance semantics."""

import pytest

from vectrify.document import import_svg
from vectrify.document.hit_test import IDENTITY
from vectrify.document.redraw import root_matrix as redraw_matrix
from vectrify.document.regions import object_matrix as region_matrix
from vectrify.document.transforms import ancestry_matrix, object_matrix, root_matrix


def test_nested_transforms_compose_outermost_first():
    document = import_svg(
        '<svg width="100" height="100" transform="translate(3 4)">'
        '<g id="g" transform="scale(2 3)">'
        '<path id="p" transform="translate(5 7)" d="M0 0L1 1"/>'
        "</g></svg>"
    )
    assert root_matrix(document, "p") == pytest.approx((2, 0, 0, 3, 13, 25))
    assert ancestry_matrix(document.ancestry("p")[:-1]) == pytest.approx(
        (2, 0, 0, 3, 3, 4)
    )
    assert object_matrix(document, "p") == root_matrix(document, "p")
    assert redraw_matrix(document, "p") == root_matrix(document, "p")
    assert ancestry_matrix(()) == IDENTITY


def test_nested_instances_include_offsets_and_skip_definition_ancestors():
    document = import_svg(
        '<svg width="100" height="100" transform="translate(3 4)">'
        '<defs transform="translate(999 999)">'
        '<path id="p" transform="scale(2 3)" d="M0 0L1 1"/>'
        '<use id="middle" href="#p" x="5" y="7" '
        'transform="translate(11 13)"/></defs>'
        '<g transform="translate(23 29)">'
        '<use id="outer" href="#middle" x="17" y="19" '
        'transform="scale(4 5)"/></g></svg>'
    )
    assert root_matrix(document, "outer") == pytest.approx((4, 0, 0, 5, 26, 33))
    assert object_matrix(document, "outer") == pytest.approx((8, 0, 0, 15, 158, 228))
    assert region_matrix(document, "outer") == object_matrix(document, "outer")
