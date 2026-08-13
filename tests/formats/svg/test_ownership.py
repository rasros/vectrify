import xml.etree.ElementTree as ET

from vectrify.formats.svg.ownership import (
    UNOWNED,
    drawable_elements,
    overlaps,
    owner_labels,
)

NS = "http://www.w3.org/2000/svg"


def _svg(body: str) -> ET.Element:
    return ET.fromstring(f'<svg xmlns="{NS}" viewBox="0 0 64 64">{body}</svg>')


def test_a_covered_element_owns_nothing():
    """The point of the pass: contribution is what survives being drawn over,
    not what the element would cover on its own."""
    root = _svg(
        '<rect x="0" y="0" width="64" height="64" fill="#eeeeee"/>'
        '<circle cx="32" cy="32" r="10" fill="#ff0000"/>'
        '<rect x="0" y="0" width="64" height="64" fill="#0000ff"/>'
    )
    labels = owner_labels(root, size=64)

    assert not (labels == 1).any()
    assert (labels == 2).all()


def test_a_ring_owns_its_annulus_not_its_disc():
    root = _svg(
        '<circle cx="32" cy="32" r="30" fill="#123456"/>'
        '<circle cx="32" cy="32" r="15" fill="#654321"/>'
    )
    labels = owner_labels(root, size=64)

    outer = int((labels == 0).sum())
    inner = int((labels == 1).sum())
    assert inner > 0
    # The outer circle's own area would be four times the inner one's; as an
    # annulus it is only three times, because the inner disc is punched out.
    assert outer < 4 * inner


def test_elements_are_counted_inside_groups():
    root = _svg(
        '<g id="dots">'
        '<circle cx="16" cy="16" r="5" fill="#111111"/>'
        '<circle cx="48" cy="48" r="5" fill="#222222"/>'
        "</g>"
    )
    assert len(drawable_elements(root)) == 2


def test_boundary_pixels_are_left_unowned_rather_than_misattributed():
    """Codes are spread across the colour cube so an antialiased blend of two
    elements decodes to nobody. With consecutive codes it would decode to the
    element between them, and the error would be invisible."""
    root = _svg(
        '<rect x="0" y="0" width="64" height="64" fill="#ffffff"/>'
        '<circle cx="32" cy="32" r="20" fill="#000000"/>'
    )
    labels = owner_labels(root, size=64)

    assert set(labels.ravel().tolist()) <= {UNOWNED, 0, 1}


def test_overlap_pairs_the_same_shape_drawn_two_ways():
    """A dot is a <circle> in one lineage and a <path> in another, so the tag
    cannot pair them and the pixels have to."""
    as_circle = _svg('<circle cx="32" cy="32" r="16" fill="#000000"/>')
    as_path = _svg(
        '<path d="M16 32 A16 16 0 1 0 48 32 A16 16 0 1 0 16 32 Z" fill="#000000"/>'
    )

    scores = overlaps(
        owner_labels(as_circle, size=64), owner_labels(as_path, size=64), 1, 1
    )
    assert scores[0, 0] > 0.9


def test_overlap_separates_elements_of_very_different_size():
    """A blade sitting inside a background overlaps it enough to look like a
    match; swapping one for the other paints over the drawing."""
    small = _svg('<circle cx="32" cy="32" r="8" fill="#000000"/>')
    large = _svg('<rect x="0" y="0" width="64" height="64" fill="#000000"/>')

    scores = overlaps(owner_labels(small, size=64), owner_labels(large, size=64), 1, 1)
    assert scores[0, 0] == 0.0
