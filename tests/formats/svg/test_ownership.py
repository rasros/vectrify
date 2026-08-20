import xml.etree.ElementTree as ET

from vectrify.formats.svg.ownership import (
    UNOWNED,
    adjacent_parts,
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


def test_touching_elements_form_one_part():
    """Crossover's unit of inheritance. A wing drawn as a sweep plus the
    feathers at its tip must travel together, or a child takes the sweep from
    one parent and the feathers from the other -- a wing neither parent has.
    """
    # A wing at the proportions real output uses: a 700 canvas with 3.5 wide
    # strokes. Proportions matter -- ownership is resolved on a 128px mask, and
    # a stroke that is wide relative to a small canvas behaves differently.
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 700">'
        '<rect x="0" y="0" width="700" height="700" fill="#ffffff" />'
        '<g fill="none" stroke="#111111" stroke-width="3.5">'
        '<path d="M 303 455 C 305 482 323 498 365 502 C 409 503 446 492 470 440" />'
        '<path d="M 470 440 C 480 425 470 415 455 410" /></g>'
        '<circle cx="120" cy="120" r="6" fill="#000000" /></svg>'
    )
    root = ET.fromstring(svg)
    units = drawable_elements(root)
    parts = adjacent_parts(owner_labels(root), len(units))
    sizes = sorted(len(part) for part in parts)
    assert sizes[-1] == 2, f"the two touching strokes were not joined: {sizes}"
    assert len(parts) == 3, "the backdrop and the far dot should stand alone"


def test_a_backdrop_does_not_join_everything_into_one_part():
    """A rect covering the canvas touches every element, so linking through it
    would make the whole drawing a single part and crossover all-or-nothing.
    """
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<rect x="0" y="0" width="100" height="100" fill="#eeeeee" />'
        '<circle cx="20" cy="20" r="5" fill="#000000" />'
        '<circle cx="80" cy="80" r="5" fill="#000000" /></svg>'
    )
    root = ET.fromstring(svg)
    units = drawable_elements(root)
    parts = adjacent_parts(owner_labels(root), len(units))
    assert len(parts) == 3, f"expected three separate parts, got {len(parts)}"
    assert all(len(part) == 1 for part in parts)


def test_every_element_lands_in_exactly_one_part():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<circle cx="20" cy="20" r="6" fill="#000000" />'
        '<circle cx="26" cy="24" r="6" fill="#111111" />'
        '<circle cx="80" cy="80" r="6" fill="#000000" /></svg>'
    )
    root = ET.fromstring(svg)
    units = drawable_elements(root)
    parts = adjacent_parts(owner_labels(root), len(units))
    seen = sorted(index for part in parts for index in part)
    assert seen == list(range(len(units)))


def _wrap(body: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" '
        'width="100" height="100">' + body + "</svg>"
    )


def test_an_element_hidden_behind_a_later_fill_is_reported():
    """The case this exists for. An eye highlight drawn before its own pupil
    paints nothing, and nothing in the search corrects it because it costs
    nothing -- so it drifts anywhere, including off the eye entirely."""
    import xml.etree.ElementTree as ET

    from vectrify.formats.svg.ownership import invisible_elements

    root = ET.fromstring(
        _wrap(
            '<circle cx="50" cy="50" r="8" fill="#ffffff"/>'
            '<circle cx="50" cy="50" r="30" fill="#000000"/>'
        )
    )
    assert invisible_elements(root) == [0]


def test_an_element_on_background_of_its_own_colour_is_reported():
    """Ownership cannot see this one: it repaints every element in a flat
    colour, so a white shape on white background reads as fully visible."""
    import xml.etree.ElementTree as ET

    from vectrify.formats.svg.ownership import invisible_elements

    root = ET.fromstring(
        _wrap(
            '<rect width="100" height="100" fill="#ffffff"/>'
            '<ellipse cx="20" cy="20" rx="9" ry="7" fill="#ffffff"/>'
            '<circle cx="70" cy="70" r="12" fill="#000000"/>'
        )
    )
    assert 1 in invisible_elements(root)


def test_a_small_but_visible_element_is_not_reported():
    """A false report tells the model to delete a feature that is really there.
    Measured on real output, a hidden nostril paints 5 px of a 512x512 canvas
    while legitimate number labels paint 12-19 -- there is no gap to threshold
    on, which is why the test is exact invisibility and nothing looser."""
    import xml.etree.ElementTree as ET

    from vectrify.formats.svg.ownership import invisible_elements

    root = ET.fromstring(
        _wrap(
            '<rect width="100" height="100" fill="#ffffff"/>'
            '<circle cx="50" cy="50" r="1.5" fill="#000000"/>'
        )
    )
    assert 1 not in invisible_elements(root)


def test_nothing_is_reported_when_every_element_paints():
    import xml.etree.ElementTree as ET

    from vectrify.formats.svg.ownership import invisible_elements

    root = ET.fromstring(
        _wrap(
            '<circle cx="30" cy="30" r="12" fill="#000000"/>'
            '<circle cx="70" cy="70" r="12" fill="#ff0000"/>'
        )
    )
    assert invisible_elements(root) == []


def test_the_description_quotes_text_that_is_really_in_the_file():
    """The model is expected to copy this into a SEARCH block. Serialising a
    subtree re-declares the namespace, and that attribute is not in the file --
    left in, the report would cause the very match failure it exists to help
    with."""
    import xml.etree.ElementTree as ET

    from vectrify.formats.svg.ownership import describe_invisible, invisible_elements

    source = _wrap(
        '<g id="eye"><ellipse cx="20" cy="20" rx="9" ry="7" fill="#ffffff" />'
        '<circle cx="70" cy="70" r="12" fill="#000000" /></g>'
    )
    root = ET.fromstring(source)
    lines = describe_invisible(root, invisible_elements(root))
    assert lines
    quoted = lines[0].split(" inside ")[0]
    assert "xmlns" not in quoted
    assert quoted in source
