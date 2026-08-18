"""Putting LLM markup into the form local search can edit.

Every rewrite here has to preserve the rendering exactly: the point is to remove
the ways a drawing can be correct on screen and unreachable to mutation, not to
change the drawing.
"""

import io
import re
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from vectrify.formats.svg.normalize import absolutize_path, normalize_svg
from vectrify.formats.svg.plugin import SvgPlugin

NS = "http://www.w3.org/2000/svg"


def _render(svg: str) -> np.ndarray:
    png = SvgPlugin().rasterize(svg, 200, 200)
    return np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=float)


def _find(svg: str, tag: str) -> ET.Element:
    return next(el for el in ET.fromstring(svg).iter() if el.tag.endswith(tag))


@pytest.mark.parametrize(
    ("relative", "absolute"),
    [
        ("m10 10 l20 0 l0 20 z", "M 10 10 L 30 10 L 30 30 Z"),
        ("M100 100 c10 0 20 10 20 20", "M 100 100 C 110 100 120 110 120 120"),
        ("M0 0 h50 v50 H10 Z", "M 0 0 H 50 V 50 H 10 Z"),
    ],
)
def test_relative_commands_become_absolute(relative, absolute):
    """A relative command says "from wherever you are", which is not a position
    anything can move or reason about."""
    assert absolutize_path(relative) == absolute


def test_a_relative_path_renders_the_same_after_rewriting():
    svg = (
        f'<svg xmlns="{NS}" viewBox="0 0 100 100">'
        '<path d="m10 10 l60 0 l0 60 z"/></svg>'
    )

    assert np.array_equal(_render(svg), _render(normalize_svg(svg)))


def test_a_named_colour_becomes_hex():
    """A name has no channels, so mutate_color skips it and the colour is frozen
    for the rest of the run."""
    svg = (
        f'<svg xmlns="{NS}">'
        '<rect width="9" height="9" fill="red" stroke="black"/></svg>'
    )

    rect = _find(normalize_svg(svg), "rect")

    assert rect.get("fill") == "#ff0000"
    assert rect.get("stroke") == "#000000"


def test_short_hex_becomes_six_digits():
    svg = f'<svg xmlns="{NS}"><rect width="9" height="9" fill="#fe1"/></svg>'

    assert _find(normalize_svg(svg), "rect").get("fill") == "#ffee11"


def test_presentation_properties_move_out_of_style():
    """A mutation reads attributes; a colour parked in a style string is
    invisible to most of them."""
    svg = (
        f'<svg xmlns="{NS}"><circle cx="5" cy="5" r="2" '
        'style="fill:#ffffff;stroke-width:3;mask:url(#m)"/></svg>'
    )

    circle = _find(normalize_svg(svg), "circle")

    assert circle.get("fill") == "#ffffff"
    assert circle.get("stroke-width") == "3"
    # Anything a mutation has no use for stays where it was.
    assert circle.get("style") == "mask:url(#m)"


def test_a_polygon_becomes_a_path():
    """Its geometry lives in `points`, which the path operators cannot see, so
    the shape is fixed in place for the whole run."""
    svg = (
        f'<svg xmlns="{NS}" viewBox="0 0 100 100">'
        '<polygon points="10,10 90,10 50,90" fill="#000000"/></svg>'
    )

    out = normalize_svg(svg)

    assert "polygon" not in out
    assert _find(out, "path").get("d", "").endswith("Z")
    assert np.array_equal(_render(svg), _render(out))


def test_a_polyline_becomes_an_unclosed_path():
    svg = (
        f'<svg xmlns="{NS}" viewBox="0 0 100 100">'
        '<polyline points="10,10 90,10 50,90" fill="none" stroke="#000000"/></svg>'
    )

    out = normalize_svg(svg)

    assert not _find(out, "path").get("d", "").endswith("Z")
    assert np.array_equal(_render(svg), _render(out))


def test_normalising_twice_changes_nothing_more():
    """It runs on every candidate the model returns, including edits of already
    normalised parents."""
    svg = (
        f'<svg xmlns="{NS}" viewBox="0 0 100 100">'
        '<polygon points="10,10 90,10 50,90" fill="red"/>'
        '<path d="m5 5 l10 10"/></svg>'
    )
    once = normalize_svg(svg)

    assert normalize_svg(once) == once


def test_markup_that_does_not_parse_comes_back_unchanged():
    """Validation is a separate step and reports its own error."""
    broken = "<svg><rect></svg>"

    assert normalize_svg(broken) == broken


def test_arc_flags_survive_absolutising():
    """An arc carries two booleans among its numbers. Treating them as
    coordinates writes sweep="-1.8", which is not path data at all."""
    d = "M10 10 a23 23 0 1 0 46 0"

    out = absolutize_path(d)

    assert re.search(r"A 23 23 0 1 0 ", out), out
