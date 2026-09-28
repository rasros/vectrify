"""Final cleanup preserves topology, paint order, and referenced geometry."""

import io
from xml.etree import ElementTree as ET

import cairosvg
import numpy as np
import pytest
from PIL import Image

from vectrify.formats.svg.cleanup import cleanup_svg_geometry

NS = "{http://www.w3.org/2000/svg}"


def document(body):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
        + body
        + "</svg>"
    )


def render(svg):
    return np.asarray(Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode()))))


def test_redundant_vertices_removed_without_changing_hole_or_sharp_tip():
    svg = document(
        '<path fill="red" fill-rule="evenodd" d="'
        "M4 4 L12 4 L12 4 L32 4 L33 1 L34 4 L60 4 L60 60 L4 60 Z "
        'M16 16 L16 32 L16 48 L48 48 L48 16 Z"/>'
    )
    cleaned, stats = cleanup_svg_geometry(svg)
    assert stats["vertices_removed"] == 3
    assert "33 1" in cleaned
    assert np.array_equal(render(svg), render(cleaned))
    assert cleanup_svg_geometry(cleaned)[0] == cleaned


def test_merge_matching_strokes_preserves_subpath_caps_and_reversals():
    svg = document(
        '<g fill="none" stroke="black" stroke-width="2" stroke-linecap="round">'
        '<path d="M4 4 L16 4 L8 4 L8 20"/>'
        '<path d="M32 32 L48 32 L48 48 Z"/></g>'
    )
    cleaned, stats = cleanup_svg_geometry(svg)
    assert stats["paths_merged"] == 1
    assert stats["vertices_removed"] == 0
    data = ET.fromstring(cleaned).find(".//" + NS + "path").get("d")
    assert data.count("M") == 2
    assert data.endswith("Z")
    assert np.array_equal(render(svg), render(cleaned))


@pytest.mark.parametrize(
    ("second", "merged"), [("M32 32 L48 32 L48 48 Z", 1), ("M4 4 L20 4 L20 20 Z", 0)]
)
def test_merge_disjoint_fills_but_keep_overlapping_evenodd_fills(second, merged):
    svg = document(
        '<g fill="red" fill-rule="evenodd">'
        '<path d="M4 4 L16 4 L16 16 Z"/>'
        f'<path d="{second}"/></g>'
    )
    cleaned, stats = cleanup_svg_geometry(svg)
    assert stats["paths_merged"] == merged
    assert np.array_equal(render(svg), render(cleaned))


@pytest.mark.parametrize(
    ("group", "attrs"),
    [
        ('opacity="0.5"', 'opacity="1"'),
        ('filter="url(#blur)"', 'filter="none"'),
        ("", 'opacity="0.5"'),
        ("", 'stroke="#ff000080"'),
        ("", 'stroke="rgba(255,0,0,0.5)"'),
        ("", 'stroke="transparent"'),
        ("", 'stroke-dasharray="2 2"'),
        ("", 'marker-mid="url(#dot)"'),
    ],
)
def test_effects_and_transparency_block_merging(group, attrs):
    svg = document(
        f'<g fill="none" stroke="black" {group}>'
        f'<path {attrs} d="M4 4 L8 4 L16 4"/>'
        f'<path {attrs} d="M16 4 L24 4 L32 4"/></g>'
    )
    _, stats = cleanup_svg_geometry(svg)
    assert stats["paths_after"] == 2


def test_references_and_clip_boundaries_are_retained():
    svg = document(
        '<defs><path id="shape" d="M4 4 L8 4 L32 4 L32 32 Z"/>'
        '<clipPath id="clip"><use href="#shape"/></clipPath></defs>'
        '<use href="#shape" fill="blue"/>'
        '<g clip-path="url(#clip)" fill="none" stroke="red">'
        '<path d="M4 4 L16 4"/><path d="M16 4 L24 4"/></g>'
        '<g fill="none" stroke="red"><path d="M24 4 L32 4"/></g>'
    )
    cleaned, stats = cleanup_svg_geometry(svg)
    root = ET.fromstring(cleaned)
    assert root.find(".//" + NS + "path[@id='shape']") is not None
    assert len(root.findall(".//" + NS + "use")) == 2
    assert stats["paths_merged"] == 1
    assert stats["paths_after"] == 3
    assert np.array_equal(render(svg), render(cleaned))


@pytest.mark.parametrize("data", ["M1 1 Q2 2 3 3", "m1 1 l2 2", "M1 1 L", "M1", "L1 1"])
def test_unsupported_or_malformed_paths_are_not_rewritten(data):
    cleaned, stats = cleanup_svg_geometry(document(f'<path d="{data}"/>'))
    assert ET.fromstring(cleaned)[0].get("d") == data
    assert stats["paths_after"] == 1


def test_css_document_is_left_untouched():
    svg = document('<style>path {stroke: red}</style><path d="M1 1 L2 2 L3 3"/>')
    cleaned, stats = cleanup_svg_geometry(svg)
    assert cleaned == svg
    assert "skipped_reason" in stats


def test_duplicate_removal_respects_paint_order_and_preserves_degenerate_dot():
    first = '<path fill="red" d="M4 4 L16 4 L16 16 Z"/>'
    middle = '<path fill="blue" d="M4 4 L16 4 L16 16 Z"/>'
    dot = '<path fill="none" stroke="red" stroke-linecap="round" d="M32 32 L32 32"/>'
    svg = document(first + first + middle + first + dot)
    cleaned, stats = cleanup_svg_geometry(svg)
    assert stats["duplicate_paths_removed"] == 1
    assert stats["paths_after"] == 4
    assert "M32 32 L32 32" in cleaned


def test_marker_on_use_preserves_vertices_in_referenced_path():
    svg = document(
        '<defs><path id="line" d="M4 4 L8 4 L16 4"/></defs>'
        '<use href="#line" marker-mid="url(#marker)"/>'
    )
    _, stats = cleanup_svg_geometry(svg)
    assert stats["vertices_removed"] == 0
