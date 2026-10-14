"""Region rendering preserves the viewport and preview overlay conventions."""

import base64
import io
import xml.etree.ElementTree as ET

import cairosvg
import numpy as np
import pytest
from PIL import Image

from vectrify.document import import_svg
from vectrify.document import svg as document_svg
from vectrify.document.topology import EdgeRef
from vectrify.operations.previews import render_previews
from vectrify.svg_render import (
    _clip_boundary,
    cached_rendering,
    render_image,
    render_png,
)


def test_clip_unions_reuse_unchanged_boundaries_and_invalidate_geometry(monkeypatch):
    _clip_boundary.cache_clear()
    parsed = []
    original_parse = document_svg.parse_path

    def parse(data):
        parsed.append(data)
        return original_parse(data)

    monkeypatch.setattr(document_svg, "parse_path", parse)
    svg = """<svg width="20" height="20"><defs>
    <clipPath id="clip" data-vectrify-clip-owner="shape"><path clip-rule="nonzero"
      d="M0 0 L20 0 L20 20 L0 20 Z M4 4 L16 4 L16 16 L4 16 Z"/>
    </clipPath></defs><rect width="20" height="20" fill="blue"
      clip-path="url(#clip)"/></svg>"""
    try:
        assert render_image(svg).getpixel((10, 10)) == (0, 0, 255)
        # Paint changes and viewport changes reuse the same normalized union.
        red = render_image(svg.replace('fill="blue"', 'fill="red"'))
        assert red.getpixel((10, 10)) == (255, 0, 0)
        render_png(svg, (0, 0, 20, 20), (40, 40))
        assert len(parsed) == 1
        hole = render_image(svg.replace('clip-rule="nonzero"', 'clip-rule="evenodd"'))
        assert hole.getpixel((10, 10)) == (255, 255, 255)
        shifted = render_image(
            svg.replace(
                "<path clip-rule=", '<path transform="translate(20 0)" clip-rule='
            )
        )
        assert shifted.getpixel((10, 10)) == (255, 255, 255)
        changed = render_image(
            svg.replace("L20 0 L20 20 L0 20", "L10 0 L10 10 L0 10").replace(
                " M4 4 L16 4 L16 16 L4 16 Z", ""
            )
        )
        assert changed.getpixel((5, 5)) == (0, 0, 255)
        assert changed.getpixel((15, 15)) == (255, 255, 255)
        assert len(parsed) == 4
    finally:
        _clip_boundary.cache_clear()


def test_direct_rgba_render_matches_cairo_with_group_opacity_and_clipping():
    svg = """<svg width="40" height="40">
    <defs><clipPath id="clip"><path d="M4 4 L36 4 L4 36 Z"/></clipPath></defs>
    <g opacity="0.6" clip-path="url(#clip)">
      <path fill="#fa8072" d="M3.2 3.7 L35.6 3.7 L35.6 35.4 L3.2 35.4 Z"/>
      <path fill="blue" fill-opacity="0.5" d="M10 10 L30 10 L30 30 L10 30 Z"/>
    </g></svg>"""
    png = cairosvg.svg2png(bytestring=svg.encode())
    assert png is not None
    expected = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    with cached_rendering():
        actual = np.asarray(render_image(svg, alpha=True))
    assert np.array_equal(actual[:, :, 3], expected[:, :, 3])
    # Straight colour can differ by one level when undoing premultiplication.
    assert np.max(abs(actual.astype(int) - expected.astype(int))) <= 1
    assert actual[39, 39].tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize("box", [(0, 0, 80, 64), (12.5, 8.5, 40, 32)])
@pytest.mark.parametrize("size", [(80, 64), (123, 77)])
def test_cached_pixels_match_cairo_with_clips_paints_markers_and_changed_geometry(
    box, size
):
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="80" height="64">
    <defs>
      <clipPath id="clip"><path d="M5 5 L65 5 L65 55 Z"/>
        <path d="M0 40 L30 40 L30 64 L0 64 Z"/></clipPath>
      <linearGradient id="ramp"><stop stop-color="red"/>
        <stop offset="1" stop-color="blue"/></linearGradient>
      <marker id="dot" markerWidth="4" markerHeight="4" refX="2" refY="2">
        <circle cx="2" cy="2" r="2" fill="green"/></marker>
    </defs>
    <g opacity="0.6" transform="translate(3 2) rotate(7)" clip-path="url(#clip)">
      <path id="fill" d="M8 8 C50 2 70 20 62 40 L8 50 Z M20 20 L30 20 L30 30 Z"
        fill="#d06040" fill-rule="evenodd"/>
      <path d="M12 36 C16 4 48 52 56 12 A8 5 30 0 1 66 24" fill="none" stroke="#102030"
        stroke-width="3" stroke-linecap="round" stroke-dasharray="3 2"/>
      <path d="M30 10 L62 10 L62 42 Z" fill="url(#ramp)"/>
      <path d="M10 40 L44 40" stroke="black" marker="url(#dot)"/>
    </g></svg>"""
    with cached_rendering():
        # Reuse the same paths, then change paint, geometry and the viewport.
        for changed in (
            svg,
            svg,
            svg.replace("50 2", "42 12"),
            svg.replace("#d06040", "#2030d0"),
        ):
            expected = Image.open(io.BytesIO(render_png(changed, box, size)))
            actual = render_image(changed, box, size)
            assert np.array_equal(np.asarray(actual), np.asarray(expected))
            expected.close()


def test_crop_stretches_to_the_requested_pixels_with_a_white_background():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" '
        'viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">'
        '<rect x="10" y="20" width="10" height="10" fill="red"/></svg>'
    )
    with Image.open(io.BytesIO(render_png(svg, (10, 20, 20, 10), (60, 80)))) as image:
        assert image.size == (60, 80)
        assert image.getpixel((10, 40)) == (255, 0, 0)
        assert image.getpixel((50, 40)) == (255, 255, 255)


def test_overlay_is_drawn_in_the_cropped_user_space():
    def overlay(root):
        ET.SubElement(
            root,
            "{http://www.w3.org/2000/svg}rect",
            {"x": "15", "y": "25", "width": "5", "height": "5", "fill": "blue"},
        )

    png = render_png(
        '<svg xmlns="http://www.w3.org/2000/svg"/>',
        (10, 20, 20, 10),
        (100, 100),
        overlay=overlay,
    )
    with Image.open(io.BytesIO(png)) as image:
        assert image.getpixel((30, 70)) == (0, 0, 255)
        assert image.getpixel((70, 70)) == (255, 255, 255)


def test_preview_highlights_only_the_after_image_in_root_coordinates():
    source = (
        '<svg width="20" height="20"><path d="M0 5 L10 5" '
        'fill="none" stroke="black" stroke-width="1"/></svg>'
    )
    before, after = import_svg(source), import_svg(source)
    geometry = after.geometries[0]
    ref = EdgeRef(
        geometry.id, geometry.subpaths[0].nodes[-1].id, matrix=(1, 0, 0, 1, 5, 5)
    )
    previews = render_previews(before, after, (0, 0, 20, 20), highlight=(ref,))
    images = {
        name: Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        for name, url in previews.items()
    }
    try:
        assert images["before"].getpixel((450, 449)) == (255, 255, 255)
        assert images["after"].getpixel((450, 449)) == (0, 200, 255)
        assert images["after"].getpixel((100, 449)) == (255, 255, 255)
    finally:
        for image in images.values():
            image.close()
