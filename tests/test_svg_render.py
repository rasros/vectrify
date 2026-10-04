"""Region rendering preserves the viewport and preview overlay conventions."""

import base64
import io
import xml.etree.ElementTree as ET

from PIL import Image

from vectrify.document import import_svg
from vectrify.document.topology import EdgeRef
from vectrify.operations.previews import render_previews
from vectrify.svg_render import render_png


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
