import io
from textwrap import dedent

import pytest
from PIL import Image

from vectrify.image_utils import (
    png_bytes_to_data_url,
    rasterize_svg_to_png_bytes,
    resize_long_side,
)


def test_resize_long_side_landscape():
    img = Image.new("RGB", (1000, 500))
    resized = resize_long_side(img, 500)
    assert resized.size == (500, 250)


def test_resize_long_side_portrait():
    img = Image.new("RGB", (500, 1000))
    resized = resize_long_side(img, 500)
    assert resized.size == (250, 500)


def test_resize_long_side_square_already_small():
    img = Image.new("RGB", (256, 256))
    resized = resize_long_side(img, 512)
    assert resized.size == (256, 256)


def test_png_bytes_to_data_url():
    png_bytes = b"fake_png_data"
    data_url = png_bytes_to_data_url(png_bytes)
    assert data_url.startswith("data:image/png;base64,")
    assert "ZmFrZV9wbmdfZGF0YQ==" in data_url


def test_rasterize_svg_to_png_bytes_valid():
    svg = dedent(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
        <rect width="10" height="10" fill="red"/>
        </svg>
        """
    )
    png_bytes = rasterize_svg_to_png_bytes(svg, out_w=100, out_h=100)
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (100, 100)
    assert img.mode == "RGB"


def test_rasterize_svg_to_png_bytes_invalid_dimensions():
    svg = "<svg></svg>"
    with pytest.raises(ValueError, match="Invalid raster target size"):
        rasterize_svg_to_png_bytes(svg, out_w=-10, out_h=100)
