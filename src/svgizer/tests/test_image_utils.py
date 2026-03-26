import base64
import io
from textwrap import dedent

import pytest
from PIL import Image

from svgizer.image_utils import (
    downscale_png_bytes,
    generate_diff_data_url,
    png_bytes_to_data_url,
    rasterize_svg_to_png_bytes,
    resize_long_side,
)


def create_test_image(width: int, height: int, color="red") -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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


def test_downscale_png_bytes_skips_if_small():
    png_bytes = create_test_image(100, 100)
    downscaled = downscale_png_bytes(png_bytes, 512)
    assert downscaled == png_bytes


def test_downscale_png_bytes_resizes():
    png_bytes = create_test_image(1000, 1000)
    downscaled = downscale_png_bytes(png_bytes, 512)

    img = Image.open(io.BytesIO(downscaled))
    assert img.size == (512, 512)


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


# ---------------------------------------------------------------------------
# generate_diff_data_url
# ---------------------------------------------------------------------------


def test_diff_data_url_returns_data_url():
    ref = create_test_image(64, 64, color="red")
    cand = create_test_image(64, 64, color="blue")
    result = generate_diff_data_url(ref, cand, long_side=64)
    assert result.startswith("data:image/png;base64,")


def test_diff_data_url_output_is_valid_png():
    ref = create_test_image(64, 64, color="red")
    cand = create_test_image(64, 64, color="blue")
    result = generate_diff_data_url(ref, cand, long_side=64)
    _, b64 = result.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert img.mode == "RGB"


def test_diff_data_url_output_respects_long_side():
    ref = create_test_image(200, 200, color="red")
    cand = create_test_image(200, 200, color="blue")
    result = generate_diff_data_url(ref, cand, long_side=64)
    _, b64 = result.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert max(img.size) <= 64


def test_diff_data_url_identical_images_are_black():
    """Identical images produce a zero-difference map (all black)."""
    ref = create_test_image(64, 64, color="red")
    result = generate_diff_data_url(ref, ref, long_side=64)
    _, b64 = result.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    assert all(p == (0, 0, 0) for p in img.get_flattened_data())


def test_diff_data_url_different_images_are_nonzero():
    """Different images produce a non-zero diff map."""
    ref = create_test_image(64, 64, color="red")
    cand = create_test_image(64, 64, color="blue")
    result = generate_diff_data_url(ref, cand, long_side=64)
    _, b64 = result.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    assert any(p != (0, 0, 0) for p in img.get_flattened_data())


def test_diff_data_url_handles_size_mismatch():
    """Candidate with different size is resized before diffing — no crash."""
    ref = create_test_image(64, 64, color="red")
    cand = create_test_image(128, 128, color="blue")
    result = generate_diff_data_url(ref, cand, long_side=64)
    assert result.startswith("data:image/png;base64,")
