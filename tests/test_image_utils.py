from PIL import Image

from vectrify.image_utils import (
    png_bytes_to_data_url,
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
