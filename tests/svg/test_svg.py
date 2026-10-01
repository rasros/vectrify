import io

from PIL import Image

from vectrify.image_utils import rasterize_svg_to_png_bytes

NS = "http://www.w3.org/2000/svg"
SVG = f'<svg xmlns="{NS}" viewBox="0 0 32 32"><rect width="32" height="32"/></svg>'


def test_rasterize_renders_at_the_requested_size():
    png = rasterize_svg_to_png_bytes(SVG, out_w=24, out_h=16)
    assert Image.open(io.BytesIO(png)).size == (24, 16)
