import base64
import io

import cairosvg
from PIL import Image
from PIL.Image import Resampling


def resize_long_side(im: Image.Image, long_side: int) -> Image.Image:
    w, h = im.size
    if max(w, h) <= long_side:
        return im
    if w >= h:
        new_w = long_side
        new_h = round(h * (long_side / float(w)))
    else:
        new_h = long_side
        new_w = round(w * (long_side / float(h)))
    return im.resize((max(1, new_w), max(1, new_h)), resample=Resampling.BILINEAR)


def png_bytes_to_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def png_url(image: Image.Image) -> str:
    return png_bytes_to_data_url(png_bytes(image))


def preview_urls(
    reference: Image.Image, before: Image.Image, after: Image.Image
) -> dict[str, str]:
    """A proposal's previews: the reference and the region before and after."""
    return {
        "reference": png_url(reference),
        "before": png_url(before),
        "after": png_url(after),
    }


def on_white(image: Image.Image) -> Image.Image:
    """*image* composited over white, so transparency never reads as black."""
    return Image.alpha_composite(
        Image.new("RGBA", image.size, "white"), image.convert("RGBA")
    ).convert("RGB")


def rasterize_svg_to_png_bytes(svg_text: str, *, out_w: int, out_h: int) -> bytes:
    """
    Rasterizes SVG to PNG and composites it over a white background
    to prevent transparency being treated as black borders/edges.
    """
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"Invalid raster target size: {out_w}x{out_h}")

    raw_png = cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        output_width=out_w,
        output_height=out_h,
    )
    if raw_png is None:
        raise ValueError(f"Failed to rasterize SVG to PNG: {svg_text}")

    return png_bytes(on_white(Image.open(io.BytesIO(raw_png))))


def rasterize_svg_to_image(svg_text: str, *, out_w: int, out_h: int) -> Image.Image:
    """The white-backed RGB render of *svg_text*, as an image."""
    png = rasterize_svg_to_png_bytes(svg_text, out_w=out_w, out_h=out_h)
    with Image.open(io.BytesIO(png)) as image:
        return image.convert("RGB")


def rasterize_svg(svg_text: str, width: int, height: int) -> bytes:
    """Positional form, for APIs that take a (svg, width, height) rasterizer."""
    return rasterize_svg_to_png_bytes(svg_text, out_w=width, out_h=height)
