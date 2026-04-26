import base64
import io

import cairosvg
from PIL import Image, ImageChops
from PIL.Image import Resampling

DIFF_BRIGHTNESS_BOOST = 3


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


def downscale_png_bytes(png_bytes: bytes, long_side: int) -> bytes:
    if long_side <= 0:
        return png_bytes

    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = im.size
    if max(w, h) <= long_side:
        return png_bytes

    im2 = resize_long_side(im, long_side)
    out = io.BytesIO()
    im2.save(out, format="PNG")
    return out.getvalue()


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

    img = Image.open(io.BytesIO(raw_png)).convert("RGBA")

    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    combined = Image.alpha_composite(bg, img).convert("RGB")

    out = io.BytesIO()
    combined.save(out, format="PNG")
    return out.getvalue()


def make_preview_data_url(full_png: bytes, image_long_side: int) -> str:
    preview_png = downscale_png_bytes(full_png, image_long_side)
    return png_bytes_to_data_url(preview_png)


def pixel_diff_png(ref_img: Image.Image, cand_png: bytes, long_side: int) -> bytes:
    """Pixel-wise RGB difference with brightness boost, returned as PNG bytes."""
    cand = Image.open(io.BytesIO(cand_png)).convert("RGB")
    if cand.size != ref_img.size:
        cand = cand.resize(ref_img.size, resample=Resampling.BILINEAR)
    diff = ImageChops.difference(ref_img, cand)
    lut = [min(255, i * DIFF_BRIGHTNESS_BOOST) for i in range(256)]
    diff = diff.point(lut * len(diff.getbands()))
    diff = resize_long_side(diff, long_side)
    buf = io.BytesIO()
    diff.save(buf, format="PNG")
    return buf.getvalue()


def generate_diff_data_url(ref_bytes: bytes, cand_bytes: bytes, long_side: int) -> str:
    """Generates a high-contrast diff map between the reference and the candidate."""
    ref_img = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
    return png_bytes_to_data_url(pixel_diff_png(ref_img, cand_bytes, long_side))
