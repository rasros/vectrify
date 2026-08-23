import base64
import io

import cairosvg
import numpy as np
from PIL import Image, ImageChops
from PIL.Image import Resampling

DIFF_BRIGHTNESS_BOOST = 3
BACKGROUND_TOLERANCE = 16
BACKGROUND_EDGE_COVERAGE = 0.9
BACKGROUND_PADDING_RATIO = 0.04
BACKGROUND_PADDING_MIN_PIXELS = 2


def crop_single_color_background(
    image: Image.Image,
    *,
    tolerance: int = BACKGROUND_TOLERANCE,
) -> Image.Image:
    """Trim a nearly solid background which encloses the whole canvas.

    The corners must agree on the background colour and that colour must cover
    nearly all of the canvas edge. This avoids cropping photos and gradients.
    A small proportional margin is retained around the detected foreground,
    without ever enlarging the original canvas.
    """
    if image.width < 2 or image.height < 2:
        return image

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    corners = np.array(
        [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], dtype=np.int16
    )
    background = np.median(corners, axis=0)
    if np.max(np.abs(corners - background)) > tolerance:
        return image

    edge = np.concatenate((rgb[0], rgb[-1], rgb[1:-1, 0], rgb[1:-1, -1]))
    edge_matches = np.max(np.abs(edge - background), axis=1) <= tolerance
    if float(edge_matches.mean()) < BACKGROUND_EDGE_COVERAGE:
        return image

    foreground = np.argwhere(np.max(np.abs(rgb - background), axis=2) > tolerance)
    if foreground.size == 0:
        return image

    top, left = foreground.min(axis=0)
    bottom, right = foreground.max(axis=0)
    if (
        top == 0
        and left == 0
        and bottom == image.height - 1
        and right == image.width - 1
    ):
        return image
    padding = max(
        BACKGROUND_PADDING_MIN_PIXELS,
        round(min(image.size) * BACKGROUND_PADDING_RATIO),
    )
    top = max(0, top - padding)
    left = max(0, left - padding)
    bottom = min(image.height - 1, bottom + padding)
    right = min(image.width - 1, right + padding)
    return image.crop((int(left), int(top), int(right) + 1, int(bottom) + 1))


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


def png_resize_exact(png_bytes: bytes, out_w: int, out_h: int) -> bytes:
    """Re-encode *png_bytes* as RGB at exactly out_w x out_h."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img = img.resize((out_w, out_h), Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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


def make_preview_data_url(full_png: bytes, resolution_llm: int) -> str:
    preview_png = downscale_png_bytes(full_png, resolution_llm)
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
