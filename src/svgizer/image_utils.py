import base64
import io

import cairosvg
from PIL import Image

from svgizer.diff.utils import resize_long_side


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
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"Invalid raster target size: {out_w}x{out_h}")
    res = cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        output_width=out_w,
        output_height=out_h,
    )
    assert isinstance(res, bytes)
    return res


def make_preview_data_url(full_png: bytes, openai_image_long_side: int) -> str:
    preview_png = downscale_png_bytes(full_png, openai_image_long_side)
    return png_bytes_to_data_url(preview_png)
