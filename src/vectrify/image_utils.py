import base64
import io

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
