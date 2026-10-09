"""Shared test helpers."""

import io
from typing import TypeVar

import cairosvg
from PIL import Image

from vectrify.image_utils import on_white, png_bytes


def make_png(color: str | tuple = "red", size: int | tuple[int, int] = 32) -> bytes:
    """Return PNG bytes of a flat-color RGB image. size is a side or (w, h)."""
    dims = (size, size) if isinstance(size, int) else size
    img = Image.new("RGB", dims, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def rasterize_svg_to_png_bytes(svg_text: str, *, out_w: int, out_h: int) -> bytes:
    """The PNG render of *svg_text* at out_w x out_h, over white."""
    raw = cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"), output_width=out_w, output_height=out_h
    )
    assert raw is not None
    return png_bytes(on_white(Image.open(io.BytesIO(raw))))


_T = TypeVar("_T")


def required(value: _T | None) -> _T:
    """Assert an optional fixture result exists before testing its contents."""
    assert value is not None
    return value
