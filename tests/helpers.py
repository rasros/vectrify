"""Shared test helpers."""

import io

from PIL import Image


def make_png(color: str | tuple = "red", size: int | tuple[int, int] = 32) -> bytes:
    """Return PNG bytes of a flat-color RGB image. size is a side or (w, h)."""
    dims = (size, size) if isinstance(size, int) else size
    img = Image.new("RGB", dims, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
