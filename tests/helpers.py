"""Shared test helpers."""

import io

from PIL import Image

# Model used by live-integration (llm-marked) tests.
TEST_MODEL = "gpt-5.4-nano"


def make_png(color: str | tuple = "red", size: int | tuple[int, int] = 32) -> bytes:
    """Return PNG bytes of a flat-color RGB image. size is a side or (w, h)."""
    dims = (size, size) if isinstance(size, int) else size
    img = Image.new("RGB", dims, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def text_blocks(blocks: list[dict]) -> list[str]:
    """Extract text content from LLM prompt blocks."""
    return [b["text"] for b in blocks if b.get("type") == "input_text"]


def image_urls(blocks: list[dict]) -> list[str]:
    """Extract image URLs from LLM prompt blocks."""
    return [b["image_url"] for b in blocks if b.get("type") == "input_image"]
