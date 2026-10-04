"""Driving the MCP server the way a client does, through the SDK."""

from __future__ import annotations

import base64
import json
import socket
from typing import Any

import numpy as np
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image

SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" \
viewBox="0 0 400 200">
<rect id="sky" width="400" height="200" fill="#e8ebe0"/>
<path id="hill" d="M0 150 L120 60 L260 140 L400 90 L400 200 L0 200 Z" \
fill="#61827a"/>
<path id="sun" d="M300 20 L340 20 L340 60 L300 60 Z" fill="#f1ba77"/>
</svg>
"""


def reference_png() -> bytes:
    """Two flat regions and their black outline, as the cel tests use."""
    pixels = np.full((200, 400, 3), 255, dtype=np.uint8)
    pixels[40:160, 40:200] = (220, 60, 50)
    pixels[40:160, 200:360] = (60, 90, 210)
    pixels[38:42, 38:362] = pixels[158:162, 38:362] = 20
    pixels[38:162, 38:42] = pixels[38:162, 358:362] = pixels[38:162, 198:202] = 20
    image = Image.fromarray(pixels)
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def data(result: CallToolResult) -> dict[str, Any]:
    assert not result.is_error, result.content
    first = result.content[0]
    assert isinstance(first, TextContent)
    return json.loads(first.text)


def error(result: CallToolResult) -> str:
    assert result.is_error, result.content
    first = result.content[0]
    assert isinstance(first, TextContent)
    return first.text


def images(result: CallToolResult) -> list[bytes]:
    return [
        base64.b64decode(c.data) for c in result.content if isinstance(c, ImageContent)
    ]


def png_size(png: bytes) -> tuple[int, int]:
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    from io import BytesIO

    with Image.open(BytesIO(png)) as image:
        return image.size


def free_port() -> int:
    """A port nothing listens on now, so tests never take a running editor's."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def restore(call, command: str = "undo", count: int = 1):
    """Resolve exact history IDs before a test restores those changes."""
    history = data(await call("history", limit=max(30, count)))
    ids = [entry["id"] for entry in history[command][:count]]
    return await call(command, ids=ids)
