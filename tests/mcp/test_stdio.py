"""The installed vectrify-mcp entry point, spoken to over stdio."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anyio
import pytest
from mcp import Client, StdioServerParameters

from tests.mcp.helpers import SVG, data, images, png_size

ENTRY = Path(sys.executable).with_name("vectrify-mcp")


@pytest.mark.skipif(not ENTRY.exists(), reason="vectrify-mcp is not installed")
def test_the_entry_point_serves_over_stdio(tmp_path, state_home):
    drawing = tmp_path / "hills.svg"
    drawing.write_text(SVG)
    parameters = StdioServerParameters(
        command=str(ENTRY),
        args=[str(drawing)],
        env={**os.environ, "XDG_STATE_HOME": str(state_home)},
    )

    async def run():
        async with Client(parameters) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert {"describe", "render", "properties", "generate", "undo"} <= names
            described = data(await client.call_tool("describe", {}))
            assert described["target"] == f"the file {drawing}"
            rendered = await client.call_tool("render", {"max_side": 100})
            assert png_size(images(rendered)[0]) == (100, 50)
            data(
                await client.call_tool(
                    "properties", {"ids": ["sun"], "fill": "#abcdef"}
                )
            )
            data(await client.call_tool("save", {}))

    anyio.run(run)
    assert "#abcdef" in drawing.read_text()
