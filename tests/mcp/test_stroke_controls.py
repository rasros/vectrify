"""Public MCP schemas and calls expose the same stroke settings as the UI."""

import asyncio

from mcp import Client

from tests.mcp.helpers import data
from vectrify.mcp.server import build_server


def test_mcp_stroke_styles_and_schema(tmp_path):
    drawing = tmp_path / "line.svg"
    drawing.write_text(
        '<svg width="100" height="100"><path id="p" d="M10 50 L90 50"'
        ' stroke="red" fill="none"/></svg>'
    )

    async def run():
        async with Client(build_server()) as client:
            schema = next(
                t.input_schema
                for t in (await client.list_tools()).tools
                if t.name == "properties"
            )
            assert set(schema["properties"]) >= {
                "stroke_linecap",
                "stroke_linejoin",
                "stroke_miterlimit",
                "start_arrow",
                "end_arrow",
                "start_taper",
                "end_taper",
            }
            data(await client.call_tool("open", {"path": str(drawing)}))
            reply = data(
                await client.call_tool(
                    "properties",
                    {
                        "ids": ["p"],
                        "stroke_linecap": "round",
                        "stroke_linejoin": "bevel",
                        "stroke_miterlimit": 3,
                        "end_arrow": "arrow",
                        "start_taper": 10,
                        "end_taper": 15,
                    },
                )
            )
            assert reply["changed"]
            described = data(await client.call_tool("describe", {}))
            [p] = described["objects"]
            assert p["paint"]["stroke-linecap"] == "round"
            svg = data(await client.call_tool("get_svg", {}))["svg"]
            assert 'data-vectrify-taper-owner="p"' in svg
            assert float(p["paint"]["data-vectrify-taper-start"]) == 10

    asyncio.run(run())
