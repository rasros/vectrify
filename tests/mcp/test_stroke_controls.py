"""Public MCP schemas and calls expose the same stroke settings as the UI."""

import asyncio

import pytest
from mcp import Client

from tests.mcp.helpers import data, error
from vectrify.document import HitIndex, import_svg
from vectrify.document.taper import END, END_WIDTH, OWNER, START, START_WIDTH
from vectrify.mcp.server import build_server

SVG = (
    '<svg width="100" height="100"><path id="p" d="M10 50 L90 50"'
    ' stroke="red" stroke-width="10" fill="none"/></svg>'
)


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
                "start_taper_width",
                "end_taper_width",
            }
            for field in ("start_taper", "end_taper"):
                number = schema["properties"][field]["anyOf"][0]
                assert number == {"type": "number", "minimum": 0}
            for field in ("start_taper_width", "end_taper_width"):
                number = schema["properties"][field]["anyOf"][0]
                assert number == {"type": "number", "minimum": 0, "maximum": 100}
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
                        "start_taper_width": 50,
                        "end_taper_width": 25,
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
            assert float(p["paint"]["data-vectrify-taper-start-width"]) == 50
            assert float(p["paint"]["data-vectrify-taper-end-width"]) == 25

    asyncio.run(run())


def test_mcp_tapers_keep_geometry_undo_disable_and_save(tmp_path):
    drawing = tmp_path / "line.svg"
    drawing.write_text(SVG)

    async def run():
        async with Client(build_server()) as client:

            async def call(name, **args):
                return data(await client.call_tool(name, args))

            await call("open", path=str(drawing))
            points = (await call("points", id="p"))["contours"]
            edited = await call(
                "properties",
                ids=["p"],
                start_taper=20,
                end_taper=30,
                start_taper_width=50,
                end_taper_width=20,
            )
            assert (await call("points", id="p"))["contours"] == points
            history = await call("history")
            assert len(history["undo"]) == 1
            assert history["undo"][0]["id"] == edited["edit_id"]

            svg = (await call("get_svg"))["svg"]
            assert HitIndex(import_svg(svg)).painted_area("p") == pytest.approx(630)
            for suffix in ("svg", "vectrify"):
                saved = tmp_path / f"tapered.{suffix}"
                await call("save", path=str(saved))
                [p] = (await call("open", path=str(saved)))["objects"]
                assert {
                    key: float(p["paint"][key])
                    for key in (START, END, START_WIDTH, END_WIDTH)
                } == {START: 20, END: 30, START_WIDTH: 50, END_WIDTH: 20}

            await call("properties", ids=["p"], start_taper=12.5)
            [p] = (await call("describe"))["objects"]
            assert float(p["paint"][START]) == 12.5
            assert float(p["paint"][END]) == 30
            assert float(p["paint"][START_WIDTH]) == 50
            assert float(p["paint"][END_WIDTH]) == 20

            full = await call(
                "properties", ids=["p"], start_taper_width=100, end_taper_width=100
            )
            assert OWNER not in (await call("get_svg"))["svg"]
            await call("undo", ids=[full["edit_id"]])
            assert OWNER in (await call("get_svg"))["svg"]
            disabled = await call("properties", ids=["p"], start_taper=0, end_taper=0)
            assert OWNER not in (await call("get_svg"))["svg"]
            await call("undo", ids=[disabled["edit_id"]])
            assert OWNER in (await call("get_svg"))["svg"]
            await call("redo", ids=[disabled["edit_id"]])
            assert OWNER not in (await call("get_svg"))["svg"]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("field", "value"),
    [(field, -1) for field in ("start_taper", "end_taper")]
    + [
        (field, value)
        for field in ("start_taper_width", "end_taper_width")
        for value in (-1, 101)
    ]
    + [
        (field, value)
        for field in (
            "start_taper",
            "end_taper",
            "start_taper_width",
            "end_taper_width",
        )
        # The SDK serializes nonfinite floats as null (an omitted setting).
        # String forms reach numeric validation instead.
        for value in ("NaN", "Infinity")
    ],
)
def test_mcp_invalid_tapers_refuse_the_whole_edit(tmp_path, field, value):
    drawing = tmp_path / "line.svg"
    drawing.write_text(SVG)

    async def run():
        async with Client(build_server()) as client:
            data(await client.call_tool("open", {"path": str(drawing)}))
            before = data(await client.call_tool("get_svg", {}))
            refused = await client.call_tool(
                "properties", {"ids": ["p"], "stroke": "blue", field: value}
            )
            assert "validation error" in error(refused)
            assert data(await client.call_tool("get_svg", {})) == before
            assert data(await client.call_tool("history", {}))["undo"] == []

    asyncio.run(run())


@pytest.mark.parametrize("lock", ["paint", START, END, START_WIDTH, END_WIDTH])
def test_mcp_tapers_respect_paint_and_property_locks(tmp_path, lock):
    drawing = tmp_path / "line.svg"
    drawing.write_text(SVG)

    async def run():
        async with Client(build_server()) as client:
            data(await client.call_tool("open", {"path": str(drawing)}))
            data(await client.call_tool("properties", {"ids": ["p"], "locks": [lock]}))
            before = data(await client.call_tool("get_svg", {}))
            refused = await client.call_tool(
                "properties",
                {
                    "ids": ["p"],
                    "start_taper": 20,
                    "end_taper": 30,
                    "start_taper_width": 50,
                    "end_taper_width": 20,
                },
            )
            assert "locked" in error(refused)
            assert data(await client.call_tool("get_svg", {})) == before
            assert len(data(await client.call_tool("history", {}))["undo"]) == 1

    asyncio.run(run())
