"""The MCP server driven through the SDK's in-memory client, on a file."""

from __future__ import annotations

import anyio
from mcp import Client

from tests.mcp.helpers import SVG, data, error, images, png_size, reference_png
from vectrify.mcp.server import build_server


def test_open_look_edit_undo_and_save(tmp_path):
    drawing = tmp_path / "hills.svg"
    drawing.write_text(SVG)

    async def session():
        async with Client(build_server()) as client:
            assert "describe" in (client.instructions or "")
            guide = await client.read_resource("vectrify://guide")
            assert "## The loop" in str(guide)

            opened = data(await client.call_tool("open", {"path": str(drawing)}))
            assert [o["id"] for o in opened["objects"]] == ["sky", "hill", "sun"]
            assert opened["artboard"] == [0, 0, 400, 200]
            sun = opened["objects"][2]
            assert sun["bounds"] == [300, 20, 40, 40]
            assert sun["paint"] == {"fill": "#f1ba77"}

            rendered = await client.call_tool("render", {"max_side": 200})
            (png,) = images(rendered)
            assert png_size(png) == (200, 100)
            close = await client.call_tool(
                "render", {"region": [300, 20, 40, 40], "max_side": 64}
            )
            assert png_size(images(close)[0]) == (64, 64)

            # A refusal is a tool error carrying the editor's reason.
            refused = await client.call_tool("paint", {"ids": ["sun"]})
            assert "Give a fill" in error(refused)
            # Every edit names its targets; none falls back to the selection.
            for tool, args in [
                ("paint", {"fill": "red"}),
                ("paint", {"ids": [], "fill": "red"}),
                ("delete", {}),
                ("move", {"dx": 1, "dy": 1}),
                ("tidy", {}),
                ("handles", {"points": [], "count": 0}),
            ]:
                assert "validation error" in error(await client.call_tool(tool, args))
            tools = await client.list_tools()
            schemas = {t.name: t.input_schema for t in tools.tools}
            assert "select" not in schemas
            for name in ("paint", "move", "group", "delete", "tidy", "cleanup"):
                assert "ids" in schemas[name]["required"], name

            painted = data(
                await client.call_tool("paint", {"ids": ["sun"], "fill": "#ff0000"})
            )
            assert painted["changed"]
            assert painted["step"] == "Agent: Change paint"
            moved = data(
                await client.call_tool("move", {"ids": ["sun"], "dx": -10, "dy": 5})
            )
            assert moved["step"] == "Agent: Move selection"

            added = data(
                await client.call_tool(
                    "add_path",
                    {"d": "M10 10 L60 10 L35 40 Z", "fill": "#123456", "name": "Bird"},
                )
            )
            # Drawing, painting and naming it are one undo step.
            assert added["step"] == "Agent: Draw path"
            (bird,) = added["created"]
            described = data(await client.call_tool("describe", {}))
            row = next(o for o in described["objects"] if o["id"] == bird)
            assert row["name"] == "Bird"
            assert row["paint"]["fill"] == "#123456"

            nodes = data(await client.call_tool("points", {"id": bird}))
            first = nodes["contours"][0]["nodes"][0]
            assert first["values"] == [10, 10]
            data(
                await client.call_tool(
                    "set_points", {"changes": {bird: {first["id"]: [0, 0]}}}
                )
            )

            history = data(await client.call_tool("history", {}))
            labels = [e["label"] for e in history["undo"]]
            assert labels == [
                "Agent: Move points",
                "Agent: Draw path",
                "Agent: Move selection",
                "Agent: Change paint",
            ]
            assert {e["author"] for e in history["undo"]} == {"agent"}

            undone = data(await client.call_tool("undo", {"steps": 2}))
            assert undone["undone"] == ["Agent: Move points", "Agent: Draw path"]
            after = data(await client.call_tool("history", {}))
            assert [e["label"] for e in after["redo"]] == [
                "Agent: Draw path",
                "Agent: Move points",
            ]
            data(await client.call_tool("redo", {}))

            saved = data(await client.call_tool("save", {}))
            assert saved["saved"] == str(drawing)
            project = tmp_path / "hills.vectrify"
            data(await client.call_tool("save", {"path": str(project)}))
            exported = tmp_path / "out.svg"
            data(await client.call_tool("export_svg", {"path": str(exported)}))

    anyio.run(session)
    svg = drawing.read_text()
    assert "#ff0000" in svg
    assert "#123456" in svg
    assert project_text(tmp_path).startswith('{"vectrify_editor": 1')
    assert (tmp_path / "out.svg").read_text().startswith("<svg")


def project_text(tmp_path) -> str:
    return (tmp_path / "hills.vectrify").read_text()


def test_an_edit_after_someone_else_changed_the_drawing_is_refused(tmp_path):
    drawing = tmp_path / "hills.svg"
    drawing.write_text(SVG)
    from vectrify.mcp.server import Vectrify
    from vectrify.mcp.target import FileTarget

    state = Vectrify()

    async def session():
        async with Client(build_server(state)) as client:
            data(await client.call_tool("open", {"path": str(drawing)}))
            # A person's edit the agent has not looked at.
            target = state.target
            assert isinstance(target, FileTarget)
            target_session = target.session
            target_session.action(
                {
                    "command": "select",
                    "objects": ["hill"],
                    "epoch": target_session.epoch,
                    "revision": 0,
                }
            )
            target_session.action(
                {
                    "command": "delete",
                    "epoch": target_session.epoch,
                    "revision": 0,
                }
            )
            stale = await client.call_tool("paint", {"ids": ["sun"], "fill": "red"})
            assert "describe()" in error(stale)
            described = data(await client.call_tool("describe", {}))
            assert "hill" not in [o["id"] for o in described["objects"]]
            history = data(await client.call_tool("history", {}))
            assert history["undo"][0] == {
                "label": "Delete selection",
                "author": "person",
                "revision": 1,
            }
            # Having looked, the agent may edit again.
            data(await client.call_tool("paint", {"ids": ["sun"], "fill": "red"}))

    anyio.run(session)


def test_reference_compare_and_a_cel_trace_applied(tmp_path):
    drawing = tmp_path / "blank.svg"
    drawing.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" '
        'viewBox="0 0 400 200"/>'
    )
    picture = tmp_path / "reference.png"
    picture.write_bytes(reference_png())

    async def session():
        async with Client(build_server()) as client:
            data(await client.call_tool("open", {"path": str(drawing)}))
            assert "load_reference" in error(await client.call_tool("reference", {}))
            loaded = data(
                await client.call_tool("load_reference", {"path": str(picture)})
            )
            assert loaded["step"] == "Load reference"
            reference = await client.call_tool("reference", {"max_side": 100})
            assert png_size(images(reference)[0]) == (100, 50)
            side = await client.call_tool(
                "render", {"overlay": "side", "max_side": 400}
            )
            assert png_size(images(side)[0]) == (404, 100)
            before = data(await client.call_tool("compare", {}))
            assert before["mse"] > 0.05
            assert len(before["worst_cells"]) == 4

            job = data(
                await client.call_tool(
                    "generate", {"method": "cel", "settings": {"regions": 3}}
                )
            )
            status = await client.call_tool(
                "job_status", {"id": job["id"], "wait_seconds": 120}
            )
            ready = data(status)
            assert ready["status"] == "ready", ready
            assert ready["previews"] == ["reference", "before", "after"]
            assert len(images(status)) == 3
            applied = data(await client.call_tool("apply", {"id": job["id"]}))
            assert applied["step"] == "Agent: Generate cel trace"
            after = data(await client.call_tool("compare", {}))
            assert after["mse"] < before["mse"] / 4

            # A worse edit, seen in compare, is taken back by undo.
            data(await client.call_tool("undo", {}))
            assert data(await client.call_tool("compare", {}))["mse"] == before["mse"]

    anyio.run(session)


def test_without_a_target_the_agent_is_told_how_to_get_one():
    async def session():
        async with Client(build_server()) as client:
            message = error(await client.call_tool("describe", {}))
            assert "open(path)" in message
            assert "connect()" in message

    anyio.run(session)
