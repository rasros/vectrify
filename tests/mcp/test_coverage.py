"""Every editor command is reachable from some MCP tool, or left out on purpose.

The commands are read from ``Session``'s dispatch registries, and the test records which
of them actually arrive at ``Session.action`` (and ``Session.operation``)
while an MCP client calls the tools. A command added to the session without
a tool, or a tool that stops sending its command, fails here. There is no
select tool: "select" arrives as every targeted tool's first step, the agent
choosing its own targets before the person's selection is given back.
"""

from __future__ import annotations

import anyio
import pytest
from mcp import Client

from tests.mcp.helpers import data, reference_png, restore
from vectrify.mcp.server import build_server
from vectrify.ui.agent import EDITS, LEFT_OUT
from vectrify.ui.session import ACTION_COMMANDS, OPERATION_COMMANDS, Session

SVG = """<svg xmlns="http://www.w3.org/2000/svg" \
xmlns:xlink="http://www.w3.org/1999/xlink" \
width="200" height="200" viewBox="0 0 200 200">
<defs><path id="shape" d="M0 0 L10 0 L10 10 Z"/></defs>
<path id="a" d="M10 10 L60 10 L60 60 L10 60 Z" fill="red"/>
<path id="side" d="M60 10 L80 10 L80 30 L60 30 Z" fill="orange"/>
<path id="b" d="M40 40 L90 40 L90 90 L40 90 Z" fill="blue"/>
<path id="ring" d="M100 100 L190 100 L190 190 L100 190 Z \
M120 120 L120 170 L170 170 L170 120 Z" fill="green"/>
<path id="line" d="M10 150 L50 150 L90 160" fill="none" stroke="black" \
stroke-width="3"/>
<path id="line2" d="M93 160 L120 150" fill="none" stroke="black" stroke-width="3"/>
<g id="grp"><path id="c" d="M150 10 L190 10 L190 50 Z" fill="#888"/></g>
<use id="inst" xlink:href="#shape" x="5" y="180"/>
</svg>
"""


def test_the_session_commands_are_found():
    # The inventory used here is the inventory dispatch actually uses.
    assert {"paint", "knife", "join_two_ends", "convert_lines", "undo"} <= (
        ACTION_COMMANDS.keys()
    )
    assert {"start", "apply", "discard", "status", "stop"} == OPERATION_COMMANDS.keys()
    assert all(
        callable(getattr(Session, spec.handler)) for spec in ACTION_COMMANDS.values()
    )
    assert all(
        callable(getattr(Session, handler)) for handler in OPERATION_COMMANDS.values()
    )


def test_left_out_commands_exist_and_say_why():
    assert set(LEFT_OUT) <= ACTION_COMMANDS.keys()
    assert all(len(why) > 20 for why in LEFT_OUT.values())


def test_every_command_arrives_from_some_tool(tmp_path, monkeypatch):
    drawing = tmp_path / "shapes.svg"
    drawing.write_text(SVG)
    picture = tmp_path / "reference.png"
    picture.write_bytes(reference_png())
    actions: set[str] = set()
    operations: set[str] = set()
    action, operation = Session.action, Session.operation

    def record_action(self, payload):
        actions.add(payload["command"])
        return action(self, payload)

    def record_operation(self, payload):
        operations.add(payload["command"])
        return operation(self, payload)

    monkeypatch.setattr(Session, "action", record_action)
    monkeypatch.setattr(Session, "operation", record_operation)

    async def session():
        async with Client(build_server()) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            # Each editing call of the agent is an MCP tool of that name.
            assert set(EDITS) - {"set_reference"} <= tools
            assert "load_reference" in tools

            async def call(tool_name: str, /, **args):
                return await client.call_tool(tool_name, args)

            data(await call("open", path=str(drawing)))

            async def nodes(oid: str) -> list[str]:
                contours = data(await call("points", id=oid))["contours"]
                return [n["id"] for c in contours for n in c["nodes"]]

            line, ring = await nodes("line"), await nodes("ring")
            await call("redraw_outline", id="b", points=[[40, 40], [65, 30], [90, 40]])
            await call("join", points=[["line", line[0]], ["line", line[-1]]])
            await call("convert", ids=["line2"], to="fill")
            await call(
                "properties", ids=["a"], fill="#ff8800", name="Square", locks=["paint"]
            )
            await call("properties", ids=["a"], locks=[])
            await call("transform", ids=["a"], dx=1, dy=1)
            await call("transform", ids=["a"], box=[0, 0, 30, 30])
            await call("arrange", ids=["a"], to="front")
            await call("arrange", ids=["a"], to="backward")
            await call("arrange", ids=["a"], parent="grp", index=0)
            grouped = data(await call("group", ids=["line", "line2"]))
            data(
                await call("group", ids=grouped["result"]["objects"], action="dissolve")
            )
            await call("join", ids=["a", "b"])
            await call("split_parts", ids=["ring"])
            await call("cut_hole", ids=["ring", "c"])
            listed = data(await call("points", id="ring", nodes=False))["contours"]
            holes = [c["id"] for c in listed if c["hole"]]
            contours = [["ring", h] for h in holes] or [["ring", "x"]]
            await call("holes", contours=contours, action="fill")
            await call("holes", contours=contours, action="shape")
            await call("convert", ids=["inst"], to="path")
            await call("convert", ids=["ring"], to="line")
            await call("convert", ids=["ring"], to="either")
            await call("add_path", d="M0 0 L20 0 L20 20 Z", fill="#000")
            await call("set_points", changes={"ring": {ring[1]: [180, 100]}})
            await call(
                "point_style", points=[["ring", ring[1]]], handles=2, pinned=True
            )
            await call("point_style", points=[["ring", ring[1]]], pinned=False)
            await call("split_edge", points=[["ring", ring[1]]])
            await call("break_points", points=[["ring", ring[2]]])
            await call("delete", points=[["ring", ring[1]]])
            await call("delete", points=[["ring", ring[0]]], contours=True)
            await call("knife", start=[0, 75], end=[200, 75])
            await call("delete", ids=["c"])
            await call("load_reference", path=str(picture))
            await call("load_reference")
            await restore(call)
            await restore(call, "redo")
            # The operations, on the drawing as it was.
            data(await call("open", path=str(drawing)))
            data(await call("extract", region=[0, 140, 70, 30], ids=["line"]))
            await call("delete", region=[140, 0, 60, 60])
            data(await restore(call, count=2))
            # Two lines join at their ends; a segment's two ends delete it.
            joined = data(await call("join", ids=["line", "line2"], reach=10))
            assert joined["joined"] == "line ends"
            (path,) = joined["result"]["objects"]
            ends = [[path, n] for n in (await nodes(path))[:2]]
            broke = data(await call("break_points", points=ends))
            assert broke["broke"] == "segment deleted"
            await call("load_reference", path=str(picture))
            job = data(await call("cleanup", ids=["b"]))
            await call("job", id=job["id"])
            await call("job", id=job["id"], action="discard")
            job = data(await call("snap_edges", ids=["a", "side"]))
            await call("job", id=job["id"], action="apply")
            job = data(await call("fit_colours", ids=["b"], resolution=32))
            await call("job", id=job["id"], action="stop")
            await call("job", id=job["id"], action="discard")

    anyio.run(session)
    missing = ACTION_COMMANDS.keys() - set(LEFT_OUT) - actions
    assert not missing, f"Editor commands no MCP tool sends: {sorted(missing)}"
    missing = OPERATION_COMMANDS.keys() - operations
    assert not missing, f"Operation commands no MCP tool sends: {sorted(missing)}"
    sent = {c for tool in EDITS.values() for c in tool}
    assert sent <= ACTION_COMMANDS.keys(), sorted(sent - ACTION_COMMANDS.keys())


@pytest.mark.parametrize("tool", sorted(EDITS))
def test_each_edit_names_commands_the_session_has(tool):
    assert set(EDITS[tool]) <= ACTION_COMMANDS.keys()
