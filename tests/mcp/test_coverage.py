"""Every editor command is reachable from some MCP tool, or left out on purpose.

The commands are read from ``Session``'s source, and the test records which
of them actually arrive at ``Session.action`` (and ``Session.operation``)
while an MCP client calls the tools. A command added to the session without
a tool, or a tool that stops sending its command, fails here. There is no
select tool: "select" arrives as every targeted tool's first step, the agent
choosing its own targets before the person's selection is given back.
"""

from __future__ import annotations

import ast
import inspect

import anyio
import pytest
from mcp import Client

from tests.mcp.helpers import data, reference_png
from vectrify.mcp.server import build_server
from vectrify.ui import session as session_module
from vectrify.ui.agent import EDITS, LEFT_OUT, OPERATIONS_LEFT_OUT
from vectrify.ui.session import Session

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


def commands(functions: list[str], names: dict[str, set[str]]) -> set[str]:
    """String literals *command* is compared against in Session's methods."""
    found: set[str] = set()
    for name in functions:
        tree = ast.parse(inspect.getsource(getattr(Session, name)).strip())
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Name)
                and node.left.id == "command"
            ):
                continue
            for right in node.comparators:
                if isinstance(right, ast.Constant) and isinstance(right.value, str):
                    found.add(right.value)
                elif isinstance(right, ast.Set | ast.Tuple | ast.List):
                    found.update(
                        e.value
                        for e in right.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    )
                elif isinstance(right, ast.Name):
                    found.update(names[right.id])
    return found


ACTION_COMMANDS = commands(
    ["action", "_edit_points", "_edit", "_lines"],
    {"POINT_COMMANDS": set(session_module.POINT_COMMANDS)},
)
OPERATION_COMMANDS = commands(["operation"], {})


def test_the_session_commands_are_found():
    # A sanity check of the parsing above.
    assert {"paint", "knife", "join_two_ends", "convert_lines", "undo"} <= (
        ACTION_COMMANDS
    )
    assert {"start", "apply", "discard", "status", "stop", "check"} == (
        OPERATION_COMMANDS
    )


def test_left_out_commands_exist_and_say_why():
    assert set(LEFT_OUT) <= ACTION_COMMANDS
    assert set(OPERATIONS_LEFT_OUT) <= OPERATION_COMMANDS
    reasons = [*LEFT_OUT.values(), *OPERATIONS_LEFT_OUT.values()]
    assert all(len(why) > 20 for why in reasons)


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
            assert {"load_reference", "remove_reference"} <= tools

            async def call(tool_name: str, /, **args):
                return await client.call_tool(tool_name, args)

            data(await call("open", path=str(drawing)))

            async def nodes(oid: str) -> list[str]:
                geometry = data(await call("points", id=oid))["geometry"]
                return [n["id"] for sp in geometry["subpaths"] for n in sp["nodes"]]

            line, ring = await nodes("line"), await nodes("ring")
            await call("redraw_outline", id="b", points=[[40, 40], [65, 30], [90, 40]])
            await call("join_points", a=["line", line[0]], b=["line", line[-1]])
            await call("convert", ids=["line2"], to="fill")
            await call("paint", ids=["a"], fill="#ff8800")
            await call("rename", id="a", name="Square")
            await call("locks", id="a", locks=[])
            await call("move", ids=["a"], dx=1, dy=1)
            await call("resize", ids=["a"], box=[0, 0, 30, 30])
            await call("reorder", ids=["a"], to="front")
            await call("reorder", ids=["a"], to="backward")
            await call("move_into", ids=["a"], parent="grp", index=0)
            grouped = data(await call("group", ids=["line", "line2"]))
            await call("ungroup", ids=grouped["result"]["objects"])
            await call("join", ids=["a", "b"])
            await call("join_ends", ids=["line", "line2"], reach=10)
            await call("split_parts", ids=["ring"])
            await call("cut_hole", ids=["ring", "c"])
            holes = data(await call("holes", id="ring"))["holes"]
            contours = [["ring", h["id"]] for h in holes] or [["ring", "x"]]
            await call("fill_holes", contours=contours)
            await call("holes_to_shapes", contours=contours)
            await call("detach", ids=["inst"])
            await call("convert", ids=["ring"], to="line")
            await call("convert", ids=["ring"], to="either")
            await call("add_path", d="M0 0 L20 0 L20 20 Z", fill="#000")
            await call("set_points", changes={"ring": {ring[1]: [180, 100]}})
            await call("handles", points=[["ring", ring[1]]], count=2)
            await call("pin", points=[["ring", ring[1]]], pinned=True)
            await call("pin", points=[["ring", ring[1]]], pinned=False)
            await call("split_edge", points=[["ring", ring[1]]])
            await call("break_points", points=[["ring", ring[2]]])
            await call("delete_segment", points=[["ring", ring[2]], ["ring", ring[3]]])
            await call("delete_points", points=[["ring", ring[1]]])
            await call("delete_contours", points=[["ring", ring[0]]])
            await call("knife", start=[0, 75], end=[200, 75])
            await call("delete", ids=["c"])
            await call("load_reference", path=str(picture))
            await call("remove_reference")
            await call("undo")
            await call("redo")
            # The operations, on the drawing as it was.
            data(await call("open", path=str(drawing)))
            await call("load_reference", path=str(picture))
            job = data(await call("cleanup", ids=["b"]))
            await call("job_status", id=job["id"])
            await call("discard", id=job["id"])
            job = data(await call("snap_edges", ids=["a", "side"]))
            await call("apply", id=job["id"])
            job = data(await call("fit_colours", ids=["b"], resolution=32))
            await call("stop", id=job["id"])
            await call("discard", id=job["id"])

    anyio.run(session)
    missing = ACTION_COMMANDS - set(LEFT_OUT) - actions
    assert not missing, f"Editor commands no MCP tool sends: {sorted(missing)}"
    missing = OPERATION_COMMANDS - set(OPERATIONS_LEFT_OUT) - operations
    assert not missing, f"Operation commands no MCP tool sends: {sorted(missing)}"
    sent = {c for tool in EDITS.values() for c in tool}
    assert sent <= ACTION_COMMANDS, sorted(sent - ACTION_COMMANDS)


@pytest.mark.parametrize("tool", sorted(EDITS))
def test_each_edit_names_commands_the_session_has(tool):
    assert set(EDITS[tool]) <= ACTION_COMMANDS
