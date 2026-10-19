"""The tools an MCP client is offered: names, descriptions and valid schemas."""

from __future__ import annotations

import json

import anyio
import pytest
from mcp import Client

from vectrify.mcp.server import Vectrify, build_server, build_window_server

# Each tool, with the arguments a client must always pass.
TOOLS = {
    "open": {"path"},
    "connect": set(),
    "save": set(),
    "load_reference": set(),
    "settings_schema": {"action", "method"},
    "describe": set(),
    "pick": {"x", "y"},
    "render": set(),
    "compare": set(),
    "trace_reference": set(),
    "get_svg": set(),
    "points": {"id"},
    "history": set(),
    "isolate_feature": set(),
    "isolate_components": {"region", "members"},
    "protect_features": {"points"},
    "linked_outline": {"id"},
    "clip": {"ids"},
    "properties": {"ids"},
    "transform": {"ids"},
    "arrange": {"ids"},
    "group": {"ids"},
    "combine": {"ids"},
    "join": set(),
    "split_parts": {"ids"},
    "cut_hole": {"ids"},
    "boolean": {"ids", "operation"},
    "holes": {"contours"},
    "convert": {"ids"},
    "delete": set(),
    "add_path": {"d"},
    "set_points": {"changes"},
    "point_style": {"points"},
    "break_points": {"points"},
    "split_edge": {"points"},
    "extract": {"region"},
    "knife": {"start", "end"},
    "redraw_outline": {"id", "points"},
    "undo": {"ids"},
    "redo": {"ids"},
    "edit_batch": set(),
    "generate": set(),
    "tidy": set(),
    "fit_colours": {"ids"},
    "snap_edges": {"ids"},
    "cleanup": {"ids"},
    "job": {"id"},
}


async def offered(server) -> list[dict]:
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
    return [t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in tools]


JSON_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def check_schema(schema, where):
    """Each subschema names JSON types, choices or alternatives a client knows."""
    assert isinstance(schema, dict), where
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind] if kind else []
    assert set(kinds) <= JSON_TYPES, where
    assert kinds or {"anyOf", "oneOf", "enum", "const", "$ref"} & schema.keys(), where
    for alternative in schema.get("anyOf", []) + schema.get("oneOf", []):
        check_schema(alternative, where)
    for name, spec in schema.get("properties", {}).items():
        check_schema(spec, f"{where}.{name}")
    if isinstance(schema.get("items"), dict):
        check_schema(schema["items"], f"{where}[]")


@pytest.fixture(scope="module")
def tools():
    return anyio.run(offered, build_server())


def test_the_server_offers_the_expected_tools(tools):
    assert [t["name"] for t in tools] == list(TOOLS)


def test_each_tool_is_described_with_a_valid_schema(tools):
    for tool in tools:
        name, schema = tool["name"], tool["inputSchema"]
        assert tool["description"].strip(), name
        # Through JSON, as a client gets it.
        assert json.loads(json.dumps(schema)) == schema, name
        assert schema["type"] == "object", name
        check_schema(schema, name)
        properties = schema.get("properties", {})
        required = set(schema.get("required", ()))
        assert required == TOOLS[name], name
        assert required <= properties.keys(), name


def test_the_window_offers_the_same_tools_less_opening_and_connecting(tools):
    window = anyio.run(offered, build_window_server(Vectrify()))
    assert window == [t for t in tools if t["name"] not in {"open", "connect"}]
