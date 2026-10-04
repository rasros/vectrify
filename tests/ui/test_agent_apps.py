"""The setup snippets the Agents popover offers for other apps, to copy."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from vectrify.ui import agent_setup
from vectrify.ui.agent_channel import AgentChannel
from vectrify.ui.agent_setup import (
    app_setup,
    claude_desktop_snippet,
    codex_command,
    codex_snippet,
    mcp_executable,
)
from vectrify.ui.server import Backend

tomllib = pytest.importorskip("tomllib" if sys.version_info >= (3, 11) else "tomli")

# Spaces, quotes and backslashes, as an install path may have.
AWKWARD = [
    "/home/me/My Apps/vectrify/.venv/bin/vectrify-mcp",
    'C:\\Users\\Ann "A"\\venv\\Scripts\\vectrify-mcp.exe',
]


def test_the_executable_is_this_installs():
    command = mcp_executable()
    assert Path(command).is_absolute()
    assert Path(command).parent == Path(sys.executable).parent
    assert Path(command).name.startswith("vectrify-mcp")


def test_the_executable_falls_back_to_path_then_the_bare_name(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_setup.sys, "executable", str(tmp_path / "python"))
    found = tmp_path / "bin" / "vectrify-mcp"
    monkeypatch.setattr(agent_setup.shutil, "which", lambda _name: str(found))
    assert mcp_executable() == str(found)
    monkeypatch.setattr(agent_setup.shutil, "which", lambda _name: None)
    assert mcp_executable() == "vectrify-mcp"


@pytest.mark.parametrize("command", AWKWARD)
def test_the_codex_table_is_valid_toml_with_the_command(command):
    snippet = codex_snippet(command)
    assert snippet.startswith("[mcp_servers.vectrify]\n")
    parsed = tomllib.loads(snippet)
    assert parsed == {
        "mcp_servers": {"vectrify": {"command": command, "startup_timeout_sec": 30}}
    }
    # It appends cleanly to a config that has other servers and sub-tables.
    existing = (
        'model = "gpt-5"\n\n[mcp_servers.node_repl]\ncommand = "node"\n\n'
        '[mcp_servers.node_repl.env]\nA = "1"\n\n'
    )
    merged = tomllib.loads(existing + snippet)
    assert merged["mcp_servers"]["node_repl"]["env"] == {"A": "1"}
    assert merged["mcp_servers"]["vectrify"]["command"] == command


def test_the_codex_command_quotes_the_path():
    command = AWKWARD[0]
    words = shlex.split(codex_command(command))
    assert words == ["codex", "mcp", "add", "vectrify", "--", command]


@pytest.mark.parametrize("command", AWKWARD)
def test_the_claude_desktop_entry_is_json_with_the_command(command):
    assert json.loads(claude_desktop_snippet(command)) == {
        "mcpServers": {"vectrify": {"command": command}}
    }


def test_the_config_locations(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(agent_setup.sys, "platform", "darwin")
    apps = app_setup()
    assert apps["codex_config"] == "~/.codex/config.toml"
    assert apps["claude_desktop_config"] == (
        "~/Library/Application Support/Claude/claude_desktop_config.json"
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(agent_setup.sys, "platform", "linux")
    apps = app_setup()
    assert apps["codex_config"] == str(tmp_path / "config.toml")
    assert apps["claude_desktop_config"] == (
        "~/.config/Claude/claude_desktop_config.json"
    )


def test_the_popover_gets_the_snippets_only_while_agents_are_allowed(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    # Not hosting MCP here: the snippets do not depend on it.
    monkeypatch.setattr(AgentChannel, "_host", lambda _self: None)
    backend = Backend()
    _, state = backend.handle("/api/session", {}, None)
    session = state["session"]
    try:
        _, off = backend.handle("/api/poll", {}, session)
        assert "apps" not in off["agent"]
        _, on = backend.handle("/api/agent", {"enabled": True}, session)
        apps = on["apps"]
        command = mcp_executable()
        assert apps["executable"] == command
        assert tomllib.loads(apps["codex"])["mcp_servers"]["vectrify"] == {
            "command": command,
            "startup_timeout_sec": 30,
        }
        assert json.loads(apps["claude_desktop"])["mcpServers"]["vectrify"] == {
            "command": command
        }
        # No token: vectrify-mcp finds the window through editor.json.
        token = (tmp_path / "state" / "vectrify" / "agent-token").read_text().strip()
        assert all(token not in value for value in apps.values())
    finally:
        backend.agents.close()
