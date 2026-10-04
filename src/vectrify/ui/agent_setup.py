"""Discovery, persistent credentials, and client setup for editor agents."""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any


def state_dir() -> Path:
    """Where the editor keeps what agents need to find it."""
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "vectrify"


def discovery_file() -> Path:
    """Where a running editor tells agents how to reach it."""
    return state_dir() / "editor.json"


def token_file() -> Path:
    """The agent token, kept across runs so a client is added only once."""
    return state_dir() / "agent-token"


def port_file() -> Path:
    """The port the MCP server was last hosted on, preferred next time so the
    URL clients were given keeps working."""
    return state_dir() / "mcp-port"


def _write_private(path: Path, text: str) -> None:
    """Write *path* owner-only (0600 in a 0700 directory), atomically."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".tmp")
    with contextlib.suppress(FileNotFoundError):
        temporary.unlink()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(text)
    temporary.replace(path)


def agent_token(regenerate: bool = False) -> str:
    """The stable agent token: read, or made once (or anew) and kept."""
    path = token_file()
    if not regenerate:
        with contextlib.suppress(OSError, UnicodeDecodeError):
            token = path.read_text().strip()
            if len(token) >= 32:
                return token
    token = secrets.token_urlsafe(32)
    _write_private(path, token + "\n")
    return token


def remembered_port() -> int | None:
    """The port the MCP server was last hosted on, if any."""
    with contextlib.suppress(OSError, ValueError):
        port = int(port_file().read_text().strip())
        if 0 < port < 65536:
            return port
    return None


def mcp_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/mcp"


def local_host(headers: Any, port: int) -> bool:
    """Whether a request was addressed to this machine, not via a renamed host."""
    return headers.get("Host", "") in {f"127.0.0.1:{port}", f"localhost:{port}"}


def claude_command(url: str, token: str) -> str:
    """The Claude Code command that adds the editor's MCP server."""
    return (
        f"claude mcp add --transport http --scope user vectrify {url} "
        f'--header "Authorization: Bearer {token}"'
    )


def mcp_executable() -> str:
    """This install's ``vectrify-mcp``: beside the running Python, else on PATH."""
    name = "vectrify-mcp.exe" if os.name == "nt" else "vectrify-mcp"
    beside = Path(sys.executable).parent / name
    if beside.is_file():
        return str(beside.absolute())
    found = shutil.which("vectrify-mcp")
    return str(Path(found).absolute()) if found else "vectrify-mcp"


def _shell_word(word: str) -> str:
    if os.name == "nt":
        return f'"{word}"' if any(c in word for c in ' \t"&()^') else word
    return shlex.quote(word)


def codex_snippet(command: str) -> str:
    """The ``config.toml`` table that adds ``vectrify-mcp`` to Codex.

    Starting it imports the vision stack, so it gets more than Codex's default
    10 s. A JSON string is also a valid TOML basic string.
    """
    return (
        "[mcp_servers.vectrify]\n"
        f"command = {json.dumps(command)}\n"
        "startup_timeout_sec = 30\n"
    )


def codex_command(command: str) -> str:
    """The Codex CLI command that adds the same server (default timeout)."""
    return f"codex mcp add vectrify -- {_shell_word(command)}"


def claude_desktop_snippet(command: str) -> str:
    """The ``claude_desktop_config.json`` entry that adds ``vectrify-mcp``."""
    return json.dumps({"mcpServers": {"vectrify": {"command": command}}}, indent=2)


def claude_desktop_config() -> str:
    """Where Claude Desktop keeps its MCP servers on this OS."""
    if sys.platform == "darwin":
        return "~/Library/Application Support/Claude/claude_desktop_config.json"
    if os.name == "nt":
        return r"%APPDATA%\Claude\claude_desktop_config.json"
    # Claude Desktop has no official Linux build; unofficial ones read this.
    return "~/.config/Claude/claude_desktop_config.json"


def app_setup() -> dict[str, str]:
    """What other agent apps need to start ``vectrify-mcp``, which then
    attaches to the window with Agents on through ``editor.json``: no token."""
    command = mcp_executable()
    codex_home = os.environ.get("CODEX_HOME")
    return {
        "executable": command,
        "codex_config": str(Path(codex_home) / "config.toml")
        if codex_home
        else "~/.codex/config.toml",
        "codex": codex_snippet(command),
        "codex_command": codex_command(command),
        "claude_desktop": claude_desktop_snippet(command),
        "claude_desktop_config": claude_desktop_config(),
    }
