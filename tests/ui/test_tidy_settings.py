"""Tidy's dialog starts with the server's default steps."""

import re
from pathlib import Path

from vectrify.operations.methods.nodes import SETTINGS


def test_the_dialog_starts_with_the_servers_default_steps():
    html = (Path(__file__).parents[2] / "src/vectrify/ui/static/index.html").read_text()
    for step in ("shape", "detail", "simplify"):
        tag = re.search(rf'<input id="nodes-{step}"[^>]*>', html)
        assert tag is not None, step
        assert ("checked" in tag[0]) == SETTINGS[step].default, step
