"""Tidy's dialog defaults: the server's steps, and a switch that discards
custom values that can prevent a fit."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from vectrify.operations.methods.nodes import SETTINGS


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_tidy_advanced_defaults_are_used_after_disabling_overrides():
    script = Path(__file__).with_name("tidy_settings.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_the_dialog_starts_with_the_servers_default_steps():
    html = (Path(__file__).parents[2] / "src/vectrify/ui/static/index.html").read_text()
    for step in ("shape", "detail", "simplify"):
        tag = re.search(rf'<input id="nodes-{step}"[^>]*>', html)
        assert tag is not None, step
        assert ("checked" in tag[0]) == SETTINGS[step].default, step
