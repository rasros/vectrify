"""Hiding objects in the tree and on the canvas, run under Node when installed."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_tree_eye_commands_and_canvas_leave_hidden_objects_out():
    script = Path(__file__).with_name("visibility.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
