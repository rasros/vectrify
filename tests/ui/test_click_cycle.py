"""Repeated canvas clicks must cycle objects even near a selection frame."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_canvas_click_cycle():
    script = Path(__file__).with_name("click_cycle.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
