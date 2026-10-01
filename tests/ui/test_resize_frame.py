"""Resizing the selection by its frame, run under Node when it is installed."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_frame_parts_cursors_and_scales():
    script = Path(__file__).with_name("resize_frame.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
