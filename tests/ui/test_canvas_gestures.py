"""Canvas gesture previews, completion and cancellation under Node.js."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_canvas_gesture_lifecycles():
    script = Path(__file__).with_name("canvas_gestures.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
