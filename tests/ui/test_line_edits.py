"""Which line edits selected points allow, run under Node when it is installed."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_line_edits_need_the_right_points():
    script = Path(__file__).with_name("line_edits.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
