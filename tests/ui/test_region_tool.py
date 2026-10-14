"""The Region tool's drawing, preview and panel, run under Node when it is
installed."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_region_tool_previews_targets_and_cuts_before_applying():
    script = Path(__file__).with_name("region_tool.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
