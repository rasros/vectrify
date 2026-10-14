"""Every entry point runs the same command and gives the same reason it cannot."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_entry_points_share_command_names_and_reasons():
    script = Path(__file__).with_name("command_entry_points.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
