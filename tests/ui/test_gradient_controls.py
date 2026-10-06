"""Manual gradient creation and direction controls, run under Node."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_manual_gradient_controls():
    script = Path(__file__).with_name("gradient_controls.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
