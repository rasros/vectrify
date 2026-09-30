"""The tool strip's overflow layout, run under Node when it is installed."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_strip_collapses_the_least_important_controls_first():
    script = Path(__file__).with_name("strip_overflow.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
