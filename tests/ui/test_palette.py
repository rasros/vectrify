"""The command palette's matching, run under Node when it is installed."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_palette_ranks_commands_by_name_group_and_keywords():
    script = Path(__file__).with_name("palette_match.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
