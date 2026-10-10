"""Tidy's defaults switch must discard custom values that can prevent a fit."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_tidy_advanced_defaults_are_used_after_disabling_overrides():
    script = Path(__file__).with_name("tidy_settings.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
