"""Command payloads and automatic catch-up in the browser editor."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_concurrent_edit_payloads_and_automatic_catch_up():
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("concurrent_edit.mjs"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
