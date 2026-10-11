"""Operation dialogs keep their job lifetime separate from live editor state."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_operation_dialog_lifetimes():
    script = Path(__file__).with_name("app_controllers.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
