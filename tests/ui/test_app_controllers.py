"""Operation dialogs keep their job lifetime separate from live editor state."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
@pytest.mark.parametrize(
    "filename",
    [
        "app_controllers.mjs",
        "application_wiring.mjs",
        "application_smoke.mjs",
        "transport.mjs",
        "inspector.mjs",
        "draw_tools.mjs",
    ],
)
def test_operation_dialog_lifetimes(filename):
    script = Path(__file__).with_name(filename)
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
