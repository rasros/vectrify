"""Selection must keep the Objects tree and its paint work between clicks."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
@pytest.mark.parametrize("script", ["selection_render.mjs", "overlay_render.mjs"])
def test_selection_rendering_keeps_work_bounded(script):
    result = subprocess.run(
        ["node", str(Path(__file__).with_name(script))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
