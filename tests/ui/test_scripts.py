"""The editor's Node harnesses, run under Node when it is installed.

Each script exits non-zero on its first failed check. Harnesses whose output a
Python test reads back run from that test instead.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent

# Run by the tests that compare their output with the server's.
DRIVEN = {"gradient_handles.mjs", "reference_render_browser.mjs", "rotate_handle.mjs"}

SCRIPTS = sorted(p for p in HERE.glob("*.mjs") if p.name not in DRIVEN)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_node_script(script):
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
