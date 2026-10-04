"""Selection must keep the Objects tree and its paint work between clicks."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_selection_retains_rows_and_refreshes_after_document_changes():
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("selection_render.mjs"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
