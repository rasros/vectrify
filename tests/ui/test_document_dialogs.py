"""New drawing's and the Artboard's dialogs, run under Node when it is
installed."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_new_drawing_and_artboard_dialogs_keep_artwork_in_place():
    script = Path(__file__).with_name("document_dialogs.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
