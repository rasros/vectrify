"""Browser downloads, native bridge saves, and recovery keep exact project bytes."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
@pytest.mark.parametrize("filename", ["project_files.mjs", "file_save_failures.mjs"])
def test_project_file_controls(filename):
    script = Path(__file__).with_name(filename)
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
