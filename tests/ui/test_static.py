"""Check the editor's JavaScript modules before they reach a browser."""

import shutil
import subprocess

import pytest

from vectrify.ui.server import STATIC


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
@pytest.mark.parametrize("script", sorted(STATIC.glob("*.js")), ids=lambda p: p.name)
def test_editor_module_syntax(script):
    # Pass module source on stdin: Node can otherwise retry a .js file as a
    # classic script and miss syntax errors inside module functions.
    result = subprocess.run(
        ["node", "--input-type=module", "--check"],
        input=script.read_text(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{script.name}: {result.stderr}"
