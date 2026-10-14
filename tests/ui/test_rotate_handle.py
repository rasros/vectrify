"""The rotation handle and pivot, run under Node when it is installed.

The harness prints the transforms its drag previews; the server must commit
the same ones, so what the canvas shows while turning is what lands.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.ui.test_session import send
from vectrify.document import import_svg
from vectrify.document.hit_test import transform
from vectrify.ui.session import Session

# The harness's two objects: one in a turned, scaled group with a transform of
# its own, one at the top level.
SVG = """<svg width="200" height="200">
<g id="turned" transform="translate(100 20) rotate(90) scale(2)">
<rect id="inner" width="10" height="5" transform="translate(3 4)"/></g>
<rect id="plain" x="10" y="150" width="20" height="10"/>
</svg>"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_the_rotation_preview_is_what_the_server_commits():
    script = Path(__file__).with_name("rotate_handle.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout.strip().splitlines()[-1])
    session = Session(import_svg(SVG))
    send(session, "select", objects=["inner", "plain"])
    send(session, "rotate", angle=preview["angle"], pivot=preview["pivot"])
    document = session.editor.snapshot.document
    for oid in ("inner", "plain"):
        committed = transform(document.element(oid).get("transform"))
        assert committed == pytest.approx(transform(preview[oid]), abs=1e-9), oid
