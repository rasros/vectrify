"""The Gradient tool's canvas handles, run under Node when it is installed.

The harness prints the endpoint edits its drags previewed; the server must
commit exactly those, leave the stops alone, and put the end where the drag
did on the page, worked out here from the committed SVG.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.ui.test_session import send
from vectrify.document import EditRejectedError, import_svg
from vectrify.document.hit_test import IDENTITY, multiply, transform
from vectrify.ui.session import Session

# The harness's drawing: a in a turned, scaled group and b share "box";
# c has "user", in drawing coordinates.
SVG = """<svg width="200" height="100" viewBox="0 0 200 100">
<defs>
<linearGradient id="box" x1="10%" y1="0%" x2="90%"
 gradientTransform="rotate(30 0.5 0.5)">
<stop id="box0" offset="0" stop-color="#336699"/>
<stop id="box1" offset="1" stop-color="#ffffff"/></linearGradient>
<linearGradient id="user" gradientUnits="userSpaceOnUse" x1="130" y1="20"
 x2="25%" y2="50%" gradientTransform="translate(5 0) scale(1.5 1)">
<stop offset="0" stop-color="#000000"/>
<stop offset="1" stop-color="#ff0000"/></linearGradient>
</defs>
<g id="turned" transform="translate(100 20) rotate(90) scale(2)">
<rect id="a" x="10" y="20" width="40" height="20" transform="translate(3 4)"
 fill="url(#box)"/></g>
<rect id="b" x="0" y="60" width="50" height="30" fill="url(#box)"/>
<rect id="c" x="120" y="10" width="60" height="40" transform="rotate(10 150 30)"
 fill="url(#user)"/>
</svg>"""

VIEWPORT = {"x": 200.0, "y": 100.0}


def drags():
    script = Path(__file__).with_name("gradient_handles.mjs")
    result = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def on_page(document, object_id, gradient_id, end):
    """Where *end* of the gradient painting *object_id* is on the page."""
    matrix = IDENTITY
    for element in document.ancestry(object_id):
        matrix = multiply(matrix, transform(element.get("transform")))
    gradient = document.element(gradient_id)
    shape = document.element(object_id)
    user = gradient.get("gradientUnits") == "userSpaceOnUse"
    if not user:
        x, y, w, h = (float(shape.get(k)) for k in ("x", "y", "width", "height"))
        matrix = multiply(matrix, (w, 0, 0, h, x, y))
    matrix = multiply(matrix, transform(gradient.get("gradientTransform")))
    defaults = {"x1": "0%", "y1": "0%", "x2": "100%", "y2": "0%"}
    keys = ("x1", "y1") if end == "start" else ("x2", "y2")
    gx, gy = (
        float(text[:-1]) / 100 * (VIEWPORT[key[0]] if user else 1)
        if (text := gradient.get(key) or defaults[key]).endswith("%")
        else float(text)
        for key in keys
    )
    a, b, c, d, e, f = matrix
    return a * gx + c * gy + e, b * gx + d * gy + f


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_a_handle_drag_commits_what_it_previewed_as_one_edit():
    for drag in drags():
        session = Session(import_svg(SVG))
        send(session, "select", objects=[drag["object"]])
        before = session.editor.snapshot.document
        send(
            session,
            "paint",
            gradient={"id": drag["gradient"], "attributes": drag["attributes"]},
        )
        after = session.editor.snapshot.document
        gradient = after.element(drag["gradient"])
        for key, value in drag["attributes"].items():
            assert gradient.get(key) == value
        # Only the dragged end changed; the stops are the same elements.
        old = before.element(drag["gradient"])
        assert gradient.children == old.children
        assert {
            k: v for k, v in gradient.attributes if k not in drag["attributes"]
        } == {k: v for k, v in old.attributes if k not in drag["attributes"]}
        x, y = on_page(after, drag["object"], drag["gradient"], drag["end"])
        assert (x, y) == pytest.approx(
            (drag["target"]["x"], drag["target"]["y"]), abs=1e-2
        )
        assert session.editor.snapshot.selection.object_ids == {drag["object"]}
        session.editor.undo()
        assert session.editor.snapshot.document == before


def test_an_endpoint_edit_moves_shared_paint_under_its_locks_but_not_its_stops():
    """A drag sends no stops, so a locked stop colour does not stop it."""
    edit = {"id": "box", "attributes": {"x2": "50%", "y2": "25%"}}
    session = Session(import_svg(SVG))
    session.editor.set_locks("box1", frozenset({"stop-color"}))
    send(session, "select", objects=["a"])
    send(session, "paint", gradient=edit)
    assert session.editor.snapshot.document.element("box").get("y2") == "25%"
    for locked, lock in [("b", "paint"), ("box", "paint"), ("turned", "paint")]:
        session = Session(import_svg(SVG))
        session.editor.set_locks(locked, frozenset({lock}))
        send(session, "select", objects=["a"])
        before = session.editor.snapshot.document
        with pytest.raises(EditRejectedError, match="locked"):
            send(session, "paint", gradient=edit)
        assert session.editor.snapshot.document == before
