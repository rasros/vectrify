"""The Pencil tool: strokes fitted in the browser module become valid,
editable paths through add_path, one undoable edit each, with the drawing
paint and container."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vectrify.document import export_svg, import_svg
from vectrify.document.transforms import object_matrix
from vectrify.ui.session import Session

STATIC = Path(__file__).parents[2] / "src/vectrify/ui/static"
SVG = """<svg width="400" height="400">
<g id="moved" transform="translate(10 5) scale(2)"><path id="a" d="M0 0 L5 5 Z"/></g>
<g id="turned" transform="rotate(30 50 50) skewX(10)"><path id="b" d="M0 0 L5 0 Z"/></g>
</svg>"""

# Fits strokes with the Pencil's own module: a shaky arc at several zooms,
# open and closed, a zigzag and a stroke barely long enough to draw.
FIT = """
import {fitStroke, strokeNodes, strokePath} from '%s';
let seed = 3;
const random = () => {
  seed = seed * 16807 %% 2147483647;
  return seed / 2147483647 - 0.5;
};
const arc = zoom => Array.from({length: 300}, (_, i) => {
  const a = 1.5 * Math.PI * i / 299;
  const x = 200 + 120 * Math.cos(a) + random(), y = 200 + 120 * Math.sin(a) + random();
  return [x / zoom, y / zoom];
});
const zigzag = Array.from({length: 301}, (_, i) => [i, 50 - Math.abs(i %% 100 - 50)]);
const strokes = [
  [arc(0.5), 0.5, false], [arc(8), 8, true], [zigzag, 1, false],
  [[[0, 0], [1.5, 1.5]], 1, true],
];
console.log(JSON.stringify(strokes.map(([points, zoom, closed]) => {
  const d = strokePath(fitStroke(points, {zoom, closed}), zoom);
  return {d, nodes: strokeNodes(d)};
})));
"""


def fitted_strokes() -> list[dict]:
    script = FIT % (STATIC / "pencil.js").as_uri()
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def send(session, command, **data):
    return session.action(
        {
            "command": command,
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            **data,
        }
    )


def root_point(session, oid, x, y):
    a, b, c, d, e, f = object_matrix(session.editor.snapshot.document, oid)
    return (a * x + c * y + e, b * x + d * y + f)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
@pytest.mark.parametrize("parent", [None, "moved", "turned"])
def test_fitted_strokes_are_valid_paths_where_drawn(parent):
    strokes = fitted_strokes()
    assert all(stroke["d"] for stroke in strokes)
    for stroke in strokes:
        session = Session(import_svg(SVG))
        before = session.editor.snapshot.document
        closed = stroke["d"].endswith("Z")
        state = send(
            session,
            "add_path",
            d=stroke["d"],
            fill="#123456" if closed else "none",
            stroke="none" if closed else "#abcdef",
            stroke_width=2,
            **({"parent": parent} if parent else {}),
        )
        assert session.editor.undo_labels == ("Draw path",)
        (oid,) = state["selection"]["objects"]
        document = session.editor.snapshot.document
        path = document.element(oid)
        assert document.ancestry(oid)[-2].id == (parent or document.root.id)
        assert path.get("fill") == ("#123456" if closed else "none")
        assert path.geometry_id is not None
        (subpath,) = document.geometry(path.geometry_id).subpaths
        assert subpath.closed == closed
        # As many points as the preview counted, at most a dozen, each where
        # the fit put it on the page.
        assert len(subpath.nodes) - closed == stroke["nodes"] <= 12
        numbers = [
            float(v)
            for v in stroke["d"].replace("M", " ").replace("C", " ").split()
            if v != "Z"
        ]
        ends = [numbers[:2]] + [
            numbers[i + 4 : i + 6] for i in range(2, len(numbers), 6)
        ]
        for node, end in zip(subpath.nodes, ends, strict=True):
            x, y = root_point(session, oid, *node.values[-2:])
            assert x == pytest.approx(end[0], abs=1e-6)
            assert y == pytest.approx(end[1], abs=1e-6)
        assert import_svg(export_svg(document)).element(oid).tag == "path"
        send(session, "undo")
        assert session.editor.snapshot.document == before
