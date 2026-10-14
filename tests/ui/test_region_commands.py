"""Extract region and Delete region from the editor: the preview the Region
tool shows, and the one undoable edit it applies, as the session command
extract makes it for agents too."""

import pytest

from vectrify.document import DocumentError, EditKind, export_svg, import_svg
from vectrify.document.hit_test import HitIndex
from vectrify.ui.session import Session

# The region [40, 20, 60, 60] (x 40-100, y 20-80) is crossed by a line and a
# blue shape, holds a red dot wholly, and in the moved group a green square
# wholly; a locked shape crosses it too, and an instance paints inside. The
# ring far below has a hole.
SVG = """<svg width="200" height="200" viewBox="0 0 200 200">
<path id="line" d="M0 50 L200 50" fill="none" stroke="black"/>
<path id="blob" d="M60 30 L140 30 L140 70 L60 70 Z" fill="blue"/>
<path id="dot" d="M50 40 L55 40 L55 45 L50 45 Z" fill="red"/>
<g id="moved" transform="translate(100 100)">
<path id="inner" d="M-50 -40 L-40 -40 L-40 -30 L-50 -30 Z" fill="green"/></g>
<path id="fixed" d="M90 60 L120 60 L120 75 L90 75 Z" fill="gray"/>
<path id="ring" d="M0 120 L80 120 L80 200 L0 200 Z M20 140 L20 180 L60 180 L60 140 Z"
 fill="teal" fill-rule="evenodd"/>
<defs><path id="src" d="M0 0 L10 0 L10 10 Z" fill="black"/></defs>
<use id="stamp" href="#src" x="80" y="25"/>
</svg>"""
REGION = [40, 20, 60, 60]


def send(session, command, **data):
    return session.action(
        {
            "command": command,
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            **data,
        }
    )


def preview(session, **data):
    return session.region(
        {"epoch": session.epoch, "revision": session.editor.snapshot.revision, **data}
    )


def outcomes(operation):
    return {t["id"]: t["outcome"] for t in operation["targets"]}


def painted(session, oid):
    bounds = HitIndex(session.editor.snapshot.document).bounds(frozenset({oid}))
    assert bounds is not None
    return bounds


def test_preview_shows_targets_and_cuts_and_leaves_the_drawing_alone():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["line", "blob", "dot"])
    before = session.editor.snapshot
    result = preview(session, region=REGION)
    # The line and the shape cross the edge and are cut; the dot is already
    # wholly inside, so Extract leaves it and Delete deletes it.
    assert outcomes(result["extract"]) == {"line": "cut", "blob": "cut"}
    assert outcomes(result["delete"]) == {"line": "cut", "blob": "cut", "dot": "whole"}
    assert result["extract"]["error"] is None
    assert sorted(result["crossing"]) == ["blob", "line"]
    assert result["revision"] == before.revision
    # Cancelling leaves nothing to undo: the preview was never applied.
    after = session.editor.snapshot
    assert after.document is before.document
    assert after.selection == before.selection
    assert session.state()["undo"] == []


def test_extract_region_is_one_undoable_edit_as_previewed():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["line", "blob", "dot"])
    original = export_svg(session.editor.snapshot.document)
    result = send(session, "extract", region=REGION, cut=True)
    assert result["undo"] == ["Extract region"]
    pieces = result["selection"]["objects"]
    assert len(pieces) == 2
    for piece in pieces:
        left, top, right, bottom = painted(session, piece)
        assert 40 - 1 <= left <= right <= 100 + 1
        assert 20 - 1 <= top <= bottom <= 80
    # What is left of the shape lies outside; the dot stays as it was.
    assert painted(session, "blob")[0] >= 100 - 1e-6
    assert session.editor.snapshot.document.element("dot")
    result = send(session, "undo")
    assert result["undo"] == []
    assert export_svg(session.editor.snapshot.document) == original


def test_without_cutting_crossing_objects_stay_whole():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["line", "blob", "dot"])
    result = preview(session, region=REGION, cut=False)
    # Extract has nothing wholly inside to take; Delete takes the dot only.
    # The objects the edge crosses are still named, to say they stay whole.
    assert result["extract"]["error"] == (
        "No contour of these paths lies inside the region: turn on Cut objects"
        " crossing the edge to take the parts inside"
    )
    assert outcomes(result["delete"]) == {"dot": "whole"}
    assert sorted(result["crossing"]) == ["blob", "line"]
    with pytest.raises(DocumentError, match="No contour"):
        send(session, "extract", region=REGION, cut=False)
    line = session.editor.snapshot.document.geometry_for("line")
    result = send(session, "extract", region=REGION, cut=False, delete=True)
    assert result["undo"] == ["Delete in region"]
    document = session.editor.snapshot.document
    assert "dot" not in {e.id for e in document.elements()}
    assert document.geometry_for("line") == line


def test_a_shape_keeps_its_holes_and_a_polygon_region_cuts_along_its_edges():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["ring"])
    # The hole alone is inside: it stays with the shape around it.
    hole = preview(session, region=[10, 130, 60, 60])
    assert (
        hole["extract"]["error"] == "No contour of these paths lies inside the region"
    )
    assert hole["delete"]["error"] == hole["extract"]["error"]
    # A triangle across the ring and its hole cuts them along its edges.
    triangle = [[-10, 110], [100, 110], [-10, 210]]
    result = preview(session, region=triangle)
    assert outcomes(result["extract"]) == {"ring": "cut"}
    result = send(session, "extract", region=triangle)
    assert result["undo"] == ["Extract region"]
    (piece,) = result["selection"]["objects"]
    assert piece != "ring"


def test_instances_are_refused_rather_than_left_out():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["stamp", "blob"])
    original = session.editor.snapshot.document
    result = preview(session, region=REGION)
    message = "is an instance of shared geometry: detach it to editable paths first"
    assert result["extract"]["error"].endswith(message)
    assert result["delete"]["error"].endswith(message)
    assert result["crossing"] is None
    with pytest.raises(DocumentError, match="instance"):
        send(session, "extract", region=REGION)
    # Every unlocked path or shape there includes the instance painting inside.
    with pytest.raises(DocumentError, match="instance"):
        send(session, "extract", region=REGION, scope="region")
    assert session.editor.snapshot.document is original


def test_region_scope_acts_on_every_unlocked_path_there_without_a_selection():
    session = Session(import_svg(SVG))
    session.editor.set_locks("fixed", frozenset({EditKind.GEOMETRY}))
    send(session, "select", objects=[])
    # Away from the instance: the line, the shape and the locked one cross.
    region = [40, 40, 60, 40]
    result = preview(session, region=region, scope="region")
    assert outcomes(result["delete"]) == {
        "line": "cut",
        "blob": "cut",
        "dot": "whole",
        "inner": "whole",
    }
    fixed = session.editor.snapshot.document.geometry_for("fixed")
    result = send(session, "extract", region=region, scope="region", delete=True)
    assert result["undo"][-1] == "Delete in region"
    document = session.editor.snapshot.document
    assert {"dot", "inner"}.isdisjoint(e.id for e in document.elements())
    assert document.geometry_for("fixed") == fixed
    # Inside the entered group only its own paths are reached.
    send(session, "undo")
    result = preview(session, region=region, scope="region", within="moved")
    assert outcomes(result["delete"]) == {"inner": "whole"}
    # The selection scope still needs a selection.
    result = preview(session, region=region)
    assert result["extract"]["error"] == "Select an object first"


def test_pinned_points_refuse_deleting_but_not_extracting():
    session = Session(import_svg(SVG))
    node = session.editor.snapshot.document.geometry_for("dot").subpaths[0].nodes[0]
    session.editor.pin_node("dot", node.id)
    send(session, "select", objects=["dot", "line"])
    result = preview(session, region=REGION)
    assert result["delete"]["error"] == "Unpin the points before deleting them"
    assert outcomes(result["extract"]) == {"line": "cut"}
    with pytest.raises(DocumentError, match="Unpin"):
        send(session, "extract", region=REGION, delete=True)


def test_a_preview_asks_for_a_known_region_scope_and_cut():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["line"])
    with pytest.raises(DocumentError, match="positive size"):
        preview(session, region=[0, 0, 0, 10])
    with pytest.raises(DocumentError, match="cut"):
        preview(session, region=REGION, cut="yes")
    result = preview(session, region=REGION, scope="everything")
    assert result["extract"]["error"] == 'The scope is "selection" or "region"'
