"""The knife and Redraw act on what is under the pointer; a selection only
narrows that."""

import pytest

from tests.ui.test_redraw import PIXEL, STROKE
from tests.ui.test_redraw import session as redraw_session
from tests.ui.test_session import send
from vectrify.document import DocumentError, import_svg
from vectrify.ui.session import Session

SHAPES = (
    '<svg width="100" height="100">'
    '<path id="a" d="M0 0H20V20H0Z" fill="red"/>'
    '<path id="b" d="M30 0H50V20H30Z" fill="blue"/>'
    '<path id="line" d="M0 40H50" fill="none" stroke="black"/>'
    '<g id="g"><path id="c" d="M60 0H80V20H60Z" fill="green"/></g>'
    "</svg>"
)


def cut(session, start, end, **data):
    return send(session, "knife", start=start, end=end, **data)


def test_without_a_selection_the_knife_cuts_every_path_it_crosses():
    session = Session(import_svg(SHAPES))
    result = cut(session, [-5, 10], [55, 10])
    # Both squares, not the line it misses or the square it does not reach.
    assert len(result["selection"]["objects"]) == 4
    assert {"a", "b"} <= set(result["selection"]["objects"])
    assert result["undo"] == ["Cut with knife"]
    send(session, "select", objects=[])
    result = cut(session, [10, 35], [10, 45])
    assert "line" in result["selection"]["objects"]


def test_a_selection_narrows_what_the_knife_cuts():
    session = Session(import_svg(SHAPES))
    send(session, "select", objects=["a"])
    result = cut(session, [-5, 10], [55, 10])
    assert set(result["selection"]["objects"]) >= {"a"}
    assert len(result["selection"]["objects"]) == 2
    assert len(session.editor.snapshot.document.geometry_for("b").subpaths) == 1
    doc = session.editor.snapshot.document
    assert [e.id for e in doc.root.children].count("b") == 1


def test_the_knife_leaves_locked_paths_and_keeps_to_the_entered_group():
    session = Session(import_svg(SHAPES))
    send(session, "select", objects=["a"])
    send(session, "locks", object="a", locks=["geometry"])
    send(session, "select", objects=[])
    result = cut(session, [-5, 10], [85, 10], within="g")
    assert set(result["selection"]["objects"]) >= {"c"}
    assert len(result["selection"]["objects"]) == 2
    send(session, "select", objects=[])
    result = cut(session, [-5, 10], [55, 10])
    assert "a" not in result["selection"]["objects"]


def test_a_knife_line_across_nothing_says_so():
    session = Session(import_svg(SHAPES))
    with pytest.raises(DocumentError, match="crosses no line"):
        cut(session, [0, 90], [90, 90])


def test_redraw_picks_the_path_its_stroke_starts_on():
    s = redraw_session()
    send(s, "select", objects=[])
    result = send(s, "redraw_outline", object="p", points=STROKE, pixel=PIXEL)
    assert result["undo"] == ["Redraw outline"]
    assert result["selection"]["objects"] == ["p"]


def test_with_a_selection_redraw_keeps_to_it():
    s = Session(
        import_svg(
            '<svg width="100" height="100"><path id="p" d="M20 20H80V80H20Z"/>'
            '<path id="q" d="M0 0H10V10H0Z"/></svg>'
        )
    )
    send(s, "select", objects=["q"])
    with pytest.raises(DocumentError, match="selected path"):
        send(s, "redraw_outline", object="p", points=STROKE, pixel=PIXEL)
