"""The editor's line tools: break, join, and fills to lines and back."""

import math

import pytest

from tests.ui.test_session import send
from vectrify.document import DocumentError, import_svg
from vectrify.document.join import curve_path
from vectrify.ui.session import Session

LINE = 'fill="none" stroke="#123456" stroke-width="2"'


def session_with(body):
    return Session(import_svg(f'<svg width="100" height="100">{body}</svg>'))


def points(session, oid):
    geometry = session.editor.snapshot.document.geometry_for(oid)
    return [[n.endpoint for n in s.nodes] for s in geometry.subpaths]


def ids(session, oid):
    geometry = session.editor.snapshot.document.geometry_for(oid)
    return [[n.id for n in s.nodes] for s in geometry.subpaths]


def test_break_at_point_keeps_both_copies_selected_and_undoes_in_one_step():
    session = session_with(f'<path id="p" d="M0 0L10 0L20 0" {LINE}/>')
    middle = ids(session, "p")[0][1]
    send(session, "select", objects=["p"], nodes=[middle])
    result = send(session, "break_points", points=[["p", middle]])
    assert result["undo"] == ["Break at point"]
    assert points(session, "p") == [[(0, 0), (10, 0)], [(10, 0), (20, 0)]]
    assert len(result["selection"]["nodes"]) == 2
    send(session, "undo")
    assert points(session, "p") == [[(0, 0), (10, 0), (20, 0)]]


def test_delete_segment_between_two_points():
    session = session_with(f'<path id="p" d="M0 0L10 0L20 0L30 0" {LINE}/>')
    chosen = ids(session, "p")[0][1:3]
    send(session, "select", objects=["p"], nodes=chosen)
    result = send(session, "delete_segment", points=[["p", n] for n in chosen])
    assert result["undo"] == ["Delete segment"]
    assert points(session, "p") == [[(0, 0), (10, 0)], [(20, 0), (30, 0)]]


def test_joining_two_selected_ends_of_two_paths():
    session = session_with(
        f'<path id="a" d="M0 0L10 0" {LINE}/><path id="b" d="M30 0L40 0" {LINE}/>'
    )
    a, b = ids(session, "a")[0][-1], ids(session, "b")[0][0]
    send(session, "select", objects=["a", "b"], nodes=[a, b])
    result = send(session, "join_two_ends", points=[["a", a], ["b", b]])
    assert result["undo"] == ["Join ends"]
    assert [o["id"] for o in result["objects"]] == ["b"]
    assert points(session, "b")[0][0] == (0, 0)
    assert points(session, "b")[0][-1] == (40, 0)


def test_join_ends_command_reaches_as_far_as_asked():
    body = f'<path id="a" d="M0 0L10 0" {LINE}/><path id="b" d="M14 0L24 0" {LINE}/>'
    session = session_with(body)
    send(session, "select", objects=["a", "b"])
    with pytest.raises(DocumentError, match="close enough"):
        send(session, "join_ends", reach=3)
    result = send(session, "join_ends", reach=5)
    assert result["selection"]["objects"] == ["b"]
    assert len(points(session, "b")) == 1
    assert session.editor.snapshot.document.element("b").get("stroke") == "#123456"


def test_fill_to_line_turns_a_thin_bar_into_a_stroke_down_its_middle():
    session = session_with(
        '<rect id="back" width="1" height="1"/>'
        '<path id="bar" d="M10 20H70V24H10Z" fill="#a00" fill-opacity=".5"/>'
        '<rect id="front" width="1" height="1"/>'
    )
    send(session, "select", objects=["bar"])
    result = send(session, "fill_to_line")
    assert result["undo"] == ["Fill to line"]
    document = session.editor.snapshot.document
    assert [c.id for c in document.root.children] == ["back", "bar", "front"]
    element = document.element("bar")
    assert element.get("fill") == "none"
    assert element.get("stroke") == "#a00"
    assert element.get("stroke-opacity") == ".5"
    assert element.get("stroke-linecap") == element.get("stroke-linejoin") == "round"
    assert float(element.get("stroke-width") or 0) == pytest.approx(4, abs=0.4)
    ((line),) = points(session, "bar")
    assert all(y == pytest.approx(22, abs=0.3) for _, y in line)
    # Round caps give back the length the centreline is short of the ends.
    xs = sorted(x for x, _ in (line[0], line[-1]))
    assert xs[0] - 2 == pytest.approx(10, abs=1)
    assert xs[1] + 2 == pytest.approx(70, abs=1)


def test_fill_to_line_follows_a_curved_band():
    outer = [(50 + 30 * math.cos(a), 50 - 30 * math.sin(a)) for a in angles()]
    inner = [(50 + 26 * math.cos(a), 50 - 26 * math.sin(a)) for a in angles()][::-1]
    data = "M" + " L".join(f"{x:.3f} {y:.3f}" for x, y in outer + inner) + "Z"
    session = session_with(f'<path id="band" d="{data}" fill="green"/>')
    send(session, "select", objects=["band"])
    send(session, "fill_to_line")
    document = session.editor.snapshot.document
    assert float(document.element("band").get("stroke-width") or 0) == pytest.approx(
        4, abs=0.4
    )
    (line,) = points(session, "band")
    assert len(line) > 1
    for x, y in line:
        assert math.hypot(x - 50, y - 50) == pytest.approx(28, abs=0.4)


def angles():
    return [math.pi * i / 60 for i in range(61)]


def test_fill_to_line_refuses_a_blob():
    session = session_with('<path id="p" d="M0 0H20V20H0Z" fill="red"/>')
    send(session, "select", objects=["p"])
    with pytest.raises(DocumentError, match="not a thin line"):
        send(session, "fill_to_line")


def test_line_to_fill_outlines_the_stroke():
    session = session_with(f'<path id="p" d="M10 10L30 10" {LINE}/>')
    send(session, "select", objects=["p"])
    result = send(session, "line_to_fill")
    assert result["undo"] == ["Line to fill"]
    document = session.editor.snapshot.document
    element = document.element("p")
    assert element.get("fill") == "#123456"
    assert element.get("stroke") == "none"
    assert abs(curve_path(document.geometry_for("p")).area) == pytest.approx(40)


def test_the_knife_cuts_a_selected_line_into_two_paths():
    session = session_with(f'<path id="p" d="M0 0L30 0" {LINE}/>')
    send(session, "select", objects=["p"])
    result = send(session, "knife", start=[10, -5], end=[10, 5])
    assert result["undo"] == ["Cut with knife"]
    assert len(result["selection"]["objects"]) == 2


def test_join_two_ends_joins_any_two_points_as_one_edit():
    # A point in the middle of a line, and the corner of a filled square.
    body = (
        f'<path id="a" d="M0 0L10 0L20 0" {LINE}/>'
        '<path id="b" d="M40 0L50 0L50 10L40 10Z" fill="#336699"/>'
    )
    session = session_with(body)
    middle, corner = ids(session, "a")[0][1], ids(session, "b")[0][2]
    send(session, "select", objects=["a", "b"], nodes=[middle, corner])
    result = send(session, "join_two_ends", points=[["a", middle], ["b", corner]])
    assert result["undo"] == ["Join ends"]
    joined = [
        contour
        for oid in ("a", "b")
        if oid in {o["id"] for o in result["objects"]}
        for contour in points(session, oid)
    ]
    # Somewhere a contour now runs from the line's middle to the corner.
    assert any((10, 0) in contour and (50, 10) in contour for contour in joined)
    send(session, "undo")
    assert points(session, "a") == [[(0, 0), (10, 0), (20, 0)]]


def test_join_two_ends_refuses_one_point():
    session = session_with(f'<path id="a" d="M0 0L10 0L20 0" {LINE}/>')
    end = ids(session, "a")[0][0]
    send(session, "select", objects=["a"], nodes=[end])
    with pytest.raises(DocumentError, match="two points"):
        send(session, "join_two_ends", points=[["a", end]])
