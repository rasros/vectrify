"""The shape tools draw native shapes in root user space."""

import pytest

from vectrify.document import DocumentError, export_svg, import_svg
from vectrify.document.project import load_project, save_project
from vectrify.document.transforms import object_matrix
from vectrify.ui.session import Session

SVG = """<svg width="200" height="200">
<g id="moved" transform="translate(10 5) scale(2)"><path id="a" d="M0 0 L5 5 Z"/></g>
<g id="turned" transform="rotate(30 50 50)"><path id="b" d="M0 0 L5 0 L5 5 Z"/></g>
<g id="mirrored" transform="matrix(-1 0 0 1 200 0)"/>
</svg>"""


def send(session, command, **data):
    return session.action(
        {
            "command": command,
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            **data,
        }
    )


def drawn(session, state):
    (oid,) = state["selection"]["objects"]
    return session.editor.snapshot.document.element(oid)


def root_point(session, oid, x, y):
    """Where the point (x, y) in *oid*'s own frame shows, in root user space."""
    a, b, c, d, e, f = object_matrix(session.editor.snapshot.document, oid)
    return (round(a * x + c * y + e, 9) + 0.0, round(b * x + d * y + f, 9) + 0.0)


@pytest.mark.parametrize(
    ("shape", "values", "label"),
    [
        ("rect", {"x": 20, "y": 30, "width": 40, "height": 10}, "Draw rectangle"),
        ("circle", {"cx": 50, "cy": 60, "r": 12.5}, "Draw circle"),
        ("ellipse", {"cx": 50, "cy": 60, "rx": 20, "ry": 8}, "Draw ellipse"),
        ("line", {"x1": 1, "y1": 2, "x2": 30, "y2": 40}, "Draw line"),
    ],
)
def test_shape_is_native_painted_selected_and_one_undo(shape, values, label):
    session = Session(import_svg(SVG))
    before = session.editor.snapshot.document
    state = send(
        session,
        "add_shape",
        shape=shape,
        **values,
        fill="#ff0000",
        stroke="#0000ff",
        stroke_width=3,
    )
    element = drawn(session, state)
    assert element.tag == shape
    assert {k: float(element.get(k)) for k in values} == values
    assert element.get("transform") is None
    # A line fills nothing; closed shapes take both colours.
    assert element.get("fill") == ("none" if shape == "line" else "#ff0000")
    assert element.get("stroke") == "#0000ff"
    assert element.get("stroke-width") == "3"
    assert state["undo"] == [label]
    document = session.editor.snapshot.document
    assert send(session, "undo")["selection"]["objects"] == []
    assert session.editor.snapshot.document == before
    send(session, "redo")
    assert session.editor.snapshot.document == document


def test_default_paint_is_the_drawing_green():
    session = Session(import_svg(SVG))
    rect = drawn(
        session, send(session, "add_shape", shape="rect", x=0, y=0, width=5, height=5)
    )
    assert (rect.get("fill"), rect.get("stroke")) == ("#83b899", "none")
    line = drawn(
        session, send(session, "add_shape", shape="line", x1=0, y1=0, x2=5, y2=0)
    )
    assert (line.get("fill"), line.get("stroke")) == ("none", "#83b899")


def test_rounded_corners_round_at_most_to_half_a_side():
    session = Session(import_svg(SVG))
    rect = drawn(
        session,
        send(session, "add_shape", shape="rect", x=0, y=0, width=40, height=10, rx=8),
    )
    assert rect.get("rx") == "5"
    sharp = drawn(
        session,
        send(session, "add_shape", shape="rect", x=0, y=0, width=40, height=10, rx=0),
    )
    assert sharp.get("rx") is None


def test_moved_and_scaled_group_takes_the_shape_in_its_own_numbers():
    session = Session(import_svg(SVG))
    state = send(
        session,
        "add_shape",
        shape="rect",
        x=30,
        y=25,
        width=40,
        height=20,
        rx=4,
        stroke="#000000",
        stroke_width=4,
        parent="moved",
    )
    rect = drawn(session, state)
    document = session.editor.snapshot.document
    assert document.ancestry(rect.id)[-2].id == "moved"
    assert rect.get("transform") is None
    assert [rect.get(k) for k in ("x", "y", "width", "height", "rx")] == [
        "10",
        "10",
        "20",
        "10",
        "2",
    ]
    # Twice as wide in a group scaled by two: as wide as drawn.
    assert rect.get("stroke-width") == "2"
    assert root_point(session, rect.id, 10, 10) == (30, 25)
    assert root_point(session, rect.id, 30, 20) == (70, 45)


def test_mirrored_group_keeps_a_positive_size():
    session = Session(import_svg(SVG))
    state = send(
        session,
        "add_shape",
        shape="rect",
        x=30,
        y=25,
        width=40,
        height=20,
        parent="mirrored",
    )
    rect = drawn(session, state)
    assert float(rect.get("width")) == 40
    x = float(rect.get("x"))
    corners = {
        root_point(session, rect.id, x, 25),
        root_point(session, rect.id, x + 40, 25),
    }
    assert corners == {(30, 25), (70, 25)}


@pytest.mark.parametrize("shape", ["ellipse", "line"])
def test_rotated_group_keeps_the_shape_where_it_was_drawn(shape):
    session = Session(import_svg(SVG))
    values = (
        {"cx": 60, "cy": 40, "rx": 20, "ry": 10}
        if shape == "ellipse"
        else {"x1": 10, "y1": 20, "x2": 90, "y2": 20}
    )
    state = send(
        session, "add_shape", shape=shape, **values, stroke_width=3, parent="turned"
    )
    element = drawn(session, state)
    document = session.editor.snapshot.document
    assert document.ancestry(element.id)[-2].id == "turned"
    assert element.tag == shape
    assert element.get("transform")
    # The transform undoes the group's: the shape's own numbers are root ones.
    matrix = object_matrix(document, element.id)
    assert matrix == pytest.approx((1, 0, 0, 1, 0, 0), abs=1e-12)
    assert {k: float(element.get(k)) for k in values} == values
    assert element.get("stroke-width") == "3"


def test_shapes_round_trip_through_projects_and_svg_and_convert_to_paths():
    session = Session(import_svg(SVG))
    ids = []
    for shape, values in (
        ("rect", {"x": 5, "y": 5, "width": 30, "height": 20, "rx": 3}),
        ("circle", {"cx": 50, "cy": 50, "r": 10}),
        ("ellipse", {"cx": 80, "cy": 50, "rx": 10, "ry": 5}),
        ("line", {"x1": 0, "y1": 100, "x2": 100, "y2": 100}),
    ):
        ids.append(drawn(session, send(session, "add_shape", shape=shape, **values)).id)
    document = session.editor.snapshot.document
    assert load_project(save_project(document))[0] == document
    exported = import_svg(export_svg(document))
    for oid in ids:
        assert exported.element(oid).tag == document.element(oid).tag
        assert exported.element(oid).attributes == document.element(oid).attributes
    send(session, "select", objects=ids)
    send(session, "convert_to_path")
    document = session.editor.snapshot.document
    assert {document.element(oid).tag for oid in ids} == {"path"}
    assert all(document.geometry_for(oid).subpaths for oid in ids)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"shape": "polygon"}, "Draw a rect"),
        ({"shape": "rect", "x": 0, "y": 0, "width": 10}, "needs height"),
        ({"shape": "rect", "x": 0, "y": 0, "width": 10, "height": 0}, "some width"),
        ({"shape": "circle", "cx": 0, "cy": 0, "r": -1}, "some width"),
        ({"shape": "ellipse", "cx": 0, "cy": 0, "rx": 0, "ry": 3}, "some width"),
        ({"shape": "line", "x1": 1, "y1": 1, "x2": 1, "y2": 1}, "some length"),
        ({"shape": "rect", "x": "a", "y": 0, "width": 1, "height": 1}, "number"),
        (
            {"shape": "rect", "x": 0, "y": 0, "width": 1, "height": 1, "fill": "u()"},
            "colour",
        ),
        (
            {"shape": "line", "x1": 0, "y1": 0, "x2": 1, "y2": 0, "stroke": "none"},
            "stroke",
        ),
        (
            {"shape": "rect", "x": 0, "y": 0, "width": 1, "height": 1, "fill": "none"},
            "fill or a stroke",
        ),
        (
            {
                "shape": "rect",
                "x": 0,
                "y": 0,
                "width": 1,
                "height": 1,
                "stroke_width": 0,
            },
            "positive",
        ),
        (
            {"shape": "rect", "x": 0, "y": 0, "width": 1, "height": 1, "parent": "a"},
            "container",
        ),
    ],
)
def test_invalid_shapes_do_not_change_the_drawing(data, message):
    session = Session(import_svg(SVG))
    before = session.editor.snapshot.document
    with pytest.raises(DocumentError, match=message):
        send(session, "add_shape", **data)
    assert session.editor.snapshot.document == before
    assert session.editor.undo_labels == ()


def test_draw_path_takes_the_drawing_paint_and_container():
    session = Session(import_svg(SVG))
    state = send(
        session,
        "add_path",
        d="M30 25 L70 25 L70 45 Z",
        fill="#123456",
        stroke="#abcdef",
        stroke_width=1.5,
        parent="moved",
    )
    path = drawn(session, state)
    assert (path.get("fill"), path.get("stroke"), path.get("stroke-width")) == (
        "#123456",
        "#abcdef",
        "1.5",
    )
    assert session.editor.snapshot.document.ancestry(path.id)[-2].id == "moved"
    assert root_point(session, path.id, 30, 25) == (30, 25)
