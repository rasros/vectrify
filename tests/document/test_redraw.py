"""Redrawing a stretch of a contour: where strokes attach and the splice."""

import pytest

from vectrify.document import (
    DocumentError,
    Editor,
    EditRejectedError,
    Selection,
    import_svg,
)
from vectrify.document.redraw import attachment

SQUARE = "M0 0 L100 0 L100 100 L0 100 Z"
BUMP = [("L", (50, -20)), ("L", (80, 0))]


def editor(d, wrapper=""):
    path = f'<path id="p" d="{d}"/>'
    if wrapper:
        path = f'<g transform="{wrapper}">{path}</g>'
    result = Editor(import_svg(f'<svg width="400" height="400">{path}</svg>'))
    result.select(Selection(object_ids=frozenset({"p"})))
    return result


def contour(e):
    return e.snapshot.document.geometry_for("p").subpaths[0]


def nodes(e):
    return contour(e).nodes


def points(e):
    return [pytest.approx(n.endpoint) for n in nodes(e)]


def redraw(e, start, end, stretch, **options):
    with e.transaction("Redraw outline") as tx:
        return tx.redraw_outline("p", contour(e).id, start, end, stretch, **options)


def test_a_stretch_of_one_side_is_replaced_and_every_other_point_stays():
    e = editor(SQUARE)
    before = nodes(e)
    top = before[1].id
    added = redraw(e, (top, 0.2), (top, 0.8), BUMP)
    assert points(e) == [
        (0, 0),
        (20, 0),
        (50, -20),
        (80, 0),
        (100, 0),
        (100, 100),
        (0, 100),
    ]
    after = nodes(e)
    assert [n.id for n in after if n.id not in added] == [n.id for n in before]
    assert all(n in after for n in before)
    assert e.undo_labels == ("Redraw outline",)
    e.undo()
    assert nodes(e) == before


def test_the_shorter_way_round_is_replaced_unless_asked_for_the_longer():
    e = editor(SQUARE)
    top, right = nodes(e)[1].id, nodes(e)[2].id
    redraw(e, (top, 0.5), (right, 0.5), [("L", (90, 10)), ("L", (100, 50))])
    # The corner between the two places went; the rest stayed.
    assert points(e) == [(0, 0), (50, 0), (90, 10), (100, 50), (100, 100), (0, 100)]
    e.undo()
    redraw(
        e, (top, 0.5), (right, 0.5), [("L", (90, 10)), ("L", (100, 50))], long_way=True
    )
    # The long way runs from the right side on round to the top, backwards
    # along the stroke, and takes the start point with it.
    assert points(e) == [(50, 0), (100, 0), (100, 50), (90, 10)]
    assert contour(e).closed


def test_a_stroke_drawn_against_the_contour_is_turned_round():
    e = editor(SQUARE)
    top = nodes(e)[1].id
    redraw(e, (top, 0.8), (top, 0.2), [("C", (70, -10, 60, -20, 50, -20)), BUMP[0]])
    assert points(e)[:5] == [(0, 0), (20, 0), (50, -20), (80, 0), (100, 0)]
    # The curve's handles are swapped with it.
    assert nodes(e)[3].values == pytest.approx((60, -20, 70, -10, 80, 0))


def test_the_closing_line_and_a_start_point_ending_the_contour_can_be_redrawn():
    e = editor(SQUARE)
    start = nodes(e)[0].id
    # The implicit closing line, from (0, 100) back to the start.
    redraw(e, (start, 0.25), (start, 0.75), [("L", (-20, 50)), ("L", (0, 25))])
    assert points(e) == [
        (0, 0),
        (100, 0),
        (100, 100),
        (0, 100),
        (0, 75),
        (-20, 50),
        (0, 25),
    ]
    assert nodes(e)[-1].command == "L"
    assert contour(e).closed
    e = editor("M0 0 L100 0 L100 100 L0 100 L0 0 Z")
    ids = [n.id for n in nodes(e)]
    # Across the start, from the left side to the top.
    redraw(e, (ids[4], 0.5), (ids[1], 0.5), [("L", (-10, -10)), ("L", (50, 0))])
    assert points(e)[0] == (50, 0)
    assert points(e)[1:4] == [(100, 0), (100, 100), (0, 100)]
    assert [n.id for n in nodes(e)][1:4] == ids[1:4]
    assert ids[0] not in {n.id for n in nodes(e)}


def test_an_open_contour_is_redrawn_between_its_places_in_either_direction():
    for forward in (True, False):
        e = editor("M0 0 L100 0 L200 0")
        first, second = (n.id for n in nodes(e)[1:])
        places = [(first, 0.5), (second, 0.5)]
        stretch = [("L", (100, -30)), ("L", (150, 0))]
        if not forward:
            places.reverse()
            stretch = [("L", (100, -30)), ("L", (50, 0))]
        redraw(e, *places, stretch)
        assert points(e) == [(0, 0), (50, 0), (100, -30), (150, 0), (200, 0)]
        assert not contour(e).closed
        assert (
            nodes(e)[0].id
            == e.snapshot.document.geometry_for("p").subpaths[0].nodes[0].id
        )
        assert nodes(e)[-1].id == second


def test_an_open_contour_can_be_redrawn_from_its_end():
    e = editor("M0 0 L100 0 L200 0")
    start, middle, end = (n.id for n in nodes(e))
    redraw(e, (middle, 0.0), (end, 0.5), [("L", (50, 20)), ("L", (150, 0))])
    assert points(e) == [(0, 0), (50, 20), (150, 0), (200, 0)]
    assert nodes(e)[0].id == start
    assert middle not in {n.id for n in nodes(e)}


def test_pinned_points_inside_the_stretch_refuse_and_at_its_ends_do_not():
    e = editor(SQUARE)
    top, right = nodes(e)[1].id, nodes(e)[2].id
    e.pin_node("p", top, pinned=True)
    with pytest.raises(EditRejectedError, match="Unpin"):
        redraw(e, (top, 0.5), (right, 0.5), [("L", (100, 50))])
    assert e.undo_labels == ("Pin node",)
    redraw(e, (top, 1.0), (right, 0.5), [("L", (90, 30)), ("L", (100, 50))])
    assert nodes(e)[1].id == top
    assert nodes(e)[1].pinned


def test_a_stretch_needs_two_different_places():
    e = editor(SQUARE)
    top = nodes(e)[1].id
    with pytest.raises(DocumentError, match="different"):
        redraw(e, (top, 0.5), (top, 0.5), BUMP)


def test_strokes_attach_to_a_near_point_else_the_nearest_place_on_the_outline():
    e = editor(SQUARE, "translate(10 20) scale(2)")
    document = e.snapshot.document
    corner = nodes(e)[1]
    # (100, 0) is at (210, 20) in root user space.
    near = attachment(document, "p", (213, 22), reach=10, node_reach=5)
    assert near is not None
    assert (near.node_id, near.t, near.point) == (corner.id, 1.0, (210, 20))
    along = attachment(document, "p", (110, 26), reach=10, node_reach=5)
    assert along is not None
    assert along.node_id == corner.id
    assert along.t == pytest.approx(0.5)
    assert along.point == pytest.approx((110, 20))
    assert attachment(document, "p", (110, 40), reach=10, node_reach=5) is None
    # An open contour's start is a point too.
    e = editor("M0 0 L100 0")
    start = attachment(e.snapshot.document, "p", (1, 1), reach=10, node_reach=5)
    assert start is not None
    assert (start.node_id, start.t) == (nodes(e)[1].id, 0.0)
