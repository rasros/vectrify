"""Redraw outline: a stroke along the reference's edge redraws that stretch."""

from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from tests.ui.helpers import PIXEL, SPIKED, SQUARE, STROKE, send, session
from vectrify.document import DocumentError, import_svg
from vectrify.document.hit_test import IDENTITY, multiply, transform
from vectrify.document.topology import mapped_point
from vectrify.refine.redraw import redraw_stretch
from vectrify.ui.session import Session


def nodes(session):
    return session.editor.snapshot.document.geometry_for("p").subpaths[0].nodes


def outline(session, samples=40):
    """Points along the contour, in root user space."""
    document = session.editor.snapshot.document
    matrix = IDENTITY
    for ancestor in document.ancestry("p"):
        matrix = multiply(matrix, transform(ancestor.get("transform")))
    found = []
    items = nodes(session)
    for before, node in zip(items, [*items[1:], items[0]], strict=True):
        controls = [before.endpoint]
        if node.command == "C":
            controls += [node.values[0:2], node.values[2:4]]
        controls = np.array([*controls, node.endpoint], dtype=np.float64)
        for t in np.linspace(0, 1, samples):
            if len(controls) == 4:
                u = 1 - t
                point = (
                    u**3 * controls[0]
                    + 3 * u**2 * t * controls[1]
                    + 3 * u * t**2 * controls[2]
                    + t**3 * controls[3]
                )
            else:
                point = (1 - t) * controls[0] + t * controls[1]
            found.append(mapped_point(tuple(point), matrix))
    return found


def strays(session, polygon):
    """How far the contour strays from *polygon*'s outline at most, and how
    close it comes to the spike's tip."""
    edge = Polygon(polygon).exterior
    points = outline(session)
    return (
        max(edge.distance(Point(p)) for p in points),
        min(np.hypot(x - 50, y - 5) for x, y in points),
    )


def test_a_stroke_along_the_spike_redraws_the_top_side_onto_it():
    s = session()
    before = nodes(s)
    result = send(s, "redraw_outline", object="p", points=STROKE, pixel=PIXEL)
    assert result["undo"] == ["Redraw outline"]
    worst, tip = strays(s, SPIKED)
    # Within two reference pixels of the edge, and of its tip.
    assert worst < 1.0
    assert tip < 1.2
    after = nodes(s)
    # The corners stay as they were, IDs and all.
    assert [n for n in after if n in before] == list(before)
    assert after[0] == before[0]
    assert after[-3:] == before[-3:]
    send(s, "undo")
    assert nodes(s) == before


def test_under_a_group_transform_the_edge_is_found_in_root_space():
    # Local (x, y) is drawn at (2x + 10, 2y - 20).
    s = session("M5 20 L35 20 L35 50 L5 50 Z", wrapper="translate(10 -20) scale(2)")
    before = nodes(s)
    send(s, "redraw_outline", object="p", points=STROKE, pixel=PIXEL)
    worst, tip = strays(s, SPIKED)
    assert worst < 1.0
    assert tip < 1.2
    assert all(n in nodes(s) for n in before if n.endpoint[1] == 50)


def test_the_stroke_ends_attach_to_the_nearest_points_on_the_outline():
    s = session()
    corner = nodes(s)[1]
    # Near the top-right corner the end attaches to it, not beside it; the
    # start, a little off the top side, attaches straight above it.
    stroke = [(30, 21.5), *STROKE[1:-1], (79, 20.8)]
    send(s, "redraw_outline", object="p", points=stroke, pixel=PIXEL)
    after = nodes(s)
    assert after[1].endpoint == pytest.approx((30, 20))
    assert corner.id in {n.id for n in after}
    assert not any(abs(n.endpoint[0] - 79) < 0.5 for n in after)


def test_a_stroke_that_misses_the_outline_changes_nothing():
    s = session()
    with pytest.raises(DocumentError, match="outline"):
        send(
            s,
            "redraw_outline",
            object="p",
            points=[(30, 10), *STROKE[1:]],
            pixel=PIXEL,
        )
    assert s.editor.undo_labels == ()


def test_pinned_points_in_the_stretch_refuse_the_redraw():
    s = session()
    corner = nodes(s)[1]
    send(s, "pin", object="p", node=corner.id, pinned=True)
    stroke = [(50, 20), (70, 10), (80, 50)]
    with pytest.raises(DocumentError, match="Unpin"):
        send(s, "redraw_outline", object="p", points=stroke, pixel=PIXEL)
    assert nodes(s)[1] == replace(corner, pinned=True)


def test_without_a_reference_the_stroke_itself_is_fitted():
    s = session(reference=False)
    before = nodes(s)
    send(s, "redraw_outline", object="p", points=SPIKED[:5], pixel=PIXEL)
    worst, tip = strays(s, SPIKED)
    # Within about a screen pixel of the stroke, its corners kept.
    assert worst < 2 * PIXEL
    assert tip < 2 * PIXEL
    assert nodes(s)[-2:] == before[2:]
    assert before[0] == nodes(s)[0]
    assert before[1].id == nodes(s)[-3].id


def test_an_open_contour_is_redrawn_between_the_ends_of_the_stroke():
    s = session("M20 20 L80 20", reference=False)
    start, end = nodes(s)
    send(s, "redraw_outline", object="p", points=[(30, 20), (50, 5), (70, 20)], pixel=1)
    after = nodes(s)
    assert (after[0], after[-1]) == (start, end)
    assert min(n.endpoint[1] for n in after) == pytest.approx(5, abs=1)


def test_redrawing_one_of_several_selected_paths_keeps_them_selected():
    s = Session(
        import_svg(
            '<svg width="100" height="100" viewBox="0 0 100 100"><g id="g">'
            f'<path id="p" fill="black" d="{SQUARE}"/>'
            '<path id="q" fill="red" d="M85 85 L95 85 L95 95 Z"/></g></svg>'
        )
    )
    send(s, "select", objects=["p", "q"])
    send(s, "redraw_outline", object="p", points=[(30, 20), (50, 5), (70, 20)], pixel=1)
    assert s.state()["selection"]["objects"] == ["p", "q"]
    # With the group selected, its paths' outlines can be redrawn too.
    send(s, "select", objects=["g"])
    send(s, "redraw_outline", object="p", points=[(20, 30), (5, 50), (20, 70)], pixel=1)
    assert s.state()["selection"]["objects"] == ["g"]
    assert min(n.endpoint[0] for n in nodes(s)) == pytest.approx(5, abs=1)


@pytest.mark.parametrize("length", [9, 10, 17])
def test_a_freehand_stroke_ending_just_past_a_segment_boundary_is_redrawn(length):
    # Its second-to-last point starts a segment, whose arriving tangent looks
    # past the end of the stroke.
    document = import_svg(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<path id="a" d="M 0 0 L 40 0"/></svg>'
    )
    xs = np.linspace(0, length, length + 1)
    stroke = list(zip(xs, 4 * np.sin(xs * 3 / length), strict=True))
    segments = redraw_stretch(document, "a", stroke, None, 1.0)
    assert segments
    assert np.allclose(segments[-1][1][-2:], stroke[-1])
