"""Compact proposals preserve corners, endpoints and canonical shared edges."""

import numpy as np

from vectrify.refine import cel
from vectrify.refine.cel_plan.geometry import Boundaries, InkBoundaries, ellipse, fitted
from vectrify.refine.cel_plan.score import render
from vectrify.refine.tracing import _loops


def circle():
    y, x = np.mgrid[:80, :80]
    points = np.array(_loops((x - 40) ** 2 + (y - 40) ** 2 <= 22**2)[0])
    return np.vstack((points, points[0]))


def test_noisy_round_contour_has_a_bounded_editable_ellipse_proposal():
    points = circle()
    model = ellipse(points, 1.2)
    assert model is not None
    assert model.kind == "ellipse"
    assert model.residual is not None
    assert model.residual <= 1.2
    assert model.contour.closed
    assert len(model.contour.nodes) == 5
    assert model.contour.nodes[0].endpoint == tuple(points[0])
    assert model.contour.nodes[-1].endpoint == tuple(points[0])
    assert all(node.command == "C" for node in model.contour.nodes[1:])


def test_cornered_closed_shape_is_not_turned_into_an_ellipse():
    points = np.array(_loops(np.pad(np.ones((36, 36), dtype=bool), 12))[0])
    points = np.vstack((points, points[0]))
    assert ellipse(points, 6) is None


def test_short_arc_cannot_be_completed_into_a_closed_shape():
    points = circle()[:30]
    assert ellipse(points, 2) is None


def test_nearly_straight_run_keeps_exact_junction_endpoints():
    x = np.linspace(10, 90, 160)
    points = np.column_stack((x, 40 + 0.2 * np.sin(x)))
    model = fitted(points, 0.5)
    assert model.kind == "straight"
    assert len(model.contour.nodes) == 2
    assert model.contour.nodes[0].endpoint == tuple(points[0])
    assert model.contour.nodes[-1].endpoint == tuple(points[-1])


def test_supported_pointed_corner_survives_geometry_fitting():
    points = np.vstack(
        (
            np.column_stack((np.arange(0, 21), np.arange(0, 21))),
            np.column_stack((np.arange(21, 42), np.arange(19, -2, -1))),
        )
    )
    model = fitted(points, 3)
    assert model.kind == "curve"
    assert (20.0, 20.0) in {node.endpoint for node in model.contour.nodes}


def test_shared_boundary_model_is_used_by_both_fills_without_a_gap():
    y, x = np.mgrid[:100, :100]
    labels = (x >= 50 + 0.3 * np.sin(y / 4)).astype(np.int32)
    callback = Boundaries()
    outlines = cel.region_outlines(labels, 1, fit_boundary=callback)
    svg = (
        '<svg width="100" height="100">'
        + "".join(
            f'<path d="{data}" fill="{paint}"/>'
            for data, paint in zip(outlines.values(), ("red", "blue"), strict=True)
        )
        + "</svg>"
    )
    pixels = render(svg, (100, 100))
    assert pixels[..., 3].min() >= 0.74  # Antialiasing at a perfectly shared edge.
    assert np.all(pixels[..., :3].sum(axis=-1) <= 1.01)
    assert sum(decision["model"] == "straight" for decision in callback.decisions) == 1


def test_closed_boundary_callback_keeps_its_serialization_anchor():
    y, x = np.mgrid[:80, :80]
    labels = ((x - 40) ** 2 + (y - 40) ** 2 <= 22**2).astype(np.int32)
    callback = Boundaries()
    outlines = cel.region_outlines(labels, 1.2, fit_boundary=callback)
    assert outlines.keys() == {0, 1}
    assert any(item["model"] == "ellipse" for item in callback.decisions)


def test_anchored_ink_line_retains_complete_source_endpoints_without_smoothing():
    x = np.linspace(10, 90, 160)
    points = np.column_stack((x, 40 + 0.2 * np.sin(x)))
    callback = InkBoundaries()
    nodes = callback(points, 0.5)
    assert nodes == [("L", tuple(points[-1]))]
    assert callback.decisions == [{"model": "straight", "nodes": 1}]


def test_anchored_ink_curve_retains_supported_pointed_corner():
    points = np.vstack(
        (
            np.column_stack((np.arange(0, 21), np.arange(0, 21))),
            np.column_stack((np.arange(21, 42), np.arange(19, -2, -1))),
        )
    )
    callback = InkBoundaries()
    nodes = callback(points, 0.75)
    assert (20.0, 20.0) in {tuple(values[-2:]) for _, values in nodes}
    assert nodes[-1][1][-2:] == tuple(points[-1])


def test_small_closed_ink_mark_keeps_the_precise_raw_interpretation():
    points = np.array(_loops(np.pad(np.ones((4, 4), bool), 4))[0], float)
    points = np.vstack((points, points[0]))
    callback = InkBoundaries()
    assert callback(points, 0.75) == cel.curve_nodes(
        points, 0.25, smooth=0, fit=cel.FILL_FIT
    )
    assert callback.decisions[0]["model"] == "precise-mark"


def test_complete_round_ink_perimeter_can_compete_as_a_bounded_ellipse():
    points = circle()
    callback = InkBoundaries()
    nodes = callback(points, 1.2)
    assert len(nodes) == 4
    assert nodes[-1][1][-2:] == tuple(points[0])
    assert callback.decisions[0]["model"] == "ellipse"
