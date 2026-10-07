"""Compact proposals preserve corners, endpoints and canonical shared edges."""

import numpy as np
import pytest

from vectrify.refine import cel
from vectrify.refine.cel_plan.geometry import (
    Boundaries,
    InkBoundaries,
    ellipse,
    fitted,
    ink_limits,
)
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


def test_attached_hairline_cannot_borrow_movement_from_a_broad_ink_shape():
    mask = np.zeros((48, 100), bool)
    mask[16:28, 8:48] = True
    mask[22:23, 48:90] = True
    broad = np.column_stack((np.arange(12, 44), np.full(32, 16)))
    narrow = np.column_stack((np.arange(52, 86), np.full(34, 23)))
    np.testing.assert_array_equal(ink_limits(broad, mask), 0.75)
    np.testing.assert_array_equal(ink_limits(narrow, mask), 0.25)


def test_ink_profile_stops_at_a_real_gap_before_another_dark_material():
    mask = np.zeros((32, 64), bool)
    mask[10, 8:56] = True
    points = np.column_stack((np.arange(12, 52), np.full(40, 10)))
    before = ink_limits(points, mask)
    mask[12:24, 8:56] = True
    np.testing.assert_array_equal(ink_limits(points, mask), before)
    np.testing.assert_array_equal(before, 0.25)


def test_ink_profile_cannot_wrap_to_paint_across_the_canvas():
    mask = np.zeros((32, 48), bool)
    mask[0, 4:44] = True
    mask[-8:, 4:44] = True
    points = np.column_stack((np.arange(8, 40), np.zeros(32)))
    np.testing.assert_array_equal(ink_limits(points, mask), 0.25)


def test_closed_ink_profile_preserves_the_serialization_seam_and_reversal():
    y, x = np.indices((64, 64))
    mask = (x - 32) ** 2 + (y - 32) ** 2 <= 18**2
    points = np.array(_loops(mask)[0])
    points = np.vstack((points, points[0]))
    limits = ink_limits(points, mask)
    assert limits[0] == limits[-1]
    np.testing.assert_array_equal(ink_limits(points[::-1], mask), limits[::-1])


def test_pointwise_ink_limit_rejects_a_line_that_erases_a_narrow_bend():
    x = np.linspace(10, 90, 160)
    points = np.column_stack((x, 40 + 0.35 * np.sin(x / 12)))
    callback = InkBoundaries()
    assert callback(points, 0.75) == [("L", tuple(points[-1]))]
    callback = InkBoundaries()
    nodes = callback(points, 0.75, limits=np.full(len(points), 0.25))
    assert nodes != [("L", tuple(points[-1]))]
    assert nodes[-1][1][-2:] == tuple(points[-1])
    assert any(d["model"] == "raw-curve" for d in callback.decisions)


def test_local_ink_profile_has_a_fixed_per_chain_work_bound():
    with pytest.raises(ValueError, match="point bound"):
        ink_limits(np.zeros((4097, 2)), np.ones((16, 16), bool))


def test_straight_ink_fit_can_use_wide_support_without_moving_narrow_ends():
    x = np.linspace(0, 100, 160)
    points = np.column_stack((x, 0.5 * np.sin(np.pi * x / 100)))
    limits = np.maximum(0.25, points[:, 1] + 0.01)
    callback = InkBoundaries()
    nodes = callback(points, 0.75, limits=limits)
    assert nodes == [("L", tuple(points[-1]))]
    assert callback.decisions == [{"model": "straight", "nodes": 1}]


def test_broad_ink_cannot_lend_width_to_a_narrow_material_projection():
    materials = np.zeros((48, 100), np.int32)
    materials[16:32, 8:90] = 1
    materials[15, 48:90] = 2
    mask = materials == 1
    points = np.column_stack((np.arange(52, 86), np.full(34, 16)))
    np.testing.assert_array_equal(ink_limits(points, mask), 0.75)
    np.testing.assert_array_equal(ink_limits(points, mask, materials=materials), 0.25)
    materials[4:16, 48:90] = 2
    np.testing.assert_array_equal(ink_limits(points, mask, materials=materials), 0.75)


def test_neighbor_profile_stops_at_a_hole_and_does_not_skip_to_same_paint():
    materials = np.ones((48, 80), np.int32)
    materials[:16] = 2
    materials[14] = 0
    mask = materials == 1
    points = np.column_stack((np.arange(8, 72), np.full(64, 16)))
    np.testing.assert_array_equal(ink_limits(points, mask, materials=materials), 0.25)
    materials[14] = 2
    np.testing.assert_array_equal(ink_limits(points, mask, materials=materials), 0.75)


def test_a_narrow_unpainted_hole_has_the_same_bound_as_narrow_paint():
    materials = np.ones((48, 80), np.int32)
    materials[15] = 0
    points = np.column_stack((np.arange(8, 72), np.full(64, 16)))
    np.testing.assert_array_equal(
        ink_limits(points, materials == 1, materials=materials), 0.25
    )


def test_material_bounds_are_independent_of_class_numbers_and_chain_direction():
    y, x = np.indices((64, 64))
    materials = np.where((x - 32) ** 2 + (y - 32) ** 2 <= 18**2, 7, 21)
    mask = materials == 7
    points = np.array(_loops(mask)[0])
    points = np.vstack((points, points[0]))
    limits = ink_limits(points, mask, materials=materials)
    np.testing.assert_array_equal(
        ink_limits(points[::-1], mask, materials=materials), limits[::-1]
    )
    np.testing.assert_array_equal(
        ink_limits(points, mask, materials=28 - materials), limits
    )
