"""Whole-guide construction preserves measured ports/corners within fit bounds."""

import numpy as np
import pytest

from vectrify.refine.cel_plan import source_centerline
from vectrify.refine.cel_plan.model import StageInterruptedError, Work


def guides():
    t = np.linspace(0, 1, 41)
    left = np.column_stack((10 + 20 * t + np.sin(t * np.pi), 10 + 25 * t))
    right = np.column_stack((30 + 20 * t - np.sin(t * np.pi), 35 - 25 * t))
    guide = np.vstack((left, right[1:]))
    measured = guide.copy()
    measured[40] += (0.2, -0.2)
    return guide, measured


@pytest.mark.parametrize("rotate", [False, True])
def test_whole_curve_keeps_raw_corner_and_bound_ports_exact_without_changing_inputs(
    rotate,
):
    guide, measured = guides()
    if rotate:
        matrix = np.array(((0.8, 0.6), (-0.6, 0.8)))
        guide = guide @ matrix + (60, 20)
        measured = measured @ matrix + (60, 20)
    before = guide.copy(), measured.copy()
    result = source_centerline.construct(guide, measured, Work.start(10))
    assert result is not None
    assert result.corners == (tuple(measured[40]),)
    subpath = result.geometry.subpaths[0]
    assert not subpath.closed
    endpoints = {node.endpoint for node in subpath.nodes}
    assert tuple(measured[40]) in endpoints
    np.testing.assert_array_equal(subpath.nodes[0].endpoint, guide[0])
    np.testing.assert_array_equal(subpath.nodes[-1].endpoint, guide[-1])
    np.testing.assert_array_equal(guide, before[0])
    np.testing.assert_array_equal(measured, before[1])
    assert result.parameters <= 24
    assert not result.points.flags.writeable
    assert len({node.id for node in subpath.nodes}) == len(subpath.nodes)


def test_unbound_ports_and_far_raw_corners_are_not_inferred_connections():
    guide, measured = guides()
    measured[0] += (0.1, 0)
    assert source_centerline.construct(guide, measured, Work.start(10)) is None
    guide, measured = guides()
    measured[40] += (0, -6)
    assert source_centerline.construct(guide, measured, Work.start(10)) is None


@pytest.mark.parametrize("invalid", ["nan", "extent", "shape", "object", "points"])
def test_invalid_and_unbounded_native_guides_are_excluded(invalid):
    guide, measured = guides()
    if invalid == "nan":
        guide[5, 0] = np.nan
    elif invalid == "extent":
        guide[5, 0] = 1000
    elif invalid == "shape":
        guide = guide[:, 0]
    elif invalid == "object":
        guide = guide.astype(object)
    else:
        guide = np.tile(guide, (100, 1))
    assert source_centerline.construct(guide, measured, Work.start(10)) is None


def test_parameter_budget_and_interruption_do_not_publish_partial_curves(monkeypatch):
    guide, measured = guides()
    with pytest.raises(StageInterruptedError):
        source_centerline.construct(guide, measured, Work.start(0))
    monkeypatch.setattr(source_centerline, "MAX_PARAMETERS", 0)
    assert source_centerline.construct(guide, measured, Work.start(10)) is None


def test_straight_nongap_geometry_does_not_invent_a_corner():
    guide = np.linspace((10, 10), (40, 10), 30)
    result = source_centerline.construct(guide, guide.copy(), Work.start(10))
    assert result is not None
    assert result.corners == ()
    assert len(result.geometry.subpaths[0].nodes) == 2
    assert result.parameters == 1
