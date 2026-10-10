"""Visible reference edges guide curves without following obscured artwork."""

import numpy as np
import pytest

from vectrify.refine.edge_profiles import EdgeProfiles
from vectrify.refine.snap import _Frame


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("transformed", [False, True])
def test_profiles_find_subpixel_edges_in_either_contour_direction(reverse, transformed):
    y = np.arange(40)[:, None]
    coverage = np.broadcast_to(np.clip(y + 1 - 18, 0, 1), (40, 40))
    desired = np.clip(y + 1 - 16.25, 0, 1)[..., None]
    base = np.broadcast_to([0.8, 0.7, 0.6], (40, 40, 3))
    response = np.broadcast_to([-0.4, 0.1, 0.2], base.shape)
    frame = (
        _Frame(np.array([[0, -2], [0.5, 0]]), np.array([4, 7]))
        if transformed
        else _Frame(np.eye(2), np.zeros(2))
    )
    guide = EdgeProfiles(base, response, base + desired * response, coverage, frame, 1)
    points = np.array(frame.local(np.array([[20, 18]]))).reshape(-1, 2)
    tangent = np.array([[1, 0]]) @ np.linalg.inv(frame.matrix).T
    reached = guide(points, -tangent if reverse else tangent)
    pixels = frame.pixels(tuple(reached.ravel()))
    assert pixels[0] == pytest.approx([20, 16.25], abs=0.15)


@pytest.mark.parametrize(
    "unknown", ["occluded", "ambiguous", "zero_tangent", "distant"]
)
def test_profiles_leave_unobservable_or_ambiguous_edges_alone(unknown):
    y = np.arange(40)[:, None]
    coverage = np.broadcast_to((y >= 18).astype(float), (40, 40)).copy()
    base = np.ones((40, 40, 3))
    response = -base.copy()
    target = np.broadcast_to(
        (y < (4 if unknown == "distant" else 16))[..., None], base.shape
    )
    if unknown == "occluded":
        response[:] = 0
    elif unknown == "ambiguous":
        coverage[:] = 1
    guide = EdgeProfiles(
        base, response, target, coverage, _Frame(np.eye(2), np.zeros(2)), 1
    )
    points = np.array([[20, 18]])
    tangent = np.array([[0, 0] if unknown == "zero_tangent" else [1, 0]])
    assert np.array_equal(guide(points, tangent), points)
