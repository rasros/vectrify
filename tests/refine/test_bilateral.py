"""Automatic curve families remain bounded proposals, without fixture rules."""

from dataclasses import replace

import numpy as np
import pytest

from vectrify.document import import_svg
from vectrify.refine.bilateral import Bilateral


def curve():
    return import_svg(
        '<svg><path id="p" d="M9 50 C13 38 26 19 32.2 8 C38.8 19 52 36 54.8 51"/></svg>'
    ).geometry_for("p")


def points(geometry):
    return np.array(
        [
            p
            for s in geometry.subpaths
            for n in s.nodes
            for p in zip(n.values[::2], n.values[1::2], strict=True)
        ]
    )


@pytest.mark.parametrize("angle", [0, 0.73, 2.1])
def test_bilateral_materialization_is_symmetric_bounded_and_preserves_topology(angle):
    geometry = curve()
    original = points(geometry)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    original = original @ rotation.T + (131, -57)

    def with_points(values):
        index, nodes = 0, []
        for node in geometry.subpaths[0].nodes:
            count = len(node.values) // 2
            nodes.append(
                replace(node, values=tuple(values[index : index + count].reshape(-1)))
            )
            index += count
        return replace(
            geometry, subpaths=(replace(geometry.subpaths[0], nodes=tuple(nodes)),)
        )

    geometry = with_points(original)
    model = Bilateral.infer(geometry, frozenset(), 4)
    assert model is not None
    # A far-away proposal exercises backtracking rather than pointwise
    # clipping, which would break symmetry around the completed curve's axis.
    proposed = original + np.random.default_rng(31).normal(size=original.shape) * 100
    result = model.geometry(with_points(proposed))
    fitted = points(result)
    assert np.linalg.norm(fitted - original, axis=1).max() <= 4 + 1e-9
    np.testing.assert_allclose(fitted, model.average(fitted), atol=1e-10)
    assert [(n.id, n.command, n.pinned) for n in result.subpaths[0].nodes] == [
        (n.id, n.command, n.pinned) for n in geometry.subpaths[0].nodes
    ]


def test_a_family_rejects_holds_pins_incompatible_topology_and_large_asymmetry():
    geometry = curve()
    subpath = geometry.subpaths[0]
    assert Bilateral.infer(geometry, frozenset(), 4) is not None
    assert Bilateral.infer(geometry, frozenset({subpath.nodes[1].id}), 4) is None
    pinned = replace(subpath.nodes[0], pinned=True)
    assert (
        Bilateral.infer(
            replace(
                geometry,
                subpaths=(replace(subpath, nodes=(pinned, *subpath.nodes[1:])),),
            ),
            frozenset(),
            4,
        )
        is None
    )
    assert (
        Bilateral.infer(
            replace(geometry, subpaths=(replace(subpath, closed=True),)), frozenset(), 4
        )
        is None
    )
    assert (
        Bilateral.infer(
            replace(geometry, subpaths=(replace(subpath, nodes=subpath.nodes[:2]),)),
            frozenset(),
            4,
        )
        is None
    )
    assert Bilateral.infer(geometry, frozenset(), 0.1) is None


def test_coupled_axis_derivatives_match_independent_numerical_differences():
    torch = pytest.importorskip("torch")
    model = Bilateral.infer(curve(), frozenset(), 4)
    assert model is not None
    values = torch.tensor(model.original, dtype=torch.float64, requires_grad=True)
    weights = np.random.default_rng(9).normal(size=model.original.shape)
    (model.controls(values) * torch.tensor(weights)).sum().backward()
    for index in [(0, 0), (1, 1), (3, 0), (6, 1)]:
        lower, upper = model.original.copy(), model.original.copy()
        lower[index] -= 1e-5
        upper[index] += 1e-5
        numerical = (
            (model.average(upper) - model.average(lower)) * weights
        ).sum() / 2e-5
        assert values.grad[index].item() == pytest.approx(numerical, abs=1e-7)
