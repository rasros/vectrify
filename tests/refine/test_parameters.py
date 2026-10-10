"""Editor fitting cannot turn redundant knots or short edges into spikes."""

import numpy as np
import pytest

from vectrify.document import import_svg
from vectrify.refine.parameters import ControlMap

torch = pytest.importorskip("torch")


def mapping(path, *, displacement=20, fixed=(), linear=None):
    geometry = import_svg(
        f'<svg width="64" height="64"><path id="p" d="{path}"/></svg>'
    ).geometry_for("p")
    local = np.array(
        [
            point
            for subpath in geometry.subpaths
            for node in subpath.nodes
            for point in zip(node.values[::2], node.values[1::2], strict=True)
        ]
    )
    original = torch.tensor(local, dtype=torch.float64)
    movable = torch.ones((len(local), 1), dtype=torch.float64)
    movable[list(fixed)] = 0
    return ControlMap(
        geometry,
        local,
        original,
        torch.eye(2, dtype=torch.float64) if linear is None else linear,
        original.new_zeros(2),
        original.new_zeros(2),
        original.new_ones(2),
        movable,
        displacement,
    )


def project(parameters, local):
    return parameters.local_from_controls(parameters.controls_from_local(local))


@pytest.mark.parametrize("fixed", [(), (0,), (8,)])
def test_duplicate_closing_knots_stay_together_and_respect_pins(fixed):
    parameters = mapping(
        "M10 10 L30 10 L30 30 C20 20 10 10 10 10 C10 10 10 10 10 10 L10 10 Z",
        fixed=fixed,
    )
    local = parameters.original.clone()
    local[0] += local.new_tensor([4, 0])
    local[5:] += local.new_tensor([0, 6])
    after = project(parameters, local)
    repeated = [0, 5, 6, 7, 8, 9]
    assert torch.equal(after[repeated], after[0].expand(len(repeated), 2))
    if fixed:
        assert torch.equal(after[0], parameters.original[0])


@pytest.mark.parametrize(
    "linear", [torch.eye(2), torch.tensor([[0.0, -3.0], [2.0, 0.0]])]
)
def test_short_edge_handles_cannot_grow_long_or_reverse_past_endpoints(linear):
    parameters = mapping(
        "M10 10 C10.5 10 11.5 10 12 10 L20 20 L10 20 Z",
        linear=linear.double(),
    )
    local = parameters.original.clone()
    local[1] = local.new_tensor([0, 20])
    local[2] = local.new_tensor([22, 0])
    after = project(parameters, local)
    handles = after[[1, 2]] - after[[0, 3]]
    assert bool((handles.norm(dim=-1) <= 2 + 1e-10).all())
    assert bool(((after[[1, 2], 0] >= 10) & (after[[1, 2], 0] <= 12)).all())


def test_existing_overhanging_curves_and_concave_outlines_are_preserved():
    parameters = mapping("M10 10 C5 20 25 20 20 10 L15 15 L20 30 L10 30 Z")
    assert torch.allclose(project(parameters, parameters.original), parameters.original)


def test_handle_projection_keeps_frozen_controls_and_absolute_movement_limit():
    parameters = mapping(
        "M10 10 C11 10 12 10 13 10 L20 20 L10 20 Z",
        displacement=2,
        fixed=(1,),
    )
    local = parameters.original + 20
    after = project(parameters, local)
    assert torch.equal(after[1], parameters.original[1])
    assert bool(((after - parameters.original).norm(dim=-1) <= 2 + 1e-10).all())


def test_projection_gradients_remain_finite_with_zero_length_segments():
    parameters = mapping("M10 10 C10 10 10 10 10 10 L30 10 L20 30 Z")
    local = parameters.original.clone().requires_grad_()
    project(parameters, local).square().sum().backward()
    assert local.grad is not None
    assert bool(torch.isfinite(local.grad).all())


@pytest.mark.parametrize("fixed", [(), (3,)])
def test_smooth_join_keeps_one_tangent_and_movement_bounds(fixed):
    parameters = mapping(
        "M10 20 C13 10 17 10 20 20 C26 40 27 30 30 20 L40 40 L10 40 Z",
        displacement=2,
        fixed=fixed,
    )
    local = parameters.original.clone()
    local[2] += local.new_tensor([-14, 15])
    local[4] += local.new_tensor([-14, -15])
    local[3] += local.new_tensor([2, -1])
    after = project(parameters, local)
    incoming, outgoing = after[3] - after[2], after[4] - after[3]
    assert torch.allclose(outgoing, 2 * incoming, atol=1e-9)
    assert torch.dot(incoming, parameters.original[3] - parameters.original[2]) > 0
    assert bool(((after - parameters.original).norm(dim=-1) <= 2 + 1e-10).all())
    if fixed:
        assert torch.equal(after[3], parameters.original[3])


def test_smooth_projection_has_finite_gradients_and_preserves_zero_movement():
    path = "M10 20 C13 10 17 10 20 20 C23 30 27 30 30 20 L40 40 Z"
    parameters = mapping(path)
    local = parameters.original.clone().requires_grad_()
    project(parameters, local).square().sum().backward()
    assert bool(torch.isfinite(local.grad).all())
    fixed = mapping(path, displacement=0)
    assert torch.allclose(project(fixed, fixed.original + 10), fixed.original)


def test_bending_loss_ignores_translation_but_penalizes_handle_wiggles():
    parameters = mapping("M10 20 C13 10 17 10 20 20 C23 30 27 30 30 20 L40 40 Z")
    controls = parameters.controls_from_local(parameters.original)
    assert parameters.bending_loss([c + 3 for c in controls]) < 1e-20
    noisy = [c.clone() for c in controls]
    noisy[0][0, 1] += 3
    assert parameters.bending_loss(noisy) > 0.1


def test_outline_fairness_penalizes_existing_dents_not_just_handle_changes():
    rough = mapping("M10 10 L14 10 L16 13 L18 10 L30 10 L30 30 L10 30 Z")
    smooth = mapping("M10 10 L14 10 L16 10 L18 10 L30 10 L30 30 L10 30 Z")
    controls = rough.controls_from_local(rough.original)
    roughness = rough.fairness_loss(controls)
    assert roughness > smooth.fairness_loss(smooth.controls_from_local(smooth.original))
    assert rough.fairness_loss([c + 17 for c in controls]) == pytest.approx(
        float(roughness)
    )


def test_outline_fairness_ignores_redundant_collinear_knots_and_has_finite_gradients():
    sparse = mapping("M10 10 L30 10 L30 30 L10 30 Z")
    dense = mapping(
        "M10 10 L10 10 L10.01 10 L17 10 L29.9 10 L30 10 L30 15 L30 30 L10 30 Z"
    )
    sparse_loss = sparse.fairness_loss(sparse.controls_from_local(sparse.original))
    local = dense.original.clone().requires_grad_()
    dense_loss = dense.fairness_loss(dense.controls_from_local(local))
    assert float(dense_loss.detach()) == pytest.approx(float(sparse_loss), rel=1e-3)
    dense_loss.backward()
    assert torch.isfinite(local.grad).all()
