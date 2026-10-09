"""Material support excludes edge halos while preserving real opacity evidence."""

import numpy as np
import pytest

from vectrify.refine.cel_plan.coverage import from_alpha
from vectrify.refine.cel_plan.local import Box, LocalPolicy
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import render, svg_metrics


def rgba(alpha):
    truth = np.zeros((*alpha.shape, 4), dtype=np.float32)
    truth[..., :3] = (0.5, 0.25, 0.125)
    truth[..., 3] = alpha / 255
    return truth


def halo():
    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[8:56, 8:56] = 128
    alpha[8:56, 8:18] = 1
    alpha[20:28, 13] = 0  # A closed slit only in the weak coverage halo.
    return alpha


def empty():
    return '<svg width="64" height="64"/>'


def test_fringe_slit_is_soft_evidence_not_a_material_hole():
    truth = rgba(halo())
    policy = Policy(truth)
    assert not policy.holes
    assert not policy.opacity_inside[20:28, 13].any()
    filled = truth.copy()
    filled[20:28, 13, 3] = 1 / 255
    evaluation = policy.evaluate(empty(), pixels=filled)
    assert evaluation.valid
    assert evaluation.terms["alpha"] > 0


def test_removing_edge_coverage_still_pays_finite_native_error():
    truth = rgba(halo())
    actual = truth.copy()
    actual[8:56, 8:18, 3] = 0
    evaluation = Policy(truth).evaluate(empty(), pixels=actual)
    assert evaluation.valid
    assert evaluation.terms["alpha"] > 0
    assert evaluation.terms["color"] > 0


@pytest.mark.parametrize("opacity", [1, 8, 64, 128, 255])
def test_material_interior_remains_protected_at_each_native_opacity(opacity):
    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[8:56, 8:56] = opacity
    truth = rgba(alpha)
    actual = truth.copy()
    actual[24:40, 24:40, 3] = 0
    policy = Policy(truth)
    assert policy.opacity_inside[24:40, 24:40].all()
    assert (
        "translucent-interior-gap" in policy.evaluate(empty(), pixels=actual).rejections
    )


@pytest.mark.parametrize("hole_opacity", [0, 4, 16])
def test_meaningful_clear_or_reduced_opacity_hole_cannot_be_filled(hole_opacity):
    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[8:56, 8:56] = 128
    alpha[28:32, 28:32] = hole_opacity
    truth = rgba(alpha)
    policy = Policy(truth)
    assert len(policy.holes) == 1
    assert policy.evaluate(empty(), pixels=truth).valid
    actual = truth.copy()
    actual[28:32, 28:32, 3] = 128 / 255
    assert "protected-hole-lost" in policy.evaluate(empty(), pixels=actual).rejections


def test_weak_ring_attached_to_stronger_body_keeps_its_hole():
    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[8:40, 8:16] = 128
    alpha[20:28, 20:28] = 8
    alpha[22:26, 22:26] = 0
    alpha[23:25, 16:20] = 8
    truth = rgba(alpha)
    policy = Policy(truth)
    assert len(policy.holes) == 1
    assert policy.coverage.plateau_holes == 1
    actual = truth.copy()
    actual[22:26, 22:26, 3] = 8 / 255
    assert "protected-hole-lost" in policy.evaluate(empty(), pixels=actual).rejections


def test_independent_faint_mark_mass_contract_does_not_follow_the_new_core():
    truth = rgba(halo())
    truth[2:6, 60:62, 3] = 1 / 255
    policy = Policy(truth)
    actual = truth.copy()
    actual[2:6, 60:62, 3] = 0
    assert (
        "translucent-component-lost"
        in policy.evaluate(empty(), pixels=actual).rejections
    )
    assert 0.95 in policy.component_retention
    assert 0.75 in policy.component_retention


def test_broad_opacity_ramp_is_not_a_uniform_paint_exemption():
    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[8:56, 8:56] = np.linspace(32, 224, 48).astype(np.uint8)[None, :]
    truth = rgba(alpha)
    actual = truth.copy()
    actual[8:56, 8:56, 3] = 128 / 255
    evaluation = Policy(truth).evaluate(empty(), pixels=actual)
    assert "translucent-opacity-excess" in evaluation.rejections
    assert "translucent-interior-gap" in evaluation.rejections


def test_padding_cannot_change_material_or_hole_support():
    alpha = halo()
    alpha[30:34, 30:34] = 4
    original = from_alpha(alpha / 255, alpha > 0)
    padded_alpha = np.pad(alpha, 20)
    padded = from_alpha(padded_alpha / 255, padded_alpha > 0)
    np.testing.assert_array_equal(original.interior, padded.interior[20:-20, 20:-20])
    np.testing.assert_array_equal(original.holes, padded.holes[20:-20, 20:-20])
    assert original.plateau_holes == padded.plateau_holes
    assert not original.interior.flags.writeable
    assert not original.holes.flags.writeable


@pytest.mark.parametrize("fill_hole", [False, True])
def test_local_native_checkpoint_agrees_for_partial_holes_and_fringe(fill_hole):
    before = (
        '<svg width="64" height="64">'
        '<path fill="#804020" fill-opacity="0.5" '
        'd="M8 8H56V56H8Z M28 28H32V32H28Z" fill-rule="evenodd"/>'
        '<path fill="#804020" fill-opacity="0.0156862745" '
        'd="M28 28H32V32H28Z"/></svg>'
    )
    after = before.replace("0.0156862745", "0.5" if fill_hole else "0.01176470588")
    policy = Policy(render(before, (64, 64)))
    baseline = policy.evaluate(before)
    assert baseline.valid
    policy.establish(baseline)
    local = LocalPolicy(policy)
    snapshot = local.start(before, baseline)
    proposed = local.update(snapshot, after, Box(20, 20, 40, 40), svg_metrics(after))
    full = policy.evaluate(after)
    assert proposed.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    assert proposed.evaluation.rejections == full.rejections
    assert bool(full.rejections) == fill_hole
