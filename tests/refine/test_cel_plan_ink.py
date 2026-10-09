"""Ink proofs distinguish a paired dark ridge from shading and blank gaps."""

import numpy as np
import pytest

from vectrify.refine.cel_plan.ink import measure


def line():
    return np.column_stack((np.arange(10.5, 90), np.full(80, 40.5)))


def test_color_boundary_with_a_dark_ridge_proposes_a_continuous_local_stroke():
    target = np.full((80, 100, 3), (220, 180, 100), dtype=float)
    target[40:] = (160, 100, 70)
    target[39:42, 10:90] = (15, 8, 5)
    proof = measure(line(), target, 2)
    assert proof is not None
    assert proof.support > 0.95
    assert 2 <= proof.width <= 4
    assert np.max(proof.paint) < 30


def test_a_shade_discontinuity_is_not_classified_as_ink():
    target = np.full((80, 100, 3), (220, 180, 100), dtype=float)
    target[40:] = (80, 60, 40)
    assert measure(line(), target, 2) is None


def test_a_large_blank_gap_cannot_be_explained_by_nearby_ink():
    target = np.full((80, 100, 3), 220.0)
    target[39:42, 10:35] = 0
    target[39:42, 65:90] = 0
    assert measure(line(), target, 2) is None


def test_an_antialias_sized_gap_has_evidence_on_both_sides():
    target = np.full((80, 100, 3), 220.0)
    target[39:42, 10:90] = 0
    target[39:42, 47:49] = 220
    proof = measure(line(), target, 2)
    assert proof is not None
    assert 0 < proof.peak_gap <= 4


def test_one_painted_side_supports_an_exterior_ink_coverage_centroid():
    target = np.full((80, 100, 3), 255.0)
    visible = np.zeros((80, 100), bool)
    visible[40:] = True
    target[40:] = 180
    target[40:44, 8:92] = 15
    points = line().copy()
    points[:, 1] = 42.5
    proof = measure(points, target, 4, visible=visible)
    assert proof is not None
    assert proof.support > 0.95
    assert 3.5 < proof.width < 4.5
    # The source band spans y=40..44; its center is 42, irrespective of the
    # first equally-dark sample selected by a plateau argmin.
    assert np.max(np.abs(proof.points[1:-1, 1] - 42)) < 0.25
    np.testing.assert_array_equal(proof.points[[0, -1]], points[[0, -1]])
    # Transparent RGB is not contrast evidence or a color/centroid hint.
    target[~visible] = (0, 220, 30)
    changed = measure(points, target, 4, visible=visible)
    assert changed is not None
    np.testing.assert_allclose(changed.points, proof.points)
    np.testing.assert_allclose(changed.paint, proof.paint)
    assert changed.width == proof.width


def test_unpainted_space_cannot_prove_a_constant_dark_material_is_ink():
    target = np.full((80, 100, 3), 255.0)
    visible = np.zeros((80, 100), bool)
    visible[40:] = True
    target[visible] = 30
    assert measure(line(), target, 2, visible=visible) is None
    visible[44:] = False
    assert measure(line(), target, 2, visible=visible) is None


def test_source_visibility_keeps_the_two_painted_side_shading_rejection():
    target = np.full((80, 100, 3), (220, 180, 100), dtype=float)
    target[40:] = (80, 60, 40)
    assert measure(line(), target, 2, visible=np.ones((80, 100), bool)) is None


def test_a_separate_profile_mark_cannot_pull_the_source_centroid_across_a_gap():
    target = np.full((80, 100, 3), 180.0)
    target[39:42, 8:92] = 15
    light = target[..., 0].copy()
    visible = np.ones((80, 100), bool)
    proof = measure(line(), target, 3, light=light, visible=visible)
    assert proof is not None
    target[43:46, 8:92] = 0
    light = target[..., 0].copy()
    changed = measure(line(), target, 3, light=light, visible=visible)
    assert changed is not None
    np.testing.assert_allclose(changed.points, proof.points)
    assert changed.width == proof.width


def exterior_profile(alpha=1):
    target = np.full((80, 100, 3), 180.0)
    target[:40] = 255
    target[40:44, 8:92] = 15
    opacity = np.zeros((80, 100), float)
    opacity[40:] = alpha
    opacity[40] *= 0.2
    points = line().copy()
    points[:, 1] = 42.5
    return points, target, opacity


@pytest.mark.parametrize("alpha", [1, 0.5, 0.25])
def test_fractional_exterior_coverage_keeps_intrinsic_width_at_each_opacity(alpha):
    points, target, opacity = exterior_profile(alpha)
    visible = opacity > 8 / 255
    proof = measure(points, target, 4, visible=visible, opacity=opacity)
    assert proof is not None
    assert 3 <= proof.width <= 3.4
    binary = measure(points, target, 4, visible=visible)
    assert binary is not None
    assert binary.width > proof.width + 0.4
    opaque = measure(points, target, 4, visible=visible, opacity=opacity / alpha)
    assert opaque is not None
    np.testing.assert_allclose(proof.points, opaque.points, atol=1e-6)
    assert proof.width == pytest.approx(opaque.width, abs=1e-6)
    np.testing.assert_array_equal(proof.points[[0, -1]], points[[0, -1]])
    np.testing.assert_array_equal(proof.paint, binary.paint)
    # The faint first pixel contributes fractional coverage. The centre moves
    # inward; a uniform translucent drawing must retain the same shape.
    assert np.mean(proof.points[1:-1, 1]) > np.mean(binary.points[1:-1, 1])


def test_fractional_coverage_cannot_use_hidden_rgb_or_complete_a_source_gap():
    points, target, opacity = exterior_profile()
    visible = opacity > 8 / 255
    proof = measure(points, target, 4, visible=visible, opacity=opacity)
    assert proof is not None
    target[~visible] = (0, 220, 30)
    hidden = measure(points, target, 4, visible=visible, opacity=opacity)
    assert hidden is not None
    np.testing.assert_allclose(hidden.points, proof.points)
    np.testing.assert_allclose(hidden.paint, proof.paint)
    assert hidden.width == proof.width
    target[40:44, 38:62] = 180
    assert measure(points, target, 4, visible=visible, opacity=opacity) is None


@pytest.mark.parametrize("invalid", ["shape", "nonfinite", "negative", "too-large"])
def test_fractional_coverage_requires_aligned_valid_source_opacity(invalid):
    points, target, opacity = exterior_profile()
    visible = opacity > 8 / 255
    if invalid == "shape":
        opacity = opacity[:-1]
    elif invalid == "nonfinite":
        opacity[:] = np.nan
    elif invalid == "negative":
        opacity[:] = -1
    else:
        opacity[:] = 2
    with pytest.raises(ValueError, match="opacity"):
        measure(points, target, 4, visible=visible, opacity=opacity)
