"""Complete RGBA fits distinguish shading ramps from material and alpha steps."""

import numpy as np
import pytest

from tests.helpers import required
from vectrify.refine.cel_plan import material_groups
from vectrify.refine.cel_plan.material_groups import grouped
from vectrify.refine.cel_plan.model import Work


def samples(*, alpha_step=False, alpha_ramp=False):
    y, x = np.mgrid[:10, :32]
    ids = (x // 8).ravel()
    rgb = np.repeat((10 + 2 * x)[..., None], 3, axis=2).astype(float)
    rgb[:, 24:] = 60
    alpha = np.ones(x.shape)
    if alpha_step:
        alpha[:, :16] = 0.25
    elif alpha_ramp:
        alpha = 0.25 + 0.75 * x / 31
        rgb[:] = 100
    basis = np.column_stack(
        (
            np.ones(x.size),
            x.ravel() / 32,
            y.ravel() / 32,
            rgb.reshape(-1, 3) / 255 * alpha.ravel()[:, None],
            alpha.ravel(),
        )
    )
    grams = np.array([basis[ids == i].T @ basis[ids == i] for i in range(4)])
    colors = np.array([rgb.reshape(-1, 3)[ids == i].mean(axis=0) for i in range(4)])
    ranges = np.array(
        [
            (alpha.ravel()[ids == i].min(), alpha.ravel()[ids == i].max())
            for i in range(4)
        ]
    )
    return colors, np.full(4, 80), grams, ranges


def invoke(colors, sizes, grams, ranges, **kwargs):
    return grouped(
        colors,
        sizes,
        [(0, 1), (1, 2), (2, 3)],
        2,
        kwargs.pop("work", Work.start(10)),
        statistics=grams,
        alpha_ranges=ranges,
        **kwargs,
    )


def test_complete_gradient_fit_preserves_a_real_material_change_at_equal_budget():
    colors, sizes, grams, ranges = samples()
    saved = grams.copy()
    fitted, eligible, details = required(invoke(colors, sizes, grams, ranges))
    flat, _, _ = required(
        grouped(colors, sizes, [(0, 1), (1, 2), (2, 3)], 2, Work.start(10))
    )
    np.testing.assert_array_equal(flat, [0, 0, 2, 2])
    np.testing.assert_array_equal(fitted, [0, 0, 0, 3])
    assert eligible.all()
    assert details["paint_cost"] == "source-rgba-fit"
    assert not details["budget_unmet"]
    assert 4 < details["model_evaluations"] < material_groups.MAX_MODEL_EVALUATIONS
    np.testing.assert_array_equal(grams, saved)
    assert not fitted.flags.writeable


def test_disabling_gradients_recovers_the_flat_material_interpretation():
    colors, sizes, grams, ranges = samples()
    fitted, _, _ = required(invoke(colors, sizes, grams, ranges, gradients=False))
    flat, _, _ = required(
        grouped(colors, sizes, [(0, 1), (1, 2), (2, 3)], 2, Work.start(10))
    )
    np.testing.assert_array_equal(fitted, flat)


def test_full_owner_alpha_step_cannot_become_a_gradient_to_meet_the_budget():
    colors, sizes, grams, ranges = samples(alpha_step=True)
    fitted, _, details = required(
        grouped(
            colors,
            sizes,
            [(0, 1), (1, 2), (2, 3)],
            1,
            Work.start(10),
            statistics=grams,
            alpha_ranges=ranges,
        )
    )
    np.testing.assert_array_equal(fitted, [0, 0, 2, 2])
    assert details["alpha_exclusions"] > 0
    assert details["budget_unmet"]


def test_supported_alpha_ramp_uses_one_common_axis_without_losing_owners():
    colors, sizes, grams, ranges = samples(alpha_ramp=True)
    fitted, _, details = required(
        grouped(
            colors,
            sizes,
            [(0, 1), (1, 2), (2, 3)],
            1,
            Work.start(10),
            statistics=grams,
            alpha_ranges=ranges,
        )
    )
    np.testing.assert_array_equal(fitted, [0, 0, 0, 0])
    assert not details["budget_unmet"]
    assert details["alpha_exclusions"] == 0


def test_ink_role_and_small_disconnected_support_remain_independent():
    colors, sizes, grams, ranges = samples(alpha_ramp=True)
    fitted, eligible, details = required(
        grouped(
            colors,
            sizes,
            [(0, 1), (1, 2)],
            1,
            Work.start(10),
            minimum_component_area=200,
            kinds=np.array([0, 1, 0, 0]),
            statistics=grams,
            alpha_ranges=ranges,
        )
    )
    np.testing.assert_array_equal(fitted, np.arange(4))
    np.testing.assert_array_equal(eligible, [True, True, True, False])
    assert details["retained_disconnected_owners"] == 1
    assert details["budget_unmet"]


def test_model_exhaustion_returns_a_complete_prefix_without_mutating_statistics(
    monkeypatch,
):
    colors, sizes, grams, ranges = samples()
    saved = grams.copy()
    monkeypatch.setattr(material_groups, "MAX_MODEL_EVALUATIONS", 6)
    fitted, _, details = required(invoke(colors, sizes, grams, ranges))
    np.testing.assert_array_equal(fitted, np.arange(4))
    np.testing.assert_array_equal(grams, saved)
    assert details["model_limit_hit"]
    assert details["model_evaluations"] == 6
    assert details["budget_unmet"]


def test_interrupted_fit_discards_the_partial_partition():
    colors, sizes, grams, ranges = samples()
    saved = grams.copy()

    class Interrupted:
        calls = 0

        @property
        def interrupted(self):
            self.calls += 1
            return self.calls > 2

    assert invoke(colors, sizes, grams, ranges, work=Interrupted()) is None
    np.testing.assert_array_equal(grams, saved)


@pytest.mark.parametrize("bad", ["shape", "nonfinite", "empty", "alpha"])
def test_invalid_complete_source_statistics_are_rejected(bad):
    colors, sizes, grams, ranges = samples()
    if bad == "shape":
        grams = grams[:, :6, :6]
    elif bad == "nonfinite":
        grams[0, 1, 1] = np.nan
    elif bad == "empty":
        grams[0, 0, 0] = 0
    else:
        ranges[0] = (1, 0)
    with pytest.raises(ValueError, match="RGBA statistics"):
        invoke(colors, sizes, grams, ranges)
