"""Exact policy behavior: local damage, topology and invariant normalization."""

import numpy as np
import pytest

from vectrify.document import import_svg
from vectrify.refine.cel_plan.policy import Feature, Policy
from vectrify.refine.cel_plan.score import render, representation


def drawing(body, size=64):
    return f'<svg width="{size}" height="{size}">{body}</svg>'


def test_used_geometry_and_inherited_strokes_are_charged_per_instance():
    svg = drawing(
        '<defs><path id="p" d="M0 0L8 0L8 8Z"/></defs>'
        '<g stroke="black"><use href="#p"/><use href="#p" x="12"/></g>'
    )
    stats = representation(import_svg(svg))
    assert (stats.paths, stats.contours, stats.nodes, stats.stroke_contours) == (
        2,
        2,
        6,
        2,
    )


def test_unused_gradients_do_not_inflate_cost():
    svg = drawing(
        '<defs><linearGradient id="unused"><stop offset="0" stop-color="red"/>'
        '</linearGradient></defs><path d="M0 0L8 0L8 8Z"/>'
    )
    assert representation(import_svg(svg)).gradients == 0


def test_primitives_and_their_used_instances_cannot_hide_representation_cost():
    svg = drawing(
        '<defs><circle id="p" cx="8" cy="8" r="4"/></defs>'
        '<use href="#p"/><use href="#p" x="16"/>'
        '<rect x="30" y="4" width="8" height="8"/>'
    )
    stats = representation(import_svg(svg))
    assert stats.paths == 0
    assert stats.primitive_objects == 3
    assert stats.contours == 3
    assert stats.cost == 18


def test_use_inherits_paint_only_when_referenced_path_does_not_specify_it():
    svg = drawing(
        '<defs><path id="p" d="M0 0L8 0L8 8Z" stroke="none"/></defs>'
        '<g stroke="black"><use href="#p"/></g>'
    )
    assert representation(import_svg(svg)).stroke_contours == 0


def test_padding_does_not_dilute_color_or_alpha_score():
    truth = np.zeros((30, 30, 4), dtype=np.float32)
    truth[8:22, 8:22] = (1, 0, 0, 1)
    actual = truth.copy()
    actual[12:16, 12:16] = (0, 0, 1, 0.5)
    empty = drawing("")
    original = Policy(truth).evaluate(empty, pixels=actual)
    padded = Policy(np.pad(truth, ((20, 20), (20, 20), (0, 0)))).evaluate(
        empty, pixels=np.pad(actual, ((20, 20), (20, 20), (0, 0)))
    )
    for term in ("color", "alpha", "edges", "visual"):
        assert original.terms[term] == pytest.approx(padded.terms[term], abs=1e-7)


def test_alpha_spill_is_visible_on_contrasting_backdrops_and_rejected():
    truth = np.zeros((64, 64, 4), dtype=np.float32)
    truth[16:48, 16:48] = 1
    actual = truth.copy()
    actual[3:12, 3:12] = 1
    policy = Policy(truth)
    evaluated = policy.evaluate(drawing(""), pixels=actual)
    assert evaluated.terms["alpha"] > 0
    assert "silhouette-spill" in evaluated.rejections


def test_filling_an_intentional_hole_is_rejected():
    ring = drawing('<path d="M8 8H56V56H8Z M24 24H40V40H24Z" fill-rule="evenodd"/>')
    filled = drawing('<path d="M8 8H56V56H8Z"/>')
    policy = Policy(render(ring, (64, 64)))
    baseline = policy.evaluate(ring)
    assert baseline.valid
    policy.establish(baseline)
    assert "protected-hole-lost" in policy.evaluate(filled).rejections


def test_reference_feature_support_cannot_disappear_from_score():
    truth = np.ones((64, 64, 4), dtype=np.float32)
    truth[30:34, 30:34, :3] = 0
    feature = Feature((30, 30, 4, 4), np.ones((4, 4), dtype=bool))
    policy = Policy(truth, features=(feature,))
    lost = policy.evaluate(drawing('<path d="M0 0H64V64H0Z" fill="white"/>'))
    assert lost.terms["features"] > 0.1
    assert lost.terms["features"] > lost.terms["color"] * 20


def test_self_crossing_proposal_is_not_retained():
    policy = Policy(np.ones((64, 64, 4), dtype=np.float32))
    evaluation = policy.evaluate(drawing('<path d="M8 8L56 56L8 56L56 8Z"/>'))
    assert "new-self-crossing" in evaluation.rejections


def test_validated_baseline_is_immutable():
    svg = drawing('<path d="M8 8H56V56H8Z"/>')
    policy = Policy(render(svg, (64, 64)))
    evaluation = policy.evaluate(svg)
    policy.establish(evaluation)
    with pytest.raises(ValueError, match="cannot change"):
        policy.establish(evaluation)


@pytest.mark.parametrize("value", [float("nan"), -0.01, 1.01])
def test_invalid_reference_score_evidence_is_rejected(value):
    truth = np.zeros((4, 4, 4), dtype=np.float32)
    truth[0, 0, 0] = value
    with pytest.raises(ValueError, match="Score evidence"):
        Policy(truth)
