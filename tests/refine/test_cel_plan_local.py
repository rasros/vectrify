"""Crop contributions agree with complete native policy evaluations."""

import numpy as np
import pytest

from vectrify.refine.cel_plan import local
from vectrify.refine.cel_plan.local import (
    Box,
    LocalLimitError,
    LocalPolicy,
    render_crop,
)
from vectrify.refine.cel_plan.policy import Feature, Policy
from vectrify.refine.cel_plan.score import render, svg_metrics


def svg(body, *, size=(160, 128), attributes=""):
    return f'<svg width="{size[0]}" height="{size[1]}" {attributes}>{body}</svg>'


def setup(truth, before, *, features=(), texture=None):
    policy = Policy(render(truth, (160, 128)), features=features, texture=texture)
    evaluated = policy.evaluate(before)
    assert evaluated.valid
    policy.establish(evaluated)
    evaluator = LocalPolicy(policy)
    return policy, evaluator, evaluator.start(before, evaluated)


def agree(policy, snapshot, candidate):
    complete = policy.evaluate(candidate)
    assert snapshot.evaluation.terms == pytest.approx(complete.terms, abs=2e-7)
    assert snapshot.evaluation.rejections == complete.rejections


@pytest.mark.parametrize("box", [Box(0, 0, 80, 64), Box(51, 39, 117, 97)])
def test_native_viewport_preserves_transform_gradient_and_opacity_context(box):
    drawing = svg(
        '<defs><linearGradient id="r"><stop offset="0" stop-color="#d04020"/>'
        '<stop offset="1" stop-color="#2080d0" stop-opacity="0.4"/>'
        "</linearGradient></defs>"
        '<path fill="#fff" d="M0 0H160V128H0Z"/>'
        '<g opacity="0.64" transform="translate(13 17) rotate(8)">'
        '<path fill="#ffe080" d="M10 10H100V75H10Z"/>'
        '<path fill="url(#r)" d="M0 0H120V90H0Z"/></g>'
    )
    np.testing.assert_array_equal(
        render_crop(drawing, box, (160, 128)), render(drawing, (160, 128))[box.slices]
    )


def test_native_viewport_can_omit_distant_paths_without_changing_local_compositing():
    drawing = svg(
        '<defs><linearGradient id="r"><stop offset="0" stop-color="#d04020"/>'
        '<stop offset="1" stop-color="#2080d0" stop-opacity="0.4"/>'
        '</linearGradient></defs><path fill="white" d="M0 0H160V128H0Z"/>'
        '<g opacity="0.64" transform="translate(13 17) rotate(8)">'
        '<path id="base" fill="#ffe080" d="M10 10H100V75H10Z"/>'
        '<path id="near" fill="url(#r)" d="M0 0H60V90H0Z"/>'
        '<path id="far" fill="url(#r)" d="M112 0H140V90H112Z"/></g>'
    )
    box = Box(0, 0, 80, 64)
    culled = render_crop(
        drawing, box, (160, 128), visible_ids=frozenset({"base", "near"})
    )
    np.testing.assert_array_equal(culled, render(drawing, (160, 128))[box.slices])


def test_crop_update_keeps_global_blur_and_feature_denominators():
    base = '<path fill="#b05030" d="M8 8H152V120H8Z"/>'
    mark = '<path fill="{color}" d="M62 45H88V70H62Z"/>'
    before = svg(base + mark.format(color="#303040"))
    after = svg(base + mark.format(color="#b05030"))
    feature = Feature((62, 45, 26, 25), np.ones((25, 26), dtype=bool))
    texture = np.random.default_rng(19).random((128, 160)).astype(np.float32)
    policy, evaluator, initial = setup(
        before, before, features=(feature,), texture=texture
    )
    updated = evaluator.update(initial, after, Box(58, 41, 92, 74), svg_metrics(after))
    agree(policy, updated, after)
    assert updated.evaluation.terms["features"] > 0
    # Damage remains concentrated on the original support after removing ink.
    assert updated.features[0] > updated.evaluation.terms["color"]
    assert initial.evaluation.terms["visual"] == 0


def test_disjoint_and_overlapping_updates_do_not_mutate_other_beam_states():
    background = '<path fill="#c08040" d="M8 8H152V120H8Z"/>'
    a = '<path fill="{color}" d="M24 24H48V48H24Z"/>'
    b = '<path fill="{color}" d="M104 80H128V104H104Z"/>'
    original = svg(background + a.format(color="#303030") + b.format(color="#404040"))
    first = svg(background + a.format(color="#b08040") + b.format(color="#404040"))
    second = svg(background + a.format(color="#b08040") + b.format(color="#c08040"))
    third = svg(background + a.format(color="#c08040") + b.format(color="#c08040"))
    policy, evaluator, initial = setup(original, original)
    branch = evaluator.update(initial, first, Box(20, 20, 52, 52), svg_metrics(first))
    fork = evaluator.update(branch, second, Box(100, 76, 132, 108), svg_metrics(second))
    final = evaluator.update(fork, third, Box(20, 20, 52, 52), svg_metrics(third))
    for snapshot, candidate in (
        (initial, original),
        (branch, first),
        (fork, second),
        (final, third),
    ):
        agree(policy, snapshot, candidate)
        np.testing.assert_array_equal(
            snapshot.canvas.crop(Box(0, 0, 160, 128)), render(candidate, (160, 128))
        )
    assert initial.canvas.root is final.canvas.root
    assert len(initial.canvas.patches) == 0
    assert len(final.canvas.patches) == 3
    x = np.array([4, 24, 38, 112, 128])
    y = np.array([4, 24, 38, 92, 104])
    np.testing.assert_array_equal(
        final.canvas.samples(x, y), render(third, (160, 128))[y, x]
    )
    with pytest.raises(ValueError, match="read-only"):
        branch.canvas.patches[0].pixels[0, 0] = 0


@pytest.mark.parametrize("change", ["hole", "thin", "opacity", "spill"])
def test_incremental_hard_alpha_checks_match_full_checkpoint(change):
    body = (
        '<path fill="#b05030" fill-opacity="0.25" fill-rule="evenodd" '
        'd="M8 8H112V112H8Z M40 40H56V56H40Z"/>'
    )
    thin = (
        '<path fill="#201008" fill-opacity="0.003921568627" d="M128 16H130V52H128Z"/>'
    )
    before = svg(body + thin)
    if change == "hole":
        after = svg(body + thin + '<path fill="red" d="M40 40H56V56H40Z"/>')
        box = Box(36, 36, 60, 60)
    elif change == "thin":
        after, box = svg(body), Box(124, 12, 134, 56)
    elif change == "opacity":
        after, box = (
            svg(body.replace('fill-opacity="0.25"', 'fill-opacity="0.1"') + thin),
            Box(4, 4, 116, 116),
        )
    else:
        after = svg(body + thin + '<path fill="red" d="M120 100H152V116H120Z"/>')
        box = Box(116, 96, 156, 120)
    policy, evaluator, initial = setup(before, before)
    updated = evaluator.update(initial, after, box, svg_metrics(after))
    agree(policy, updated, after)
    assert updated.evaluation.rejections


@pytest.mark.parametrize("has_source_ink", [False, True])
def test_appearance_and_disappearance_of_last_ink_keep_global_edge_semantics(
    has_source_ink,
):
    base = '<path fill="#c09050" d="M8 8H152V120H8Z"/>'
    mark = '<path fill="#101010" d="M32 32H72V40H32Z"/>'
    blank, drawn = svg(base), svg(base + mark)
    truth = drawn if has_source_ink else blank
    policy, evaluator, initial = setup(truth, blank)
    updated = evaluator.update(initial, drawn, Box(28, 28, 76, 44), svg_metrics(drawn))
    agree(policy, updated, drawn)
    restored = evaluator.update(updated, blank, Box(28, 28, 76, 44), svg_metrics(blank))
    agree(policy, restored, blank)
    assert restored.observed == initial.observed


def test_changed_worst_feature_uses_unchanged_other_feature_errors():
    background = '<path fill="#c09050" d="M8 8H152V120H8Z"/>'
    patches = (
        '<path fill="{color}" d="M28 28H44V44H28Z"/>'
        '<path fill="#404040" d="M112 80H128V96H112Z"/>'
    )
    truth = svg(background)
    before = svg(background + patches.format(color="#101010"))
    after = svg(background + patches.format(color="#c09050"))
    features = (
        Feature((28, 28, 16, 16), np.ones((16, 16), dtype=bool)),
        Feature((112, 80, 16, 16), np.ones((16, 16), dtype=bool)),
    )
    policy, evaluator, initial = setup(truth, before, features=features)
    updated = evaluator.update(initial, after, Box(24, 24, 48, 48), svg_metrics(after))
    agree(policy, updated, after)
    assert updated.features[1] == initial.features[1]
    assert updated.evaluation.terms["features"] > 0


def test_incomplete_native_bounds_are_rejected_before_scoring():
    background = '<path fill="#c09050" d="M8 8H152V120H8Z"/>'
    before, after = svg(background), svg(background.replace("#c09050", "#101010"))
    _policy, evaluator, initial = setup(before, before)
    with pytest.raises(ValueError, match="declared native bounds"):
        evaluator.update(initial, after, Box(32, 32, 48, 48), svg_metrics(after))
    assert not initial.canvas.patches


def test_crop_and_root_memory_limits_fail_before_rendering(monkeypatch):
    before = svg('<path d="M8 8H152V120H8Z"/>')
    policy, evaluator, initial = setup(before, before)
    monkeypatch.setattr(local, "MAX_CROP_PIXELS", 100)
    with pytest.raises(LocalLimitError, match="crop limit"):
        evaluator.update(initial, before, Box(32, 32, 48, 48), svg_metrics(before))
    monkeypatch.setattr(local, "MAX_RASTER_BYTES", 100)
    with pytest.raises(LocalLimitError, match="raster limit"):
        LocalPolicy(policy).start(before, initial.evaluation)
