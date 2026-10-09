"""Disjoint tile contributions match native full renders and hard checks."""

import numpy as np
import pytest

from vectrify.refine.cel_plan import local
from vectrify.refine.cel_plan.local import Box, LocalLimitError, LocalPolicy, tile_boxes
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.policy import Feature, Policy
from vectrify.refine.cel_plan.score import render, svg_metrics


def drawing(
    *, ink=False, thin=True, hole=False, opacity="0.25", color="#b05030", spill=False
):
    return (
        '<svg width="160" height="128">'
        f'<g opacity="{opacity}"><path fill="{color}" fill-rule="evenodd" '
        'd="M8 8H112V112H8Z M40 40H72V72H40Z"/>'
        + ('<path fill="#101010" d="M24 80H104V84H24Z"/>' if ink else "")
        + ('<path fill="red" d="M40 40H72V72H40Z"/>' if hole else "")
        + "</g>"
        + (
            '<path fill="#201008" fill-opacity="0.003921568627" '
            'd="M130 16H132V56H130Z"/>'
            if thin
            else ""
        )
        + ('<path fill="red" d="M144 100H156V124H144Z"/>' if spill else "")
        + "</svg>"
    )


def initial(before, truth=None):
    features = (
        Feature((22, 76, 86, 14), np.ones((14, 86), dtype=bool)),
        Feature((128, 16, 6, 40), np.ones((40, 6), dtype=bool)),
    )
    texture = np.random.default_rng(43).random((128, 160)).astype(np.float32)
    policy = Policy(
        render(truth or before, (160, 128)), features=features, texture=texture
    )
    evaluation = policy.evaluate(before)
    assert evaluation.valid
    policy.establish(evaluation)
    evaluator = LocalPolicy(policy)
    return policy, evaluator, evaluator.start(before, evaluation)


@pytest.mark.parametrize(
    ("shape", "box"),
    [
        ((1200, 600, 4), Box(30, 10, 570, 1190)),
        ((3000, 160, 4), Box(16, 8, 144, 2992)),
        ((160, 3000, 4), Box(8, 16, 2992, 144)),
    ],
)
def test_output_ownership_is_complete_disjoint_and_input_halos_are_bounded(shape, box):
    tiles = tile_boxes(box, shape)
    assert len(tiles) > 1
    assert len(tiles) <= local.MAX_TILES
    expected = box.expand(local.HALO, shape)
    assert sum(tile.area for tile in tiles) == expected.area
    for index, tile in enumerate(tiles):
        assert tile.intersection(expected) == tile
        assert tile.expand(local.HALO, shape).area <= local.MAX_CROP_PIXELS
        assert all(not tile.intersection(other).area for other in tiles[index + 1 :])


@pytest.mark.parametrize(
    "change", ["color", "hole", "thin", "opacity", "spill", "ink-first", "ink-last"]
)
def test_multi_tile_native_score_and_hard_checks_agree(change, monkeypatch):
    monkeypatch.setattr(local, "MAX_CROP_PIXELS", 4096)
    before = drawing(ink=change == "ink-last")
    after = drawing(
        **{
            "color": {"color": "#2040a0"},
            "hole": {"hole": True},
            "thin": {"thin": False},
            "opacity": {"opacity": "0.1"},
            "spill": {"spill": True},
            "ink-first": {"ink": True},
            "ink-last": {},
        }[change]
    )
    policy, evaluator, snapshot = initial(
        before, after if change == "ink-first" else before
    )
    candidate = evaluator.update(
        snapshot, after, Box(0, 0, 160, 128), svg_metrics(after)
    )
    complete = policy.evaluate(after)
    assert candidate.evaluation.terms == pytest.approx(complete.terms, abs=2e-7)
    assert candidate.evaluation.rejections == complete.rejections
    assert candidate.canvas.matches(render(after, (160, 128)))
    assert evaluator.tiles_scored > 1
    assert evaluator.native_context_renders == 1
    assert not snapshot.canvas.patches
    if change in {"hole", "thin", "opacity", "spill"}:
        assert candidate.evaluation.rejections
    if change == "ink-first":
        assert snapshot.observed == 0
        assert candidate.observed > 0
    if change == "ink-last":
        assert snapshot.observed > 0
        assert candidate.observed == 0


def test_gradient_and_transformed_opacity_group_have_exact_native_pixels_across_tiles(
    monkeypatch,
):
    monkeypatch.setattr(local, "MAX_CROP_PIXELS", 4096)
    body = (
        '<svg width="160" height="128"><defs><linearGradient id="g">'
        '<stop offset="0" stop-color="#d04020"/>'
        '<stop offset="1" stop-color="#2080d0" stop-opacity="0.4"/>'
        '</linearGradient></defs><path fill="white" d="M0 0H160V128H0Z"/>'
        '<g opacity="0.64" transform="translate(13 17) rotate(8)">'
        '<path fill="#ffe080" d="M10 10H100V75H10Z"/>'
        '<path fill="{paint}" d="M0 0H120V90H0Z"/></g></svg>'
    )
    before = body.format(paint="url(#g)")
    after = body.format(paint="#705090")
    policy, evaluator, snapshot = initial(before)
    candidate = evaluator.update(
        snapshot, after, Box(0, 0, 160, 128), svg_metrics(after)
    )
    assert candidate.canvas.matches(render(after, (160, 128)))
    assert candidate.evaluation.terms == pytest.approx(
        policy.evaluate(after).terms, abs=2e-7
    )
    assert evaluator.native_context_renders == 1


def test_stopping_between_tiles_discards_the_entire_working_edit(monkeypatch):
    monkeypatch.setattr(local, "MAX_CROP_PIXELS", 4096)
    before, after = drawing(), drawing(color="#2040a0")
    _policy, evaluator, snapshot = initial(before)
    work = Work.start(10)
    original = evaluator._tile

    def stop(*args):
        result = original(*args)
        work.stop.set()
        return result

    monkeypatch.setattr(evaluator, "_tile", stop)
    with pytest.raises(StageInterruptedError, match="tile evaluation"):
        evaluator.update(
            snapshot, after, Box(0, 0, 160, 128), svg_metrics(after), work=work
        )
    assert evaluator.tiles_scored == 1
    assert not snapshot.canvas.patches
    assert snapshot.canvas.matches(render(before, (160, 128)))


def test_tile_limit_rejects_work_before_native_rendering(monkeypatch):
    monkeypatch.setattr(local, "MAX_CROP_PIXELS", 4096)
    monkeypatch.setattr(local, "MAX_TILES", 1)
    _policy, evaluator, snapshot = initial(drawing())

    def forbidden(*_args):
        pytest.fail("Oversized work must not reach the renderer")

    monkeypatch.setattr(local, "_native_raster", forbidden)
    with pytest.raises(LocalLimitError, match="tile limit"):
        evaluator.update(snapshot, drawing(color="#2040a0"), Box(0, 0, 160, 128), {})


def test_cumulative_tiled_edits_keep_disjoint_and_overlapping_sibling_histories(
    monkeypatch,
):
    monkeypatch.setattr(local, "MAX_CROP_PIXELS", 4096)
    body = (
        '<svg width="160" height="128"><g opacity="0.6">'
        '<path fill="{top}" d="M8 8H152V56H8Z"/>'
        '<path fill="{bottom}" d="M8 72H152V120H8Z"/></g></svg>'
    )
    original = body.format(top="#c08040", bottom="#805030")
    first = body.format(top="#c09050", bottom="#805030")
    second = body.format(top="#c09050", bottom="#906040")
    third = body.format(top="#b09060", bottom="#906040")
    sibling = body.format(top="#c08040", bottom="#a06040")
    policy, evaluator, initial_state = initial(original)
    top, bottom = Box(4, 4, 156, 60), Box(4, 68, 156, 124)
    branch = evaluator.update(initial_state, first, top, svg_metrics(first))
    fork = evaluator.update(branch, second, bottom, svg_metrics(second))
    final = evaluator.update(fork, third, top, svg_metrics(third))
    other = evaluator.update(initial_state, sibling, bottom, svg_metrics(sibling))
    for snapshot, svg in (
        (initial_state, original),
        (branch, first),
        (fork, second),
        (final, third),
        (other, sibling),
    ):
        assert snapshot.evaluation.terms == pytest.approx(
            policy.evaluate(svg).terms, abs=2e-7
        )
        assert snapshot.canvas.matches(render(svg, (160, 128)))
        assert snapshot.canvas.root is initial_state.canvas.root
    assert len(branch.canvas.patches) > 1
    assert len(final.canvas.patches) > len(fork.canvas.patches)
    assert not initial_state.canvas.patches
    with pytest.raises(ValueError, match="read-only"):
        final.canvas.patches[-1].pixels[0, 0] = 0


def test_long_gradient_beyond_real_crop_limit_matches_a_full_native_checkpoint():
    body = (
        '<svg width="160" height="2000"><defs><linearGradient id="g">'
        '<stop offset="0" stop-color="#b05030"/>'
        '<stop offset="1" stop-color="#b85030"/>'
        '</linearGradient></defs><g opacity="0.75">'
        '<path fill="{paint}" d="M8 8H152V1992H8Z"/>'
        '<path fill="#201008" d="M76 24H80V1976H76Z"/></g></svg>'
    )
    before, after = body.format(paint="url(#g)"), body.format(paint="#b45030")
    policy = Policy(render(before, (160, 2000)))
    evaluated = policy.evaluate(before)
    assert evaluated.valid
    policy.establish(evaluated)
    evaluator = LocalPolicy(policy)
    snapshot = evaluator.start(before, evaluated)
    changed = Box(4, 4, 156, 1996)
    assert (
        changed.expand(2 * local.HALO, policy.truth.shape).area > local.MAX_CROP_PIXELS
    )
    updated = evaluator.update(snapshot, after, changed, svg_metrics(after))
    complete = policy.evaluate(after)
    assert updated.evaluation.terms == pytest.approx(complete.terms, abs=2e-7)
    assert updated.evaluation.rejections == complete.rejections
    assert updated.canvas.matches(render(after, (160, 2000)))
    assert evaluator.tiles_scored > 1
    assert evaluator.native_context_renders == 1
