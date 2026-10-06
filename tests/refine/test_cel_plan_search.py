"""Individual edits enter bounded beams and independent native checkpoints."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from vectrify.document import Editor, Selection, export_svg, import_svg
from vectrify.refine.cel_plan import search as beam
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.model import Boundary, Options, Work
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import Operators, bounds
from vectrify.refine.cel_plan.score import render, representation
from vectrify.refine.cel_plan.search import Proposal, Rejections, State, search

INITIAL = (
    '<svg width="160" height="128"><defs>'
    '<linearGradient id="a"><stop offset="0" stop-color="#ac5030"/>'
    '<stop offset="1" stop-color="#b45030"/></linearGradient>'
    '<linearGradient id="b"><stop offset="0" stop-color="#40907c"/>'
    '<stop offset="1" stop-color="#409084"/></linearGradient></defs>'
    '<path id="left" fill="url(#a)" d="M8 8H72V120H8Z"/>'
    '<path id="right" fill="url(#b)" d="M88 8H152V120H88Z"/></svg>'
)
TARGET = INITIAL.replace('fill="url(#a)"', 'fill="#b05030"').replace(
    'fill="url(#b)"', 'fill="#409080"'
)


def setup(initial=INITIAL, target=TARGET, *, size=(160, 128)):
    options = Options(refine=False)
    truth = render(target, size)
    evidence = collect(
        Image.fromarray(np.rint(truth * 255).astype(np.uint8)),
        None,
        options,
        Work.start(10),
    )
    frontier = Frontier(Policy(truth))
    assert frontier.add(initial, "Initial")
    frontier.freeze_normalizer()
    return frontier, evidence, options


def color(document, oid, value):
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Test proposal") as transaction:
        transaction.set_fill(oid, value)
    return editor.snapshot.document


def proposal(state, oid, value, *, operator="paint", parent=None):
    document = color(state.document, oid, value)
    return Proposal(
        operator,
        (oid,),
        (value,),
        parent or state.key,
        document,
        bounds(state.document, document, (oid,)),
    )


def test_real_paint_operators_remove_two_noisy_gradients_one_edit_at_a_time():
    frontier, evidence, options = setup()
    before = frontier.select(50).metrics["objective"]
    result = search(
        frontier, options, Work.start(10), Operators(evidence, build(evidence), options)
    )
    selected = frontier.select(50)
    assert result["attempted"] >= 2
    assert result["accepted"] >= 2
    assert result["checkpointed"] >= 1
    assert result["score_disagreements"] == 0
    assert selected.metrics["gradients"] == 0
    assert selected.metrics["objective"] < before
    assert [edit["operator"] for edit in selected.metrics["local_edits"]] == [
        "flat-paint",
        "flat-paint",
    ]
    assert frontier.baseline.svg == INITIAL
    assert frontier.policy.evaluate(selected.svg).valid
    assert result["native_context_renders"] > 0


@pytest.mark.parametrize("quality", ["fast", "balanced", "high"])
def test_quality_caps_and_common_frontier_selection(quality):
    frontier, _evidence, options = setup()
    options = replace(options, quality=quality)

    def edits(state, _work):
        for oid, value in (("left", "#b05030"), ("right", "#409080")):
            if state.document.element(oid).get("fill") != value:
                yield proposal(state, oid, value)

    result = search(frontier, options, Work.start(10), edits)
    assert result["beam_limit"] == beam.LIMITS[quality][0]
    assert result["evaluation_limit"] == beam.LIMITS[quality][1]
    assert result["beam_states"] <= result["beam_limit"]
    assert result["attempted"] <= result["evaluation_limit"]
    costs = [frontier.select(c).metrics["representation_cost"] for c in range(101)]
    assert costs == sorted(costs)


def test_rejected_individual_edit_leaves_seed_and_other_branch_unchanged():
    frontier, _evidence, options = setup()

    def edits(state, _work):
        if not state.edits:
            yield proposal(state, "left", "#ffffff")
            yield proposal(state, "right", "#409080")

    result = search(frontier, options, Work.start(10), edits)
    assert any(
        "local-objective-regression" in d["rejections"] for d in result["decisions"]
    )
    assert frontier.baseline.svg == INITIAL
    selected = import_svg(frontier.select(50).svg)
    assert selected.element("left").get("fill") == "url(#a)"
    assert selected.element("right").get("fill") == "#409080"


def test_hard_coverage_rejection_cannot_be_bought_with_representation_savings():
    frontier, _evidence, options = setup()

    def edits(state, _work):
        yield proposal(state, "left", "none")

    result = search(frontier, options, Work.start(10), edits)
    assert result["accepted"] == 0
    assert result["checkpointed"] == 0
    assert "opaque-interior-gap" in result["decisions"][0]["rejections"]
    assert frontier.select(50).svg == INITIAL


def test_stale_parent_revision_never_reaches_native_evaluation():
    frontier, _evidence, options = setup()

    def edits(state, _work):
        yield proposal(state, "left", "#b05030", parent="old")

    result = search(frontier, options, Work.start(10), edits)
    assert result["attempted"] == 0
    assert result["accepted"] == 0
    assert result["decisions"][0]["rejections"] == ["stale-proposal"]


def test_full_checkpoint_rejects_an_incorrect_local_score(monkeypatch):
    frontier, _evidence, options = setup()
    original = beam.LocalPolicy.update

    def inaccurate(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        evaluation = replace(
            result.evaluation, terms={**result.evaluation.terms, "visual": 0.0}
        )
        return replace(result, evaluation=evaluation)

    monkeypatch.setattr(beam.LocalPolicy, "update", inaccurate)

    def edits(state, _work):
        if not state.edits:
            yield proposal(state, "left", "#b05030")

    result = search(frontier, options, Work.start(10), edits)
    assert result["accepted"] == 1
    assert result["score_disagreements"] == 1
    assert frontier.select(50).svg == INITIAL
    assert frontier.decisions[-1]["rejections"] == ["local-score-disagreement"]


def test_existing_candidate_does_not_skip_independent_checkpoint_agreement():
    frontier, _evidence, _options = setup()
    evaluation = frontier.entries[0].evaluation
    inaccurate = replace(evaluation, terms={**evaluation.terms, "visual": -1})
    assert not frontier.checkpoint(INITIAL, "Repeated local checkpoint", {}, inaccurate)
    assert frontier.decisions[-1]["rejections"] == ["local-score-disagreement"]


def test_checkpoint_rejects_wrong_patch_history_despite_a_correct_expected_score():
    frontier, _evidence, _options = setup()
    entry = frontier.entries[0]
    initial = beam.LocalPolicy(frontier.policy).start(entry.svg, entry.evaluation)
    proposed = export_svg(color(import_svg(entry.svg), "left", "#b05030"))
    expected = frontier.policy.evaluate(proposed)
    assert not frontier.checkpoint(
        proposed, "Wrong local pixels", {}, expected, raster=initial.canvas
    )
    assert frontier.decisions[-1]["rejections"] == ["local-raster-disagreement"]
    assert frontier.select(50).svg == INITIAL


def test_stop_after_local_acceptance_keeps_only_the_fully_validated_seed():
    frontier, _evidence, options = setup()
    work = Work.start(10)

    def edits(state, _work):
        yield proposal(state, "left", "#b05030")
        work.stop.set()

    result = search(frontier, options, work, edits)
    assert result["accepted"] == 1
    assert result["checkpointed"] == 0
    assert frontier.select(50).svg == INITIAL


def test_rejection_cache_reuses_a_proof_and_invalidates_changed_dependencies():
    frontier, _evidence, _options = setup()
    entry = frontier.entries[0]
    evaluator = beam.LocalPolicy(frontier.policy)
    state = State(
        import_svg(entry.svg),
        entry.svg,
        evaluator.start(entry.svg, entry.evaluation),
        entry.key,
        entry.details,
    )
    bad = proposal(state, "left", "#ffffff")
    cache = Rejections()
    key = cache.key(state, bad, -12)
    cache.put(key, ("local-objective-regression",))
    assert cache.get(cache.key(state, bad, -12)) == ("local-objective-regression",)
    changed = proposal(state, "left", "#b05030")
    updated = evaluator.update(
        state.snapshot,
        export_svg(changed.document),
        changed.bounds,
        representation(changed.document).metrics(),
    )
    next_state = replace(state, document=changed.document, snapshot=updated)
    assert cache.get(cache.key(next_state, bad, -12)) is None
    for index in range(beam.MAX_REJECTIONS + 2):
        cache.put(str(index), ("rejected",))
    assert len(cache.values) == beam.MAX_REJECTIONS
    assert cache.get(key) is None


def test_hidden_paint_and_layer_order_are_rejection_dependencies():
    initial = INITIAL.replace(
        '<path id="left"',
        '<path id="under" fill="red" d="M8 8H72V120H8Z"/><path id="left"',
    )
    frontier, _evidence, _options = setup(initial=initial)
    entry = frontier.entries[0]
    evaluator = beam.LocalPolicy(frontier.policy)
    state = State(
        import_svg(entry.svg),
        entry.svg,
        evaluator.start(entry.svg, entry.evaluation),
        entry.key,
        entry.details,
    )
    transparent = color(state.document, "left", "none")
    edit = Proposal(
        "paint",
        ("left",),
        ("none",),
        state.key,
        transparent,
        bounds(state.document, transparent, ("left",)),
    )
    cache = Rejections()
    key = cache.key(state, edit, -12)
    cache.put(key, ("rejected",))
    changed = color(state.document, "under", "blue")
    # The opaque foreground hides this change completely, but removing it
    # would reveal different paint. A visible-pixel hash alone is insufficient.
    np.testing.assert_array_equal(
        render(export_svg(changed), (160, 128)), render(initial, (160, 128))
    )
    assert cache.get(cache.key(replace(state, document=changed), edit, -12)) is None


def test_independent_color_edit_can_reuse_a_rejection_proof():
    def wide(svg):
        return svg.replace('width="160"', 'width="240"').replace(
            "M88 8H152V120H88Z", "M200 8H232V120H200Z"
        )

    frontier, _evidence, _options = setup(
        initial=wide(INITIAL), target=wide(TARGET), size=(240, 128)
    )
    entry = frontier.entries[0]
    evaluator = beam.LocalPolicy(frontier.policy)
    state = State(
        import_svg(entry.svg),
        entry.svg,
        evaluator.start(entry.svg, entry.evaluation),
        entry.key,
        entry.details,
    )
    bad = proposal(state, "left", "#ffffff")
    cache = Rejections()
    key = cache.key(state, bad, -12)
    cache.put(key, ("rejected",))
    other = proposal(state, "right", "#409080")
    updated = evaluator.update(
        state.snapshot,
        export_svg(other.document),
        other.bounds,
        representation(other.document).metrics(),
    )
    after = replace(state, document=other.document, snapshot=updated)
    assert cache.get(cache.key(after, proposal(after, "left", "#ffffff"), -12)) == (
        "rejected",
    )


def test_state_memory_limit_and_scan_limit_bound_unproductive_work(monkeypatch):
    frontier, _evidence, options = setup()
    monkeypatch.setattr(beam, "MAX_BYTES", 160 * 128 * 4 + len(INITIAL.encode()) + 100)

    def edits(state, _work):
        for _ in range(1000):
            yield proposal(state, "left", "#b05030")

    result = search(frontier, options, Work.start(10), edits)
    assert result["accepted"] == 0
    assert result["bounded_proposals"] > 0
    assert result["attempted"] <= beam.LIMITS["balanced"][1]
    assert frontier.select(50).svg == INITIAL


def test_scanning_stale_proposals_has_an_independent_cap():
    frontier, _evidence, options = setup()
    options = replace(options, quality="fast")

    def edits(state, _work):
        rejected = proposal(state, "left", "#ffffff", parent="old")
        while True:
            yield rejected

    result = search(frontier, options, Work.start(10), edits)
    assert result["attempted"] == 0
    assert result["scanned"] == beam.LIMITS["fast"][1] * 8
    assert result["status"] == "bounded"


def test_preview_reports_a_real_local_search_stage(monkeypatch):
    from vectrify.refine.cel_plan import pipeline
    from vectrify.refine.cel_plan.pipeline import vectorize

    _frontier, evidence, options = setup()
    captured = []
    original = pipeline.Operators

    def operator_options(evidence, graph, settings):
        captured.append(settings)
        return original(evidence, graph, settings)

    monkeypatch.setattr(pipeline, "Operators", operator_options)
    options = replace(options, complexity=75, protection=2)
    result = vectorize(
        Image.fromarray(np.rint(evidence.rgba * 255).astype(np.uint8)),
        options=options,
        seconds=10,
    )
    assert result.metrics["structural_search"]["status"] != "unavailable"
    assert result.metrics["structural_search"]["beam_limit"] == 4
    assert result.metrics["structural_search"]["evaluation_limit"] == 48
    assert captured[0].complexity == 50
    assert result.metrics["score_weights"]["features"] == 1


def test_optional_search_failure_retains_the_independent_validated_checkpoint(
    monkeypatch,
):
    from vectrify.refine.cel_plan import pipeline

    _frontier, evidence, options = setup()

    def failed(*_args):
        raise ValueError("Injected optional search failure")

    monkeypatch.setattr(pipeline, "local_search", failed)
    result = pipeline.vectorize(
        Image.fromarray(np.rint(evidence.rgba * 255).astype(np.uint8)),
        options=options,
        seconds=10,
    )
    assert result.metrics["structural_search"]["status"] == "failed"
    assert result.metrics["structural_search"]["validation_seconds_unavailable"]
    assert Policy(evidence.rgba).evaluate(result.svg).valid


def test_continuous_ink_is_an_individually_validated_evidence_proposal():
    source = np.full((128, 160, 4), (220, 180, 100, 255), dtype=np.uint8)
    source[64:, :, :3] = (160, 100, 70)
    source[63:66, 10:150, :3] = (15, 8, 5)
    initial = (
        '<svg width="160" height="128">'
        '<path fill="#dcb464" d="M0 0H160V64H0Z"/>'
        '<path fill="#a06446" d="M0 64H160V128H0Z"/></svg>'
    )
    options = Options(refine=False, line_width=3)
    evidence = collect(Image.fromarray(source), None, options, Work.start(10))
    graph = build(evidence)
    ids = [
        region.id
        for region in graph.regions
        if region.area and region.id not in graph.hidden
    ]
    assert len(ids) >= 2
    points = np.column_stack((np.arange(10.5, 150), np.full(140, 64.5)))
    graph = replace(graph, boundaries=(Boundary(0, ids[0], ids[1], points, 1.0),))
    frontier = Frontier(Policy(source.astype(np.float32) / 255))
    assert frontier.add(initial, "Missing line")
    frontier.freeze_normalizer()
    before = frontier.select(100).metrics["objective"]
    result = search(
        frontier, options, Work.start(10), Operators(evidence, graph, options)
    )
    selected = frontier.select(100)
    assert any(
        edit["operator"] == "boundary-ink" and edit["accepted"]
        for edit in result["decisions"]
    )
    assert result["score_disagreements"] == 0
    assert selected.metrics["objective"] < before
    assert selected.metrics["stroke_contours"] == 1
    document = import_svg(selected.svg)
    assert document.element("cel-local-ink-0").get("stroke-width") == "3"
    assert frontier.policy.evaluate(selected.svg).valid
