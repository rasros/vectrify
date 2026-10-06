"""Individual edits enter bounded beams and independent native checkpoints."""

import time
from dataclasses import replace
from itertools import islice

import numpy as np
import pathops
import pytest
from PIL import Image

from vectrify.document import Editor, Selection, export_svg, import_svg
from vectrify.refine.cel_plan import local
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


@pytest.mark.parametrize(
    "failed_slot", ["families", "overlays", "paint", "geometry", "ink", "replacements"]
)
def test_native_boolean_failure_does_not_lose_checkpoint_or_other_operators(
    failed_slot, monkeypatch
):
    frontier, evidence, options = setup()
    before = frontier.baseline
    operators = Operators(evidence, build(evidence), options)
    names = ("families", "overlays", "paint", "geometry", "ink", "replacements")
    useful = "paint" if failed_slot != "paint" else "families"
    closed = []
    for name in names:

        def edits(state, _work, name=name):
            try:
                if name == failed_slot:
                    raise pathops.PathOpsError("Injected curve boolean failure")
                if (
                    name == useful
                    and state.document.element("left").get("fill") != "#b05030"
                ):
                    yield proposal(state, "left", "#b05030", operator=name)
            finally:
                closed.append(name)

        monkeypatch.setattr(operators, name, edits)
    report = search(frontier, options, Work.start(10), operators)
    assert operators.schedule_diagnostics["native_boolean_failures"] > 0
    assert report["accepted"] == 1
    assert report["score_disagreements"] == 0
    assert frontier.baseline is before
    assert frontier.select(50).metrics["gradients"] == 1
    assert set(closed) == set(names)


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


@pytest.mark.parametrize("stop", [False, True])
def test_expired_local_slice_can_checkpoint_only_with_live_shared_validation_time(stop):
    frontier, _evidence, options = setup()
    phase = Work.start(10)
    checkpoint = Work(phase.deadline + 5, phase.stop, phase.timings)

    def edits(state, local_work):
        yield proposal(state, "left", "#b05030")
        local_work.deadline = time.monotonic() - 1
        phase.deadline = time.monotonic() - 1
        if stop:
            phase.stop.set()

    report = search(frontier, options, phase, edits, checkpoint_work=checkpoint)
    assert phase.interrupted
    assert report["accepted"] == 1
    assert report["checkpointed"] == (0 if stop else 1)
    assert report["score_disagreements"] == 0
    selected = frontier.select(50)
    assert selected.metrics["gradients"] == (2 if stop else 1)
    assert frontier.policy.evaluate(selected.svg).valid


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


def test_fast_search_reserves_evaluations_for_cumulative_edits():
    definitions, paths, flat = [], [], []
    for index in range(20):
        x, y = 8 + (index % 10) * 14, 8 + (index // 10) * 56
        definitions.append(
            f'<linearGradient id="g{index}"><stop offset="0" stop-color="#ac5030"/>'
            '<stop offset="1" stop-color="#b45030"/></linearGradient>'
        )
        shape = f'<path id="p{index:02}" fill="{{paint}}" d="M{x} {y}h10v40h-10Z"/>'
        paths.append(shape.format(paint=f"url(#g{index})"))
        flat.append(shape.format(paint="#b05030"))
    initial = (
        '<svg width="160" height="128"><defs>'
        + "".join(definitions)
        + "</defs>"
        + "".join(paths)
        + "</svg>"
    )
    target = '<svg width="160" height="128">' + "".join(flat) + "</svg>"
    frontier, _evidence, options = setup(initial=initial, target=target)

    def edits(state, _work):
        for index in range(20):
            oid = f"p{index:02}"
            if state.document.element(oid).get("fill") != "#b05030":
                yield proposal(state, oid, "#b05030")

    result = search(frontier, replace(options, quality="fast"), Work.start(10), edits)
    assert result["attempted"] == 16
    assert result["bounded_expansions"] >= 3
    assert result["expansion_evaluation_limit"] == 4
    assert len(frontier.select(50).metrics["local_edits"]) >= 3
    assert result["score_disagreements"] == 0


def test_rejected_prefix_resumes_without_consuming_the_first_useful_tail_edit():
    frontier, _evidence, options = setup()
    closed = []

    def edits(state, _work):
        try:
            if state.edits:
                return
            for value in ("#ffffff", "#fffffe", "#fffeff", "#feffff", "#b05030"):
                yield proposal(state, "left", value)
        finally:
            closed.append(state.key)

    result = search(frontier, replace(options, quality="fast"), Work.start(10), edits)
    assert result["resumed_expansions"] >= 1
    assert result["attempted"] == 5
    assert result["accepted"] == 1
    assert result["decisions"][-1]["parameters"] == ["#b05030"]
    assert frontier.select(50).metrics["gradients"] == 1
    assert result["score_disagreements"] == 0
    assert result["proposal_cursor_peak"] == 1
    assert len(closed) == 2  # The resumed seed and the completed child.


def test_stop_during_proposal_discovery_prevents_scoring_and_closes_the_cursor():
    frontier, _evidence, options = setup()
    closed = []

    def edits(state, work):
        try:
            work.stop.set()
            yield proposal(state, "left", "#b05030")
        finally:
            closed.append(state.key)

    result = search(frontier, options, Work.start(10), edits)
    assert result["status"] == "interrupted"
    assert result["attempted"] == result["scanned"] == 0
    assert result["accepted"] == 0
    assert frontier.select(50).svg == INITIAL
    assert len(closed) == 1


@pytest.mark.parametrize(("pressure", "burst"), [(2, 2), (1, 1)])
def test_compaction_burst_keeps_each_reserved_operator_opportunity(
    pressure, burst, monkeypatch
):
    frontier, evidence, options = setup()
    entry = frontier.entries[0]
    state = State(
        import_svg(entry.svg),
        entry.svg,
        beam.LocalPolicy(frontier.policy).start(entry.svg, entry.evaluation),
        entry.key,
        {"search_budget": {"representation_target": entry.evaluation.cost / pressure}},
    )
    operators = Operators(evidence, build(evidence), options)
    names = ("families", "overlays", "paint", "geometry", "ink", "replacements")
    for name in names:

        def edits(state, _work, name=name):
            for _ in range(4):
                yield proposal(state, "left", "#b05030", operator=name)

        monkeypatch.setattr(operators, name, edits)
    iterator = operators(state, Work.start(10))
    first = list(islice(iterator, burst + 5))
    iterator.close()
    assert [p.operator for p in first] == ["families"] * burst + list(names[1:])
    assert operators.schedule_diagnostics["family_proposals"] == burst
    assert operators.schedule_diagnostics["reserved_proposals"] == 5
    assert operators.schedule_diagnostics["compaction_parents"] == (pressure > 1.25)


def test_node_ceiling_can_prioritize_compaction_without_changing_acceptance(
    monkeypatch,
):
    frontier, evidence, options = setup()
    entry = frontier.entries[0]
    state = State(
        import_svg(entry.svg),
        entry.svg,
        beam.LocalPolicy(frontier.policy).start(entry.svg, entry.evaluation),
        entry.key,
        {
            "search_budget": {
                "representation_target": entry.evaluation.cost * 4,
                "node_target": 1,
            }
        },
    )
    operators = Operators(evidence, build(evidence), options)

    def family(state, _work):
        for _ in range(2):
            # Scheduling cannot admit this visually destructive paint edit.
            yield proposal(state, "left", "#ffffff", operator="family-surface")

    monkeypatch.setattr(operators, "families", family)
    first = list(islice(operators(state, Work.start(10)), 2))
    assert all(p.operator == "family-surface" for p in first)
    assert operators.schedule_diagnostics["compaction_parents"] == 1
    result = search(frontier, replace(options, node_budget=1), Work.start(10), family)
    assert result["accepted"] == 0
    assert all(
        "local-objective-regression" in d["rejections"] for d in result["decisions"]
    )
    assert frontier.select(50, node_budget=1).metrics["budget_unmet"]


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
    assert result["proposal_cursor_peak"] <= result["beam_limit"]
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


def test_stop_between_native_tiles_keeps_the_validated_seed(monkeypatch):
    monkeypatch.setattr(local, "MAX_CROP_PIXELS", 4096)
    frontier, _evidence, options = setup()
    work = Work.start(10)
    original = beam.LocalPolicy._tile

    def stop(self, *args):
        result = original(self, *args)
        work.stop.set()
        return result

    monkeypatch.setattr(beam.LocalPolicy, "_tile", stop)

    def edits(state, _work):
        yield proposal(state, "left", "#b05030")

    result = search(frontier, options, work, edits)
    assert result["status"] == "interrupted"
    assert result["accepted"] == 0
    assert result["tiles_scored"] == 1
    assert result["checkpointed"] == 0
    assert result["decisions"][-1]["rejections"] == ["local-search-interrupted"]
    assert frontier.select(50).svg == INITIAL


def test_empty_native_bounds_are_rejected_without_rendering_an_edit():
    frontier, _evidence, options = setup()

    def edits(state, _work):
        yield replace(proposal(state, "left", "#b05030"), bounds=local.Box(0, 0, 0, 0))

    result = search(frontier, options, Work.start(10), edits)
    assert result["attempted"] == 0
    assert result["decisions"][0]["rejections"] == ["invalid-local-bounds"]
    assert frontier.select(50).svg == INITIAL


def test_rejection_context_streams_bounded_bytes_and_keys_spatial_bounds(monkeypatch):
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
    edit = proposal(state, "left", "#ffffff")
    calls = []
    original = local.Canvas.values

    def values(self, box):
        calls.append(box.area)
        return original(self, box)

    monkeypatch.setattr(local.Canvas, "values", values)
    monkeypatch.setattr(beam, "MAX_CROP_PIXELS", 128)
    cache = Rejections()
    key = cache.key(state, edit, -12)
    assert len(calls) > 1
    assert max(calls) <= 128
    # Equal expanded context/pixels still represent different affected areas.
    first = replace(edit, bounds=local.Box(0, 0, 160, 128))
    second = replace(edit, bounds=local.Box(1, 1, 159, 127))
    assert first.bounds.expand(32, (128, 160)) == second.bounds.expand(32, (128, 160))
    assert cache.key(state, first, -12) != cache.key(state, second, -12)
    assert cache.key(state, edit, -12) == key


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

    def failed(*_args, **_kwargs):
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
