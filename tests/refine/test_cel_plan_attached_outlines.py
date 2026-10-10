"""Complete attached strokes use the ordinary ownership and search gates."""

from dataclasses import replace
from itertools import islice

import pytest

from tests.refine.test_cel_plan_attached_spans import fixture
from vectrify.document import export_svg, load_project, save_project
from vectrify.refine.cel_plan.attached_outlines import AttachedOutlines
from vectrify.refine.cel_plan.attached_spans import AttachedSpans
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.search import State, identity, search


def setup():
    document, partition, evidence, guard = fixture()
    svg = export_svg(document)
    frontier = Frontier(Policy(evidence.rgba))
    assert frontier.add(
        svg, "Material parent", {"planning_surfaces": partition.metadata()}
    )
    frontier.freeze_normalizer()
    entry = frontier.baseline
    assert entry is not None
    state = State(
        document,
        svg,
        LocalPolicy(frontier.policy).start(svg, entry.evaluation),
        identity(svg, partition),
        entry.details,
        partition=partition,
    )
    options = Options(quality="high", refine=False)
    operator = AttachedOutlines(evidence, options, guard=lambda _work: guard)
    return state, frontier, evidence, options, guard, operator


def test_complete_proposal_seals_every_dependency_and_preserves_native_ownership():
    state, _, _, _, _, operator = setup()
    assert state.partition is not None
    candidates = list(islice(operator(state, Work.start(30)), 1))
    assert candidates
    proposal = candidates[0]
    assert proposal.operator == "attached-outline"
    assert proposal.parent == state.key
    assert proposal.partition is not None
    family = proposal.partition.families[0]
    assert family.junction == "bar"
    assert set(proposal.ids) == family.dependencies
    assert proposal.dependencies == ("component",)
    assert proposal.component is not None
    assert proposal.component.validate(
        state.document,
        proposal.document,
        state.partition,
        proposal.partition,
        proposal.ids,
        proposal.bounds,
        Work.start(10),
    )
    assert proposal.partition.owners == state.partition.owners
    assert proposal.partition.surfaces == state.partition.surfaces
    assert proposal.details is not None
    assert not proposal.details["attached_outline"]["accepted"]
    assert family.dependencies <= set(proposal.details["geometry_constraints"])
    loaded, _ = load_project(save_project(proposal.document))
    proposal.partition.validate(loaded)


def test_common_search_does_not_waive_complexity_for_a_valid_stroke_proposal():
    state, frontier, _, options, _, operator = setup()
    proposed = []

    def only_attached(current, work):
        if current.edits:
            return
        for candidate in islice(operator(current, work), 1):
            proposed.append(candidate)
            yield candidate

    report = search(
        frontier, options, Work.start(30), only_attached, seed_document=state.document
    )
    assert report["attempted"] == 1
    assert report["accepted"] == 0
    assert report["checkpointed"] == 0
    assert report["score_disagreements"] == 0
    assert proposed[0].partition is not None
    decision = report["decisions"][0]
    assert decision["component_dependency"]["declared_objects"] == 5
    assert decision["visual_delta"] < 0
    assert decision["representation_delta"] == 27
    assert decision["rejections"] == ["local-objective-regression"]


@pytest.mark.parametrize("kind", ["missing-partition", "fixed-width", "interrupted"])
def test_unsupported_or_interrupted_state_never_publishes_partial_family(kind):
    state, _, evidence, options, guard, operator = setup()
    if kind == "missing-partition":
        state = replace(state, partition=None)
    elif kind == "fixed-width":
        operator = AttachedOutlines(
            evidence, replace(options, line_width=2), guard=lambda _work: guard
        )
    work = Work.start(0 if kind == "interrupted" else 10)
    if kind == "interrupted":
        with pytest.raises(StageInterruptedError):
            list(operator(state, work))
    else:
        assert list(operator(state, work)) == []
    assert operator.diagnostics["proposals"] == 0


def test_complete_operator_is_scheduled_only_in_explicit_experimental_high():
    _, _, evidence, options, _, _ = setup()
    graph = build(evidence)
    ordinary = Operators(evidence, graph, options)
    assert ordinary.attached_outlines is None
    experimental = Operators(evidence, graph, options, filled_bands=True)
    assert isinstance(experimental.attached_outlines, AttachedOutlines)


def test_explicit_scheduler_tries_complete_strokes_on_the_material_parent(monkeypatch):
    state, _, evidence, options, guard, _ = setup()
    state = replace(state, details={"planned_band_stroke": {"verified": True}})
    operators = Operators(evidence, build(evidence), options, filled_bands=True)
    operators._filled_band_guard = guard
    for name in (
        "families",
        "overlays",
        "paint",
        "geometry",
        "ink",
        "replacements",
        "ridges",
        "opacity_fields",
        "strokes",
        "bands",
    ):
        monkeypatch.setattr(operators, name, lambda _state, _work: iter(()))
    candidates = list(islice(operators(state, Work.start(30)), 1))
    assert candidates
    assert candidates[0].operator == "attached-outline"
    assert candidates[0].partition is not None
    assert candidates[0].partition.families[0].junction == "bar"
    assert operators.schedule_diagnostics["composition_parents"] == 1


def test_fitting_limit_does_not_construct_an_extra_discovery_candidate(monkeypatch):
    state, _, evidence, _, guard, operator = setup()
    seed = next(
        AttachedSpans(evidence, guard)(state.document, state.partition, Work.start(10))
    )
    fetched = []

    class RepeatedDiscovery:
        def __init__(self, _evidence, _guard):
            pass

        def __call__(self, _document, _partition, _work):
            for i in range(100):
                fetched.append(i)
                yield seed

    monkeypatch.setattr(
        "vectrify.refine.cel_plan.attached_outlines.AttachedSpans", RepeatedDiscovery
    )
    assert len(list(operator(state, Work.start(30)))) == 4
    assert fetched == [0, 1, 2, 3]
