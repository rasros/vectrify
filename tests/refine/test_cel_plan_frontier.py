"""A frozen frontier gives ordered slider choices and preserves safe fallbacks."""

import pytest

from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.policy import Policy, Weights
from vectrify.refine.cel_plan.score import render

SIMPLE = '<svg width="40" height="40"><path d="M4 4H36V36H4Z" fill="#e08080"/></svg>'
DETAILED = (
    '<svg width="40" height="40"><path d="M4 4H20V36H4Z" fill="#e08080"/>'
    '<path d="M20 4H36V36H20Z" fill="#808080"/></svg>'
)


def frontier(observe=None):
    policy = Policy(render(DETAILED, (40, 40)), weights=Weights(edges=0))
    result = Frontier(policy, observe)
    assert result.add(DETAILED, "Detailed")
    assert result.add(SIMPLE, "Simple")
    return result


def test_slider_selection_is_monotonic_on_one_common_frontier():
    result = frontier()
    choices = [result.select(level) for level in range(101)]
    costs = [choice.metrics["representation_cost"] for choice in choices]
    assert costs == sorted(costs)
    assert costs[0] < costs[-1]
    assert {choice.svg for choice in choices} == {SIMPLE, DETAILED}
    assert {choice.metrics["cost_normalizer"] for choice in choices} == {20}


def test_invalid_cheaper_proposal_cannot_replace_fallback():
    result = frontier()
    before = result.baseline
    assert not result.add('<svg width="40" height="40"/>', "Missing drawing")
    assert result.baseline is before
    assert result.select(50).svg in {SIMPLE, DETAILED}
    assert "opaque-interior-gap" in result.decisions[-1]["rejections"]


def test_explicit_node_budget_selects_a_feasible_candidate():
    result = frontier()
    selected = result.select(100, node_budget=4)
    assert selected.svg == SIMPLE
    assert not selected.metrics["budget_unmet"]
    infeasible = result.select(50, node_budget=1)
    assert infeasible.metrics["budget_unmet"]
    assert infeasible.svg in {SIMPLE, DETAILED}


def test_dominated_candidate_does_not_change_slider_choices():
    result = frontier()
    choices = [result.select(level).svg for level in (0, 25, 50, 75, 100)]
    redundant = DETAILED.replace("M4 4H20", "M4 4L12 4H20")
    assert not result.add(redundant, "Redundant node")
    assert [result.select(level).svg for level in (0, 25, 50, 75, 100)] == choices
    assert result.decisions[-1]["rejections"] == ["dominated"]


def test_no_duplicate_user_alternatives():
    result = frontier()
    alternatives = result.alternatives(50)
    assert len(alternatives) == 2
    assert len({candidate.svg for candidate in alternatives}) == 2
    assert alternatives[0].label == "Selected"


def test_frontier_needs_a_valid_checkpoint_before_selection():
    result = Frontier(Policy(render(DETAILED, (40, 40))))
    with pytest.raises(ValueError, match="validated"):
        result.select(50)


def test_invalid_document_proposal_keeps_the_existing_checkpoint():
    result = frontier()
    before = result.select(50)
    assert not result.add(
        '<svg width="40" height="40"><path d="M0 0L1e999 4Z"/></svg>',
        "Nonfinite proposal",
    )
    assert result.decisions[-1]["rejections"] == ["invalid-candidate"]
    assert result.select(50).svg == before.svg


def test_observer_sees_valid_dominated_and_hard_rejected_proposals():
    observed = []
    result = frontier(observed.append)
    redundant = DETAILED.replace("M4 4H20", "M4 4L12 4H20")
    assert not result.add(redundant, "Redundant")
    assert not result.add('<svg width="40" height="40"/>', "Missing")
    assert [item.label for item in observed] == [
        "Detailed",
        "Simple",
        "Redundant",
        "Missing",
    ]
    assert observed[-2].evaluation.valid
    assert observed[-2].decision["rejections"] == ["dominated"]
    assert not observed[-1].evaluation.valid
    assert observed[-1].svg not in {item.svg for item in result.alternatives(50)}


def test_observer_mutations_cannot_change_production_scores_or_decisions():
    def mutate(observation):
        if observation.evaluation:
            observation.evaluation.terms["visual"] = 1e6
            observation.evaluation.structure["nodes"] = 0
        observation.decision["accepted"] = False
        observation.details["changed"] = True

    result = frontier(mutate)
    ordinary = frontier()
    assert result.select(50).svg == ordinary.select(50).svg
    assert result.select(50).metrics == ordinary.select(50).metrics
    assert result.decisions == ordinary.decisions
    assert result.baseline is not None
    assert "changed" not in result.baseline.details


def test_observer_reports_invalid_svg_without_a_fabricated_evaluation():
    observed = []
    result = frontier(observed.append)
    before = result.select(50)
    assert not result.add('<svg><path d="M0 0L1e999 3Z"/></svg>', "Invalid")
    assert observed[-1].evaluation is None
    assert observed[-1].decision["rejections"] == ["invalid-candidate"]
    assert result.select(50).svg == before.svg
