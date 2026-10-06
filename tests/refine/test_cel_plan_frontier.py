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


def test_soft_budget_schedule_reports_target_floor_and_achieved_cost():
    result = frontier()
    result.freeze_normalizer()
    floor = min(entry.evaluation.cost for entry in result.entries)
    targets = []
    for complexity in range(101):
        selected = result.select(complexity)
        budget = selected.metrics["representation_budget"]
        assert budget["target"] == pytest.approx(
            max(floor, 20 * 2 ** ((complexity - 100) / 50))
        )
        assert budget["floor"] == floor
        assert budget["achieved"] == selected.metrics["representation_cost"]
        assert budget["unmet"] == (budget["achieved"] > budget["target"])
        targets.append(budget["target"])
    assert targets == sorted(targets)
    for candidate in result.alternatives(50):
        assert (
            candidate.metrics["representation_budget"]["target"]
            == result.budget(50)["target"]
        )


def test_node_ceiling_and_soft_representation_target_are_separate():
    result = frontier()
    normal = result.select(50)
    unmet = result.select(50, node_budget=1)
    assert unmet.metrics["budget_unmet"]
    assert (
        unmet.metrics["representation_budget"]
        == normal.metrics["representation_budget"]
    )
    assert unmet.svg == normal.svg


@pytest.mark.parametrize("complexity", [-1, 101])
def test_soft_budget_rejects_out_of_range_complexity(complexity):
    with pytest.raises(ValueError, match="Complexity"):
        frontier().budget(complexity)


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


def test_detailed_cost_scale_is_fixed_separately_from_coverage_baseline():
    result = frontier()
    baseline = result.baseline
    # Valid but dominated detailed geometry still supplies its own cost scale.
    redundant = DETAILED.replace("M4 4H20", "M4 4L12 4H20")
    assert not result.add(redundant, "Detailed geometry")
    result.freeze_normalizer(redundant)
    assert result.normalizer == 21
    assert result.baseline is baseline
    assert result.normalizer_fixed
    with pytest.raises(ValueError, match="already fixed"):
        result.freeze_normalizer(SIMPLE)
    costs = [
        result.select(level).metrics["representation_cost"] for level in range(101)
    ]
    assert costs == sorted(costs)


def test_invalid_or_unmeasured_geometry_cannot_set_the_cost_scale():
    result = frontier()
    with pytest.raises(ValueError, match="validated"):
        result.freeze_normalizer('<svg width="40" height="40"/>')
    result.select(50)
    with pytest.raises(ValueError, match="already fixed or in use"):
        result.freeze_normalizer()


def test_refinement_requires_its_anchor_objective_to_improve():
    result = frontier()
    result.freeze_normalizer()
    before = result.entries[-1].evaluation.objective(50, result.normalizer)
    proposal = DETAILED.replace("#e08080", "#000000").replace("#808080", "#000000")
    assert not result.refine(proposal, "Worse paint", {}, complexity=50, before=before)
    assert result.decisions[-1]["rejections"] == ["objective-regression"]
    assert result.decisions[-1]["objective"] > before


def test_refinement_pruning_retains_every_initial_slider_tradeoff(monkeypatch):
    from vectrify.refine.cel_plan import frontier as module

    monkeypatch.setattr(module, "MAX_CANDIDATES", 2)
    result = frontier()
    result.freeze_normalizer()
    before = [result.select(level).svg for level in range(101)]
    proposal = SIMPLE.replace("#e08080", "#b08080").replace("H36", "L12 4L20 4L28 4H36")
    assert not result.refine(proposal, "Intermediate", {}, complexity=50, before=1)
    assert result.decisions[-1]["rejections"] == ["frontier-limit"]
    assert len(result.entries) == 2
    assert [result.select(level).svg for level in range(101)] == before


def test_replacement_that_exceeds_pool_memory_restores_dominated_entries(monkeypatch):
    from vectrify.refine.cel_plan import frontier as module

    monkeypatch.setattr(
        module, "MAX_BYTES", len(DETAILED.encode()) + len(SIMPLE.encode())
    )
    result = frontier()
    result.freeze_normalizer()
    before = tuple(result.entries)
    # Better paint at the same geometric cost, but more serialized storage.
    proposal = SIMPLE.replace("#e08080", "#b08080").replace(
        "</svg>", "<!--" + "x" * 50 + "--></svg>"
    )
    assert not result.refine(proposal, "Large replacement", {}, complexity=50, before=1)
    assert result.decisions[-1]["rejections"] == ["frontier-limit"]
    assert tuple(result.entries) == before


def test_lower_cost_with_more_nodes_cannot_destroy_node_budget_feasibility():
    low_nodes = (
        '<svg width="40" height="40">'
        + "".join(
            f'<path fill="#e08080" d="M{x} 4H{x + 8}V36H{x}Z"/>'
            for x in (4, 12, 20, 28)
        )
        + "</svg>"
    )
    lower_cost = (
        '<svg width="40" height="40"><path fill="#e08080" d="M4 4'
        + "".join(f"L{x} 4" for x in range(6, 37, 2))
        + 'V36H4Z"/></svg>'
    )
    result = Frontier(Policy(render(low_nodes, (40, 40)), weights=Weights(edges=0)))
    assert result.add(low_nodes, "Fewer nodes")
    result.freeze_normalizer()
    assert result.refine(lower_cost, "Fewer contours", {}, complexity=50, before=1)
    assert len(result.entries) == 2
    assert result.select(50).svg == lower_cost
    feasible = result.select(50, node_budget=16)
    assert feasible.svg == low_nodes
    assert not feasible.metrics["budget_unmet"]
