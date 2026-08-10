import math

import pytest

from vectrify.search.stats import RATE_SPECS, SearchStats, derived_rates, score_std


def _populated() -> SearchStats:
    s = SearchStats()
    s.tasks_completed = 200
    s.accepted_count = 50
    s.pool_rejected_count = 30
    s.invalid_count = 20
    s.llm_call_count = 40
    s.llm_accepted_count = 12
    s.llm_invalid_count = 8
    s.mutation_call_count = 160
    s.mutation_accepted_count = 38
    return s


def test_derived_rates_values():
    rates = _populated().derived_rates()
    assert rates["accept_rate"] == pytest.approx(0.25)
    assert rates["pool_rejected_rate"] == pytest.approx(0.15)
    assert rates["invalid_rate"] == pytest.approx(0.10)
    assert rates["llm_valid_rate"] == pytest.approx(0.80)
    assert rates["llm_accept_rate"] == pytest.approx(0.30)
    assert rates["mutation_accept_rate"] == pytest.approx(0.2375)


def test_methods_agree_with_derived_rates():
    s = _populated()
    rates = s.derived_rates()
    assert s.accept_rate() == rates["accept_rate"]
    assert s.pool_rejected_rate() == rates["pool_rejected_rate"]
    assert s.invalid_rate() == rates["invalid_rate"]
    assert s.llm_valid_rate() == rates["llm_valid_rate"]
    assert s.llm_accept_rate() == rates["llm_accept_rate"]
    assert s.mutation_accept_rate() == rates["mutation_accept_rate"]


def test_every_rate_spec_has_an_accessor():
    s = SearchStats()
    for name in RATE_SPECS:
        assert callable(getattr(s, name)), f"no {name}() accessor"


def test_rates_are_zero_with_no_activity():
    rates = SearchStats().derived_rates()
    assert set(rates) == set(RATE_SPECS)
    assert all(v == 0.0 for v in rates.values())


def test_derived_rates_accepts_a_plain_mapping():
    rates = derived_rates({"tasks_completed": 10.0, "accepted_count": 4.0})
    assert rates["accept_rate"] == pytest.approx(0.4)
    assert rates["llm_accept_rate"] == 0.0  # missing counters degrade to zero


def test_score_std_matches_population_formula():
    scores = [0.1, 0.2, 0.4, 0.9]
    mean = sum(scores) / len(scores)
    expected = math.sqrt(sum((x - mean) ** 2 for x in scores) / len(scores))
    assert score_std(scores) == pytest.approx(expected)


def test_score_std_needs_two_samples():
    assert score_std([]) == 0.0
    assert score_std([0.5]) == 0.0


def test_score_std_zero_for_identical_scores():
    assert score_std([0.3, 0.3, 0.3]) == pytest.approx(0.0)


def test_stagnation_fraction_is_capped():
    s = SearchStats()
    s.epoch_patience = 10
    s.epoch_no_improve = 25
    assert s.stagnation_fraction() == 1.0


def test_stagnation_fraction_zero_when_disabled():
    assert SearchStats().stagnation_fraction() == 0.0
