"""Event handling and flush rules of the stats collector.

The CSV schema itself is covered by test_collector_schema.py; this file is
about which events move which counter and which of them write a row.
"""

import csv
from pathlib import Path

from vectrify.search.collector import StatCollector
from vectrify.search.models import ChainState, Result, SearchNode
from vectrify.search.stats import SearchStats


def _result(llm_type: str | None = None) -> Result:
    return Result(
        task_id=1,
        parent_id=0,
        valid=True,
        score=0.5,
        payload=None,
        llm_type=llm_type,
    )


def _node(score: float = 0.25) -> SearchNode:
    return SearchNode(
        score=score, id=1, parent_id=0, state=ChainState(score=score, payload=None)
    )


def _rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "stats.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _collector(tmp_path: Path) -> tuple[StatCollector, SearchStats]:
    stats = SearchStats()
    return StatCollector(stats, tmp_path), stats


def test_configure_run_records_the_epoch_thresholds(tmp_path):
    collector, stats = _collector(tmp_path)
    collector.configure_run(epoch_max_tasks=300)
    assert stats.epoch_max_tasks == 300


def test_seed_initial_score_anchors_the_history_at_zero():
    stats = SearchStats()
    StatCollector(stats, None).seed_initial_score(0.8)
    assert stats.best_score == 0.8
    assert list(stats.score_history) == [(0.0, 0.8)]


def test_on_run_start_sets_the_clock_and_patience():
    stats = SearchStats()
    StatCollector(stats, None).on_run_start(start_time=123.0, epoch_patience=7)
    assert stats.start_time == 123.0
    assert stats.epoch_patience == 7


def test_on_shutdown_writes_a_final_row(tmp_path):
    collector, stats = _collector(tmp_path)
    collector.on_shutdown()
    assert stats.shutting_down
    assert len(_rows(tmp_path)) == 1


def test_on_phase_seed_resets_the_seed_counter_and_flushes(tmp_path):
    collector, stats = _collector(tmp_path)
    stats.seeds_completed = 4
    collector.on_phase("seed", 6)
    assert (stats.phase, stats.seeds_target, stats.seeds_completed) == ("seed", 6, 0)
    assert _rows(tmp_path)[-1]["phase"] == "seed"


def test_on_phase_local_keeps_the_seed_counter(tmp_path):
    collector, stats = _collector(tmp_path)
    stats.seeds_completed = 4
    collector.on_phase("local", 6)
    assert stats.seeds_completed == 4


def test_on_result_splits_llm_and_mutation_calls(tmp_path):
    collector, stats = _collector(tmp_path)
    collector.on_result(
        _result(llm_type="generate"),
        tasks_completed=1,
        epoch_no_improve=0,
        seeds_completed=1,
        llm_in_flight=2,
    )
    collector.on_result(
        _result(),
        tasks_completed=2,
        epoch_no_improve=3,
        seeds_completed=1,
        llm_in_flight=1,
    )
    assert stats.llm_call_count == 1
    assert stats.mutation_call_count == 1
    assert stats.tasks_completed == 2
    assert stats.epoch_no_improve == 3
    assert stats.llm_calls_in_flight == 1


def test_on_result_does_not_write_a_row(tmp_path):
    collector, _ = _collector(tmp_path)
    collector.on_result(
        _result(llm_type="generate"),
        tasks_completed=1,
        epoch_no_improve=0,
        seeds_completed=0,
        llm_in_flight=0,
    )
    assert _rows(tmp_path) == []


def test_on_invalid_counts_llm_failures_separately(tmp_path):
    collector, stats = _collector(tmp_path)
    collector.on_invalid(_result(llm_type="generate"))
    collector.on_invalid(_result())
    assert stats.invalid_count == 2
    assert stats.llm_invalid_count == 1


def test_invalid_llm_result_flushes_but_a_mutation_one_does_not(tmp_path):
    collector, stats = _collector(tmp_path)
    stats.tasks_completed = 7
    collector.on_invalid(_result())
    assert _rows(tmp_path) == []
    collector.on_invalid(_result(llm_type="generate"))
    assert len(_rows(tmp_path)) == 1


def test_on_pool_rejected_counts_and_follows_the_flush_rule(tmp_path):
    collector, stats = _collector(tmp_path)
    stats.tasks_completed = 7
    collector.on_pool_rejected()
    assert stats.pool_rejected_count == 1
    assert _rows(tmp_path) == []
    collector.on_pool_rejected(is_llm=True)
    assert stats.pool_rejected_count == 2
    assert len(_rows(tmp_path)) == 1


def test_every_hundredth_task_flushes(tmp_path):
    collector, stats = _collector(tmp_path)
    stats.tasks_completed = 100
    collector.on_pool_rejected()
    assert len(_rows(tmp_path)) == 1


def test_on_accepted_splits_llm_and_mutation_acceptances(tmp_path):
    collector, stats = _collector(tmp_path)
    collector.on_accepted(_node(), is_new_best=False, elapsed=1.0, llm_type="generate")
    collector.on_accepted(_node(), is_new_best=False, elapsed=2.0, llm_type=None)
    assert stats.accepted_count == 2
    assert stats.llm_accepted_count == 1
    assert stats.mutation_accepted_count == 1


def test_a_new_best_records_the_score_and_writes_a_row(tmp_path):
    collector, stats = _collector(tmp_path)
    collector.on_accepted(_node(0.25), is_new_best=True, elapsed=4.5, llm_type=None)
    assert stats.best_score == 0.25
    assert list(stats.score_history) == [(4.5, 0.25)]
    assert float(_rows(tmp_path)[-1]["best_score"]) == 0.25


def test_a_non_best_mutation_acceptance_does_not_write_a_row(tmp_path):
    collector, stats = _collector(tmp_path)
    stats.tasks_completed = 7
    collector.on_accepted(_node(), is_new_best=False, elapsed=1.0, llm_type=None)
    assert _rows(tmp_path) == []


def test_on_no_improve_reset_clears_the_stagnation_counter():
    stats = SearchStats()
    stats.epoch_no_improve = 9
    StatCollector(stats, None).on_no_improve_reset()
    assert stats.epoch_no_improve == 0


def test_on_pool_state_records_diversity_and_spread():
    stats = SearchStats()
    StatCollector(stats, None).on_pool_state(diversity=0.4, score_std=0.02)
    assert (stats.pool_diversity, stats.pool_score_std) == (0.4, 0.02)


def test_epoch_transition_resets_stagnation_and_writes_a_row(tmp_path):
    collector, stats = _collector(tmp_path)
    stats.epoch_no_improve = 5
    collector.on_epoch_transition(3)
    assert (stats.epoch, stats.epoch_no_improve) == (3, 0)
    assert _rows(tmp_path)[-1]["epoch"] == "3"


def test_on_idle_updates_in_flight_and_pool_spread():
    stats = SearchStats()
    StatCollector(stats, None).on_idle(llm_in_flight=3, valid_scores=[0.2, 0.4])
    assert stats.llm_calls_in_flight == 3
    assert stats.pool_score_std == 0.1


def test_on_idle_keeps_the_last_spread_when_the_pool_is_too_small():
    stats = SearchStats()
    stats.pool_score_std = 0.05
    StatCollector(stats, None).on_idle(llm_in_flight=0, valid_scores=[0.2])
    assert stats.pool_score_std == 0.05


def test_header_is_written_once_across_many_flushes(tmp_path):
    collector, _ = _collector(tmp_path)
    for epoch in range(3):
        collector.on_epoch_transition(epoch)
    rows = _rows(tmp_path)
    assert [r["epoch"] for r in rows] == ["0", "1", "2"]
    with (tmp_path / "stats.csv").open(encoding="utf-8") as f:
        assert f.read().count("tasks_completed") == 1


def test_an_unwritable_run_dir_does_not_kill_the_search(tmp_path):
    stats = SearchStats()
    StatCollector(stats, tmp_path / "does" / "not" / "exist").on_epoch_transition(1)
    assert stats.epoch == 1  # the event still took effect
