"""What the live panel has to say about a run.

The old panel was built on a blended score that updated every task and on pool
statistics that no longer exist. What a run has now is an evaluator that speaks
a few dozen times, three criteria that can end an epoch, and a large class of
candidates that change nothing -- none of which the old layout could show.
"""

import io

import pytest
from rich.console import Console

from vectrify.dashboard import _build_renderable
from vectrify.search.stats import SearchStats


def _render(stats: SearchStats, width: int = 100) -> str:
    console = Console(width=width, record=True, file=io.StringIO())
    console.print(_build_renderable(stats))
    return console.export_text()


@pytest.mark.parametrize(
    ("stale", "patience", "checks", "eval_patience", "expected"),
    [
        (450, 500, 1, 5, "staleness"),
        (100, 500, 4, 5, "evaluator"),
        (0, 0, 0, 0, "staleness"),  # nothing live: reports the zero-progress one
    ],
)
def test_the_panel_names_the_criterion_closest_to_ending_the_epoch(
    stale, patience, checks, eval_patience, expected
):
    """A run has three that can fire and no way to tell them apart from the
    outside: across four runs the answer moved from the evaluator to staleness
    with no setting changed, only the acceptance rate."""
    stats = SearchStats(
        epoch_no_improve=stale,
        epoch_patience=patience,
        eval_checks_without_gain=checks,
        eval_patience=eval_patience,
    )

    assert stats.nearest_epoch_end()[0] == expected


def test_the_epoch_budget_competes_with_the_other_two():
    stats = SearchStats(
        epoch_no_improve=50,
        epoch_patience=500,
        epoch_tasks=9500,
        epoch_max_tasks=10000,
    )

    name, closeness = stats.nearest_epoch_end()

    assert name == "budget"
    assert closeness == pytest.approx(0.95)


def test_unchanged_is_reported_apart_from_invalid():
    """They call for opposite responses: a rising invalid rate means candidates
    are breaking, a rising unchanged rate means the operators are spending the
    run standing still."""
    stats = SearchStats(tasks_completed=1000, unchanged_count=580, invalid_count=20)

    out = _render(stats)

    assert "unchanged 58.0%" in out
    assert "invalid 2.0%" in out


def test_the_evaluator_row_shows_how_long_since_it_approved_anything():
    """A row labelled "score" that moves three times in an hour reads as a
    stalled run rather than a rare judgement."""
    stats = SearchStats(
        best_score=0.327818,
        eval_checks=28,
        eval_checks_without_gain=4,
        eval_patience=5,
    )

    out = _render(stats)

    assert "0.327818" in out
    assert "checks 28" in out
    assert "since gain 4/5" in out


def test_the_seed_batch_is_shown_only_while_a_batch_is_running():
    seeding = SearchStats(phase="seed", seeds_completed=2, seeds_target=5)
    refining = SearchStats(phase="local", seeds_completed=5, seeds_target=5)

    assert "batch 2/5" in _render(seeding)
    assert "batch" not in _render(refining)


def test_an_unevaluated_run_shows_no_score_rather_than_infinity():
    assert "inf" not in _render(SearchStats())
