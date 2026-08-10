import csv

from vectrify.search.collector import STATS_COLUMNS, STATS_FIELDS, StatCollector
from vectrify.search.models import INVALID_SCORE
from vectrify.search.stats import SearchStats


def _row(stats: SearchStats, tmp_path) -> dict[str, str]:
    StatCollector(stats, tmp_path)._flush_row()
    with (tmp_path / "stats.csv").open(encoding="utf-8", newline="") as f:
        return next(iter(csv.DictReader(f)))


def test_columns_derive_from_fields():
    assert list(STATS_FIELDS) == STATS_COLUMNS


def test_written_row_matches_the_header(tmp_path):
    row = _row(SearchStats(), tmp_path)
    assert list(row) == STATS_COLUMNS
    assert None not in row.values()  # no extra/short fields


def test_counters_are_written_through(tmp_path):
    s = SearchStats()
    s.tasks_completed = 7
    s.accepted_count = 3
    s.epoch = 2
    row = _row(s, tmp_path)
    assert row["tasks_completed"] == "7"
    assert row["accepted_count"] == "3"
    assert row["epoch"] == "2"


def test_infinite_best_score_is_written_blank(tmp_path):
    s = SearchStats()
    s.best_score = INVALID_SCORE
    assert _row(s, tmp_path)["best_score"] == ""


def test_real_best_score_is_written(tmp_path):
    s = SearchStats()
    s.best_score = 0.125
    assert float(_row(s, tmp_path)["best_score"]) == 0.125


def test_no_run_dir_is_a_noop():
    StatCollector(SearchStats(), None)._flush_row()  # must not raise
