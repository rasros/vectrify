import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from bench_search import _bootstrap_ci, discover_cases, plant_seed, read_curve


def _write_lineage(run: Path, scores: list[str]) -> None:
    run.mkdir(parents=True)
    rows = ["id,score,evicted"]
    rows += [f"{i},{s}," for i, s in enumerate(scores, start=1)]
    (run / "lineage.csv").write_text("\n".join(rows) + "\n")


def test_curve_is_the_running_best(tmp_path):
    _write_lineage(tmp_path / "runs" / "2024-01-01_00-00-00", ["0.5", "0.7", "0.3"])
    assert read_curve(tmp_path) == [0.5, 0.5, 0.3]


def test_curve_skips_rows_without_a_score(tmp_path):
    """Eviction rows carry no score and inf marks a candidate that never
    rendered; counting either as a data point corrupts the AUC."""
    _write_lineage(tmp_path / "runs" / "2024-01-01_00-00-00", ["0.5", "", "inf", "0.4"])
    assert read_curve(tmp_path) == [0.5, 0.4]


def test_curve_reads_the_newest_run(tmp_path):
    _write_lineage(tmp_path / "runs" / "1970-01-01_00-00-00", ["9.0"])
    _write_lineage(tmp_path / "runs" / "2024-01-01_00-00-00", ["0.2"])
    assert read_curve(tmp_path) == [0.2]


def test_plant_seed_lays_out_a_resumable_run(tmp_path):
    seed = tmp_path / "seed.svg"
    seed.write_text("<svg/>")
    output = plant_seed(tmp_path / "work", seed)
    planted = list((tmp_path / "work" / "out" / "runs").glob("*/nodes/*.svg"))
    assert len(planted) == 1
    assert planted[0].read_text() == "<svg/>"
    assert output.suffix == ".svg"


def test_discover_cases_requires_both_files(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "target.png").write_bytes(b"x")
    (tmp_path / "good" / "seed.svg").write_text("<svg/>")
    (tmp_path / "partial").mkdir()
    (tmp_path / "partial" / "seed.svg").write_text("<svg/>")
    assert [c.name for c in discover_cases(tmp_path)] == ["good"]


def test_discover_cases_rejects_an_empty_corpus(tmp_path):
    with pytest.raises(SystemExit):
        discover_cases(tmp_path)


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = _bootstrap_ci([-0.02, -0.03, -0.025, -0.021, -0.028])
    assert lo < -0.02 < hi or lo < -0.026 < hi
    assert hi < 0


def test_bootstrap_ci_of_noise_spans_zero():
    lo, hi = _bootstrap_ci([0.02, -0.03, 0.025, -0.021, 0.001])
    assert lo < 0 < hi


def test_results_json_round_trips(tmp_path):
    payload = {"config": {"tasks": 10}, "runs": [{"case": "a", "seed": 1}]}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(payload))
    assert json.loads(path.read_text()) == payload
