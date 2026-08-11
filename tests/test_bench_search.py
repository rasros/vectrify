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


def _make_case(root: Path, name: str, seeds: int = 2, target: bool = True) -> Path:
    case = root / name
    (case / "seeds").mkdir(parents=True)
    if target:
        (case / "target.png").write_bytes(b"x")
    for i in range(seeds):
        (case / "seeds" / f"{i + 1}.svg").write_text(f"<svg id='{i}'/>")
    return case


def test_plant_seed_plants_every_seed(tmp_path):
    """The pool must start as several lineages; planting one would leave
    crossover recombining a candidate with itself."""
    case = _make_case(tmp_path, "case", seeds=3)
    output = plant_seed(tmp_path / "work", case)
    planted = list((tmp_path / "work" / "out" / "runs").glob("*/nodes/*.svg"))
    assert len(planted) == 3
    assert {p.read_text() for p in planted} == {f"<svg id='{i}'/>" for i in range(3)}
    assert output.suffix == ".svg"


def test_discover_cases_requires_a_target_and_seeds(tmp_path):
    _make_case(tmp_path, "good")
    _make_case(tmp_path, "no-target", target=False)
    _make_case(tmp_path, "no-seeds", seeds=0)
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
