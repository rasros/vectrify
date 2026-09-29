import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from bench_search import _bootstrap_ci, discover_cases


def _make_case(root: Path, name: str, seeds: int = 2, target: bool = True) -> Path:
    case = root / name
    (case / "seeds").mkdir(parents=True)
    if target:
        (case / "target.png").write_bytes(b"x")
    for i in range(seeds):
        (case / "seeds" / f"{i + 1}.svg").write_text(f"<svg id='{i}'/>")
    return case


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
