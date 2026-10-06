"""Clean targets stay independent of generation and incomplete pools are explicit."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from scripts import bench_cel_pairs as bench
from scripts.cel_pairs import cases, degrade
from vectrify.refine.cel_plan.frontier import Observation
from vectrify.refine.cel_plan.score import render

SVG = (
    '<svg width="96" height="96"><rect width="96" height="96" fill="white"/>'
    '<path d="M16 16H80V80H16Z" fill="#e08080" stroke="#202020" stroke-width="3"/>'
    '<path d="M30 40H66" stroke="#202020" stroke-width="3"/></svg>'
)


def test_oracle_excludes_invalid_candidates_and_reports_cost_tradeoff():
    def row(key, cost, error, valid=True, nodes=10):
        return {
            "key": key,
            "label": key,
            "representation_cost": cost,
            "nodes": nodes,
            "hard_valid": valid,
            "clean": {"mse": error},
        }

    selected = row("selected", 20, 12)
    candidates = [
        selected,
        row("cheaper", 19, 10),
        row("detailed", 40, 3, nodes=20),
        row("invalid", 1, 0, valid=False),
    ]
    found = bench.oracle(candidates, selected)
    assert found["best_in_pool"]["key"] == "detailed"
    assert found["best_at_selected_cost"]["key"] == "cheaper"
    assert found["best_at_selected_cost"]["selected_mse_gap"] == 2
    bounded = bench.oracle(candidates, selected, node_budget=10)
    assert bounded["best_in_pool"]["key"] == "cheaper"


def test_incomplete_candidate_pool_is_explicit_and_bounded():
    pool = bench.Pool(max_bytes=len(SVG.encode()))
    item = Observation(SVG, "one", "Candidate", None, {}, {"accepted": False})
    pool.observe(item)
    pool.observe(item)
    assert not pool.complete
    assert pool.bytes == len(SVG.encode())
    assert len(pool.observations) == 1
    assert pool.omitted[0]["reason"] == "benchmark-pool-memory-limit"


def test_nonfinite_line_metrics_serialize_as_unavailable(tmp_path):
    path = tmp_path / "metrics.json"
    bench.write_json(path, {"lines": {"line_f": float("nan")}, "count": np.int64(4)})
    assert json.loads(path.read_text()) == {"lines": {"line_f": None}, "count": 4}
    assert "NaN" not in path.read_text()


@pytest.mark.parametrize("case", cases(), ids=lambda case: case["name"])
def test_tuning_feature_rectangles_are_valid_at_multiple_resolutions(case):
    assert case["features"]
    for size in ((37, 41), (900, 1000)):
        boxes = bench.feature_boxes(case, size)
        assert boxes.keys() == case["features"].keys()
        for x, y, width, height in boxes.values():
            assert 0 <= x < x + width <= size[0]
            assert 0 <= y < y + height <= size[1]


def test_feature_rectangles_reject_out_of_canvas_annotations():
    with pytest.raises(ValueError, match="within"):
        bench.feature_boxes({"features": {"bad": [0.9, 0.9, 0.2, 0.2]}}, (100, 100))


@pytest.mark.parametrize("composition_opacity", [1, 0.5])
def test_paired_run_logs_actual_candidates_without_passing_clean_truth_to_planner(
    tmp_path, monkeypatch, composition_opacity
):
    from scripts.cel_pairs import composition

    source = composition(SVG, composition_opacity)
    clean = Image.fromarray((render(source, (96, 96)) * 255).round().astype(np.uint8))
    degraded = degrade(clean, "noise", 4)
    case = {
        "name": "synthetic",
        "family": "test",
        "set": "tuning",
        "features": {"mark": [0.25, 0.3, 0.5, 0.3]},
    }
    monkeypatch.setattr(bench, "pair", lambda *_: (source, clean, degraded))
    original = bench.vectorize
    supplied = []

    def observed_generator(image, **kwargs):
        supplied.append((image.tobytes(), set(kwargs)))
        return original(image, **kwargs)

    monkeypatch.setattr(bench, "vectorize", observed_generator)
    row = bench.compare(
        case,
        "noise",
        "cel-planned",
        {"quality": "fast", "gradients": False},
        tmp_path,
        seconds=10,
        composition_opacity=composition_opacity,
    )
    assert row["status"] == "ready"
    assert row["composition_opacity"] == composition_opacity
    assert row["family"] == "test"
    assert row["split"] == "tuning"
    assert supplied == [(degraded.tobytes(), {"options", "seconds", "observe"})]
    assert row["clean_pixels_sha256"] != row["input_pixels_sha256"]
    assert row["clean"]["features"]["mark"] > 0
    assert row["candidate_pool_complete"]
    assert row["candidate_oracle"]["available"]
    assert row["selected_key"] in {item["key"] for item in row["candidates"]}
    assert (tmp_path / "feature-mark.png").exists()
    for item in row["candidates"]:
        saved = tmp_path / item["svg"]
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == item["key"]
        assert item["exact_evaluated"]
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert report["candidate_oracle"] == row["candidate_oracle"]


def test_failed_planner_run_preserves_rejection_artifacts(tmp_path, monkeypatch):
    clean = Image.new("RGBA", (96, 96), "white")
    case = {"name": "synthetic", "family": "test", "set": "tuning"}
    monkeypatch.setattr(bench, "pair", lambda *_: (SVG, clean, clean))

    def failed(_image, *, observe, **_kwargs):
        observe(
            Observation(
                "<svg/>",
                "invalid",
                "Rejected",
                None,
                {},
                {"accepted": False, "rejections": ["invalid-candidate"]},
            )
        )
        raise ValueError("No validated candidate")

    monkeypatch.setattr(bench, "vectorize", failed)
    (tmp_path / "drawing.svg").write_text("previous success")
    (tmp_path / "drawing.png").write_bytes(b"previous success")
    row = bench.compare(case, "clean", "cel-planned", {}, tmp_path)
    assert row["status"] == "failed"
    assert row["error"] == "No validated candidate"
    assert row["candidates"][0]["hard_valid"] is False
    assert not row["candidate_oracle"]["available"]
    assert (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "drawing.svg").exists()
    assert not (tmp_path / "drawing.png").exists()


def test_planner_benchmark_matches_blank_document_operation_pixels(
    tmp_path, monkeypatch
):
    clean = Image.fromarray((render(SVG, (96, 96)) * 255).round().astype(np.uint8))
    case = {"name": "synthetic", "family": "test", "set": "tuning"}
    monkeypatch.setattr(bench, "pair", lambda *_: (SVG, clean, clean))
    settings = {"quality": "fast", "gradients": False, "refine": False}
    row = bench.compare(case, "clean", "cel-planned", settings, tmp_path, seconds=10)
    assert row["status"] == "ready"
    operation_svg, _, _ = bench.generate(clean, "cel-planned", settings, seconds=10)
    direct_svg = (tmp_path / "drawing.svg").read_text()
    np.testing.assert_array_equal(
        render(direct_svg, clean.size), render(operation_svg, clean.size)
    )


def test_default_cli_does_not_silently_accept_heldout_case(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_cel_pairs.py",
            "--cases",
            cases(True)[0]["name"],
            "--out",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        bench.main()
    assert caught.value.code == 2
    assert not (tmp_path / "summary.json").exists()


def test_candidate_pool_bounds_entry_count_as_well_as_svg_bytes():
    pool = bench.Pool(max_bytes=10000, max_entries=1)
    item = Observation("<svg/>", "small", "Tiny", None, {}, {"accepted": False})
    pool.observe(item)
    pool.observe(item)
    assert len(pool.observations) == 1
    assert not pool.complete
    assert pool.bytes <= pool.max_bytes
