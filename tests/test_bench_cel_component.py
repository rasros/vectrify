"""Offline component role controls are explicit and reach the intended factory."""

import pytest

from scripts import bench_cel_component as bench


def test_cli_passes_explicit_fitted_roles_and_anchored_boundary(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bench, "run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_cel_component",
            "--grouping",
            "ward",
            "--ink-support",
            "connected",
            "--ink-roles",
            "fitted",
            "--boundary-fit",
            "anchored",
            "--source-line-diagnostics",
            "--ink-coverage",
            "fractional",
            "--out",
            str(tmp_path),
        ],
    )
    bench.main()
    assert len(calls) == 1
    assert calls[0][0][0]["name"] == "sword"
    assert calls[0][1]["ink_roles"] == "fitted"
    assert calls[0][1]["ink_support"] == "connected"
    assert calls[0][1]["boundary_fit"] == "anchored"
    assert calls[0][1]["source_line_diagnostics"] is True
    assert calls[0][1]["ink_coverage"] == "fractional"


@pytest.mark.parametrize(
    "extra",
    [[], ["--ink-support", "connected"], ["--proposal", "source-strokes"]],
)
def test_cli_rejects_inapplicable_fitted_roles_before_loading_source(
    monkeypatch, extra
):
    monkeypatch.setattr(
        "sys.argv", ["bench_cel_component", "--ink-roles", "fitted", *extra]
    )
    monkeypatch.setattr(
        bench,
        "run",
        lambda *_a, **_k: pytest.fail("Invalid role option reached benchmark"),
    )
    with pytest.raises(SystemExit) as raised:
        bench.main()
    assert raised.value.code == 2


@pytest.mark.parametrize(
    "extra",
    [[], ["--ink-support", "connected"], ["--proposal", "source-strokes"]],
)
def test_cli_rejects_inapplicable_source_line_diagnostics_before_loading_source(
    monkeypatch, extra
):
    monkeypatch.setattr(
        "sys.argv", ["bench_cel_component", "--source-line-diagnostics", *extra]
    )
    monkeypatch.setattr(
        bench,
        "run",
        lambda *_a, **_k: pytest.fail("Invalid diagnostics reached benchmark"),
    )
    with pytest.raises(SystemExit) as raised:
        bench.main()
    assert raised.value.code == 2


@pytest.mark.parametrize(
    "extra",
    [[], ["--ink-support", "connected"], ["--proposal", "source-strokes"]],
)
def test_cli_rejects_inapplicable_fractional_coverage_before_loading_source(
    monkeypatch, extra
):
    monkeypatch.setattr(
        "sys.argv", ["bench_cel_component", "--ink-coverage", "fractional", *extra]
    )
    monkeypatch.setattr(
        bench,
        "run",
        lambda *_a, **_k: pytest.fail("Invalid coverage option reached benchmark"),
    )
    with pytest.raises(SystemExit) as raised:
        bench.main()
    assert raised.value.code == 2
