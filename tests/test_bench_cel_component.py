"""Offline component role controls are explicit and reach the intended factory."""

import pytest

from scripts import bench_cel_component as bench


@pytest.mark.parametrize(
    "ink_fit", ["carrier", "source-gaps", "source-intervals", "source-widths"]
)
@pytest.mark.parametrize("atom_layout", ["retired", "residual"])
def test_cli_passes_explicit_fitted_roles_and_anchored_boundary(
    monkeypatch, tmp_path, ink_fit, atom_layout
):
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
            "--ink-fit",
            ink_fit,
            "--opacity-model",
            "components",
            "--facet-fit",
            "regional",
            "--atom-layout",
            atom_layout,
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
    assert calls[0][1]["ink_fit"] == ink_fit
    assert calls[0][1]["opacity_model"] == "components"
    assert calls[0][1]["facet_fit"] == "regional"
    assert calls[0][1]["atom_layout"] == atom_layout


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--ink-fit", "carrier"],
        ["--proposal", "source-strokes", "--ink-fit", "carrier"],
    ],
)
def test_cli_rejects_inapplicable_residual_atoms_before_loading_source(
    monkeypatch, extra
):
    monkeypatch.setattr(
        "sys.argv", ["bench_cel_component", "--atom-layout", "residual", *extra]
    )
    monkeypatch.setattr(
        bench,
        "run",
        lambda *_a, **_k: pytest.fail("Invalid atom mode reached benchmark"),
    )
    with pytest.raises(SystemExit) as raised:
        bench.main()
    assert raised.value.code == 2


@pytest.mark.parametrize(
    "extra", [[], ["--ink-support", "connected"], ["--proposal", "source-strokes"]]
)
def test_cli_rejects_inapplicable_regional_facets_before_loading_source(
    monkeypatch, extra
):
    monkeypatch.setattr(
        "sys.argv", ["bench_cel_component", "--facet-fit", "regional", *extra]
    )
    monkeypatch.setattr(
        bench,
        "run",
        lambda *_a, **_k: pytest.fail("Invalid facet mode reached benchmark"),
    )
    with pytest.raises(SystemExit) as raised:
        bench.main()
    assert raised.value.code == 2


def test_cli_rejects_component_opacity_for_stroke_only_comparison(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_cel_component",
            "--proposal",
            "source-strokes",
            "--opacity-model",
            "components",
        ],
    )
    monkeypatch.setattr(
        bench,
        "run",
        lambda *_a, **_k: pytest.fail("Invalid opacity model reached benchmark"),
    )
    with pytest.raises(SystemExit) as raised:
        bench.main()
    assert raised.value.code == 2


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


@pytest.mark.parametrize(
    "extra", [[], ["--ink-support", "connected"], ["--proposal", "source-strokes"]]
)
@pytest.mark.parametrize(
    "ink_fit", ["carrier", "source-gaps", "source-intervals", "source-widths"]
)
def test_cli_rejects_inapplicable_carrier_fitting_before_loading_source(
    monkeypatch, extra, ink_fit
):
    monkeypatch.setattr(
        "sys.argv", ["bench_cel_component", "--ink-fit", ink_fit, *extra]
    )
    monkeypatch.setattr(
        bench,
        "run",
        lambda *_a, **_k: pytest.fail("Invalid carrier option reached benchmark"),
    )
    with pytest.raises(SystemExit) as raised:
        bench.main()
    assert raised.value.code == 2
