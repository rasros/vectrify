"""Offline component role controls reach the intended factory."""

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
