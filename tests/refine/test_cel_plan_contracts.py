"""Planner contracts and diagnostic observation have independent owners."""

import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace
from types import SimpleNamespace
from typing import Any

import pytest

from tests.refine.test_cel_plan_ownership import stripes
from vectrify.document import import_svg
from vectrify.refine.cel_plan.bounds import conservative_bounds
from vectrify.refine.cel_plan.contracts import Proposal, State
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import Box
from vectrify.refine.cel_plan.model import Options
from vectrify.refine.cel_plan.proposals import Operators


@pytest.mark.parametrize("module", ["contracts", "limits", "bounds", "component_edits"])
def test_shared_modules_do_not_load_search_or_optimizer(module):
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import importlib
import sys
importlib.import_module('vectrify.refine.cel_plan.{module}')
for name in (
    'vectrify.refine.cel_plan.search',
    'vectrify.refine.cel_plan.refine',
    'vectrify.refine.cel_plan.frontier',
    'scipy.optimize',
    'torch',
):
    assert name not in sys.modules, name
if '{module}' == 'contracts':
    assert 'vectrify.refine.cel_plan.local' not in sys.modules
    assert 'vectrify.document' not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_contract_fields_defaults_and_legacy_identity_are_preserved():
    from vectrify.refine.cel_plan import limits, search
    from vectrify.refine.cel_plan.refine import _bounds

    assert search.State is State
    assert search.Proposal is Proposal
    assert _bounds is conservative_bounds
    assert search.EXPANSIONS is limits.EXPANSIONS
    assert search.MAX_EDIT_OBJECTS == limits.MAX_EDIT_OBJECTS == 256
    assert search.MAX_SEED_NODES == limits.MAX_SEED_NODES == 32_000
    assert search.MAX_SEED_OBJECTS == limits.MAX_SEED_OBJECTS == 8_192
    assert limits.EXPANSIONS == {"fast": 4, "balanced": 8, "high": 16}
    assert [field.name for field in fields(State)] == [
        "document",
        "svg",
        "snapshot",
        "key",
        "details",
        "edits",
        "partition",
    ]
    assert [field.name for field in fields(Proposal)] == [
        "operator",
        "ids",
        "parameters",
        "parent",
        "document",
        "bounds",
        "estimate",
        "details",
        "dependencies",
        "partition",
        "component",
    ]
    defaults = {field.name: field.default for field in fields(Proposal)[6:]}
    assert defaults == {
        "estimate": 0,
        "details": None,
        "dependencies": (),
        "partition": None,
        "component": None,
    }
    proposal = Proposal("test", (), (), "parent", import_svg("<svg/>"), Box(0, 0, 1, 1))
    with pytest.raises(FrozenInstanceError):
        proposal.operator = "changed"


def test_conservative_bounds_preserve_control_hull_transform_and_stroke_padding():
    document = import_svg(
        '<svg width="100" height="100"><g transform="matrix(2 0 0 3 7 11)" '
        'stroke="#222" stroke-width="2"><path id="curve" '
        'd="M1 2C3 -4 5 6 7 8"/></g></svg>'
    )
    # Control points span [1, -4]..[7, 8]. The transformed stroke padding is
    # 4 + 2 * 3 * 5 = 34, including the existing conservative miter allowance.
    assert conservative_bounds(document, "curve") == (-25, -35, 55, 69)
    plain = import_svg('<svg><path id="plain" d="M1 2L7 8"/></svg>')
    assert conservative_bounds(plain, "plain") == (-3, -2, 11, 12)
    geometry = plain.geometry_for("plain")
    empty = replace(
        plain,
        geometries=tuple(
            replace(g, subpaths=()) if g.id == geometry.id else g
            for g in plain.geometries
        ),
    )
    assert conservative_bounds(empty, "plain") == (0, 0, 0, 0)


@pytest.fixture
def operators():
    evidence = stripes()
    return Operators(evidence, build(evidence), Options(quality="high"))


def test_cold_reports_do_not_construct_models_or_mutate_caches(operators, monkeypatch):
    def unexpected_model(_self):
        pytest.fail("Reporting must never access the lazy surface_models property")

    monkeypatch.setattr(
        type(operators.families), "surface_models", property(unexpected_model)
    )
    before = repr(vars(operators))
    families_before = repr(vars(operators.families))
    report = operators.report()
    assert report == operators.report()
    assert operators.families.report() == operators.families.report()
    assert repr(vars(operators)) == before
    assert repr(vars(operators.families)) == families_before
    for key in (
        "material_surfaces",
        "source_splits",
        "piecewise_surfaces",
        "core_material_cells",
        "joint_material_cells",
    ):
        assert report[key] == {}
    assert operators.strokes.silhouettes is not None
    assert (
        report["source_silhouette_strokes"] == operators.strokes.silhouettes.diagnostics
    )
    assert not operators._branches
    assert not operators._live_branches
    assert set(report) == {
        "scheduling",
        "surface_families",
        "material_surfaces",
        "source_splits",
        "piecewise_surfaces",
        "core_material_cells",
        "joint_material_cells",
        "opacity_fields",
        "ink_replacement",
        "source_ridges",
        "source_strokes",
        "source_silhouette_strokes",
        "source_ridge_models",
        "source_ridge_underpaint_rejections",
        "source_ridge_restoration_rejections",
        "closed_overlays",
        "nested_surface_rejections",
        "nested_overlay_rejections",
        "paint_restoration_rejections",
    }


def test_reports_preserve_observed_metadata_and_detach_nested_values(
    operators, monkeypatch
):
    sources: dict[str, dict[str, Any]] = {}
    for slot, key in (
        ("_surface_models", "material_surfaces"),
        ("_split_models", "source_splits"),
        ("_piecewise_models", "piecewise_surfaces"),
        ("_core_models", "core_material_cells"),
        ("_joint_models", "joint_material_cells"),
    ):
        diagnostic: dict[str, Any] = {"proposals": 2, "observed": [{"source": key}]}
        monkeypatch.setattr(
            operators.families, slot, SimpleNamespace(diagnostics=diagnostic)
        )
        sources[key] = diagnostic
    for owner, attribute, key in (
        (operators, "schedule_diagnostics", "scheduling"),
        (operators.families, "diagnostics", "surface_families"),
        (operators.families, "nesting_rejections", "nested_surface_rejections"),
        (operators.ridges, "rim_diagnostics", "source_ridge_models"),
        (
            operators.replacements,
            "restoration_rejections",
            "paint_restoration_rejections",
        ),
    ):
        diagnostic = {"observed": [{"source": key}]}
        monkeypatch.setattr(owner, attribute, diagnostic)
        sources[key] = diagnostic
    family_report = operators.families.report()
    report = operators.report()
    for key, diagnostic in sources.items():
        assert report[key] == diagnostic
        assert report[key] is not diagnostic
        report[key]["observed"][0]["source"] = "detached"
        assert diagnostic["observed"][0]["source"] == key
        if key in family_report:
            assert family_report[key] == diagnostic
