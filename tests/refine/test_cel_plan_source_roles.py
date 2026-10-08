"""Joint source roles recover lines hidden inside existing paint owners."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_core_cells import material
from tests.refine.test_cel_plan_families import prepared
from vectrify.document import export_svg, load_project, save_project
from vectrify.document.join import path_style
from vectrify.refine.cel_plan import core_cells, source_roles
from vectrify.refine.cel_plan.core_cells import CoreCells
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.materials import moments
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render


def mixed(alpha=128, gap=False):
    evidence = material(alpha, hole=True)
    y, x = np.indices(evidence.labels.shape)
    ink = (y >= 18) & (y < 21) & (x >= 15) & (x < 81)
    if gap:
        ink &= (x < 42) | (x >= 54)
    rgb, rgba = evidence.target.copy(), evidence.rgba.copy()
    rgb[ink], rgba[ink, :3] = 20, 20 / 255
    # Every original atom retains both paint and ink, without a separate line
    # owner. Owner-majority classification incorrectly assigns all of it paint.
    return replace(
        evidence, target=rgb, smooth=rgb, coarse=rgb, rgba=rgba, drawn=ink, line=ink
    ), ink


def test_virtual_roles_keep_true_gaps_and_do_not_change_source_atoms():
    evidence, ink = mixed(gap=True)
    graph = build(evidence)
    own = ~evidence.empty & (graph.labels != 8)
    source = graph.labels - 1
    before = graph.labels.copy()
    roles = source_roles.observations(source, own, ink, Work.start(10))
    assert roles is not None
    np.testing.assert_array_equal(graph.labels, before)
    np.testing.assert_array_equal(roles.source >= 0, own)
    np.testing.assert_array_equal(roles.ink[roles.source[own]], ink[own])
    np.testing.assert_array_equal(roles.owners[roles.source[own]], source[own])
    assert roles.source[19, 48] >= 0
    assert not roles.ink[roles.source[19, 48]]
    assert len(roles.owners) > len(np.unique(source[own]))
    assert all(not a.flags.writeable for a in (roles.source, roles.owners, roles.ink))


def test_virtual_role_bounds_and_cancellation_publish_no_partial_classification(
    monkeypatch,
):
    evidence, ink = mixed()
    own, source = ~evidence.empty, evidence.labels - 1
    monkeypatch.setattr(source_roles, "MAX_MATERIALS", 8)
    assert source_roles.observations(source, own, ink, Work.start(10)) is None
    work = Work.start(10)
    work.stop.set()
    assert source_roles.observations(source, own, ink, work) is None
    np.testing.assert_array_equal(source, evidence.labels - 1)


def test_virtual_intrinsic_statistics_recompose_original_weighted_atoms(monkeypatch):
    evidence, ink = mixed()
    graph = build(evidence)
    graph = replace(
        graph,
        regions=tuple(
            replace(r, feature=r.id / 10, texture=r.id / 20) for r in graph.regions
        ),
    )
    roles = source_roles.observations(
        (graph.labels - 1) // 2, ~evidence.empty, ink, Work.start(10)
    )
    assert roles is not None
    options = Options(refine=False)
    expected = moments(replace(evidence, opacity=None), graph, options, Work.start(10))
    actual = source_roles.moments(roles, evidence, graph, options, Work.start(10))
    monkeypatch.setattr(source_roles, "CHUNK_PIXELS", 113)
    tiled = source_roles.moments(roles, evidence, graph, options, Work.start(10))
    np.testing.assert_allclose(tiled, actual, rtol=1e-12, atol=1e-9)
    for owner in np.unique(roles.owners):
        np.testing.assert_allclose(
            actual[roles.owners == owner].sum(axis=0),
            expected[2 * owner + 1 : 2 * owner + 3].sum(axis=0),
            rtol=1e-12,
            atol=1e-9,
        )
    work = Work.start(10)
    work.stop.set()
    with pytest.raises(StageInterruptedError, match="role statistics"):
        source_roles.moments(roles, evidence, graph, options, work)


@pytest.mark.parametrize("alpha", [128, 64])
@pytest.mark.parametrize("gap", [False, True])
@pytest.mark.parametrize("grouping", ["ward", "paint-fit"])
@pytest.mark.parametrize("ink_roles", ["connected", "fitted"])
@pytest.mark.parametrize("ink_coverage", ["visible", "fractional"])
def test_joint_edit_recovers_editable_lines_inside_mixed_owners(
    alpha, gap, grouping, ink_roles, ink_coverage
):
    evidence, ink = mixed(alpha, gap)
    frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    factory = CoreCells(
        Families(evidence, graph, options),
        options,
        joint=True,
        grouping=grouping,
        boundary_fit="anchored",
        ink_support="connected",
        ink_roles=ink_roles,
        ink_coverage=ink_coverage,
    )
    before = state.svg
    edits = list(factory(state, Work.start(20)))
    assert edits, factory.diagnostics
    assert factory.diagnostics["source_role_mixed_owners_peak"] >= 6
    supported = [e for e in edits if e.details["core_material_cells"]["stroke_models"]]
    assert supported, factory.diagnostics
    for edit in supported:
        assert edit.details["core_material_cells"]["ink_coverage"] == ink_coverage
        assert edit.details["core_material_cells"]["source_roles"] == (
            "stroke-body-and-material"
            if ink_roles == "fitted"
            else "pixel-ink-and-material"
        )
        assert edit.partition.follows(state.partition)
        assert edit.partition.atoms.cuts
        Operators(evidence, graph, options).validate_partition(
            edit.partition, Work.start(10)
        )
        assert edit.component.validate(
            state.document,
            edit.document,
            state.partition,
            edit.partition,
            edit.ids,
            edit.bounds,
            Work.start(10),
        )
        svg = export_svg(edit.document)
        full = frontier.policy.evaluate(svg)
        assert full.valid, full.rejections
        actual = render(svg, evidence.source_size)
        np.testing.assert_array_equal(
            actual[..., 3], render(before, evidence.source_size)[..., 3]
        )
        assert actual[24:40, 40:56, 3].max() == 0
        assert np.mean(actual[ink, :3].mean(axis=1) < 0.2) > 0.9
        assert full.terms["color"] < state.snapshot.evaluation.terms["color"]
        for element in edit.document.elements():
            style = path_style(edit.document, element)
            if style["stroke"] not in (None, "none"):
                assert style["fill"] == "none"
                assert all(
                    not s.closed
                    for s in edit.document.geometry_for(element.id).subpaths
                )
        if gap:
            np.testing.assert_allclose(
                actual[19, 46:50, :3] * 255, evidence.target[19, 46:50], atol=2
            )
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        restored, _ = load_project(save_project(edit.document))
        np.testing.assert_array_equal(
            render(export_svg(restored), evidence.source_size), actual
        )
    assert state.svg == before
    assert state.partition.atoms is None


def test_final_role_cut_exhaustion_retains_the_complete_original_owner_namespace(
    monkeypatch,
):
    from vectrify.refine.cel_plan import atoms

    evidence, _ink = mixed()
    _frontier, state, options = prepared(evidence, layers=True)
    factory = CoreCells(
        Families(evidence, build(evidence), options),
        options,
        joint=True,
        grouping="ward",
        boundary_fit="anchored",
        ink_support="connected",
    )
    before = export_svg(state.document)
    monkeypatch.setattr(atoms, "MAX_CUTS", 0)
    assert list(factory(state, Work.start(10))) == []
    assert factory.diagnostics["atom_exclusions"] > 0
    assert state.partition.atoms is None
    assert export_svg(state.document) == before


def test_mid_role_discovery_and_statistics_cancellation_publish_nothing(monkeypatch):
    evidence, ink = mixed()
    graph = build(evidence)
    own = ~evidence.empty
    roles = source_roles.observations(graph.labels - 1, own, ink, Work.start(10))
    work = Work.start(10)
    bincount = source_roles.np.bincount

    def stopped_count(*args, **kwargs):
        value = bincount(*args, **kwargs)
        work.stop.set()
        return value

    with monkeypatch.context() as patch:
        patch.setattr(source_roles.np, "bincount", stopped_count)
        with pytest.raises(StageInterruptedError, match="role statistics"):
            source_roles.moments(roles, evidence, graph, Options(), work)
    unique = source_roles.np.unique
    work = Work.start(10)

    def stopped_unique(*args, **kwargs):
        value = unique(*args, **kwargs)
        work.stop.set()
        return value

    monkeypatch.setattr(source_roles.np, "unique", stopped_unique)
    assert source_roles.observations(graph.labels - 1, own, ink, work) is None
    np.testing.assert_array_equal(graph.labels, evidence.labels)


def test_source_discovery_precedes_budget_and_keeps_an_independent_owned_chain(
    monkeypatch,
):
    from vectrify.refine.cel_plan import core_cells

    evidence, ink = mixed(gap=True)
    frontier, state, options = prepared(evidence, layers=True)
    held = state.partition.owners[int(evidence.labels[19, 40])]
    state = replace(state, details={**state.details, "paint_constraints": [held]})
    factory = CoreCells(
        Families(evidence, build(evidence), options),
        options,
        joint=True,
        grouping="ward",
        boundary_fit="anchored",
        ink_support="connected",
    )
    masks = []
    discover = core_cells.ink_models

    def recorded(mask, *args, **kwargs):
        masks.append(mask.copy())
        assert kwargs["prune_spurs"]
        assert kwargs["boundary_contacts"]
        return discover(mask, *args, **kwargs)

    monkeypatch.setattr(core_cells, "ink_models", recorded)
    edits = list(factory(state, Work.start(20)))
    assert edits, factory.diagnostics
    assert len(masks) == 1
    np.testing.assert_array_equal(masks[0], ink)
    assert factory.diagnostics["source_stroke_owner_exclusions"] > 0
    for edit in edits:
        assert edit.document.geometry_for(held) == state.document.geometry_for(held)
        assert edit.document.element(held) == state.document.element(held)
        stroke_models = edit.details["core_material_cells"]["stroke_models"]
        assert stroke_models
        assert all(
            m["runs"] == 1 and m["owner_excluded_runs"] == 1 for m in stroke_models
        )
        paths = [
            e
            for e in edit.document.elements()
            if path_style(edit.document, e)["stroke"] not in (None, "none")
        ]
        assert paths
        for e in paths:
            assert all(
                n.values[-2] > 48
                for s in edit.document.geometry_for(e.id).subpaths
                for n in s.nodes
            )
        svg = export_svg(edit.document)
        full = frontier.policy.evaluate(svg)
        assert full.valid, full.rejections
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert local.canvas.matches(render(svg, evidence.source_size))
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)


def test_unmodeled_tiny_mark_stays_filled_without_a_fabricated_stroke_connection():
    evidence, ink = mixed(gap=True)
    ink = ink.copy()
    ink[42:44, 78:81] = True
    rgb, rgba = evidence.target.copy(), evidence.rgba.copy()
    rgb[42:44, 78:81] = 20
    rgba[42:44, 78:81, :3] = 20 / 255
    evidence = replace(
        evidence, target=rgb, smooth=rgb, coarse=rgb, rgba=rgba, drawn=ink, line=ink
    )
    frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    factory = CoreCells(
        Families(evidence, graph, options),
        options,
        joint=True,
        grouping="ward",
        boundary_fit="anchored",
        ink_support="connected",
    )
    edits = list(factory(state, Work.start(20)))
    assert edits, factory.diagnostics
    for edit in edits:
        branch = Operators(evidence, graph, options).branch(
            edit.partition, Work.start(10)
        )
        surfaces = {s.id: s for s in edit.partition.surfaces}
        for atom in np.unique(branch.graph.labels[42:44, 78:81]):
            owner = edit.partition.owners[int(atom)]
            assert surfaces[owner].role == "surface"
            assert (
                path_style(edit.document, edit.document.element(owner))["stroke"]
                == "none"
            )
        svg = export_svg(edit.document)
        assert frontier.policy.evaluate(svg).valid
        actual = render(svg, evidence.source_size)
        np.testing.assert_allclose(actual[42:44, 78:81, :3] * 255, 20, atol=2)
        for element in edit.document.elements():
            if path_style(edit.document, element)["stroke"] not in (None, "none"):
                assert all(
                    n.values[-1] < 25
                    for s in edit.document.geometry_for(element.id).subpaths
                    for n in s.nodes
                )


def test_fitted_role_competitor_uses_the_same_complete_physical_discovery_once(
    monkeypatch,
):
    evidence, _ink = mixed(gap=True)
    _frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    calls = []
    original = core_cells.ink_models

    def observed(mask, *args, **kwargs):
        models = original(mask, *args, **kwargs)
        calls.append(
            (
                mask.copy(),
                [(m.geometry.path_data(), m.paint.tolist(), m.details) for m in models],
            )
        )
        return models

    monkeypatch.setattr(core_cells, "ink_models", observed)
    for interpretation in ("connected", "fitted"):
        factory = CoreCells(
            Families(evidence, graph, options),
            options,
            joint=True,
            grouping="ward",
            boundary_fit="anchored",
            ink_support="connected",
            ink_roles=interpretation,
        )
        assert list(factory(state, Work.start(20)))
        assert factory.diagnostics["source_stroke_attempts"] == 1
        assert bool(factory.diagnostics["source_fitted_role_pixels"]) == (
            interpretation == "fitted"
        )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    assert calls[0][1] == calls[1][1]


def test_cancelled_early_stroke_discovery_cannot_publish_material_roles(monkeypatch):
    evidence, _ink = mixed()
    _frontier, state, options = prepared(evidence, layers=True)
    factory = CoreCells(
        Families(evidence, build(evidence), options),
        options,
        joint=True,
        grouping="ward",
        ink_support="connected",
        ink_roles="fitted",
    )
    work = Work.start(20)
    original = core_cells.ink_models

    def cancelled(*args, **kwargs):
        result = original(*args, **kwargs)
        work.stop.set()
        return result

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Interrupted discovery must not classify material roles")

    monkeypatch.setattr(core_cells, "ink_models", cancelled)
    monkeypatch.setattr(core_cells, "observations", forbidden)
    assert list(factory(state, work)) == []
    assert factory.diagnostics["proposals"] == 0
    assert factory.diagnostics["source_fitted_role_pixels"] == 0
    assert state.partition.atoms is None
    assert state.snapshot.canvas.matches(render(state.svg, evidence.source_size))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ink_roles": "unknown"},
        {"ink_roles": "fitted"},
        {
            "joint": True,
            "grouping": "ward",
            "ink_support": "connected",
            "ink_roles": "fitted",
            "layout": "ink-planes",
        },
    ],
)
def test_fitted_roles_cannot_be_silently_ignored_in_another_layout(kwargs):
    with pytest.raises(ValueError, match="Fitted ink roles"):
        CoreCells(None, Options(), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ink_coverage": "unknown"},
        {"ink_coverage": "fractional"},
        {
            "joint": True,
            "grouping": "ward",
            "ink_support": "connected",
            "ink_coverage": "fractional",
            "layout": "ink-planes",
        },
    ],
)
def test_fractional_coverage_cannot_be_silently_ignored_in_another_layout(kwargs):
    with pytest.raises(ValueError, match="Fractional ink coverage"):
        CoreCells(None, Options(), **kwargs)
