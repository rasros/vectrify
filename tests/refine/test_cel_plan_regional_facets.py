"""Regional facets retain source ink and advance only complete source ledgers."""

from dataclasses import replace

import numpy as np
import pytest

from tests.helpers import required
from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_final_cells import fragmented
from tests.refine.test_cel_plan_piecewise_surfaces import step
from vectrify.document import export_svg, load_project, save_project
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import core_cells
from vectrify.refine.cel_plan.core_cells import Cell, CoreCells
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import Box, LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.opacity import Paint
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import Proposal


def factory(evidence, options, **kwargs):
    return CoreCells(
        Families(evidence, build(evidence), options),
        options,
        joint=True,
        grouping="ward",
        ink_support="connected",
        ink_fit="source-intervals",
        **kwargs,
    )


@pytest.mark.parametrize("alpha", [255, 128])
@pytest.mark.parametrize("hole", [False, True])
def test_source_facet_cuts_improve_paint_with_complete_native_ownership_and_reload(
    alpha, hole
):
    evidence = step(alpha, hole=hole)
    evidence = replace(evidence, opacity=evidence.rgba[..., 3])
    frontier, state, options = prepared(evidence, layers=True)
    plain = list(factory(evidence, options)(state, Work.start(20)))
    fitted = factory(evidence, options, facet_fit="regional")
    edits = list(fitted(state, Work.start(20)))
    baselines = [p for p in edits if "regional_facets" not in required(p.details)]
    assert len(baselines) == len(plain)
    for baseline, parent in zip(baselines, plain, strict=True):
        assert baseline.parameters == parent.parameters
        assert parent.partition is not None
        assert baseline.partition is not None
        assert baseline.partition.metadata() == parent.partition.metadata()
        np.testing.assert_array_equal(
            render(export_svg(baseline.document), evidence.source_size),
            render(export_svg(parent.document), evidence.source_size),
        )
    facets = [p for p in edits if "regional_facets" in required(p.details)]
    assert facets, fitted.diagnostics
    ops = Operators(evidence, build(evidence), options)
    for edit in facets:
        parent = next(p for p in plain if p.parameters[2] == edit.parameters[2])
        assert edit.details is not None
        assert parent.details is not None
        assert (
            edit.details["core_material_cells"]["source_squared_error"]
            < (parent.details["core_material_cells"]["source_squared_error"])
        )
        assert edit.details["regional_facets"]["cuts"][0]["rho"] == pytest.approx(
            43, abs=0.1
        )
        assert edit.partition is not None
        ops.validate_partition(edit.partition, Work.start(10))
        assert state.partition is not None
        assert edit.partition.follows(state.partition)
        assert edit.component is not None
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
        np.testing.assert_array_equal(actual[..., 3], evidence.rgba[..., 3])
        if hole:
            assert actual[24:40, 40:56, 3].max() == 0
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        document, _ = load_project(save_project(edit.document))
        required(Partition.from_metadata(edit.partition.metadata())).validate(document)
        np.testing.assert_array_equal(
            render(export_svg(document), evidence.source_size), actual
        )


def test_material_cuts_keep_actual_source_strokes_exact_and_above_new_paint():
    evidence, _ink = fragmented(gap=True)
    frontier, state, options = prepared(evidence, layers=True)
    options = replace(options, line_width=6)
    common = {
        "boundary_fit": "anchored",
        "ink_roles": "fitted",
        "ink_coverage": "fractional",
    }
    plain = list(factory(evidence, options, **common)(state, Work.start(30)))
    edits = list(
        factory(evidence, options, facet_fit="regional", **common)(
            state, Work.start(30)
        )
    )
    facets = [p for p in edits if "regional_facets" in required(p.details)]
    assert facets

    def strokes(document):
        return [
            (element.attributes, document.geometry_for(element.id).path_data())
            for element in document.elements()
            if element.geometry_id is not None
            and element.get("stroke") not in {None, "none"}
        ]

    for edit in facets:
        parent = next(p for p in plain if p.parameters[2] == edit.parameters[2])
        # Generated IDs may depend on the source cut ledger; stroke properties,
        # exact endpoints, caps and geometry must all stay unchanged.
        expected = [
            (tuple((k, v) for k, v in a if k != "id"), g)
            for a, g in strokes(parent.document)
        ]
        actual = [
            (tuple((k, v) for k, v in a if k != "id"), g)
            for a, g in strokes(edit.document)
        ]
        assert expected
        assert actual == expected
        assert edit.details is not None
        assert parent.details is not None
        assert (
            edit.details["core_material_cells"]["stroke_models"]
            == parent.details["core_material_cells"]["stroke_models"]
        )
        assert frontier.policy.evaluate(export_svg(edit.document)).valid
        assert edit.partition is not None
        Operators(evidence, build(evidence), options).validate_partition(
            edit.partition, Work.start(10)
        )
        for surface in edit.partition.surfaces:
            if surface.role == "overlay":
                siblings = edit.document.ancestry(surface.id)[-2].children
                position = next(i for i, e in enumerate(siblings) if e.id == surface.id)
                material_ids = {
                    s.id
                    for s in required(edit.partition).surfaces
                    if s.role != "overlay" and s.id in edit.ids
                }
                assert all(
                    i < position for i, e in enumerate(siblings) if e.id in material_ids
                )


def fixture_cells():
    xy = np.column_stack((np.arange(96) + 0.5, np.full(96, 0.5)))
    classes = np.repeat(np.arange(3, dtype=np.uint8), 32)[None, :]
    shape = parse_path("M0 0H96V2H0Z")
    paint = Paint("#80664c", 1)
    stroke = {"width": 3, "linecap": "round"}
    cells = [
        Cell(
            f"r{i}",
            np.arange(i * 32, (i + 1) * 32),
            shape,
            shape,
            paint,
            100 - i,
            20,
            support=i,
            covered_classes=(2,) if i < 2 else (),
            ink=i == 2,
            stroke=stroke if i == 2 else None,
            footprint=shape if i == 2 else None,
        )
        for i in range(3)
    ]
    splits = {}
    for cell in cells[:2]:
        middle = len(cell.indices) // 2
        splits[cell.key] = (
            cell.error,
            np.array((1.0, 0)),
            float(xy[cell.indices[middle], 0]),
            replace(cell, key=cell.key + "0", indices=cell.indices[:middle], error=0),
            replace(cell, key=cell.key + "1", indices=cell.indices[middle:], error=0),
        )
    return xy, classes, cells, splits


def test_class_remapping_preserves_secondary_ownership_and_does_not_mutate_parent():
    xy, classes, cells, splits = fixture_cells()
    before = classes.copy()
    result, changed = CoreCells._facet_cells(cells, classes, 0, splits["r0"], xy)
    np.testing.assert_array_equal(classes, before)
    assert not changed.flags.writeable
    np.testing.assert_array_equal(changed[0], np.repeat([0, 1, 2, 3], [16, 16, 32, 32]))
    assert result[0].covered_classes == (3, 1)
    assert result[1].covered_classes == (3,)
    assert result[2].covered_classes == (3,)
    assert result[3].stroke is cells[2].stroke
    assert result[3].draw is cells[2].draw
    assert result[3].footprint is cells[2].footprint
    assert result[3].paint is cells[2].paint
    assert cells[0].covered_classes == (2,)


@pytest.mark.parametrize("consumer_mutation", [False, True])
def test_failed_cut_cannot_poison_later_prefix_or_consumer_feedback(
    monkeypatch, consumer_mutation
):
    evidence = step(128)
    _frontier, state, options = prepared(evidence, layers=True)
    fitted = factory(evidence, options, facet_fit="regional")
    xy, classes, cells, splits = fixture_cells()
    monkeypatch.setattr(
        fitted, "_ink_support", lambda _work: np.zeros(evidence.empty.shape, bool)
    )
    monkeypatch.setattr(
        fitted, "_split", lambda _state, _base, cell, *_a: splits[cell.key]
    )
    seen = []

    def proposal(_state, _base, _selected, current, cuts, *_a, **_k):
        seen.append(([c.key for c in current], list(cuts)))
        if cuts[0][0] == "r0":
            return None
        return Proposal(
            "regional", (), (), state.key, state.document, Box(0, 0, 96, 64), details={}
        )

    monkeypatch.setattr(fitted, "_proposal", proposal)
    edits = fitted._regional_facets(
        state,
        None,
        (),
        cells,
        classes,
        32,
        xy,
        None,
        classes == 0,
        None,
        None,
        None,
        Work.start(10),
    )
    found = next(edits)
    if consumer_mutation:
        monkeypatch.setattr(fitted, "_last_regional_feasible", False, raising=False)
    assert list(edits) == []
    assert seen[0][0] == ["r0/0", "r0/1", "r1", "r2"]
    assert seen[1][0] == ["r0", "r1/0", "r1/1", "r2"]
    assert [
        c["parent"] for c in required(found.details)["regional_facets"]["cuts"]
    ] == ["r1"]
    assert fitted.diagnostics["regional_facet_attempts"] == 2
    assert fitted.diagnostics["regional_facet_proposals"] == 1


def test_child_identity_cannot_alias_an_unsplit_decimal_region_key():
    xy, classes, cells, splits = fixture_cells()
    cells = [
        replace(cell, key=key)
        for cell, key in zip(cells, ("r1", "r10", "r11"), strict=True)
    ]
    first, changed = CoreCells._facet_cells(cells, classes, 0, splits["r0"], xy)
    assert [c.key for c in first] == ["r1/0", "r1/1", "r10", "r11"]
    index = next(i for i, c in enumerate(first) if c.key == "r10")
    final, final_classes = CoreCells._facet_cells(
        first, changed, index, splits["r1"], xy
    )
    assert [c.key for c in final] == ["r1/0", "r1/1", "r10/0", "r10/1", "r11"]
    np.testing.assert_array_equal(
        final_classes[0], np.repeat([0, 1, 2, 3, 4], [16, 16, 16, 16, 32])
    )
    assert (
        sum(c.error for c in final)
        == sum(c.error for c in cells) - splits["r0"][0] - splits["r1"][0]
    )
    assert final[-1].stroke is cells[-1].stroke


@pytest.mark.parametrize("stop", ["before", "during"])
def test_cancelled_facet_stage_publishes_no_new_partial_drawing(monkeypatch, stop):
    evidence = step(128)
    _frontier, state, options = prepared(evidence, layers=True)
    fitted = factory(evidence, options, facet_fit="regional")
    xy, classes, cells, splits = fixture_cells()
    work = Work.start(10)
    if stop == "before":
        work.stop.set()
    monkeypatch.setattr(
        fitted, "_ink_support", lambda _w: np.zeros(evidence.empty.shape, bool)
    )

    def split(_state, _base, cell, *_a):
        work.stop.set()
        return splits[cell.key]

    monkeypatch.setattr(fitted, "_split", split)
    monkeypatch.setattr(
        fitted,
        "_proposal",
        lambda *_a, **_k: pytest.fail("Canceled stage reached export"),
    )
    assert (
        list(
            fitted._regional_facets(
                state,
                None,
                (),
                cells,
                classes,
                32,
                xy,
                None,
                classes == 0,
                None,
                None,
                None,
                work,
            )
        )
        == []
    )
    assert fitted.diagnostics["regional_facet_proposals"] == 0


def test_cell_budget_precedes_facet_votes_and_export(monkeypatch):
    evidence = step(128)
    _frontier, state, options = prepared(evidence, layers=True)
    fitted = factory(evidence, options, facet_fit="regional")
    xy, classes, cells, _splits = fixture_cells()
    monkeypatch.setattr(core_cells, "MAX_REGION_CELLS", len(cells))
    monkeypatch.setattr(
        fitted, "_split", lambda *_a: pytest.fail("Cell bound reached votes")
    )
    assert (
        list(
            fitted._regional_facets(
                state,
                None,
                (),
                cells,
                classes,
                32,
                xy,
                None,
                classes == 0,
                None,
                None,
                None,
                Work.start(10),
            )
        )
        == []
    )
    assert fitted.diagnostics["regional_facet_cell_limits"] == 1


def test_consumer_cannot_reset_a_successful_prefix_and_bounds_keep_complete_parent(
    monkeypatch,
):
    evidence = step(128)
    _frontier, state, options = prepared(evidence, layers=True)
    fitted = factory(evidence, options, facet_fit="regional")
    xy, classes, cells, splits = fixture_cells()
    monkeypatch.setattr(
        fitted, "_ink_support", lambda _w: np.zeros(evidence.empty.shape, bool)
    )
    monkeypatch.setattr(fitted, "_split", lambda _s, _b, cell, *_a: splits[cell.key])
    monkeypatch.setattr(
        fitted,
        "_proposal",
        lambda *_a, **_k: Proposal(
            "regional", (), (), state.key, state.document, Box(0, 0, 96, 64), details={}
        ),
    )
    edits = fitted._regional_facets(
        state,
        None,
        (),
        cells,
        classes,
        32,
        xy,
        None,
        classes == 0,
        None,
        None,
        None,
        Work.start(10),
    )
    first = next(edits)
    assert first.details is not None
    assert len(first.details["regional_facets"]["cuts"]) == 1
    monkeypatch.setattr(fitted, "_last_regional_feasible", False, raising=False)
    second = next(edits)
    assert [
        c["parent"] for c in required(second.details)["regional_facets"]["cuts"]
    ] == [
        "r0",
        "r1",
    ]
    assert list(edits) == []
    # Reaching a structural cell limit retains the already yielded parent.
    monkeypatch.setattr(core_cells, "MAX_REGION_CELLS", len(cells) + 1)
    bounded = list(
        fitted._regional_facets(
            state,
            None,
            (),
            cells,
            classes,
            32,
            xy,
            None,
            classes == 0,
            None,
            None,
            None,
            Work.start(10),
        )
    )
    assert len(bounded) == 1
    assert bounded[0].details is not None
    assert len(bounded[0].details["regional_facets"]["cuts"]) == 1


def test_raw_ink_and_empty_pixels_cannot_supply_material_votes(monkeypatch):
    evidence = step(128)
    rgb = evidence.target.copy()
    rgb[~evidence.empty] = 180
    ink = np.zeros(evidence.empty.shape, bool)
    ink[12:52, 40:44] = True
    rgb[ink] = 5
    evidence = replace(evidence, target=rgb)
    _frontier, state, options = prepared(evidence, layers=True)
    fitted = factory(evidence, options, facet_fit="regional")
    xy, classes, cells, _splits = fixture_cells()
    monkeypatch.setattr(fitted, "_ink_support", lambda _w: ink)

    def vote(_s, _b, _c, *_a):
        voter = fitted._facet_lines
        assert voter is not None
        np.testing.assert_array_equal(voter.visible, ~evidence.empty & ~ink)
        assert list(voter(~evidence.empty, Work.start(10))) == []
        assert voter.diagnostics["points"] == 0
        return

    monkeypatch.setattr(fitted, "_split", vote)
    assert (
        list(
            fitted._regional_facets(
                state,
                None,
                (),
                cells,
                classes,
                32,
                xy,
                None,
                classes == 0,
                None,
                None,
                None,
                Work.start(10),
            )
        )
        == []
    )


def test_cancellation_during_complete_ledger_check_publishes_no_cut(monkeypatch):
    evidence = step(128)
    _frontier, state, options = prepared(evidence, layers=True)
    fitted = factory(evidence, options, facet_fit="regional")
    xy, classes, cells, splits = fixture_cells()
    work = Work.start(10)
    monkeypatch.setattr(
        fitted, "_ink_support", lambda _w: np.zeros(evidence.empty.shape, bool)
    )
    monkeypatch.setattr(fitted, "_split", lambda _s, _b, cell, *_a: splits[cell.key])

    def proposal(*_a, **_k):
        work.stop.set()
        return Proposal(
            "regional", (), (), state.key, state.document, Box(0, 0, 96, 64), details={}
        )

    monkeypatch.setattr(fitted, "_proposal", proposal)
    assert (
        list(
            fitted._regional_facets(
                state,
                None,
                (),
                cells,
                classes,
                32,
                xy,
                None,
                classes == 0,
                None,
                None,
                None,
                work,
            )
        )
        == []
    )
    assert fitted.diagnostics["regional_facet_proposals"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"joint": True},
        {"joint": True, "layout": "planes"},
        {"facet_fit": "unknown"},
    ],
)
def test_inapplicable_regional_mode_rejected_before_work(kwargs):
    evidence = step(128)
    _frontier, _state, options = prepared(evidence, layers=True)
    with pytest.raises(ValueError, match="Regional facets"):
        CoreCells(
            Families(evidence, build(evidence), options),
            options,
            **({"facet_fit": "regional"} | kwargs),
        )
