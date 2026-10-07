"""Source-supported editable ink retains junctions, gaps and native paint."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from tests.refine.test_cel_plan_families import prepared
from vectrify.document import Editor, Selection, export_svg, load_project, save_project
from vectrify.document.hit_test import multiply
from vectrify.document.join import curve_path, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.document.topology import inverse_matrix
from vectrify.refine import cel
from vectrify.refine.cel_plan.core_cells import CoreCells
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.ink_models import decoded
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render


def drawing(alpha=128, *, gap=False):
    line = "M20 28 L76 60 M20 60 L76 28"
    if gap:
        line = "M20 28 L42 28 M54 28 L76 28"
    svg = (
        '<svg width="96" height="96"><g opacity="'
        + str(alpha / 255)
        + '"><path d="M8 8 H88 V88 H8Z" fill="#ad8665"/>'
        '<path d="M48 8 H88 V88 H48Z" fill="#9d8065"/>'
        f'<path d="{line}" fill="none" stroke="#202020" '
        'stroke-width="3" stroke-linecap="round"/></g></svg>'
    )
    rgba = render(svg, (96, 96))
    image = Image.fromarray((rgba * 255).round().astype(np.uint8))
    options = Options(refine=False)
    evidence = collect(image, None, options, Work.start(10))
    ink = (~evidence.empty) & (cel.lightness(evidence.target) < 65)
    return evidence, ink


@pytest.mark.parametrize("alpha", [255, 128, 64])
def test_source_crossing_has_connected_editable_centerlines_and_native_width(alpha):
    evidence, ink = drawing(alpha)
    model = decoded(ink, evidence, Options(refine=False), Work.start(10))
    assert model is not None
    assert model.details["model"] == "source-stroke"
    assert 2 <= model.details["width"] <= 4
    assert model.details["support"] >= 0.45
    assert sum(len(s.nodes) for s in model.geometry.subpaths) < 30
    # Incident centerlines retain their shared source junction rather than
    # separately fitted filled outlines. The footprint covers that junction.
    junction = np.array((48, 44))
    assert any(
        np.linalg.norm(np.array(s.nodes[0].values) - junction) < 3
        or np.linalg.norm(np.array(s.nodes[-1].values[-2:]) - junction) < 3
        for s in model.geometry.subpaths
    )
    assert curve_path(model.footprint).contains(tuple(junction))
    assert not model.selected.flags.writeable
    assert np.all(ink[model.selected])


def test_explicit_source_gap_is_not_joined_to_match_an_imagined_reference():
    evidence, ink = drawing(gap=True)
    model = decoded(ink, evidence, Options(refine=False), Work.start(10))
    assert model is not None
    assert len(model.geometry.subpaths) == 2
    assert not curve_path(model.footprint).contains((48, 28))
    assert not model.selected[28, 48]


def test_unsupported_disconnected_dot_is_not_owned_by_a_nearby_stroke():
    from vectrify.refine.cel_plan.ink_models import models

    evidence, ink = drawing(gap=True)
    target = evidence.target.copy()
    target[72:77, 78:83] = 32
    ink[72:77, 78:83] = True
    found = models(
        ink,
        replace(evidence, target=target),
        Options(),
        Work.start(10),
        prune_spurs=True,
    )
    assert found
    assert all(not m.selected[72:77, 78:83].any() for m in found)


def test_broad_dark_material_and_cancelled_work_do_not_become_strokes():
    evidence, _ink = drawing()
    mask = ~evidence.empty
    flat = replace(evidence, target=np.full(evidence.target.shape, 100.0))
    assert decoded(mask, flat, Options(), Work.start(10)) is None
    work = Work.start(10)
    work.stop.set()
    assert decoded(mask, evidence, Options(), work) is None


@pytest.mark.parametrize("alpha", [128, 64])
@pytest.mark.parametrize("matrix", [None, (1.2, 0.15, 0.2, 1.0, 6, -10)])
@pytest.mark.parametrize("layout", ["regions", "ink-planes"])
def test_component_replaces_ink_and_underpaint_together_with_true_stroke_paths(
    alpha, matrix, layout, monkeypatch
):
    from vectrify.refine.cel_plan import core_cells

    monkeypatch.setattr(core_cells, "MAX_PLANES", 4)
    evidence, ink = drawing(alpha)
    # Independent source paint fragments provide an intentionally dense start.
    y, x = np.indices(evidence.labels.shape)
    labels = np.where(evidence.empty, 0, 1 + (y // 8) * 12 + x // 8)
    labels[ink] += 200
    evidence = replace(evidence, labels=labels.astype(np.int32), drawn=ink, line=ink)
    frontier, state, options = prepared(evidence, layers=True)
    assert state.partition is not None
    if matrix is not None:
        carrier = next(s.id for s in state.partition.surfaces if s.role == "underlay")
        editor = Editor(state.document, selection=Selection(whole_document=True))
        with editor.transaction("Reexpress carrier frame") as tx:
            tx.replace_geometry(
                carrier,
                transformed_geometry(
                    state.document.geometry_for(carrier),
                    multiply(
                        inverse_matrix(matrix), root_matrix(state.document, carrier)
                    ),
                ),
            )
            tx.set_attributes(
                carrier, {"transform": f"matrix({' '.join(str(v) for v in matrix)})"}
            )
        document = editor.snapshot.document
        svg = export_svg(document)
        state = replace(
            state,
            document=document,
            svg=svg,
            snapshot=LocalPolicy(frontier.policy).start(
                svg, frontier.policy.evaluate(svg)
            ),
        )
    graph = build(evidence)
    factory = CoreCells(
        Families(evidence, graph, options),
        options,
        joint=True,
        grouping="ward",
        boundary_fit="anchored" if layout == "ink-planes" else "curve",
        ink_support="connected",
        layout=layout,
    )
    edits = list(factory(state, Work.start(20)))
    assert edits
    assert all(p.details is not None for p in edits)
    stroke_edits = [
        p
        for p in edits
        if p.details is not None and p.details["core_material_cells"]["stroke_models"]
    ]
    assert stroke_edits
    for edit in stroke_edits:
        assert edit.partition is not None
        assert state.partition is not None
        svg = export_svg(edit.document)
        full = frontier.policy.evaluate(svg)
        assert full.valid
        assert full.structure["stroke_contours"] > 0
        assert edit.partition.follows(state.partition)
        Operators(evidence, graph, options).validate_partition(
            edit.partition, Work.start(10)
        )
        strokes = [
            e for e in edit.document.elements() if e.get("stroke", "none") != "none"
        ]
        assert strokes
        assert all(e.get("fill") == "none" for e in strokes)
        actual = render(svg, evidence.source_size)
        np.testing.assert_array_equal(
            actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
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


def two_inks():
    svg = (
        '<svg width="96" height="96"><g opacity="0.5">'
        '<path d="M8 8H88V88H8Z" fill="#ad8665"/>'
        '<path d="M20 28H76" fill="none" stroke="#202020" stroke-width="3"/>'
        '<path d="M20 60H76" fill="none" stroke="#184060" stroke-width="7"/>'
        "</g></svg>"
    )
    image = Image.fromarray((render(svg, (96, 96)) * 255).round().astype(np.uint8))
    evidence = collect(image, None, Options(refine=False), Work.start(10))
    mask = (~evidence.empty) & (cel.lightness(evidence.target) < 105)
    return evidence, mask


def test_distinct_source_widths_and_paints_get_separate_editable_models():
    from vectrify.refine.cel_plan.ink_models import models

    evidence, mask = two_inks()
    found = models(mask, evidence, Options(refine=False), Work.start(10))
    assert len(found) == 2
    assert not (found[0].selected & found[1].selected).any()
    assert abs(found[0].details["width"] - found[1].details["width"]) > 2
    colors = sorted(tuple(np.round(m.paint).astype(int)) for m in found)
    assert colors == [(24, 64, 96), (32, 32, 32)]
    assert all(not m.selected.flags.writeable for m in found)
    assert sum(len(m.geometry.subpaths) for m in found) == 2
    # A consumer asking for one paint must not silently flatten the two.
    assert decoded(mask, evidence, Options(), Work.start(10)) is None


def test_bounded_discovery_retains_unexamined_source_ink(monkeypatch):
    from vectrify.refine.cel_plan import ink_models

    evidence, mask = two_inks()
    monkeypatch.setattr(ink_models, "MAX_RUNS", 1)
    found = ink_models.models(mask, evidence, Options(), Work.start(10))
    assert len(found) == 1
    assert found[0].details["source_runs_scanned"] == 1
    assert (mask & ~found[0].selected).sum() > 100
    assert not curve_path(found[0].footprint).contains((48, 44))


def test_interruption_during_final_geometry_discards_discovered_models(monkeypatch):
    from vectrify.refine.cel_plan import ink_models

    evidence, mask = two_inks()
    work = Work.start(10)
    original = ink_models.footprint

    def stopped(geometry, width):
        result = original(geometry, width)
        work.stop.set()
        return result

    monkeypatch.setattr(ink_models, "footprint", stopped)
    assert ink_models.models(mask, evidence, Options(), work) == ()


def test_degree_two_source_junctions_join_without_joining_gaps_or_real_branches():
    from vectrify.refine.cel_plan.ink_models import connected_runs

    runs = [
        np.array(v, float)
        for v in (
            [(0, 0), (1, 0)],
            [(2, 0), (1, 0)],
            [(2, 0), (3, 0)],
            [(3.25, 0), (4, 0)],
            [(10, 0), (11, 0)],
            [(11, 0), (12, 1)],
            [(11, 0), (12, -1)],
        )
    ]
    result = connected_runs(runs, Work.start(10))
    assert len(result) == 5
    np.testing.assert_array_equal(result[0], [[0, 0], [1, 0], [2, 0], [3, 0]])
    assert len(result[1]) == 2  # A real gap has distinct source endpoints.
    assert sum(tuple(r[0]) == (11, 0) or tuple(r[-1]) == (11, 0) for r in result) == 3
    work = Work.start(10)
    work.stop.set()
    assert connected_runs(runs, work) == []


def test_pruning_preserves_a_real_short_branch_and_an_explicit_source_gap():
    from vectrify.refine.cel_plan.ink_models import models

    svg = (
        '<svg width="96" height="96"><path d="M8 8H88V88H8Z" fill="#ad8665"/>'
        '<path d="M20 48H76M48 48V32" stroke="#202020" stroke-width="3" '
        'fill="none" stroke-linecap="round"/></svg>'
    )
    pixels = render(svg, (96, 96))
    evidence = collect(
        Image.fromarray((pixels * 255).round().astype(np.uint8)),
        None,
        Options(),
        Work.start(10),
    )
    mask = ~evidence.empty & (cel.lightness(evidence.target) < 65)
    found = models(mask, evidence, Options(), Work.start(10), prune_spurs=True)
    assert len(found) == 1
    shape = curve_path(found[0].footprint)
    assert all(shape.contains(p) for p in ((48, 48), (48, 34), (22, 48), (74, 48)))
    evidence, mask = drawing(gap=True)
    found = models(mask, evidence, Options(), Work.start(10), prune_spurs=True)
    assert len(found) == 1
    assert len(found[0].geometry.subpaths) == 2
    assert not curve_path(found[0].footprint).contains((48, 28))
