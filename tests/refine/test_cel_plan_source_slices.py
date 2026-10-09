"""A whole physical stroke can be removed from a connected filled shadow."""

from dataclasses import replace

import numpy as np
import pathops
import pytest
from PIL import Image

from vectrify.document import (
    Editor,
    Selection,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.join import curve_path, path_style
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import atoms as atom_module
from vectrify.refine.cel_plan.atoms import Atoms
from vectrify.refine.cel_plan.band_plans import (
    BandPlans,
    attached_locality,
    color_quantization,
)
from vectrify.refine.cel_plan.component_edits import ComponentEdit
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import Operators, bounds
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import Proposal, State
from vectrify.refine.cel_plan.source_slices import source_slice


def fixture(*, alpha=1, group_alpha=1, gap=False):
    prefix = (
        '<svg width="96" height="96"><g id="scene" '
        f'opacity="{alpha}"><path id="bg" d="M0 0H96V96H0Z" fill="#ad8665"/>'
    )
    suffix = "</g></svg>"
    if group_alpha != 1:
        prefix += f'<g id="paint" opacity="{group_alpha}">'
        suffix = "</g>" + suffix
    shadow = '<path d="M76.25 25.75H90V70H76.25Z" fill="#121008"/>'
    line = "M20.25 28.25H44 M50 28.25H76.25" if gap else "M20.25 28.25H76.25"
    source = (
        prefix
        + shadow
        + f'<path d="{line}" fill="none" stroke="#121008" stroke-width="3"/>{suffix}'
    )
    document = import_svg(
        prefix + '<path id="ink" d="M20.25 25.75H90V70H76.25V30.75H20.25Z" '
        f'fill="#121008"/>{suffix}'
    )
    rgba = render(source, (96, 96))
    e = collect(
        Image.fromarray(np.rint(rgba * 255).astype(np.uint8)),
        None,
        Options(),
        Work.start(10),
    )
    points = np.column_stack((np.linspace(20.25, 76.25, 24), np.full(24, 28.25)))
    guard = SourceLineGuard(e.rgba, (SourceProfile.at(points, 3),))
    return e, guard, document


@pytest.mark.parametrize("alpha", [1, 0.75])
def test_subtracted_band_keeps_original_shadow_controls_and_has_no_old_ink_fill(alpha):
    _, guard, document = fixture(alpha=alpha)
    observed = guard.source_breaks(guard.original_profiles()[0])
    result = source_slice(document, "ink", observed, Work.start(10))
    assert result is not None
    removed, retained = result
    original = document.geometry_for("ink")
    assert retained.subpaths[: len(original.subpaths)] == original.subpaths
    difference = pathops.op(
        curve_path(original), curve_path(removed), pathops.PathOp.DIFFERENCE
    )
    assert pathops.op(curve_path(retained), difference, pathops.PathOp.XOR).area == 0
    assert (
        pathops.op(
            curve_path(retained), curve_path(removed), pathops.PathOp.INTERSECTION
        ).area
        == 0
    )


def test_real_gap_fragment_and_broad_shadow_are_not_sliced_into_fake_complete_lines():
    _, guard, document = fixture(gap=True)
    observed = guard.source_breaks(guard.original_profiles()[0])
    assert observed is not None
    assert observed.gaps.any()
    assert source_slice(document, "ink", observed, Work.start(10)) is None
    _, guard, document = fixture()
    observed = guard.source_breaks(guard.original_profiles()[0])
    assert observed is not None
    assert (
        source_slice(document, "ink", replace(observed, anchors=None), Work.start(10))
        is None
    )
    broad = import_svg(
        '<svg width="96" height="96"><path id="ink" d="M10 10H90V70H10Z"/></svg>'
    )
    assert source_slice(broad, "ink", observed, Work.start(10)) is None
    with pytest.raises(ValueError, match="tolerance"):
        source_slice(document, "ink", observed, Work.start(10), tolerance=float("nan"))
    with pytest.raises(StageInterruptedError):
        source_slice(document, "ink", observed, Work.start(0))


def test_color_quantization_cannot_hide_alpha_or_more_than_one_premultiplied_byte():
    before = np.array([[[80, 90, 100, 255], [32, 32, 32, 10]]], np.uint8)
    outside = np.ones((1, 2), bool)
    after = before.copy()
    after[0, 0, 0] += 1
    assert color_quantization(before, after, outside) == {
        "changed_pixels": 1,
        "max_premultiplied_byte_delta": 1.0,
        "alpha_exact": True,
    }
    after[0, 0, 0] += 1
    assert color_quantization(before, after, outside) is None
    after = before.copy()
    after[0, 1, 3] += 1
    assert color_quantization(before, after, outside) is None


@pytest.mark.parametrize(
    ("alpha", "group_alpha"), [(1, 1), (0.75, 1), (0.5, 1), (0.75, 0.6)]
)
def test_attached_stroke_replays_ancestor_and_roundtrips_partial_group_opacity(
    alpha, group_alpha, monkeypatch
):
    e, guard, document = fixture(alpha=alpha, group_alpha=group_alpha)
    labels = np.zeros(e.labels.shape, np.int32)
    labels[e.target.max(axis=-1) < e.target[0, 0].max() - 25] = 1
    e = replace(e, labels=labels)
    graph = build(e)
    old = Partition((Surface("bg", (0,), covered=(1,)), Surface("ink", (1,))))
    policy = Policy.from_evidence(e, graph)
    svg = export_svg(document)
    state = State(
        document,
        svg,
        LocalPolicy(policy).start(svg, policy.evaluate(svg)),
        "ancestor",
        {},
        partition=old,
    )
    classes = (np.indices(labels.shape)[1] >= 48).astype(np.int32)
    atoms, groups = Atoms.original(graph).partition(
        graph, (1,), classes, 2, Work.start(10), compact=True, retain_parent=True
    )
    part = old.split(
        ("ink",), (Surface("ink", tuple(sorted(groups[0] + groups[1]))),), atoms
    )
    edit = Proposal(
        "joint-core-cells",
        ("ink",),
        (),
        state.key,
        document,
        bounds(document, document, ("ink",)),
        partition=part,
        component=ComponentEdit.bind(
            document, old, document.ancestry("ink")[-2].id, Work.start(10)
        ),
    )
    monkeypatch.setattr(atom_module, "MAX_CUTS", 1)
    planner = BandPlans(e, graph, graph, guard=lambda _work: guard)
    alternatives = list(
        planner.proposals(state, edit, Work.start(20), source_fit=True, attached=True)
    )
    assert len(alternatives) == 1, planner.diagnostics
    candidate = alternatives[0]
    assert candidate.partition.follows(old)
    assert len(candidate.partition.atoms.cuts) == 1
    Operators(e, graph, Options()).validate_partition(
        candidate.partition, Work.start(10)
    )
    candidate.component.validate(
        document,
        candidate.document,
        old,
        candidate.partition,
        candidate.ids,
        candidate.bounds,
        Work.start(10),
    )
    assert (
        path_style(candidate.document, candidate.document.element("ink"))["fill"]
        == "none"
    )
    assert len(candidate.document.geometry_for("ink").subpaths[0].nodes) == 2
    record = candidate.details["planned_band_stroke"]
    assert record["attached"]
    assert record["source_line_comparison"]["rejections"] == []
    assert record["source_fit"]["fit"]["native_body_absence"]
    assert record["cut_quantization"]["alpha_exact"]
    proof = record["source_fit"]["native_locality"]
    assert proof["scope"] == "complete-final-candidate-vs-material-parent"
    assert proof["native_alpha_exact"]
    assert proof["native_footprint_exact"]
    removed = parse_path(record["removed"])
    mutations: tuple[tuple[str, dict[str, str | None]], ...] = (
        ("bg", {"fill": "#ffffff"}),
        ("scene", {"opacity": "0.25"}),
        (record["marks"], {"transform": "translate(8 0)"}),
    )
    for target, attrs in mutations:
        editor = Editor(candidate.document, selection=Selection(whole_document=True))
        with editor.transaction("Unrelated change must not pass locality") as tx:
            tx.set_attributes(target, attrs)
        assert (
            attached_locality(
                document,
                editor.snapshot.document,
                "ink",
                removed,
                (),
                (96, 96),
                Work.start(10),
            )
            is None
        )
    actual = render(export_svg(candidate.document), (96, 96))
    np.testing.assert_array_equal(actual[..., 3], render(svg, (96, 96))[..., 3])
    residual = candidate.document.geometry_for(record["marks"])
    original = document.geometry_for("ink")
    # Published residuals receive fresh editing ids; commands and controls
    # must still retain the original shadow geometry exactly.
    assert (
        replace(
            residual, subpaths=residual.subpaths[: len(original.subpaths)]
        ).path_data()
        == original.path_data()
    )
    actual = render(export_svg(candidate.document), (96, 96))
    restored, _ = load_project(save_project(candidate.document))
    assert np.array_equal(actual, render(export_svg(restored), (96, 96)))
