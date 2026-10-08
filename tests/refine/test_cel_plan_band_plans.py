"""Compound strokes preserve exact marks, ancestor ledgers and source gaps."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from vectrify.document import export_svg, import_svg, load_project, save_project
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import atoms as atom_module
from vectrify.refine.cel_plan.atoms import Atoms
from vectrify.refine.cel_plan.band_plans import mark_components, separate
from vectrify.refine.cel_plan.component_edits import ComponentEdit
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import Operators, bounds
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import Proposal, State


def fixture(alpha=0.75, frame="", missing=False):
    svg = (
        '<svg width="96" height="64"><g id="scene" '
        f'opacity="{alpha}"><path id="bg" d="M0 0H96V64H0Z" fill="#c4b79c"/>'
        f'<g id="frame" transform="{frame or "matrix(1 0 0 1 0 0)"}">'
        '<path id="shade" d="M8 42H88V58H8Z" fill="#a18c6d"/>'
        '<path id="ink" d="M10 30.5H86V33.5H10Z M75 20h2v2h-2z" fill="#202020"/>'
        "</g></g></svg>"
    )
    document = import_svg(svg)
    rgba = render(svg, (96, 64))
    if missing:
        rgba[10:12, 80:82] = (32 / 255, 32 / 255, 32 / 255, 1)
    options = Options()
    e = collect(
        Image.fromarray((rgba * 255).round().astype(np.uint8)),
        None,
        options,
        Work.start(10),
    )
    labels = np.zeros(e.labels.shape, np.int32)
    # Deliberately one original source atom contains line and independent marks.
    labels[e.target.max(axis=-1) < 110] = 1
    labels[(e.target[..., 0] < 180) & (e.target[..., 0] > 100) & (labels == 0)] = 2
    e = replace(e, labels=labels)
    graph = build(e)
    old = Partition(
        (
            Surface("bg", (0,), covered=(1, 2)),
            Surface("ink", (1,)),
            Surface("shade", (2,)),
        )
    )
    policy = Policy.from_evidence(e, graph)
    exported = export_svg(document)
    state = State(
        document,
        exported,
        LocalPolicy(policy).start(exported, policy.evaluate(exported)),
        "ancestor",
        {},
        partition=old,
    )
    # The ordinary proposal has already spent its one permitted split on this
    # root, grouping both children in its compound filled ink owner.
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
        component=ComponentEdit.bind(document, old, "frame", Work.start(10)),
    )
    return e, graph, options, policy, state, edit


@pytest.mark.parametrize("alpha", [0.75, 0.5])
@pytest.mark.parametrize("frame", ["", "matrix(0.9 0.03 0.08 0.85 0.25 1.5)"])
def test_compound_stroke_reallocates_from_ancestor_at_saturated_cut_limit(
    alpha, frame, monkeypatch
):
    e, graph, options, policy, state, edit = fixture(alpha, frame)
    monkeypatch.setattr(atom_module, "MAX_CUTS", 1)
    operators = Operators(e, graph, options, filled_bands=True)
    planner = operators.families.band_planner
    alternatives = list(planner(state, edit, Work.start(20)))
    assert len(alternatives) == 5, planner.diagnostics
    assert len({p.partition.atoms for p in alternatives}) == 1
    assert len({p.details["planned_band_stroke"]["width"] for p in alternatives}) == 5
    for variant in alternatives:
        operators.validate_partition(variant.partition, Work.start(10))
        variant.component.validate(
            state.document,
            variant.document,
            state.partition,
            variant.partition,
            variant.ids,
            variant.bounds,
            Work.start(10),
        )
        assert (
            variant.details["planned_band_stroke"]["source_line_comparison"][
                "rejections"
            ]
            == []
        )
    proposed = alternatives[0]
    assert len(proposed.partition.atoms.cuts) == 1
    assert proposed.partition.follows(state.partition)
    # This really is a distinct alternative namespace, never an edit claiming
    # to extend the already finished material ledger.
    assert not proposed.partition.atoms.extends(edit.partition.atoms)
    operators.validate_partition(proposed.partition, Work.start(10))
    proposed.component.validate(
        state.document,
        proposed.document,
        state.partition,
        proposed.partition,
        proposed.ids,
        proposed.bounds,
        Work.start(10),
    )
    details = proposed.details["planned_band_stroke"]
    stroke = proposed.document.element("ink")
    assert stroke.get("fill") == "none"
    assert stroke.get("stroke") == "#202020"
    assert len(proposed.document.geometry_for("ink").subpaths[0].nodes) == 2
    oldmark = state.document.geometry_for("ink").subpaths[1]
    mark = proposed.document.geometry_for(details["marks"]).subpaths[0]
    assert (
        replace(state.document.geometry_for("ink"), subpaths=(mark,)).path_data()
        == replace(state.document.geometry_for("ink"), subpaths=(oldmark,)).path_data()
    )
    assert details["source_line_comparison"]["rejections"] == []
    for oid in ("bg", "shade"):
        assert proposed.document.element(oid) == state.document.element(oid)
        assert proposed.document.geometry_for(oid) == state.document.geometry_for(oid)
    svg = export_svg(proposed.document)
    actual = render(svg, e.source_size)
    restored, _ = load_project(save_project(proposed.document))
    np.testing.assert_array_equal(actual, render(export_svg(restored), e.source_size))
    native = policy.evaluate(svg)
    assert native.valid
    local = LocalPolicy(policy).update(
        state.snapshot, svg, proposed.bounds, native.structure
    )
    assert local.canvas.matches(actual)
    assert (
        max(abs(local.evaluation.terms[k] - v) for k, v in native.terms.items()) < 2e-7
    )
    assert edit.document is state.document
    assert state.partition.atoms is None


def test_unrepresented_source_pixels_keep_existing_owner_without_fake_support():
    e, graph, options, _, state, edit = fixture(missing=True)
    ops = Operators(e, graph, options, filled_bands=True)
    proposed = next(ops.families.band_planner(state, edit, Work.start(20)))
    assert (
        proposed.details["planned_band_stroke"]["inherited_unrepresented_source_pixels"]
        == 4
    )
    labels = proposed.partition.atoms.labels(graph, Work.start(10))
    assert set(map(int, labels[10:12, 80:82].ravel())) <= set(
        next(s.members for s in proposed.partition.surfaces if s.id == "ink")
    )
    assert (
        render(export_svg(proposed.document), e.source_size)[10:12, 80:82, 0].min()
        > 0.5
    )


@pytest.mark.parametrize(
    ("rule", "data"),
    [
        ("evenodd", "M10 10H80V50H10Z M20 20H70V40H20Z"),
        ("nonzero", "M10 10H80V50H10Z M20 20H70V40H20Z"),
        ("nonzero", "M10 10H60V30H10Z M50 20H80V40H50Z"),
    ],
)
def test_holes_nested_marks_and_interacting_fills_cannot_be_separated(rule, data):
    assert separate(parse_path(data), rule, Work.start(10)) is None


def test_connected_source_touching_both_parts_is_ambiguous():
    own = np.zeros((20, 40), bool)
    own[10, 2:38] = True
    main = np.zeros(own.shape)
    marks = main.copy()
    main[10, 2:12] = 1
    marks[10, 30:38] = 1
    assert mark_components(own, main, marks, (1, 1), Work.start(10)) is None


def test_cancelled_planning_publishes_no_partial_document_or_namespace():
    e, graph, options, _, state, edit = fixture()
    ops = Operators(e, graph, options, filled_bands=True)
    work = Work.start(10)
    work.stop.set()
    assert list(ops.families.band_planner(state, edit, work)) == []
    assert state.partition.atoms is None
    assert len(edit.partition.atoms.cuts) == 1


def test_new_atom_cut_limit_excludes_whole_alternative(monkeypatch):
    e, graph, options, _, state, edit = fixture()
    ops = Operators(e, graph, options, filled_bands=True)
    monkeypatch.setattr(atom_module, "MAX_CUTS", 0)
    assert list(ops.families.band_planner(state, edit, Work.start(20))) == []
    assert ops.families.band_planner.diagnostics["bounded"] == 1


def test_default_operators_leave_co_planning_disabled():
    e, graph, options, _, _, _ = fixture()
    assert Operators(e, graph, options).families.band_planner is None


def test_explicit_width_setting_does_not_offer_width_adjustments():
    e, graph, options, _, state, edit = fixture()
    ops = Operators(e, graph, replace(options, line_width=3), filled_bands=True)
    edits = list(ops.families.band_planner(state, edit, Work.start(20)))
    assert len(edits) == 1
    details = edits[0].details["planned_band_stroke"]
    assert details["width"] == details["intrinsic_width"]


def test_branch_co_planning_extends_actual_ancestor_prefix():
    e, graph, options, _, state, first = fixture()
    state = replace(state, partition=first.partition)
    current = first.partition.atoms.graph(e, graph, Work.start(10))
    members = next(s.members for s in state.partition.surfaces if s.id == "ink")
    classes = (np.indices(current.labels.shape)[0] >= 32).astype(np.int32)
    ledger, groups = first.partition.atoms.partition(
        current, members, classes, 2, Work.start(10), compact=True, retain_parent=True
    )
    owned = state.partition.split(
        ("ink",), (Surface("ink", tuple(sorted(groups[0] + groups[1]))),), ledger
    )
    ordinary = replace(
        first,
        partition=owned,
        component=ComponentEdit.bind(
            state.document, state.partition, "frame", Work.start(10)
        ),
    )
    ops = Operators(e, graph, options, filled_bands=True)
    branch = ops.branch(state.partition, Work.start(10))
    proposed = next(branch.families.band_planner(state, ordinary, Work.start(20)))
    assert proposed.partition.atoms.extends(state.partition.atoms)
    assert not proposed.partition.atoms.extends(ordinary.partition.atoms)
    ops.validate_partition(proposed.partition, Work.start(10))
    proposed.component.validate(
        state.document,
        proposed.document,
        state.partition,
        proposed.partition,
        proposed.ids,
        proposed.bounds,
        Work.start(10),
    )


def test_joint_cursor_offers_material_first_then_complete_band_alternative(monkeypatch):
    from vectrify.refine.cel_plan import joint_cells
    from vectrify.refine.cel_plan.joint_cells import JointCells

    e, graph, options, _, state, ordinary = fixture()
    ops = Operators(e, graph, options, filled_bands=True)
    closed = []

    class Cursor:
        def __init__(self, *_args, **kwargs):
            self.mode = kwargs["ink_fit"]

        def __call__(self, _state, _work):
            try:
                yield ordinary
            finally:
                closed.append(self.mode)

    monkeypatch.setattr(joint_cells, "CoreCells", Cursor)
    scheduled = JointCells(ops.families, options, bands=ops.families.band_planner)
    edits = list(scheduled(state, Work.start(20)))
    assert len(edits) == 7
    assert edits[0].details["joint_cell_search"]["ink_fit"] == "source-intervals"
    assert edits[1].details["joint_cell_search"]["ink_fit"] == "source-widths"
    assert edits[1].document is ordinary.document
    assert edits[2].details["planned_band_stroke"]["stroke_nodes"] == 2
    assert edits[2].details["joint_cell_search"]["ink_fit"] == "source-widths"
    assert sorted(closed) == ["source-intervals", "source-widths"]


@pytest.mark.parametrize("reason", ["limit", "cancel", "close"])
def test_joint_band_budget_and_exit_close_without_computing_next_edit(
    reason, monkeypatch
):
    from vectrify.refine.cel_plan import joint_cells
    from vectrify.refine.cel_plan.joint_cells import JointCells

    e, graph, options, _, state, ordinary = fixture()
    ops = Operators(e, graph, options, filled_bands=True)
    offered, closed = [], []

    class Cursor:
        def __init__(self, *_args, **kwargs):
            self.mode = kwargs["ink_fit"]

        def __call__(self, _state, _work):
            try:
                yield ordinary
            finally:
                closed.append(self.mode)

    def alternatives(_state, planned, _work):
        try:
            for i in range(5):
                offered.append(i)
                yield planned
        finally:
            closed.append("bands")

    monkeypatch.setattr(joint_cells, "CoreCells", Cursor)
    if reason == "limit":
        monkeypatch.setattr(joint_cells, "MAX_PROPOSALS", 3)
    scheduler = JointCells(ops.families, options, bands=alternatives)
    work = Work.start(20)
    iterator = iter(scheduler(state, work))
    assert [next(iterator).operator for _ in range(3)] == ["joint-core-cells"] * 3
    if reason == "close":
        iterator.close()
    else:
        if reason == "cancel":
            work.stop.set()
        assert list(iterator) == []
    assert offered == [0]
    assert sorted(closed) == ["bands", "source-intervals", "source-widths"]


def mixed_fixture(alpha=0.75, frame=""):
    """A narrow outline and a broad shadow share one original owner/atom."""
    svg = (
        '<svg width="160" height="100"><g id="scene" '
        f'opacity="{alpha}">'
        '<path id="bg" d="M0 0H160V100H0Z" fill="#c4b79c"/>'
        '<g id="frame" '
        f'transform="{frame or "matrix(1 0 0 1 0 0)"}">'
        '<path id="ink" d="M10 30.5H150V33.5H10Z M145 20h2v2h-2z" '
        'fill="#202020"/>'
        '<path id="mixed" d="M10 50.5H150V53.5H10Z '
        'M25 70H100L135 72V90H25Z M30 75V85H130V75Z" fill="#2f211c"/>'
        "</g></g></svg>"
    )
    document = import_svg(svg)
    rgba = render(svg, (160, 100))
    options = Options()
    e = collect(
        Image.fromarray(np.rint(rgba * 255).astype(np.uint8)),
        None,
        options,
        Work.start(10),
    )
    labels = np.zeros(e.labels.shape, np.int32)
    yy, xx = np.indices(labels.shape)
    dark = (e.target.max(axis=-1) < 110) & ~e.empty
    labels[dark & (yy < 40)] = 1
    labels[dark & (yy >= 40)] = 2
    e = replace(e, labels=labels)
    graph = build(e)
    old = Partition(
        (
            Surface("bg", (0,), covered=(1, 2)),
            Surface("ink", (1,)),
            Surface("mixed", (2,)),
        )
    )
    policy = Policy.from_evidence(e, graph)
    exported = export_svg(document)
    state = State(
        document,
        exported,
        LocalPolicy(policy).start(exported, policy.evaluate(exported)),
        "ancestor",
        {},
        partition=old,
    )
    atoms, groups = Atoms.original(graph).partition(
        graph,
        (1, 2),
        (xx >= 80).astype(np.int32),
        2,
        Work.start(10),
        compact=True,
        retain_parent=True,
    )
    children = tuple(sorted(groups[0] + groups[1]))
    split = atoms.labels(graph, Work.start(10))
    ink = tuple(i for i in children if np.any((split == i) & (labels == 1)))
    mixed = tuple(i for i in children if np.any((split == i) & (labels == 2)))
    part = old.split(
        ("ink", "mixed"), (Surface("ink", ink), Surface("mixed", mixed)), atoms
    )
    proposal = Proposal(
        "joint-core-cells",
        ("ink", "mixed"),
        (),
        state.key,
        document,
        bounds(document, document, ("ink", "mixed")),
        partition=part,
        component=ComponentEdit.bind(document, old, "frame", Work.start(10)),
    )
    return e, graph, options, policy, state, proposal


@pytest.mark.parametrize("frame", ["", "matrix(0.9 0.03 0.08 0.85 0.25 1.5)"])
def test_separable_outline_in_mixed_shadow_owner_becomes_a_second_stroke(
    frame, monkeypatch
):
    e, graph, options, policy, state, proposal = mixed_fixture(frame=frame)
    monkeypatch.setattr(atom_module, "MAX_CUTS", 2)
    assert len(proposal.partition.atoms.cuts) == 2
    # The largest contour belongs to the broad shadow with a real hole, so
    # the preceding largest-contour route cannot isolate the narrow outline.
    assert (
        separate(state.document.geometry_for("mixed"), "nonzero", Work.start(10))
        is None
    )
    operators = Operators(e, graph, options, filled_bands=True)
    variants = list(operators.families.band_planner(state, proposal, Work.start(30)))
    originals = [p for p in variants if "planned_band_strokes" not in p.details]
    extended = [p for p in variants if "planned_band_strokes" in p.details]
    assert len(originals) == len(extended) == 5
    for parent, child in zip(originals, extended, strict=True):
        assert child.document.geometry_for("ink") == parent.document.geometry_for("ink")
        assert child.document.element("ink") == parent.document.element("ink")
        assert child.document.element("bg") == state.document.element("bg")
        assert child.document.geometry_for("bg") == state.document.geometry_for("bg")
        residual = child.document.geometry_for("mixed-band-marks")
        assert (
            residual.path_data()
            == replace(
                state.document.geometry_for("mixed"),
                subpaths=state.document.geometry_for("mixed").subpaths[1:],
            ).path_data()
        )
        for oid in ("ink", "mixed"):
            assert child.document.element(oid).get("fill") == "none"
            assert child.document.element(oid).get("stroke") != "none"
            assert len(child.document.geometry_for(oid).subpaths[0].nodes) == 2
        assert len(child.details["planned_band_strokes"]) == 2
        assert len(child.partition.atoms.cuts) == 2
        operators.validate_partition(child.partition, Work.start(10))
        child.component.validate(
            state.document,
            child.document,
            state.partition,
            child.partition,
            child.ids,
            child.bounds,
            Work.start(10),
        )
        full = policy.evaluate(export_svg(child.document))
        assert full.valid, full.rejections
        local = LocalPolicy(policy).update(
            state.snapshot, export_svg(child.document), child.bounds, full.structure
        )
        actual = render(export_svg(child.document), e.source_size)
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        restored, _ = load_project(save_project(child.document))
        np.testing.assert_array_equal(
            render(export_svg(restored), e.source_size), actual
        )
        comparison = child.details["planned_band_stroke"]["source_line_comparison"]
        assert not comparison["rejections"]
        assert comparison["new_gap_completed"] == 0


def test_isolated_band_requires_its_own_source_ink_role():
    e, graph, options, _policy, state, proposal = mixed_fixture()
    drawn = e.drawn.copy()
    drawn[45:60] = False  # Explicit uncertain source role on the smaller contour.
    e = replace(e, drawn=drawn)
    planner = Operators(e, graph, options, filled_bands=True).families.band_planner
    variants = list(planner(state, proposal, Work.start(20)))
    assert len(variants) == 5
    assert all("planned_band_strokes" not in p.details for p in variants)


def test_second_band_keeps_the_existing_source_child_bound(monkeypatch):
    e, graph, options, _policy, state, proposal = mixed_fixture()
    monkeypatch.setattr(atom_module, "MAX_CHILDREN", 1)
    planner = Operators(e, graph, options, filled_bands=True).families.band_planner
    variants = list(planner(state, proposal, Work.start(20)))
    assert len(variants) == 5
    assert all("planned_band_strokes" not in p.details for p in variants)
    assert planner.diagnostics["bounded"] == 1
