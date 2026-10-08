"""Actual filled-band inversion, painted contracts and exact material retention."""

from dataclasses import replace

import numpy as np
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
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import filled_bands
from vectrify.refine.cel_plan.atoms import Atoms
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.filled_bands import invert
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition, Surface
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.search import State


def scene(alpha=0.5, frame=""):
    transform = f' transform="{frame}"' if frame else ""
    svg = (
        '<svg width="96" height="64">'
        f'<g id="scene" opacity="{alpha}">'
        '<path id="bg" d="M0 0H96V64H0Z" fill="#c4b79c"/>'
        f'<g id="frame"{transform}>'
        '<path id="shade" d="M8 42H88V58H8Z" fill="#a18c6d"/>'
        '<path id="ink" d="M10 30.5H86V33.5H10Z" fill="#202020"/>'
        "</g></g></svg>"
    )
    rgba = render(svg, (96, 64))
    options = Options()
    evidence = collect(
        Image.fromarray((rgba * 255).round().astype(np.uint8)),
        None,
        options,
        Work.start(10),
    )
    labels = np.zeros(evidence.labels.shape, np.int32)
    labels[evidence.target.max(axis=-1) < 65] = 1
    labels[(evidence.target[..., 0] < 180) & (evidence.target[..., 0] > 100)] = 2
    evidence = replace(evidence, labels=labels)
    graph = build(evidence)
    document = import_svg(svg)
    partition = Partition(
        (
            Surface("bg", (0,), covered=(1, 2)),
            Surface("ink", (1,), covered=(0,)),
            Surface("shade", (2,)),
        )
    )
    policy = Policy.from_evidence(evidence, graph)
    exported = export_svg(document)
    full = policy.evaluate(exported)
    state = State(
        document,
        exported,
        LocalPolicy(policy).start(exported, full),
        "parent",
        {},
        partition=partition,
    )
    return evidence, graph, options, policy, state


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ("M10 30.5H86V33.5H10Z", ((10, 32), (86, 32))),
        ("M10 20H14V65H10Z", ((12, 20), (12, 65))),
    ],
)
def test_uniform_band_recovers_a_complete_two_node_stroke(data, expected):
    model = invert(parse_path(data), "nonzero", Work.start(10))
    assert model is not None
    sub = model.geometry.subpaths[0]
    assert len(sub.nodes) == 2
    assert not sub.closed
    np.testing.assert_allclose([n.endpoint for n in sub.nodes], expected, atol=1e-6)


def test_closed_band_keeps_the_hole_and_never_creates_a_serialization_cap():
    def circle(radius):
        k = radius * 0.55228475
        return (
            f"M{48 - radius} 32 C{48 - radius} {32 - k} "
            f"{48 - k} {32 - radius} 48 {32 - radius} "
            f"C{48 + k} {32 - radius} {48 + radius} {32 - k} {48 + radius} 32 "
            f"C{48 + radius} {32 + k} {48 + k} {32 + radius} 48 {32 + radius} "
            f"C{48 - k} {32 + radius} {48 - radius} {32 + k} {48 - radius} 32Z"
        )

    geometry = parse_path(circle(20) + circle(17))
    assert invert(geometry, "nonzero", Work.start(10)) is None
    model = invert(geometry, "evenodd", Work.start(10))
    assert model is not None
    assert model.geometry.subpaths[0].closed
    actual = render(
        '<svg width="96" height="64">'
        f'<path d="{model.geometry.path_data()}" fill="none" stroke="black" '
        f'stroke-width="{model.width}"/></svg>',
        (96, 64),
    )
    assert actual[32, 48, 3] == 0
    assert actual[32, 28:33, 3].max() > 0


@pytest.mark.parametrize(
    "data",
    [
        "M10 20H86V65H10Z",  # broad material
        "M10 30H60V33H10Z M75 30H85V33H75Z",  # real physical gap
        "M10 30H60V33H10Z M70 30h1v1h-1z",  # independent tiny mark
        "M10 30H60V33H10Z M30 20H33V45H30Z",  # genuine branch
        "M10 30H60V33H10Z M60 20H80V45H60Z",  # broad attached material
        "M10 30H60",  # open fill is not an observed complete band
    ],
)
def test_unsupported_parts_are_never_pruned_to_publish_a_longest_chain(data):
    assert invert(parse_path(data), "nonzero", Work.start(10)) is None


@pytest.mark.parametrize("alpha", [0.25, 0.5])
@pytest.mark.parametrize("frame", ["", "matrix(0.9 0.03 0.08 0.85 0.25 1.5)"])
def test_complete_owner_becomes_stroke_with_exact_shadow_and_native_scoring(
    alpha, frame
):
    evidence, graph, options, policy, state = scene(alpha, frame)
    operators = Operators(evidence, graph, options, filled_bands=True)
    edits = list(operators.bands(state, Work.start(20)))
    assert edits, operators.bands.diagnostics
    before = render(state.svg, evidence.source_size)
    for edit in edits:
        assert edit.ids == ("ink",)
        assert edit.partition is state.partition
        operators.validate_partition(edit.partition, Work.start(10))
        edit.component.validate(
            state.document,
            edit.document,
            state.partition,
            edit.partition,
            edit.ids,
            edit.bounds,
            Work.start(10),
        )
        stroke = edit.document.element("ink")
        assert stroke.get("fill") == "none"
        assert stroke.get("stroke") == "#202020"
        assert len(edit.document.geometry_for("ink").subpaths[0].nodes) == 2
        for oid in ("bg", "shade"):
            assert edit.document.element(oid) == state.document.element(oid)
        for oid in ("scene", "frame"):
            assert (
                edit.document.element(oid).attributes
                == state.document.element(oid).attributes
            )
        svg = export_svg(edit.document)
        actual = render(svg, evidence.source_size)
        np.testing.assert_array_equal(actual[40:64], before[40:64])
        full = policy.evaluate(svg)
        assert full.valid, full.rejections
        assert full.structure["stroke_contours"] == 1
        comparison = edit.details["filled_band_stroke"]["source_line_comparison"]
        assert comparison["new_gap_completed"] == 0
        assert not comparison["rejections"]
        local = LocalPolicy(policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        restored, _ = load_project(save_project(edit.document))
        np.testing.assert_array_equal(
            render(export_svg(restored), evidence.source_size), actual
        )


@pytest.mark.parametrize("protected", ["clip", "lock", "pin"])
def test_protected_ink_is_retained(protected):
    evidence, graph, options, _policy, state = scene()
    editor = Editor(state.document, selection=Selection(whole_document=True))
    if protected == "clip":
        svg = state.svg.replace('id="scene"', 'id="scene" clip-path="url(#cut)"')
        svg = svg.replace(
            "</svg>",
            '<defs><clipPath id="cut"><path d="M0 0H96V64H0Z"/>'
            "</clipPath></defs></svg>",
        )
        document = import_svg(svg)
    else:
        if protected == "lock":
            editor.set_locks("frame", frozenset({"transform"}))
        else:
            editor.pin_node(
                "ink", state.document.geometry_for("ink").subpaths[0].nodes[0].id
            )
        document = editor.snapshot.document
    state = replace(state, document=document)
    operators = Operators(evidence, graph, options, filled_bands=True)
    assert not list(operators.bands(state, Work.start(10)))


def test_interruption_and_allocation_limits_publish_no_partial_band(monkeypatch):
    work = Work.start(10)
    work.stop.set()
    with pytest.raises(StageInterruptedError):
        invert(parse_path("M10 30H86V33H10Z"), "nonzero", work)
    monkeypatch.setattr(filled_bands, "MAX_CROP_PIXELS", 1)
    monkeypatch.setattr(
        filled_bands, "render", lambda *_: pytest.fail("allocation happened")
    )
    assert invert(parse_path("M10 30H86V33H10Z"), "nonzero", Work.start(10)) is None


def test_operator_is_explicit_and_original_source_observations_are_reused():
    evidence, graph, options, _policy, state = scene()
    assert Operators(evidence, graph, options).bands is None
    operators = Operators(evidence, graph, options, filled_bands=True)
    list(operators.bands(state, Work.start(20)))
    first = operators.band_guard(Work.start(10))
    assert operators.band_guard(Work.start(10)) is first


def test_an_owned_branch_uses_its_actual_namespace_and_shared_original_observer():
    evidence, graph, options, _policy, state = scene()
    partition = Partition(state.partition.surfaces, Atoms.original(graph))
    state = replace(state, partition=partition)
    operators = Operators(evidence, graph, options, filled_bands=True)
    assert not list(operators.bands(state, Work.start(10)))
    assert operators.bands.diagnostics["bounded"] == 1
    branch = operators.branch(partition, Work.start(10))
    assert list(branch.bands(state, Work.start(10)))
    assert branch.band_guard(Work.start(10)) is operators.band_guard(Work.start(10))


def test_source_observation_limits_exclude_conversion_without_partial_results():
    evidence, graph, options, _policy, state = scene()

    def unavailable(_work):
        raise ValueError("Source observation limit")

    factory = filled_bands.FilledBands(evidence, graph, options, guard=unavailable)
    assert not list(factory(state, Work.start(10)))
    assert factory.diagnostics["bounded"] == 1
    assert factory.diagnostics["proposals"] == 0


def test_transverse_chunking_keeps_the_complete_model_exact(monkeypatch):
    geometry = parse_path("M10 30.5H86V33.5H10Z")
    original = invert(geometry, "nonzero", Work.start(10))
    monkeypatch.setattr(filled_bands, "PROFILE_CHUNK", 3)
    chunked = invert(geometry, "nonzero", Work.start(10))
    assert original is not None
    assert chunked is not None
    assert original.width == chunked.width
    assert original.geometry.path_data() == chunked.geometry.path_data()
