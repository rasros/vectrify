"""Exterior hypotheses become actual editable strokes only after native proofs."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from tests.helpers import required
from tests.refine.test_cel_plan_families import prepared
from vectrify.document import export_svg, load_project, save_project
from vectrify.document.join import path_style
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.ink import measure_closed
from vectrify.refine.cel_plan.line_fidelity import SourceProfile
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.silhouette_ink import SilhouetteInk
from vectrify.refine.cel_plan.source_strokes import SourceStrokes
from vectrify.refine.tracing import _loops


def circle(alpha=1, *, opaque=False, gap=False):
    svg = '<svg width="96" height="96">'
    if opaque:
        svg += '<path d="M0 0H96V96H0Z" fill="white"/>'
    svg += (
        f'<g opacity="{alpha}"><circle cx="48" cy="48" r="30" '
        'fill="#c4b79c" stroke="#202020" stroke-width="3"/></g></svg>'
    )
    rgba = render(svg, (96, 96))
    if gap:
        rgba[14:23, 45:51, :3] = np.array((196, 183, 156)) / 255
    evidence = collect(
        Image.fromarray(np.rint(rgba * 255).astype(np.uint8)),
        None,
        Options(refine=False),
        Work.start(10),
    )
    ink = ~evidence.empty & (evidence.target.max(-1) < 65)
    return replace(evidence, drawn=ink, line=ink)


def perimeter(evidence):
    mask = ~evidence.empty & evidence.foreground
    if evidence.opacity is not None:
        mask &= evidence.opacity >= evidence.opacity.max() * 0.5
    loop = _loops(mask)[0]
    return np.array([*loop, loop[0]], float)


@pytest.mark.parametrize("alpha", [1, 0.5, 0.25])
def test_closed_measure_has_no_artificial_seam_or_transparent_rgb_evidence(alpha):
    evidence = circle(alpha)
    original = perimeter(evidence)
    proof = measure_closed(
        original, evidence.target, 3, visible=~evidence.empty, opacity=evidence.opacity
    )
    assert proof is not None
    np.testing.assert_array_equal(proof.points[0], proof.points[-1])
    assert not np.array_equal(proof.points[0], original[0])
    shifted = np.roll(original[:-1], 37, axis=0)
    shifted = np.vstack((shifted, shifted[0]))
    rotated = measure_closed(
        shifted, evidence.target, 3, visible=~evidence.empty, opacity=evidence.opacity
    )
    assert rotated is not None
    np.testing.assert_allclose(
        rotated.points[:-1], np.roll(proof.points[:-1], 37, axis=0)
    )
    assert rotated.width == pytest.approx(proof.width, abs=0.01)
    target = evidence.target.copy()
    target[evidence.empty] = (0, 255, 100)
    hidden = measure_closed(
        original, target, 3, visible=~evidence.empty, opacity=evidence.opacity
    )
    assert hidden is not None
    np.testing.assert_array_equal(hidden.points, proof.points)
    np.testing.assert_array_equal(hidden.paint, proof.paint)
    assert hidden.width == proof.width
    p = SourceProfile.from_ink(original, proof, evidence, cyclic=True)
    assert p.terminals == p.dense().terminals == (False, False)
    np.testing.assert_array_equal(p.direction[0], p.direction[-1])
    assert np.linalg.norm(p.direction, axis=1).min() > 0.99
    for name in ("points", "sides", "direction", "tolerance"):
        assert not getattr(p, name).flags.writeable


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf")])
def test_closed_measure_rejects_invalid_styles(width):
    evidence = circle()
    with pytest.raises(ValueError, match="closed perimeter"):
        measure_closed(perimeter(evidence), evidence.target, width)


def test_closed_measure_rejects_open_geometry_and_constant_dark_material():
    evidence = circle()
    points = perimeter(evidence)
    with pytest.raises(ValueError, match="closed perimeter"):
        measure_closed(points[:-1], evidence.target, 3)
    assert (
        measure_closed(
            points,
            np.full_like(evidence.target, 32),
            3,
            visible=~evidence.empty,
            opacity=evidence.opacity,
        )
        is None
    )


@pytest.mark.parametrize("alpha", [1, 0.5, 0.25])
def test_source_perimeter_is_independent_of_opaque_carrier_and_keeps_actual_gaps(alpha):
    evidence = circle(alpha)
    factory = SilhouetteInk(evidence, Options())
    found = factory(Work.start(10))
    assert found
    for model in found:
        assert model.geometry.subpaths[0].closed
        assert model.details["model"] == "source-silhouette-stroke"
        assert 2.5 < model.details["width"] < 3.2
        assert model.details["source_absence_samples"] == 0
        assert not model.selected.flags.writeable
    broken = SilhouetteInk(circle(alpha, gap=True), Options())
    assert broken(Work.start(10)) == ()
    assert broken.diagnostics["absence_exclusions"] > 0


def test_outline_replaces_selected_fill_with_native_editable_stroke(
    tmp_path, monkeypatch
):
    from vectrify.refine.cel_plan import export as exporter

    evidence = circle(opaque=True)
    with monkeypatch.context() as context:
        context.setattr(exporter, "strokes", lambda *_args, **_kwargs: ([], {}))
        frontier, state, options = prepared(evidence)
    ops = Operators(evidence, build(evidence), options)
    factory = SourceStrokes(
        evidence,
        ops.graph,
        options,
        resolver=ops.branch,
        boundary_contacts=True,
        outlines=True,
        perimeter_only=True,
    )
    edits = [
        p
        for p in factory.proposals(state, Work.start(20))
        if required(p.details)["source_strokes"]["model"] == "source-silhouette-stroke"
    ]
    assert edits
    original = render(state.svg, evidence.source_size)
    for edit in edits:
        assert edit.partition is not None
        ops.validate_partition(edit.partition, Work.start(10))
        svg = export_svg(edit.document)
        full = frontier.policy.evaluate(svg)
        assert full.valid, full.rejections
        strokes = [
            p
            for p in edit.document.elements()
            if path_style(edit.document, p)["stroke"] != "none" and p.tag == "path"
        ]
        assert len(strokes) == 1
        stroke = strokes[0]
        assert stroke.get("fill") == "none"
        assert edit.document.geometry_for(stroke.id).subpaths[0].closed
        assert full.structure["stroke_contours"] == 1
        actual = render(svg, evidence.source_size)
        np.testing.assert_array_equal(actual[..., 3], original[..., 3])
        np.testing.assert_array_equal(actual[:8], original[:8])
        np.testing.assert_array_equal(actual[40:56, 40:56], original[40:56, 40:56])
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        path = tmp_path / "outline.vectrify"
        path.write_text(save_project(edit.document))
        loaded, _ = load_project(path.read_text())
        np.testing.assert_array_equal(
            render(export_svg(loaded), evidence.source_size), actual
        )
        # Discovery can be reused; native ownership/restoration remains per-state.
    assert factory._outline_models is not None


def test_perimeter_hypothesis_does_not_duplicate_a_retained_stroke():
    evidence = circle(opaque=True)
    _frontier, state, options = prepared(evidence)
    factory = SourceStrokes(evidence, build(evidence), options, outlines=True)
    edits = list(factory.proposals(state, Work.start(10)))
    assert not any(
        required(p.details)["source_strokes"]["model"] == "source-silhouette-stroke"
        for p in edits
    )
    assert factory.diagnostics["outline_retained_exclusions"] > 0


def test_cancellation_in_raw_discovery_or_after_footprint_never_publishes_prefix(
    monkeypatch,
):
    from vectrify.refine.cel_plan import silhouette_ink

    evidence = circle()
    work = Work.start(10)

    def cancelled(*_args, **_kwargs):
        raise StageInterruptedError("test")

    with monkeypatch.context() as context:
        context.setattr(silhouette_ink, "source_models", cancelled)
        factory = SilhouetteInk(evidence, Options())
        assert factory(work) == ()
        assert factory.diagnostics["cancelled"] == 1
    footprint = silhouette_ink.footprint

    def stopped(*args, **kwargs):
        value = footprint(*args, **kwargs)
        work.stop.set()
        return value

    monkeypatch.setattr(silhouette_ink, "footprint", stopped)
    factory = SilhouetteInk(evidence, Options())
    assert factory(work) == ()
    assert factory.diagnostics["models"] == 0


def test_perimeter_bank_bounds_exclude_work_without_partial_models(monkeypatch):
    from vectrify.refine.cel_plan import silhouette_ink

    evidence = circle()
    monkeypatch.setattr(silhouette_ink, "MAX_POINTS", 8)
    factory = SilhouetteInk(evidence, Options())
    assert factory(Work.start(10)) == ()
    assert factory.diagnostics["bounds_exclusions"] > 0


def test_artificial_cycle_chunk_ends_cannot_authorize_facing_gap_probes():
    from tests.refine.test_cel_plan_line_fidelity import source
    from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard

    truth = source(gap=True)
    a = SourceProfile.at(np.column_stack((np.arange(12.5, 46), np.full(34, 32.5))), 3)
    b = SourceProfile.at(np.column_stack((np.arange(50.5, 84), np.full(34, 32.5))), 3)
    physical = SourceLineGuard(truth, [a, b])
    physical.establish(truth)
    assert physical.metrics(truth)["endpoint_gap_profiles"] == 1
    packaged = SourceLineGuard(truth, [replace(a, terminals=(False, False)), b])
    packaged.establish(truth)
    assert packaged.metrics(truth)["endpoint_gap_profiles"] == 0
    assert (
        packaged.metrics(truth)["qualified_samples"]
        == physical.metrics(truth)["qualified_samples"]
    )


def test_a_complete_outline_outside_the_carrier_is_excluded(monkeypatch):
    from vectrify.document.join import curve_path
    from vectrify.document.svg import parse_path
    from vectrify.refine.cel_plan import export as exporter

    evidence = circle(opaque=True)
    with monkeypatch.context() as context:
        context.setattr(exporter, "strokes", lambda *_args, **_kwargs: ([], {}))
        _frontier, state, options = prepared(evidence)
    graph = build(evidence)
    factory = SourceStrokes(evidence, graph, options, outlines=True)
    monkeypatch.setattr(
        factory,
        "carriers",
        lambda *_args: iter(
            [
                (
                    tuple(r.id for r in graph.regions),
                    curve_path(parse_path("M0 0H2V2H0Z")),
                )
            ]
        ),
    )
    assert list(factory.proposals(state, Work.start(10))) == []
    assert factory.diagnostics["outline_carrier_exclusions"] > 0


@pytest.mark.parametrize("quality", ["fast", "balanced", "high"])
def test_new_native_stroke_cursor_gets_a_high_quality_reserved_turn(
    quality, monkeypatch
):
    from tests.refine.test_cel_plan_search import proposal, setup
    from vectrify.document import import_svg
    from vectrify.refine.cel_plan.search import State

    frontier, evidence, options = setup()
    entry = frontier.entries[0]
    state = State(
        import_svg(entry.svg),
        entry.svg,
        LocalPolicy(frontier.policy).start(entry.svg, entry.evaluation),
        entry.key,
        {},
    )
    ops = Operators(evidence, build(evidence), replace(options, quality=quality))
    names = (
        "families",
        "overlays",
        "paint",
        "geometry",
        "ink",
        "replacements",
        "ridges",
        "opacity_fields",
        "strokes",
    )
    for name in names:

        def cursor(current, _work, name=name):
            yield proposal(current, "left", "#b05030", operator=name)

        monkeypatch.setattr(ops, name, cursor)
    assert [p.operator for p in ops(state, Work.start(10))] == list(
        names if quality == "high" else names[:7]
    )


def test_perimeter_only_cursor_cannot_fall_back_to_legacy_source_strokes(monkeypatch):
    from vectrify.refine.cel_plan import source_strokes

    evidence = circle(opaque=True)
    _frontier, state, options = prepared(evidence)
    factory = SourceStrokes(
        evidence, build(evidence), options, outlines=True, perimeter_only=True
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Legacy stroke discovery is outside this cursor")

    monkeypatch.setattr(source_strokes, "models", forbidden)
    assert list(factory.proposals(state, Work.start(10))) == []
    assert factory.diagnostics["outline_retained_exclusions"] > 0
    with pytest.raises(ValueError, match="outline discovery"):
        SourceStrokes(evidence, build(evidence), options, perimeter_only=True)
