"""Native source gaps, never packaging seams, authorize editable cycle runs."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_silhouette_ink import circle, perimeter
from vectrify.document import Geometry
from vectrify.refine.cel_plan.ink import measure_closed
from vectrify.refine.cel_plan.line_fidelity import SourceProfile
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.silhouette_ink import SilhouetteInk
from vectrify.refine.cel_plan.source_absence import SourceAbsence
from vectrify.refine.cel_plan.source_cycles import SourceCycle
from vectrify.refine.crossings import crossings


def measured(alpha=1, *, gap=True):
    evidence = circle(alpha, gap=gap)
    original = perimeter(evidence)
    ink = measure_closed(
        original,
        evidence.target,
        3,
        visible=~evidence.empty,
        opacity=evidence.opacity,
    )
    assert ink is not None
    profile = SourceProfile.from_ink(
        original,
        ink,
        evidence,
        component=("fixture", 1),
        cyclic=True,
    )
    return evidence, ink, profile


def absence(evidence, cycle):
    return SourceAbsence(evidence, cycle.profiles, Work.start(10), intervals=True)


@pytest.mark.parametrize("alpha", [1, 0.5, 0.25])
def test_broken_outline_recovers_one_complete_open_run_and_preserves_real_gap(alpha):
    evidence, ink, profile = measured(alpha)
    cycle = SourceCycle(profile)
    guard = absence(evidence, cycle)
    observation = cycle.observations(guard, Work.start(10))
    assert observation.gaps.any()
    runs = cycle.recover(guard, ink.width, "round", Options(), Work.start(10))
    assert len(runs) == 1
    assert not runs[0].closed
    geometry = Geometry("open-ink", runs)
    assert not crossings(geometry)
    assert guard.permits(geometry, ink.width, "round", Work.start(10))
    assert len(runs[0].nodes) < len(profile.points) / 4
    assert cycle.diagnostics["unobserved_samples"] == 0
    assert cycle.diagnostics["gaps"] == int(observation.gaps.sum())
    for end in (runs[0].nodes[0], runs[0].nodes[-1]):
        # Every new cap retains an actual qualified source center.
        assert np.any(
            np.all(observation.points[observation.qualified] == end.endpoint, axis=1)
        )
    assert (
        np.linalg.norm(
            np.subtract(runs[0].nodes[0].endpoint, runs[0].nodes[-1].endpoint)
        )
        > ink.width * 2
    )


def test_serialization_rotation_never_changes_caps_or_introduces_a_chunk_terminal():
    evidence, ink, profile = measured()
    original = SourceCycle(profile)

    def roll(values):
        shifted = np.roll(values[:-1], 37, axis=0)
        return np.concatenate((shifted, shifted[:1]))

    rotated = SourceCycle(
        replace(
            profile,
            **{
                name: roll(getattr(profile, name))
                for name in ("points", "sides", "direction", "tolerance")
            },
        )
    )
    first = original.recover(
        absence(evidence, original), ink.width, "round", Options(), Work.start(10)
    )
    second = rotated.recover(
        absence(evidence, rotated), ink.width, "round", Options(), Work.start(10)
    )
    assert len(first) == len(second) == 1
    np.testing.assert_allclose(
        [v for n in first[0].nodes for v in n.values],
        [v for n in second[0].nodes for v in n.values],
        atol=1e-10,
    )
    for cycle in (original, rotated):
        assert all(p.terminals == (False, False) for p in cycle.profiles)
        assert all(
            not getattr(p, n).flags.writeable
            for p in cycle.profiles
            for n in ("points", "sides", "direction", "tolerance")
        )


def test_unbroken_cycle_cannot_be_cut_at_a_seam_or_a_neighbouring_gap():
    evidence, ink, profile = measured(gap=False)
    cycle = SourceCycle(profile)
    guard = absence(evidence, cycle)
    assert not cycle.observations(guard, Work.start(10)).gaps.any()
    assert cycle.recover(guard, ink.width, "round", Options(), Work.start(10)) == ()


def test_unknown_observers_cannot_authorize_cycle_reconstruction():
    evidence, ink, profile = measured()
    first, second = SourceCycle(profile), SourceCycle(profile)
    with pytest.raises(ValueError, match="complete source cycle"):
        first.recover(
            absence(evidence, second), ink.width, "round", Options(), Work.start(10)
        )


def test_native_observations_are_required():
    evidence, ink, profile = measured()
    cycle = SourceCycle(profile)
    guard = SourceAbsence(evidence, cycle.profiles, Work.start(10))
    with pytest.raises(ValueError, match="retained native observations"):
        cycle.recover(guard, ink.width, "round", Options(), Work.start(10))


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf")])
def test_invalid_styles_cannot_publish_an_interval(width):
    evidence, _ink, profile = measured()
    cycle = SourceCycle(profile)
    with pytest.raises(ValueError, match="finite stroke style"):
        cycle.recover(
            absence(evidence, cycle), width, "round", Options(), Work.start(10)
        )


@pytest.mark.parametrize("change", ["open", "terminals", "fields"])
def test_invalid_or_unbounded_cycle_profiles_are_rejected(change):
    _evidence, _ink, profile = measured()
    if change == "open":
        profile = replace(profile, points=profile.points[:-1])
    elif change == "terminals":
        profile = replace(profile, terminals=(True, True))
    else:
        profile = replace(profile, tolerance=np.zeros(1))
    with pytest.raises(ValueError, match=r"profile|nonterminal"):
        SourceCycle(profile)


def test_fit_capacity_never_publishes_a_prefix(monkeypatch):
    from vectrify.refine.cel_plan import source_cycles

    evidence, ink, profile = measured()
    cycle = SourceCycle(profile)
    monkeypatch.setattr(source_cycles, "MAX_FITS", 0)
    with pytest.raises(ValueError, match="fit bound"):
        cycle.recover(
            absence(evidence, cycle), ink.width, "round", Options(), Work.start(10)
        )
    assert cycle.diagnostics["recovered_runs"] == 0


def test_interrupted_recovery_cannot_publish_a_prefix():
    evidence, ink, profile = measured()
    cycle = SourceCycle(profile)
    guard = absence(evidence, cycle)
    with pytest.raises(StageInterruptedError):
        cycle.recover(guard, ink.width, "round", Options(), Work.start(0))


@pytest.mark.parametrize("alpha", [1, 0.5, 0.25])
def test_opt_in_bank_recovers_actual_source_gap_without_closing_it(alpha):
    evidence = circle(alpha, gap=True)
    factory = SilhouetteInk(evidence, Options(), intervals=True)
    found = factory(Work.start(10))
    assert found
    assert factory.diagnostics["recovered_runs"] > 0
    assert all(not m.geometry.subpaths[0].closed for m in found)
    assert all(m.details["source_cycle"]["unobserved_samples"] == 0 for m in found)
    assert all(m.details["source_cycle"]["gaps"] > 0 for m in found)
    assert all(not m.selected.flags.writeable for m in found)


def test_gap_replacement_exports_an_open_editable_stroke_and_roundtrips(
    tmp_path, monkeypatch
):
    from tests.refine.test_cel_plan_families import prepared
    from vectrify.document import export_svg, load_project, save_project
    from vectrify.document.join import path_style
    from vectrify.refine.cel_plan import export as exporter
    from vectrify.refine.cel_plan.graph import build
    from vectrify.refine.cel_plan.local import LocalPolicy
    from vectrify.refine.cel_plan.proposals import Operators
    from vectrify.refine.cel_plan.score import render
    from vectrify.refine.cel_plan.source_strokes import SourceStrokes

    evidence = circle(opaque=True, gap=True)
    with monkeypatch.context() as context:
        context.setattr(exporter, "strokes", lambda *_args, **_kwargs: ([], {}))
        frontier, state, options = prepared(evidence)
    ops = Operators(evidence, build(evidence), options)
    factory = SourceStrokes(
        evidence,
        ops.graph,
        options,
        resolver=ops.branch,
        outlines=True,
        perimeter_only=True,
        outline_intervals=True,
        boundary_contacts=True,
    )
    edits = list(factory.proposals(state, Work.start(20)))
    assert edits
    before = render(state.svg, evidence.source_size)
    for edit in edits:
        ops.validate_partition(edit.partition, Work.start(10))
        svg = export_svg(edit.document)
        full = frontier.policy.evaluate(svg)
        assert full.valid, full.rejections
        strokes = [
            p
            for p in edit.document.elements()
            if p.tag == "path" and path_style(edit.document, p)["stroke"] != "none"
        ]
        assert len(strokes) == 1
        assert strokes[0].get("fill") == "none"
        assert not edit.document.geometry_for(strokes[0].id).subpaths[0].closed
        assert edit.details["source_strokes"]["cuts"] > 0
        actual = render(svg, evidence.source_size)
        np.testing.assert_array_equal(actual[..., 3], before[..., 3])
        np.testing.assert_array_equal(actual[:8], before[:8])
        np.testing.assert_array_equal(actual[40:56, 40:56], before[40:56, 40:56])
        np.testing.assert_array_equal(actual[14:22, 47:49], before[14:22, 47:49])
        local = LocalPolicy(frontier.policy).update(
            state.snapshot, svg, edit.bounds, full.structure
        )
        assert local.canvas.matches(actual)
        assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
        path = tmp_path / "open-outline.vectrify"
        path.write_text(save_project(edit.document))
        loaded, _ = load_project(path.read_text())
        np.testing.assert_array_equal(
            render(export_svg(loaded), evidence.source_size), actual
        )


@pytest.mark.parametrize("limit", ["MAX_RUNS", "MAX_NODES"])
def test_interval_and_node_bounds_never_publish_a_prefix(monkeypatch, limit):
    from vectrify.refine.cel_plan import source_cycles

    evidence, ink, profile = measured()
    cycle = SourceCycle(profile)
    monkeypatch.setattr(source_cycles, limit, 0)
    with pytest.raises(ValueError, match=r"interval bound|node bound"):
        cycle.recover(
            absence(evidence, cycle), ink.width, "round", Options(), Work.start(10)
        )
    assert cycle.diagnostics["recovered_runs"] == 0


def test_body_rejection_cannot_be_waived_by_recovery(monkeypatch):
    evidence, ink, profile = measured()
    cycle = SourceCycle(profile)
    guard = absence(evidence, cycle)
    monkeypatch.setattr(guard, "permits", lambda *_args: False)
    assert cycle.recover(guard, ink.width, "round", Options(), Work.start(10)) == ()
    assert cycle.diagnostics["absence_exclusions"] > 0


def test_large_cycle_observations_cover_chunk_seams_and_keep_both_gaps():
    from PIL import Image

    from vectrify.refine.cel_plan.evidence import collect
    from vectrify.refine.cel_plan.score import render

    rgba = render(
        '<svg width="480" height="480"><circle cx="240" cy="240" r="200" '
        'fill="#c4b79c" stroke="#202020" stroke-width="3"/></svg>',
        (480, 480),
    )
    rgba[36:44, 237:243, :3] = np.array((196, 183, 156)) / 255
    rgba[436:444, 237:243, :3] = np.array((196, 183, 156)) / 255
    evidence = collect(
        Image.fromarray(np.rint(rgba * 255).astype(np.uint8)),
        None,
        Options(),
        Work.start(10),
    )
    original = perimeter(evidence)
    ink = measure_closed(
        original, evidence.target, 3, visible=~evidence.empty, opacity=evidence.opacity
    )
    assert ink is not None
    cycle = SourceCycle(
        SourceProfile.from_ink(
            original, ink, evidence, component=("large", 1), cyclic=True
        )
    )
    assert len(cycle.profiles) > 2
    guard = absence(evidence, cycle)
    observation = cycle.observations(guard, Work.start(10))
    assert observation.gaps.any()
    assert cycle.diagnostics["unobserved_samples"] == 0
    runs = cycle.recover(guard, ink.width, "round", Options(), Work.start(10))
    assert len(runs) == 2
    assert all(not run.closed for run in runs)
    assert guard.permits(Geometry("two-runs", runs), ink.width, "round", Work.start(10))


def test_recovery_is_an_explicit_experiment_until_artwork_composition_is_usable():
    from vectrify.refine.cel_plan.graph import build
    from vectrify.refine.cel_plan.proposals import Operators
    from vectrify.refine.cel_plan.source_strokes import SourceStrokes

    evidence = circle(gap=True)
    graph = build(evidence)
    ops = Operators(evidence, graph, Options(quality="high"))
    assert ops.strokes.silhouettes is not None
    assert not ops.strokes.silhouettes.intervals
    explicit = SourceStrokes(
        evidence, graph, Options(), outlines=True, outline_intervals=True
    )
    assert explicit.silhouettes is not None
    assert explicit.silhouettes.intervals
    with pytest.raises(ValueError, match="source outline discovery"):
        SourceStrokes(evidence, graph, Options(), outline_intervals=True)
