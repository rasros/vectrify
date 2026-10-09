"""Coherent initialization offers linear materials without erasing source atoms."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_ownership import stripes
from vectrify.document import export_svg, import_svg, load_project, save_project
from vectrify.refine.cel_plan import materials
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.materials import (
    coherent_labels,
    model,
    moments,
    retain_thin_paint,
)
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.pipeline import material_seed
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import render


def thin_material(values):
    original = stripes(alpha=128)
    rgba = np.zeros_like(original.rgba)
    rgba[8, 8 : 8 + len(values), :3] = (0.4, 0.2, 0.1)
    rgba[8, 8 : 8 + len(values), 3] = np.array(values) / 255
    shown = rgba[..., 3] > 0
    target = np.where(shown[..., None], rgba[..., :3] * 255, 255)
    labels = np.zeros_like(original.labels)
    labels[8, 8 : 8 + len(values)] = np.arange(1, len(values) + 1)
    return replace(
        original,
        rgba=rgba,
        opacity=rgba[..., 3].copy(),
        target=target,
        smooth=target.copy(),
        coarse=target.copy(),
        labels=labels,
        empty=~shown,
        foreground=shown,
    )


def test_actual_flat_paint_screen_restores_faint_atoms_with_native_coverage():
    evidence = thin_material([1, 1, 3, 4, 2, 3, 1, 1])
    labels = (~evidence.empty).astype(np.int32)
    options = Options()
    unsafe, _ = export(evidence, labels, options, Work.start(10), cost_normalizer=5000)
    assert (
        "translucent-component-lost"
        in Policy(evidence.rgba).evaluate(unsafe).rejections
    )
    repaired, details = retain_thin_paint(
        evidence, labels, options, Work.start(10), normalizer=5000
    )
    assert details["restored_families"] == 1
    assert len(np.unique(repaired[~evidence.empty])) == 8
    svg, exported = export(
        evidence, repaired, options, Work.start(10), cost_normalizer=5000
    )
    assert Policy(evidence.rgba).evaluate(svg).valid
    np.testing.assert_array_equal(
        render(svg, evidence.source_size)[..., 3], evidence.opacity
    )
    partition = Partition.from_metadata(exported["planning_surfaces"])
    assert partition is not None
    assert set(partition.owners) == set(range(1, 9))
    np.testing.assert_array_equal(labels, (~evidence.empty).astype(np.int32))


def test_renderer_half_byte_alpha_is_used_instead_of_rounding_an_analytic_fit():
    evidence = thin_material([1, 2, 1, 2])
    labels = (~evidence.empty).astype(np.int32)
    options = Options(gradients=False)
    svg, _ = export(evidence, labels, options, Work.start(10), cost_normalizer=5000)
    # Cairo's native half-byte alpha is not np.rint(alpha * 255).
    actual = render(svg, evidence.source_size)
    assert actual[8, 8][3] == pytest.approx(1 / 255)
    assert (
        "translucent-component-lost" in Policy(evidence.rgba).evaluate(svg).rejections
    )
    repaired, details = retain_thin_paint(
        evidence, labels, options, Work.start(10), normalizer=5000
    )
    assert details["restored_families"] == 1
    safe, _ = export(evidence, repaired, options, Work.start(10), cost_normalizer=5000)
    assert Policy(evidence.rgba).evaluate(safe).valid


@pytest.mark.parametrize("values", [[1] * 8, list(range(64, 177, 16))])
def test_thin_flat_and_actual_gradient_models_remain_compact_when_supported(values):
    evidence = thin_material(values)
    labels = (~evidence.empty).astype(np.int32)
    repaired, details = retain_thin_paint(
        evidence, labels, Options(), Work.start(10), normalizer=1e9
    )
    assert details["restored_families"] == 0
    np.testing.assert_array_equal(repaired, labels)
    svg, _ = export(evidence, repaired, Options(), Work.start(10), cost_normalizer=1e9)
    evaluation = Policy(evidence.rgba).evaluate(svg)
    assert evaluation.valid
    assert evaluation.structure["paths"] == 1
    assert evaluation.structure["gradients"] == int(max(values) > min(values))


def test_thin_paint_work_bound_and_stop_keep_complete_original_atoms(monkeypatch):
    evidence = thin_material([1] * 8)
    labels = (~evidence.empty).astype(np.int32)
    monkeypatch.setattr(materials, "MAX_THIN_PAINT_PIXELS", 2)
    repaired, details = retain_thin_paint(
        evidence, labels, Options(), Work.start(10), normalizer=5000
    )
    assert details["bounded_families"] == 1
    assert len(np.unique(repaired[~evidence.empty])) == 8
    stopped = Work.start(10)
    stopped.stop.set()
    with pytest.raises(StageInterruptedError):
        retain_thin_paint(evidence, labels, Options(), stopped, normalizer=5000)
    np.testing.assert_array_equal(labels, (~evidence.empty).astype(np.int32))


def test_material_seed_grows_coarse_checkpoint_without_losing_original_owners():
    evidence = stripes(alpha=128, gradient=True, hole=True)
    frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    starting = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4])[evidence.labels]
    before = starting.copy()
    details = material_seed(
        frontier, evidence, graph, options, Work.start(10), 10, starting_labels=starting
    )
    assert details["initial_partition"] == "validated-detailed"
    assert details["source_regions"] == 5
    assert details["merges"] == 3
    assert details["status"] == "retained"
    entry = next(e for e in frontier.entries if e.label == "Coherent RGBA materials")
    partition = Partition.from_metadata(entry.details["planning_surfaces"])
    assert partition is not None
    assert state.partition is not None
    assert set(partition.owners) == set(state.partition.owners)
    assert entry.evaluation.valid
    assert entry.evaluation.cost < state.snapshot.evaluation.cost
    np.testing.assert_array_equal(starting, before)
    np.testing.assert_array_equal(graph.labels, evidence.labels)


@pytest.mark.parametrize("chunk_pixels", [47, 137, 65_536])
def test_streamed_source_moments_match_dense_weighted_outer_products(
    chunk_pixels, monkeypatch
):
    evidence = stripes(alpha=128, gradient=True, hole=True)
    graph = build(evidence)
    options = Options(protection=2)
    monkeypatch.setattr(materials, "CHUNK_PIXELS", chunk_pixels)
    actual = moments(evidence, graph, options, Work.start(10))
    for r in graph.regions:
        support = (graph.labels == r.id) & ~evidence.empty
        y, x = np.nonzero(support)
        assert evidence.opacity is not None
        alpha = evidence.opacity[support]
        basis = np.column_stack(
            (
                np.ones(len(x)),
                (x + 0.5) / 96,
                (y + 0.5) / 96,
                evidence.target[support] / 255 * alpha[:, None],
                alpha,
            )
        )
        weight = 1 + 8 * options.protection * r.feature * (1 - r.texture)
        np.testing.assert_allclose(
            actual[r.id], basis.T @ basis * weight, rtol=1e-12, atol=1e-10
        )


def test_a_common_axis_gradient_fits_but_two_independent_spatial_axes_do_not():
    y, x = np.mgrid[:16, :24] / 24
    one = np.column_stack(
        (
            np.ones(x.size),
            x.ravel(),
            y.ravel(),
            x.ravel() / 2,
            x.ravel() / 3,
            x.ravel() / 4,
            np.ones(x.size),
        )
    )
    fitted = model(one.T @ one, gradients=True, gradient_price=0.01)
    assert fitted.gradient
    assert fitted.error == pytest.approx(0.01, abs=1e-10)
    assert fitted.alpha_residual == 0
    two = one.copy()
    two[:, 4] = y.ravel() / 3
    assert model(two.T @ two, gradients=True, gradient_price=0).error > 0.1
    assert not model(one.T @ one, gradients=False, gradient_price=0).gradient
    assert not model(one.T @ one, gradients=True, gradient_price=1000).gradient


@pytest.mark.parametrize("hole", [False, True])
def test_gradient_material_replaces_bands_and_retains_original_ownership_and_holes(
    hole,
):
    evidence = stripes(alpha=128, gradient=True, hole=hole)
    graph = build(evidence)
    frontier, state, options = prepared(evidence, layers=True)
    labels, details = coherent_labels(
        evidence, graph, options, Work.start(10), normalizer=5000
    )
    assert details["merges"] == 7
    assert details["linear_estimate_families"] == 1
    assert len(np.unique(labels[~evidence.empty])) == 1
    svg, exported = export(
        evidence, labels, options, Work.start(10), layers=True, cost_normalizer=5000
    )
    evaluation = frontier.policy.evaluate(svg)
    assert evaluation.valid
    assert evaluation.cost < state.snapshot.evaluation.cost
    assert evaluation.structure["gradients"] == 1
    assert frontier.add(svg, "Coherent materials", exported)
    partition = Partition.from_metadata(exported["planning_surfaces"])
    assert partition is not None
    assert state.partition is not None
    assert set(partition.owners) == set(state.partition.owners)
    assert len(set(partition.owners.values())) == 1
    document, _ = load_project(save_project(import_svg(svg)))
    partition.validate(document)
    np.testing.assert_array_equal(
        render(export_svg(document), evidence.source_size),
        render(svg, evidence.source_size),
    )
    actual = render(svg, evidence.source_size)
    np.testing.assert_array_equal(actual[..., 3], evidence.rgba[..., 3])
    if hole:
        assert actual[24:40, 40:56, 3].max() == 0


def test_broad_continuous_alpha_can_use_a_gradient_but_an_abrupt_step_cannot():
    original = stripes(alpha=64)
    shown = ~original.empty
    assert original.opacity is not None
    alpha = original.opacity.copy()
    alpha[shown] = np.broadcast_to(np.linspace(64, 200, 80) / 255, (48, 80)).ravel()
    ramp = replace(
        original,
        opacity=alpha,
        rgba=np.concatenate((original.rgba[..., :3], alpha[..., None]), axis=-1),
    )
    labels, details = coherent_labels(
        ramp, build(ramp), Options(), Work.start(10), normalizer=5000
    )
    assert len(np.unique(labels[shown])) == 1
    assert details["linear_estimate_families"] == 1
    frontier, _state, options = prepared(ramp, layers=True)
    svg, _details = export(ramp, labels, options, Work.start(10), layers=True)
    assert frontier.policy.evaluate(svg).valid
    actual = render(svg, ramp.source_size)
    np.testing.assert_allclose(
        actual[10:54, 10:86, 3], alpha[10:54, 10:86], atol=1 / 255
    )
    step_alpha = original.opacity.copy()
    step_alpha[8:56, 48:88] = 128 / 255
    step = replace(
        original,
        opacity=step_alpha,
        rgba=np.concatenate((original.rgba[..., :3], step_alpha[..., None]), axis=-1),
    )
    labels, details = coherent_labels(
        step, build(step), Options(), Work.start(10), normalizer=5000
    )
    assert labels[32, 20] != labels[32, 70]
    assert details["alpha_exclusions"] > 0


def test_a_weak_alternate_route_cannot_erase_a_supported_ridge(monkeypatch):
    evidence = stripes(alpha=128)
    labels = evidence.labels.copy()
    labels[8:32, 8:48] = 1
    labels[8:32, 48:88] = 2
    labels[32:56, 8:48] = 3
    labels[32:56, 48:88] = 4
    evidence = replace(evidence, labels=labels)
    graph = build(evidence)
    edges = tuple(
        replace(b, line_support=1) if {b.left, b.right} == {1, 2} else b
        for b in graph.boundaries
    )
    graph = replace(graph, boundaries=edges)
    monkeypatch.setattr(
        materials.Families, "_protected", lambda _self, _edge, _work: True
    )
    proposed, details = coherent_labels(
        evidence, graph, Options(), Work.start(10), normalizer=5000
    )
    assert details["merges"] > 0
    assert proposed[20, 20] != proposed[20, 60]


def test_explicit_width_source_atom_stays_separate_from_neighboring_material():
    evidence = stripes(alpha=128)
    graph = build(evidence)
    owner = int(graph.labels[32, 12])
    graph = replace(
        graph,
        regions=tuple(
            replace(r, fixed=True) if r.id == owner else r for r in graph.regions
        ),
    )
    proposed, details = coherent_labels(
        evidence, graph, Options(), Work.start(10), normalizer=5000
    )
    assert details["merges"] > 0
    np.testing.assert_array_equal(proposed == proposed[32, 12], graph.labels == owner)


def test_optional_material_failure_returns_a_valid_native_candidate(monkeypatch):
    from PIL import Image

    from vectrify.refine.cel_plan import pipeline

    pixels = np.zeros((48, 48, 4), dtype=np.uint8)
    pixels[8:40, 8:40] = (180, 80, 40, 128)

    def failed(*_args, **_kwargs):
        raise np.linalg.LinAlgError("Material fit failed")

    monkeypatch.setattr(pipeline, "material_seed", failed)
    candidate = pipeline.vectorize(
        Image.fromarray(pixels), options=Options(refine=False), seconds=10
    )
    assert candidate.metrics["material_initialization"]["status"] == "failed"
    assert (
        candidate.metrics["material_initialization"]["detail"] == "Material fit failed"
    )
    actual = render(candidate.svg, (48, 48))
    assert actual[16:32, 16:32, 3].min() == pytest.approx(128 / 255)
    assert actual[:4, :, 3].max() == 0


@pytest.mark.parametrize("bound", ["regions", "models"])
def test_discovery_limits_do_not_publish_a_partial_source_partition(bound, monkeypatch):
    evidence = stripes(alpha=128)
    if bound == "regions":
        monkeypatch.setattr(materials, "MAX_REGIONS", 2)
    else:
        monkeypatch.setattr(materials, "MAX_MODELS", 1)
    labels, details = coherent_labels(
        evidence, build(evidence), Options(), Work.start(10), normalizer=5000
    )
    assert details["merges"] == 0
    np.testing.assert_array_equal(labels, evidence.labels)
    assert details["status"] == "graph-limit" or details["model_limit_hit"]


def test_interrupted_statistics_leave_the_validated_frontier_and_evidence_unchanged(
    monkeypatch,
):
    evidence = stripes(alpha=128)
    graph = build(evidence)
    frontier, state, options = prepared(evidence, layers=True)
    original = materials.moments
    work = Work.start(10)

    def stopped(evidence, graph, options, discovery):
        discovery.stop.set()
        return original(evidence, graph, options, discovery)

    monkeypatch.setattr(materials, "moments", stopped)
    result = material_seed(frontier, evidence, graph, options, work, 10)
    assert result["status"] == "discovery-interrupted"
    assert len(frontier.entries) == 1
    assert frontier.entries[0].key == state.key
    np.testing.assert_array_equal(graph.labels, evidence.labels)
    with pytest.raises(StageInterruptedError):
        original(evidence, graph, options, work)
