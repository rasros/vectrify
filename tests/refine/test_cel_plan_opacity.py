"""Transparent content keeps native opacity, topology and ink interpretations."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import label

from vectrify.document import export_svg, import_svg, load_project, save_project
from vectrify.refine import cel
from vectrify.refine.cel_plan import opacity
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.pipeline import vectorize
from vectrify.refine.cel_plan.planning import merged_labels
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import foreground_mask, opacity_measurements, render
from vectrify.refine.colour_regions import boundary_chains


def pixels(alpha=64):
    values = np.zeros((64, 64, 4), dtype=np.uint8)
    values[8:56, 8:56] = (176, 80, 48, alpha)
    return values


def empty():
    return '<svg width="64" height="64"/>'


def test_sparse_runs_match_four_connected_equal_value_components():
    values = np.random.default_rng(17).integers(0, 5, size=(32, 48))
    values[10:20, 10:30] = 0
    values[13:17, 15:25] = 2
    actual = opacity.components(values)
    seen = set()
    for value in np.unique(values):
        expected, count = label(values == value)
        for index in range(1, count + 1):
            ids = np.unique(actual[expected == index])
            assert len(ids) == 1
            assert int(ids[0]) not in seen
            seen.add(int(ids[0]))
    assert len(seen) == len(np.unique(actual))


def test_diagonal_contact_does_not_join_surfaces():
    values = np.array([[1, 0], [0, 1]])
    assert len(np.unique(opacity.components(values))) == 4


def test_almost_opaque_byte_variation_does_not_fragment_one_surface():
    target = np.full((32, 32, 3), (176, 80, 48), dtype=float)
    alpha = np.full((32, 32), 253 / 255)
    alpha[:, ::2] = 252 / 255
    labels = opacity.labels(
        target, alpha, np.ones_like(alpha, dtype=bool), 1, Work.start(10)
    )
    assert len(np.unique(labels)) == 1


def test_flat_merges_preserve_relative_opacity_discontinuities():
    source = pixels(4)
    source[8:56, 32:56, 3] = 16
    options = Options(complexity=0, palette=1, gradients=False, refine=False)
    evidence = collect(Image.fromarray(source), None, options, Work.start(10))
    graph = build(evidence)
    labels, _ = merged_labels(graph, options, Work.start(10))
    assert labels[30, 20] != labels[30, 40]
    svg, _ = export(evidence, labels, options, Work.start(10))
    actual = render(svg, (64, 64))
    np.testing.assert_allclose(actual[12:52, 12:28, 3], 4 / 255, atol=1 / 255)
    np.testing.assert_allclose(actual[12:52, 36:52, 3], 16 / 255, atol=1 / 255)
    assert Policy(evidence.rgba).evaluate(svg).valid


def test_native_alpha_holes_and_thin_marks_survive_compact_boundary_fitting():
    source = pixels()
    # A diagonal, one-pixel hole is very easy for a fitted closed curve to erase.
    for index in range(20, 30):
        source[index, index] = 0
    source[2:6, 2:4] = (176, 80, 48, 1)
    source[12:52, 32:52, :3] = (160, 72, 48)
    options = Options(complexity=0, palette=4, refine=False)
    evidence = collect(Image.fromarray(source), None, options, Work.start(10))
    graph = build(evidence)
    labels, _ = merged_labels(graph, options, Work.start(10))
    svg, metadata = export(evidence, labels, options, Work.start(10), structure=True)
    actual = render(svg, (64, 64))
    assert actual[np.arange(20, 30), np.arange(20, 30), 3].max() == 0
    np.testing.assert_allclose(actual[2:6, 2:4, 3], 1 / 255, atol=0.5 / 255)
    assert metadata["native_alpha_regions"] > 0
    assert metadata["geometry_constraints"]
    assert Policy(evidence.rgba).evaluate(svg).valid


def test_transparent_padding_does_not_raise_the_soft_representation_budget():
    source = pixels()
    source[20:44, 20:44, :3] = (160, 72, 48)
    options = Options(palette=4, refine=False)
    original = vectorize(Image.fromarray(source), options=options, seconds=10)
    padded = vectorize(
        Image.fromarray(np.pad(source, ((20, 20), (20, 20), (0, 0)))),
        options=options,
        seconds=10,
    )
    assert padded.metrics["cost_normalizer"] == original.metrics["cost_normalizer"]
    assert (
        padded.metrics["representation_budget"]
        == original.metrics["representation_budget"]
    )


def test_gradient_alpha_does_not_extrapolate_beyond_sampled_opacity():
    y, x = np.mgrid[:32, :32]
    target = np.stack((x * 7, y * 5, x * 3), axis=-1).astype(float)
    alpha = (60 + x % 3) / 255
    paint = opacity.fit(target, alpha, np.ones_like(alpha, dtype=bool))
    assert paint.gradient is not None
    assert all(
        alpha.min() <= stop.opacity <= alpha.max() for stop in paint.gradient.stops
    )


def test_gradient_proposal_competes_with_its_representation_price():
    _y, x = np.mgrid[:32, :32]
    target = np.stack((x * 7, x * 5, x * 3), axis=-1).astype(float)
    alpha = np.full(x.shape, 64 / 255)
    mask = np.ones_like(alpha, dtype=bool)
    assert opacity.fit(target, alpha, mask).gradient is not None
    priced = opacity.fit(target, alpha, mask, gradient_price=1)
    assert priced.gradient is None
    assert priced.opacity == pytest.approx(64 / 255)


def test_complexity_prices_paint_using_one_fixed_normalizer():
    source = pixels()
    source[8:56, 8:56, 0] = np.linspace(150, 210, 48).round().astype(np.uint8)[None, :]
    options = Options(palette=4, refine=False)
    evidence = collect(Image.fromarray(source), None, options, Work.start(10))
    labels = opacity.components((~evidence.empty).astype(np.int32))
    counts = []
    for complexity in (0, 50, 100):
        svg, metadata = export(
            evidence,
            labels,
            replace(options, complexity=complexity),
            Work.start(10),
            cost_normalizer=1500,
        )
        assert Policy(evidence.rgba).evaluate(svg).valid
        counts.append(metadata["gradients"])
        assert metadata["paint_cost_normalizer"] == 1500
    assert counts == [0, 0, 1]


def test_opaque_antialias_fringe_keeps_legacy_evidence_path():
    alpha = pixels(255)[..., 3].astype(float) / 255
    alpha[8, 8:56] = 0.3
    assert not opacity.needed(alpha)
    alpha[30, 30] = 0.3
    assert opacity.needed(alpha)
    thin = np.zeros_like(alpha)
    thin[20:44, 28] = 1 / 255
    assert opacity.needed(thin)


def test_low_opacity_padding_and_cache_keep_the_same_immutable_evidence():
    source = pixels(1)
    options = Options(palette=4)
    first = collect(Image.fromarray(source), None, options, Work.start(10))
    padded = collect(
        Image.fromarray(np.pad(source, ((20, 20), (20, 20), (0, 0)))),
        None,
        options,
        Work.start(10),
    )
    assert first.opacity is not None
    assert first.foreground.sum() == 48 * 48
    assert first.background is None
    np.testing.assert_array_equal(first.labels, padded.labels)
    np.testing.assert_array_equal(first.opacity, padded.opacity)
    assert (
        collect(
            Image.fromarray(source),
            None,
            replace(options, complexity=0),
            Work.start(10),
        )
        is first
    )
    with pytest.raises(ValueError, match="read-only"):
        first.opacity[10, 10] = 1


def test_long_translucent_crop_keeps_native_samples_within_pixel_allowance():
    source = np.zeros((1800, 64, 4), dtype=np.uint8)
    source[8:1792, 8:56] = (176, 80, 48, 64)
    source[900, 30:34] = 0
    evidence = collect(Image.fromarray(source), None, Options(), Work.start(10))
    assert evidence.scale == (1, 1)
    assert evidence.empty[900, 30:34].all()


@pytest.mark.parametrize("alpha", [1, 25, 64])
def test_native_export_preserves_low_opacity_surface_and_hole(alpha):
    source = pixels(alpha)
    source[24:40, 24:40] = 0
    options = Options(refine=False, palette=4)
    evidence = collect(Image.fromarray(source), None, options, Work.start(10))
    svg, metadata = export(
        evidence,
        evidence.labels,
        options,
        Work.start(10),
        conservative=True,
        conservative_tolerance=0,
    )
    actual = render(svg, (64, 64))
    np.testing.assert_allclose(actual[..., 3], source[..., 3] / 255, atol=1 / 255)
    assert actual[24:40, 24:40, 3].max() == 0
    assert metadata["alpha_model"] == "adjacent-rgba-surfaces"
    assert Policy(evidence.rgba).evaluate(svg).valid
    document, _ = load_project(save_project(import_svg(svg)))
    np.testing.assert_array_equal(render(export_svg(document), (64, 64)), actual)


def test_pipeline_retains_byte_opacity_thin_component():
    source = np.zeros((64, 64, 4), dtype=np.uint8)
    source[20:44, 28:30] = (176, 80, 48, 1)
    candidate = vectorize(
        Image.fromarray(source), options=Options(refine=False), seconds=10
    )
    actual = render(candidate.svg, (64, 64))
    assert actual[20:44, 28:30, 3].sum() >= source[20:44, 28:30, 3].sum() / 255 * 0.95
    assert not candidate.metrics["validation_rejections"]


def test_rgba_ramp_uses_stop_opacity_and_preserves_native_interior():
    source = pixels()
    source[8:56, 8:56, 3] = np.linspace(32, 224, 48).round().astype(np.uint8)[None, :]
    options = Options(refine=False, palette=4)
    evidence = collect(Image.fromarray(source), None, options, Work.start(10))
    # This proposal compares one RGBA ramp to the conservative byte partitions.
    labels = opacity.components((~evidence.empty).astype(np.int32))
    svg, metadata = export(evidence, labels, options, Work.start(10))
    assert metadata["gradients"] == 1
    assert 'stop-opacity="' in svg
    actual = render(svg, (64, 64))
    np.testing.assert_allclose(
        actual[12:52, 12:52, 3], source[12:52, 12:52, 3] / 255, atol=2 / 255
    )
    assert Policy(evidence.rgba).evaluate(svg).valid


def test_one_body_palette_color_still_preserves_translucent_ink():
    source = pixels()
    source[20:44, 28:32, :3] = 0
    candidate = vectorize(
        Image.fromarray(source), options=Options(palette=1, refine=False), seconds=10
    )
    actual = render(candidate.svg, (64, 64))
    assert actual[24:40, 29:31, :3].max() < 2 / 255
    assert actual[24:40, 29:31, 3].min() == pytest.approx(64 / 255)


def test_explicit_filled_ink_is_fixed_in_merge_and_refinement_proposals():
    source = pixels()
    source[20:44, 28:32, :3] = 0
    options = Options(palette=1, line_width=2)
    evidence = collect(Image.fromarray(source), None, options, Work.start(10))
    graph = build(evidence)
    fixed = {r.id for r in graph.regions if r.fixed}
    assert fixed
    changed, _ = merged_labels(graph, replace(options, complexity=0), Work.start(10))
    for index in fixed:
        own = graph.labels == index
        assert len(np.unique(changed[own])) == 1
        assert (changed == changed[own][0]).sum() == own.sum()
    _, metadata = export(
        evidence,
        changed,
        options,
        Work.start(10),
        conservative=True,
        conservative_tolerance=0,
    )
    assert metadata["paint_constraints"]
    assert set(metadata["paint_constraints"]) <= set(metadata["geometry_constraints"])
    other = collect(
        Image.fromarray(source), None, replace(options, line_width=6), Work.start(10)
    )
    assert other is not evidence
    assert other.drawn.sum() > evidence.drawn.sum()


def test_native_width_coverage_honors_even_widths():
    target = np.full((32, 32, 3), 200.0)
    smooth = target.copy()
    drawn = np.zeros((32, 32), dtype=bool)
    drawn[4:28, 16] = True
    target[drawn] = 0
    for width in (2, 6):
        changed, _ = opacity.ink_width(
            target, smooth, drawn, np.ones_like(drawn), width, (1, 1)
        )
        assert ((200 - changed[16, :, 0]) / 200).sum() == pytest.approx(width)


@pytest.mark.parametrize("alpha", [1 / 255, 0.1, 0.25])
def test_missing_or_excess_thin_opacity_is_rejected(alpha):
    truth = np.zeros((64, 64, 4), dtype=np.float32)
    truth[20:44, 28:30] = (0.5, 0.2, 0.1, alpha)
    policy = Policy(truth)
    missing = truth.copy()
    missing[20:23, 28:30] = 0
    assert (
        "translucent-component-lost"
        in policy.evaluate(empty(), pixels=missing).rejections
    )
    excess = truth.copy()
    excess[20:44, 28:30, 3] = min(1, alpha * 3)
    assert (
        "translucent-component-opacity-excess"
        in policy.evaluate(empty(), pixels=excess).rejections
    )


def test_equal_white_composite_cannot_hide_wrong_opacity():
    truth = pixels().astype(np.float32) / 255
    actual = truth.copy()
    inside = truth[..., 3] > 0
    actual[inside, 3] *= 2
    actual[inside, :3] = 1 - (1 - truth[inside, :3]) / 2
    policy = Policy(truth)
    evaluation = policy.evaluate(empty(), pixels=actual)
    assert evaluation.terms["alpha"] > 0
    assert "translucent-opacity-excess" in evaluation.rejections


def test_low_opacity_diagnostics_are_not_empty_and_frozen_mask_stays_unchanged():
    truth = pixels().astype(np.float32) / 255
    assert not foreground_mask(truth).any()
    measured = opacity_measurements(np.zeros_like(truth), truth)
    assert measured["support_pixels"] == 48 * 48
    assert measured["alpha_mse"] > 0
    assert measured["black_mse"] > 0
    assert measured["white_mse"] > 0


def test_alpha_only_boundary_is_not_ink_and_translucent_holes_are_protected():
    truth = pixels().astype(np.float32) / 255
    truth[8:56, 32:56, 3] *= 2
    truth[24:40, 24:40] = 0
    policy = Policy(truth)
    assert not policy.ink.any()
    actual = truth.copy()
    actual[24:40, 24:40] = (0.5, 0.2, 0.1, 0.1)
    assert "protected-hole-lost" in policy.evaluate(empty(), pixels=actual).rejections


def test_graph_and_surface_export_stop_before_publishing_partial_geometry():
    options = Options(palette=4)
    evidence = collect(Image.fromarray(pixels()), None, options, Work.start(10))
    work = Work.start(10)
    work.stop.set()
    with pytest.raises(StageInterruptedError):
        build(evidence, work=work)
    with pytest.raises(StageInterruptedError):
        export(evidence, evidence.labels, options, work)


def test_large_boundary_extraction_checks_cancellation_during_assembly():
    labels = np.indices((64, 64)).sum(axis=0) % 2
    checks = 0

    def check():
        nonlocal checks
        checks += 1
        if checks == 3:
            raise StageInterruptedError("Stopped while assembling boundaries")

    with pytest.raises(StageInterruptedError):
        boundary_chains(labels, check=check)
    assert checks == 3


def test_bad_interior_curve_restores_both_sides_of_its_shared_boundary(monkeypatch):
    source = pixels()
    source[20:44, 20:44, :3] = (80, 120, 180)
    options = Options(palette=4, refine=False, gradients=False)
    evidence = collect(Image.fromarray(source), None, options, Work.start(10))
    labels = np.zeros(evidence.empty.shape, dtype=np.int32)
    labels[~evidence.empty] = 1
    labels[20:44, 20:44] = 2
    calls = []

    def crossing(points, _bound, **_kwargs):
        calls.append(points.copy())
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        return [
            ("L", (right, bottom)),
            ("L", (left, bottom)),
            ("L", (right, top)),
            ("L", tuple(points[-1])),
        ]

    monkeypatch.setattr(cel, "curve_nodes", crossing)
    svg, metadata = export(evidence, labels, options, Work.start(10))
    assert calls
    assert metadata["repaired_crossing_regions"]
    actual = render(svg, (64, 64))
    np.testing.assert_allclose(actual[..., 3], source[..., 3] / 255, atol=1 / 255)
    evaluation = Policy(evidence.rgba).evaluate(svg)
    assert evaluation.structure["self_crossings"] == 0
    assert evaluation.valid
    assert actual[28:36, 28:36, 2].mean() > actual[12:18, 12:18, 2].mean()
