"""Native opacity groups close subdivision seams without filling source holes."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import label

from vectrify.document import export_svg, import_svg, load_project, save_project
from vectrify.refine import cel
from vectrify.refine.cel_plan.bases import propose
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.pipeline import vectorize
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import render


def surfaces(alpha=64, *, gradient=False):
    source = np.zeros((64, 64, 4), dtype=np.uint8)
    source[8:56, 8:56] = (176, 80, 48, alpha)
    y, x = np.mgrid[:64, :64]
    second = (x + y >= 64) & (source[..., 3] > 0)
    source[second, :3] = (112, 128, 144)
    if gradient:
        source[8:56, 8:56, 0] += np.arange(48, dtype=np.uint8)[None, :]
    source[24:30, 26:32] = 0
    return source


def evidence_for(source):
    evidence = collect(Image.fromarray(source), None, Options(), Work.start(10))
    y, x = np.mgrid[: evidence.empty.shape[0], : evidence.empty.shape[1]]
    labels = np.where(
        evidence.empty,
        0,
        1 + (x + evidence.offset[0] + y + evidence.offset[1] >= 64).astype(np.int32),
    )
    return replace(evidence, labels=labels)


@pytest.mark.parametrize("gradients", [False, True])
def test_opacity_core_closes_diagonal_seams_and_round_trips(gradients):
    source = surfaces(gradient=gradients)
    evidence = evidence_for(source)
    options = Options(refine=False, gradients=gradients)
    adjacent, _ = export(evidence, evidence.labels, options, Work.start(10))
    svg, details = export(
        evidence, evidence.labels, options, Work.start(10), layers=True
    )
    actual = render(svg, (64, 64))
    assert details["alpha_model"] == "opacity-core-layers"
    assert len(details["base_models"]) == 1
    assert details["base_models"][0]["opacity"] == pytest.approx(64 / 255)
    np.testing.assert_allclose(actual[..., 3], source[..., 3] / 255, atol=0.5 / 255)
    support = source[..., 3] > 0
    assert (
        actual[..., 3][support].min()
        - render(adjacent, (64, 64))[..., 3][support].min()
        >= 3 / 255
    )
    assert actual[24:30, 26:32, 3].max() == 0
    assert np.linalg.norm(actual[12, 12, :3] * 255 - source[12, 12, :3]) < 8
    assert np.linalg.norm(actual[48, 48, :3] * 255 - source[48, 48, :3]) < 8
    assert details["gradients"] == (2 if gradients else 0)
    assert Policy(evidence.rgba).evaluate(svg).valid
    document, _ = load_project(save_project(import_svg(svg)))
    np.testing.assert_array_equal(render(export_svg(document), (64, 64)), actual)


def test_low_opacity_fringe_outside_core_retains_its_original_alpha():
    source = surfaces()
    source[8:56, 8:12, 3] = 16
    evidence = evidence_for(source)
    labels = evidence.labels.copy()
    # The weaker fringe is its own surface, rather than a constant-alpha fit
    # spanning the core and fringe. Both paints normalize inside the group.
    labels[(evidence.opacity < 32 / 255) & ~evidence.empty] = 3
    svg, details = export(
        evidence, labels, Options(gradients=False), Work.start(10), layers=True
    )
    assert details["base_models"]
    actual = render(svg, (64, 64))
    np.testing.assert_allclose(actual[..., 3], source[..., 3] / 255, atol=0.5 / 255)
    assert Policy(evidence.rgba).evaluate(svg).valid


def test_broad_variable_alpha_does_not_propose_one_uniform_base():
    source = surfaces()
    source[8:56, 8:56, 3] = np.linspace(32, 224, 48).round().astype(np.uint8)[None, :]
    source[24:30, 26:32] = 0
    evidence = evidence_for(source)
    components, _ = label(~evidence.empty)
    assert not propose(evidence, evidence.labels, components, Work.start(10))
    svg, details = export(
        evidence, evidence.labels, Options(), Work.start(10), layers=True
    )
    assert not details["base_models"]
    actual = render(svg, (64, 64))
    np.testing.assert_allclose(actual[..., 3], source[..., 3] / 255, atol=3 / 255)


def test_native_thin_mark_keeps_an_independent_component():
    source = surfaces()
    source[2:6, 2:4] = (32, 16, 8, 1)
    evidence = evidence_for(source)
    labels = evidence.labels.copy()
    labels[(evidence.opacity < 2 / 255) & ~evidence.empty] = 3
    svg, details = export(
        evidence, labels, Options(gradients=False), Work.start(10), layers=True
    )
    assert all(3 not in base["members"] for base in details["base_models"])
    actual = render(svg, (64, 64))
    np.testing.assert_allclose(actual[2:6, 2:4, 3], 1 / 255, atol=0.5 / 255)
    assert Policy(evidence.rgba).evaluate(svg).valid


def test_disconnected_materials_have_separate_opacity_groups():
    source = np.zeros((64, 64, 4), dtype=np.uint8)
    source[8:28, 8:28] = (176, 80, 48, 64)
    source[12:24, 18:28, :3] = (112, 128, 144)
    source[36:56, 36:56] = (176, 80, 48, 128)
    source[40:52, 46:56, :3] = (112, 128, 144)
    evidence = evidence_for(source)
    y, x = np.mgrid[: evidence.empty.shape[0], : evidence.empty.shape[1]]
    x, y = x + evidence.offset[0], y + evidence.offset[1]
    labels = np.zeros_like(evidence.labels)
    labels[(source[y, x, 3] == 64)] = 1
    labels[(source[y, x, 3] == 64) & (source[y, x, 0] == 112)] = 2
    labels[(source[y, x, 3] == 128)] = 3
    labels[(source[y, x, 3] == 128) & (source[y, x, 0] == 112)] = 4
    svg, details = export(
        evidence, labels, Options(gradients=False), Work.start(10), layers=True
    )
    assert len(details["base_models"]) == 2
    assert sorted(base["members"] for base in details["base_models"]) == [
        [1, 2],
        [3, 4],
    ]
    actual = render(svg, (64, 64))
    np.testing.assert_allclose(actual[..., 3], source[..., 3] / 255, atol=0.5 / 255)


def test_modal_core_avoids_fragmentation_from_small_byte_variation():
    source = surfaces(253)
    source[8:56:3, 8:56:3, 3] = 252
    source[12, 12, 3] = 255
    source[24:30, 26:32] = 0
    evidence = evidence_for(source)
    components, _ = label(~evidence.empty)
    bases = propose(evidence, evidence.labels, components, Work.start(10))
    assert len(bases) == 1
    assert bases[0].opacity == pytest.approx(253 / 255)
    assert bases[0].core_fraction == 1
    assert bases[0].data.count("M") == 2  # Outer contour and intentional hole.


def test_detailed_normalization_seed_competes_with_a_native_opacity_base(monkeypatch):
    from vectrify.refine.cel_plan import pipeline

    source = surfaces(253)
    evidence = evidence_for(source)
    observations = []
    monkeypatch.setattr(pipeline, "collect", lambda *_args: evidence)
    result = vectorize(
        Image.fromarray(source),
        options=Options(refine=False, gradients=False),
        seconds=10,
        observe=observations.append,
    )
    detailed = next(o for o in observations if o.label == "Traced at complexity 100")
    assert detailed.evaluation is not None
    assert detailed.evaluation.valid
    assert detailed.details["alpha_model"] == "opacity-core-layers"
    assert result.metrics["cost_normalizer_source"] == "validated-detailed"
    assert result.metrics["cost_normalizer"] == detailed.evaluation.cost


def test_stop_during_core_outline_discards_the_partial_proposal(monkeypatch):
    evidence = evidence_for(surfaces())
    components, _ = label(~evidence.empty)
    work = Work.start(10)
    original = cel.region_outlines

    def stopped(*args, **kwargs):
        work.stop.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(cel, "region_outlines", stopped)
    with pytest.raises(StageInterruptedError):
        propose(evidence, evidence.labels, components, work)
