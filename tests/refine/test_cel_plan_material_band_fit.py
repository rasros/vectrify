"""Partial silhouette coverage needs joint material and genuine stroke planning."""

from dataclasses import replace

import numpy as np
import pathops
import pytest
from PIL import Image

from vectrify.document import export_svg, import_svg, load_project, save_project
from vectrify.document.join import curve_path, path_style
from vectrify.refine.cel_plan.band_fit import BandFit
from vectrify.refine.cel_plan.band_plans import BandPlans
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.material_band_fit import MaterialBandFit
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.source_bands import SourceBands
from vectrify.refine.cel_plan.source_slices import source_slice


def test_long_implicit_closing_edge_controls_errors_far_from_its_endpoints():
    from vectrify.document.svg import parse_path
    from vectrify.refine.cel_plan.material_band_fit import _edge_variables

    patch = parse_path("M0 0L100 0L0 100Z")
    errors = np.array([[0.5, 50.0]])
    variables = _edge_variables(patch, errors, 2)
    selected = {
        patch.subpaths[s].nodes[n].endpoint
        for occurrences in variables
        for s, n in occurrences
    }
    assert selected == {(0.0, 0.0), (0.0, 100.0)}
    # An open contour has no closing edge and cannot acquire one by fitting.
    opened = replace(patch, subpaths=(replace(patch.subpaths[0], closed=False),))
    assert _edge_variables(opened, errors, 2) == []


def fixture(alpha=1, gradient=False):
    definitions = (
        '<defs><linearGradient id="paint" gradientUnits="userSpaceOnUse" '
        'x1="20" x2="90"><stop stop-color="#ad8665"/>'
        '<stop offset="1" stop-color="#927359"/></linearGradient></defs>'
        if gradient
        else ""
    )
    paint = "url(#paint)" if gradient else "#ad8665"
    prefix = f'<svg width="96" height="96">{definitions}<g opacity="{alpha}">'
    material = f'<path id="material" d="M20.25 28.25H76.25V70H20.25Z" fill="{paint}"/>'
    shadow = '<path d="M76.25 25.75H90V70H76.25Z" fill="#121008"/>'
    source = (
        prefix + material + shadow + '<path d="M24 28.25H72" fill="none" '
        'stroke="#121008" stroke-width="3"/></g></svg>'
    )
    original = import_svg(
        prefix + material + '<path id="ink" d="M20.25 25.75H90V70H76.25V30.75H20.25Z" '
        'fill="#121008"/></g></svg>'
    )
    rgba = render(source, (96, 96))
    evidence = collect(
        Image.fromarray(np.rint(rgba * 255).astype(np.uint8)),
        None,
        Options(),
        Work.start(10),
    )
    points = np.column_stack((np.linspace(24, 72, 24), np.full(24, 28.25)))
    guard = SourceLineGuard(evidence.rgba, (SourceProfile.at(points, 3),))
    observed = guard.source_breaks(guard.original_profiles()[0])
    parts = source_slice(original, "ink", observed, Work.start(10))
    assert parts is not None
    removed, retained = parts
    seed = SourceBands(evidence, guard).seed(
        original, "ink", removed, "nonzero", Work.start(10)
    )
    assert seed is not None
    # The former field supplies partially covered silhouette pixels above the
    # adjacent material. A stroke alone cannot preserve these native alpha rows.
    assembled = BandPlans.document(
        original,
        "ink",
        "marks",
        seed.band,
        retained,
        {**path_style(original, original.element("ink")), "fill": seed.paint},
    )
    return evidence, guard, original, assembled, seed, removed


@pytest.mark.parametrize(("alpha", "gradient"), [(1, False), (0.6, False), (1, True)])
def test_joint_material_restores_fractional_silhouette_without_filled_outline(
    alpha, gradient
):
    evidence, guard, before, assembled, seed, removed = fixture(alpha, gradient)
    original_svg = export_svg(before)
    assembled_svg = export_svg(assembled)
    assert (
        BandFit(evidence, guard).fit(
            before, assembled, "ink", seed, Work.start(20), preserve_alpha=True
        )
        is None
    )
    result = MaterialBandFit(evidence, guard).fit(
        before, assembled, "ink", seed, "material", Work.start(20)
    )
    assert result is not None
    candidate, witness = result
    assert witness["mode"] == "source-material-joint"
    assert witness["native_alpha_exact"]
    assert witness["native_body_absence"]
    assert witness["native_body_support"]["missing_samples"] == 0
    assert witness["native_body_support"]["qualified_samples"] > 90
    assert witness["source_line_comparison"]["rejections"] == []
    actual = candidate.geometry_for("ink")
    assert len(actual.subpaths) == 1
    assert not actual.subpaths[0].closed
    assert actual.subpaths[0].nodes[0].endpoint == tuple(seed.anchors[0])
    assert actual.subpaths[0].nodes[-1].endpoint == tuple(seed.anchors[-1])
    assert path_style(candidate, candidate.element("ink"))["fill"] == "none"
    assert candidate.geometry_for("marks") == assembled.geometry_for("marks")
    assert (
        pathops.op(
            curve_path(candidate.geometry_for("marks")),
            curve_path(removed),
            pathops.PathOp.INTERSECTION,
        ).area
        == 0
    )
    old = before.geometry_for("material")
    new = candidate.geometry_for("material")
    assert new.subpaths[: len(old.subpaths)] == old.subpaths
    patch = replace(new, subpaths=new.subpaths[len(old.subpaths) :])
    assert (
        pathops.op(curve_path(patch), curve_path(old), pathops.PathOp.INTERSECTION).area
        == 0
    )
    assert candidate.element("material") == before.element("material")
    pixels = render(export_svg(candidate), evidence.source_size)
    assert np.array_equal(
        pixels[..., 3], render(original_svg, evidence.source_size)[..., 3]
    )
    restored, _ = load_project(save_project(candidate))
    assert np.array_equal(pixels, render(export_svg(restored), evidence.source_size))
    assert export_svg(before) == original_svg
    assert export_svg(assembled) == assembled_svg


def test_joint_fit_rejects_remote_alpha_change_after_crop_feasibility(monkeypatch):
    from vectrify.document import Editor, Element, Selection
    from vectrify.document.svg import parse_path
    from vectrify.refine.cel_plan import material_band_fit

    evidence, guard, before, assembled, seed, _ = fixture()
    remote = parse_path("M1 88H5V92H1Z")
    editor = Editor(assembled, selection=Selection(whole_document=True))
    with editor.transaction("Unrelated remote paint must fail full native proof") as tx:
        tx.insert_object(
            assembled.root.id,
            Element(
                "remote", "path", attributes=(("fill", "black"),), geometry_id=remote.id
            ),
            geometries=(remote,),
        )
    changed = editor.snapshot.document
    initial = export_svg(changed)
    optimized = []
    original_minimize = material_band_fit.minimize

    def observed_minimize(*args, **kwargs):
        optimized.append(True)
        return original_minimize(*args, **kwargs)

    monkeypatch.setattr(material_band_fit, "minimize", observed_minimize)
    result = MaterialBandFit(evidence, guard).fit(
        before, changed, "ink", seed, "material", Work.start(20)
    )
    assert optimized
    assert result is None
    assert export_svg(changed) == initial


def test_joint_fit_keeps_fixed_width_and_rejects_unstable_material_paint(monkeypatch):
    from vectrify.document import Editor, Selection
    from vectrify.refine.cel_plan import material_band_fit

    evidence, guard, before, assembled, seed, _ = fixture(gradient=True)
    result = MaterialBandFit(evidence, guard).fit(
        before, assembled, "ink", seed, "material", Work.start(20), width_fixed=True
    )
    assert result is not None
    assert result[1]["width"] == seed.band.width
    gradient = next(e for e in before.elements() if e.tag == "linearGradient")
    editor = Editor(before, selection=Selection(whole_document=True))
    with editor.transaction("Bounding-box paint cannot survive expanded bounds") as tx:
        tx.set_attributes(gradient.id, {"gradientUnits": "objectBoundingBox"})
    unsupported = editor.snapshot.document
    initial = export_svg(unsupported)

    def unexpected_optimizer(*_args, **_kwargs):
        pytest.fail("Unsupported material paint must be excluded before fitting")

    monkeypatch.setattr(material_band_fit, "minimize", unexpected_optimizer)
    assert (
        MaterialBandFit(evidence, guard).fit(
            unsupported, assembled, "ink", seed, "material", Work.start(20)
        )
        is None
    )
    assert export_svg(unsupported) == initial


def test_joint_fit_checks_retained_width_instead_of_last_optimizer_trial(monkeypatch):
    from vectrify.refine.cel_plan import material_band_fit

    evidence, guard, before, assembled, seed, _ = fixture()

    def end_on_unretained_width(score, initial, **_kwargs):
        trial = initial.copy()
        trial[-1] = seed.band.width * 1.49
        score(trial)

    monkeypatch.setattr(material_band_fit, "minimize", end_on_unretained_width)
    result = MaterialBandFit(evidence, guard).fit(
        before, assembled, "ink", seed, "material", Work.start(20)
    )
    assert result is not None
    candidate, witness = result
    assert witness["width"] == max(0.8, seed.band.width * 0.5)
    assert (
        float(path_style(candidate, candidate.element("ink"))["stroke-width"])
        == witness["width"]
    )


def test_joint_fit_rechecks_actual_native_stroke_support_after_crop_hint(monkeypatch):
    from vectrify.refine.cel_plan import material_band_fit
    from vectrify.refine.cel_plan.local import Canvas

    evidence, guard, before, assembled, seed, _ = fixture(alpha=0.6)
    original = material_band_fit._native_raster
    checked = []

    def missing_body(root, size):
        paths = [e for e in root.iter() if e.tag.endswith("path")]
        if len(paths) == 1 and paths[0].get("stroke") == "white":
            checked.append(True)
            return Canvas(np.zeros((size[1], size[0], 4), dtype=np.uint8))
        return original(root, size)

    monkeypatch.setattr(material_band_fit, "_native_raster", missing_body)
    assert (
        MaterialBandFit(evidence, guard).fit(
            before, assembled, "ink", seed, "material", Work.start(20)
        )
        is None
    )
    assert checked
