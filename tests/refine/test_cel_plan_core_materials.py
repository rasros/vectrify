"""A source-owned material silhouette competes without replacing native truth."""

import time
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

from vectrify.document import export_svg, import_svg, load_project, save_project
from vectrify.refine.cel_plan.core_materials import candidate, prepare
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.nested import in_core
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import render, svg_metrics
from vectrify.refine.cel_plan.search import State


def source(*, shades=True, faint=False):
    rgba = np.zeros((80, 88, 4), dtype=np.uint8)
    alpha = np.zeros((80, 88), dtype=float)
    alpha[16:64, 20:68] = 128
    rgba[..., 3] = np.rint(gaussian_filter(alpha, 1)).astype(np.uint8)
    rgba[rgba[..., 3] > 0, :3] = (176, 80, 48)
    if shades:
        rgba[16:64, 44:68, :3] = (80, 144, 176)
    if faint:
        rgba[4:8, 4:6] = (32, 16, 8, 1)
    return rgba


def setup(rgba, **settings):
    options = Options(refine=False, **settings)
    evidence = collect(Image.fromarray(rgba), None, options, Work.start(10))
    graph = build(evidence, work=Work.start(10))
    policy = Policy.from_evidence(evidence, graph)
    fallback, _ = export(
        evidence, evidence.labels, options, Work.start(10), conservative=True
    )
    baseline = policy.evaluate(fallback)
    assert baseline.valid
    policy.establish(baseline)
    return evidence, graph, policy, options, fallback


@pytest.mark.parametrize("shades", [False, True])
def test_native_material_candidate_is_compact_owned_and_round_trips(shades):
    evidence, graph, policy, options, fallback = setup(source(shades=shades))
    before = evidence.opacity.copy()
    result, details = candidate(
        evidence, graph, policy, options, Work.start(10), normalizer=1000
    )
    assert result is not None, details
    svg, exported = result
    assert details["models"][0]["fringe_atoms"] > 0
    evaluation = policy.evaluate(svg)
    assert evaluation.valid, evaluation.rejections
    assert svg_metrics(svg)["nodes"] < svg_metrics(fallback)["nodes"]
    partition = Partition.from_metadata(exported["planning_surfaces"])
    assert partition is not None
    partition.validate(import_svg(svg))
    assert set(partition.owners) == {
        r.id for r in graph.regions if r.id not in graph.hidden and r.area
    }
    np.testing.assert_array_equal(evidence.opacity, before)
    assert evidence.coverage_fit is None
    document, _ = load_project(save_project(import_svg(svg)))
    np.testing.assert_array_equal(
        render(export_svg(document), evidence.source_size),
        render(svg, evidence.source_size),
    )


def test_native_faint_component_keeps_source_ownership_and_mass():
    evidence, graph, policy, options, _ = setup(source(faint=True))
    result, details = candidate(
        evidence, graph, policy, options, Work.start(10), normalizer=1000
    )
    assert result is not None, details
    svg, exported = result
    actual = render(svg, evidence.source_size)
    np.testing.assert_allclose(actual[4:8, 4:6, 3], 1 / 255, atol=0.5 / 255)
    assert policy.evaluate(svg).valid
    partition = Partition.from_metadata(exported["planning_surfaces"])
    assert partition is not None
    assert all(r.id in partition.owners for r in graph.regions if r.area)


@pytest.mark.parametrize("kind", ["ramp", "partial-hole", "width", "scale"])
def test_unsupported_models_keep_native_evidence(kind):
    rgba = source()
    if kind == "ramp":
        rgba[16:64, 20:68, 3] = np.linspace(32, 224, 48).astype(np.uint8)[None, :]
    if kind == "partial-hole":
        rgba[32:36, 32:36, 3] = 8
    evidence, graph, policy, _, _ = setup(
        rgba, **({"line_width": 2} if kind == "width" else {})
    )
    if kind == "scale":
        evidence = replace(evidence, scale=(0.5, 0.5))
    virtual, cores, details = prepare(evidence, graph, policy, Work.start(10))
    assert not cores, details
    np.testing.assert_array_equal(virtual.opacity, evidence.opacity)
    np.testing.assert_array_equal(virtual.empty, evidence.empty)


def test_stop_discards_material_discovery_without_mutating_evidence():
    evidence, graph, policy, _, _ = setup(source())
    work = Work.start(10)
    work.stop.set()
    before = evidence.empty.copy()
    with pytest.raises(StageInterruptedError):
        prepare(evidence, graph, policy, work)
    np.testing.assert_array_equal(evidence.empty, before)


def test_completed_growth_prefix_exports_with_separate_live_budget(monkeypatch):
    from vectrify.refine.cel_plan import core_materials

    evidence, graph, policy, options, _ = setup(source())
    growth = Work.start(10)
    output = Work.start(10)
    original = core_materials.coherent_labels

    def expire_growth(*args, **kwargs):
        labels, details = original(*args, **kwargs)
        growth.deadline = time.monotonic() - 1
        return labels, {**details, "status": "interrupted"}

    monkeypatch.setattr(core_materials, "coherent_labels", expire_growth)
    result, details = candidate(
        evidence,
        graph,
        policy,
        options,
        output,
        normalizer=1000,
        discovery=growth,
    )
    assert growth.interrupted
    assert details["growth"]["status"] == "interrupted"
    assert result is not None, details
    assert policy.evaluate(result[0]).valid


def test_growth_stop_cannot_export_with_separate_live_budget(monkeypatch):
    from vectrify.refine.cel_plan import core_materials

    evidence, graph, policy, options, _ = setup(source())
    output = Work.start(10)
    growth = Work(output.deadline, output.stop, output.timings)
    original = core_materials.coherent_labels

    def stop_growth(*args, **kwargs):
        labels, details = original(*args, **kwargs)
        growth.stop.set()
        return labels, {**details, "status": "interrupted"}

    monkeypatch.setattr(core_materials, "coherent_labels", stop_growth)
    with pytest.raises(StageInterruptedError):
        candidate(
            evidence,
            graph,
            policy,
            options,
            output,
            normalizer=1000,
            discovery=growth,
        )


def test_primary_fringe_owner_can_prove_secondary_core_coverage():
    rgba = source()
    rgba[28:44, 36:52, :3] = (32, 16, 8)
    evidence, graph, policy, options, _ = setup(rgba)
    result, details = candidate(
        evidence, graph, policy, options, Work.start(10), normalizer=1000
    )
    assert result is not None, details
    svg, exported = result
    partition = Partition.from_metadata(exported["planning_surfaces"])
    assert partition is not None
    base = next(surface for surface in partition.surfaces if surface.covered)
    original = int(evidence.labels[34 - evidence.offset[1], 42 - evidence.offset[0]])
    oid = partition.owners[original]
    surface = next(surface for surface in partition.surfaces if surface.id == oid)
    assert set(surface.members).issubset(base.covered)
    document = import_svg(svg)
    evaluation = policy.evaluate(svg)
    snapshot = LocalPolicy(policy).start(svg, evaluation)
    state = State(document, svg, snapshot, "test", exported, partition=partition)
    assert in_core(
        state, surface.members, oid, document.geometry_for(oid), Work.start(10)
    )
    # Keep membership/paint, but move the actual base away: metadata cannot
    # substitute for containment in the same transformed coordinate frame.
    import xml.etree.ElementTree as ET

    tree = ET.fromstring(svg)
    node = next(node for node in tree.iter() if node.get("id") == base.id)
    node.set("transform", "translate(9999 9999)")
    changed = ET.tostring(tree, encoding="unicode")
    outside = replace(state, document=import_svg(changed), svg=changed)
    assert not in_core(
        outside,
        surface.members,
        oid,
        outside.document.geometry_for(oid),
        Work.start(10),
    )


def test_material_compaction_keeps_supported_closed_ink():
    rgba = source(shades=False)
    y, x = np.mgrid[:80, :88]
    distance = np.hypot(x - 44, y - 40)
    rgba[distance <= 16, :3] = (24, 16, 8)
    rgba[distance <= 12, :3] = (56, 152, 112)
    evidence, graph, policy, options, _ = setup(rgba)
    result, details = candidate(
        evidence, graph, policy, options, Work.start(10), normalizer=1000
    )
    assert result is not None, details
    svg, _ = result
    actual = render(svg, evidence.source_size)
    assert actual[26, 44, :3].mean() < 0.15
    assert actual[40, 44, 1] > 0.4
    assert policy.evaluate(svg).valid
