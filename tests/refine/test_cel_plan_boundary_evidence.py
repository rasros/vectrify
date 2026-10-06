"""Short monotone shades can compete, while ambiguous marks remain barriers."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from tests.refine.test_cel_plan_ownership import stripes
from vectrify.refine import cel
from vectrify.refine.cel_plan import families
from vectrify.refine.cel_plan.boundary_evidence import shade_fragment
from vectrify.refine.cel_plan.export import export
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.ink import measure
from vectrify.refine.cel_plan.materials import coherent_labels
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.policy import Policy
from vectrify.refine.cel_plan.score import render


def source(kind):
    evidence = stripes(alpha=128)
    target = np.full_like(evidence.target, 180)
    target[:, :48] = 100 if kind in {"shade", "alpha"} else 180
    if kind == "ridge":
        target[:, 46:50] = 24
    elif kind == "pale":
        target[:, 47:49] = 240
    elif kind == "dot":
        target[31:33, 47:49] = 0
    opacity = evidence.opacity.copy()
    if kind == "alpha":
        opacity[:, :48] *= 0.25
    return replace(evidence, target=target, opacity=opacity)


@pytest.mark.parametrize("kind", ["shade", "ridge", "pale", "dot", "alpha"])
def test_complete_short_profiles_preserve_ridges_pale_marks_and_opacity_steps(kind):
    evidence = source(kind)
    light = gaussian_filter(cel.lightness(evidence.target), 0.5)
    points = np.array(((48, 31), (48, 33)), dtype=float)
    assert shade_fragment(points, evidence, light) == (kind == "shade")
    assert shade_fragment(points[::-1], evidence, light) == (kind == "shade")


@pytest.mark.parametrize(
    "points", [((8, 30), (8, 32)), ((0, 0), (0, 1)), ((48, 31), (48, 31))]
)
def test_incomplete_or_degenerate_support_remains_protected(points):
    evidence = source("shade")
    light = gaussian_filter(cel.lightness(evidence.target), 0.5)
    assert not shade_fragment(np.array(points, dtype=float), evidence, light)


def test_fragment_limit_and_stop_preserve_the_uncertain_barrier(monkeypatch):
    evidence = source("shade")
    graph = build(evidence)
    edge = next(e for e in graph.boundaries if {e.left, e.right} == {4, 5})
    edge = replace(edge, points=np.array(((48, 31), (48, 33))), line_support=1)
    factory = Families(evidence, graph, Options())
    work = Work.start(10)
    assert not factory._protected(edge, work)
    cached = dict(factory.diagnostics)
    assert not factory._protected(edge, work)
    assert factory.diagnostics == cached
    assert factory.diagnostics["shade_fragments"] == 1
    monkeypatch.setattr(families, "MAX_FRAGMENT_PROOFS", 0)
    assert factory._protected(replace(edge, id=10000), work)
    assert 10000 not in factory.fragments
    work.stop.set()
    assert factory._protected(replace(edge, id=10001), work)
    assert 10001 not in factory.fragments


def test_stop_during_profile_never_caches_a_shade_proof(monkeypatch):
    evidence = source("shade")
    graph = build(evidence)
    edge = replace(
        graph.boundaries[0], points=np.array(((48, 31), (48, 33))), line_support=1
    )
    factory = Families(evidence, graph, Options())
    work = Work.start(10)

    def stopped(*_args):
        work.stop.set()
        return True

    monkeypatch.setattr(families, "shade_fragment", stopped)
    assert factory._protected(edge, work)
    assert factory.fragments == {}
    assert factory.diagnostics["shade_fragments"] == 0


def test_compatible_fragments_join_a_continuous_dark_mark_without_its_background():
    evidence = stripes(alpha=128)
    labels = np.where(evidence.empty, 0, 1).astype(np.int32)
    labels[16:48, 46:48] = 2
    labels[16:48, 48:50] = 3
    target = evidence.target.copy()
    target[labels == 2] = 40
    target[labels == 3] = 44
    drawn = np.isin(labels, (2, 3))
    rgba = evidence.rgba.copy()
    rgba[..., :3] = target / 255
    evidence = replace(evidence, labels=labels, target=target, drawn=drawn, rgba=rgba)
    graph = build(evidence)
    edge = next(e for e in graph.boundaries if {e.left, e.right} == {2, 3})
    assert measure(edge.points, target, 1.5) is not None
    graph = replace(
        graph,
        boundaries=tuple(replace(e, line_support=1) for e in graph.boundaries),
    )
    proposed, details = coherent_labels(
        evidence, graph, Options(gradients=False), Work.start(10), normalizer=1000
    )
    assert proposed[32, 47] == proposed[32, 49]
    assert proposed[32, 44] != proposed[32, 47]
    assert details["ridge_diagnostics"]["compatible_ink_contacts"] == 1
    svg, _ = export(
        evidence, proposed, Options(gradients=False), Work.start(10), layers=True
    )
    assert Policy(evidence.rgba).evaluate(svg).valid
    actual = render(svg, evidence.source_size)
    assert actual[32, 47, :3].max() < 0.2
    assert actual[32, 44, 0] > 0.6


def test_native_trough_inside_a_broad_dark_atom_cannot_claim_same_ink():
    evidence = source("shade")
    target = np.full_like(evidence.target, 40)
    target[:, 46:50] = 10
    evidence = replace(evidence, target=target, drawn=~evidence.empty)
    graph = build(evidence)
    edge = next(e for e in graph.boundaries if {e.left, e.right} == {4, 5})
    edge = replace(edge, line_support=1)
    factory = Families(evidence, graph, Options())
    assert factory._protected(edge, Work.start(10))
    assert factory.diagnostics["compatible_ink_contacts"] == 0
    assert factory.diagnostics["supported_ridges"] == 1


def test_explicit_ink_width_disallows_the_internal_ink_exemption():
    evidence = source("shade")
    evidence = replace(
        evidence, target=np.full_like(evidence.target, 40), drawn=~evidence.empty
    )
    graph = build(evidence)
    graph = replace(graph, regions=tuple(replace(r, fixed=True) for r in graph.regions))
    edge = next(e for e in graph.boundaries if {e.left, e.right} == {4, 5})
    factory = Families(evidence, graph, Options())
    assert not factory._same_ink(edge, Work.start(10))
    assert factory.ink_contacts == set()
