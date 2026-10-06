"""CPU fitting changes real output without changing the checkpoint policy."""

import builtins
from dataclasses import replace
from threading import Event

import numpy as np
from PIL import Image

from vectrify.document import import_svg
from vectrify.refine import shared
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.frontier import Frontier
from vectrify.refine.cel_plan.model import Options, Work
from vectrify.refine.cel_plan.policy import Policy, Weights
from vectrify.refine.cel_plan.refine import _geometry_proposal, refine
from vectrify.refine.cel_plan.score import render, svg_metrics

INITIAL = (
    '<svg width="64" height="64"><path id="surface" '
    'd="M8 8H56V56H8Z" fill="#b05030"/></svg>'
)
TARGET = INITIAL.replace("#b05030", "#dc5030")


def setup(
    target=TARGET, initial=INITIAL, *, gradients=False, constraints=(), detail=0.04
):
    truth = render(target, (64, 64))
    options = Options(quality="fast", gradients=gradients)
    evidence = collect(
        Image.fromarray((truth * 255).round().astype(np.uint8)),
        None,
        options,
        Work.start(20),
    )
    frontier = Frontier(
        Policy(truth, weights=Weights(edges=0, features=0, detail=detail))
    )
    assert frontier.add(initial, "Initial", {"geometry_constraints": constraints})
    frontier.freeze_normalizer()
    return frontier, evidence, options


def geometry(svg, oid="surface"):
    return [
        (subpath.closed, [(node.command, node.values) for node in subpath.nodes])
        for subpath in import_svg(svg).geometry_for(oid).subpaths
    ]


def test_cpu_paint_improves_the_exact_score_without_torch(monkeypatch):
    frontier, evidence, options = setup(constraints=("surface",))
    baseline = frontier.baseline
    original_import = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No optional vision dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    result = refine(frontier, evidence, options, Work.start(10))
    selected = frontier.select(50)
    assert result["backend"] == "cpu"
    assert result["accepted"] >= 1
    assert selected.metrics["score_terms"]["visual"] < baseline.evaluation.visual
    assert selected.metrics["validation_rejections"] == []
    assert geometry(selected.svg) == geometry(INITIAL)
    assert frontier.baseline is baseline


def test_disabled_refinement_retains_the_original_frontier():
    frontier, evidence, _options = setup()
    before = tuple(frontier.entries)
    decisions = list(frontier.decisions)
    result = refine(frontier, evidence, Options(refine=False), Work.start(10))
    assert result["status"] == "disabled"
    assert result["attempted"] == 0
    assert tuple(frontier.entries) == before
    assert frontier.decisions == decisions


def test_stop_after_an_accepted_edit_retains_the_validated_checkpoint(monkeypatch):
    frontier, evidence, options = setup(constraints=("surface",))
    stop = Event()
    original = frontier.refine

    def stop_after_validation(*args, **kwargs):
        accepted = original(*args, **kwargs)
        if accepted:
            stop.set()
        return accepted

    monkeypatch.setattr(frontier, "refine", stop_after_validation)
    result = refine(frontier, evidence, options, Work.start(10, stop))
    selected = frontier.select(50)
    assert result["status"] == "interrupted"
    assert result["accepted"] == 1
    assert (
        selected.metrics["score_terms"]["visual"] < frontier.baseline.evaluation.visual
    )
    assert not selected.metrics["validation_rejections"]


def test_gradient_competes_with_flat_paint_and_pays_its_real_cost():
    target = (
        '<svg width="64" height="64"><defs><linearGradient id="r" '
        'gradientUnits="userSpaceOnUse" x1="8" y1="8" x2="56" y2="8">'
        '<stop stop-color="#200000"/><stop offset="1" stop-color="#ff8080"/>'
        '</linearGradient></defs><path id="surface" d="M8 8H56V56H8Z" '
        'fill="url(#r)"/></svg>'
    )
    frontier, evidence, options = setup(
        target=target, gradients=True, constraints=("surface",), detail=0.002
    )
    result = refine(frontier, evidence, options, Work.start(10))
    selected = frontier.select(50)
    assert result["accepted"] >= 1
    assert selected.metrics["gradients"] == 1
    assert (
        selected.metrics["representation_cost"]
        == svg_metrics(INITIAL)["representation_cost"] + 12
    )
    assert selected.metrics["score_terms"]["visual"] < 0.001


def test_flat_reference_does_not_gain_a_redundant_gradient():
    frontier, evidence, options = setup(gradients=True, constraints=("surface",))
    refine(frontier, evidence, options, Work.start(10))
    assert frontier.select(50).metrics["gradients"] == 0


def test_flat_color_with_injected_noise_does_not_gain_a_gradient():
    _frontier, evidence, options = setup(gradients=True, constraints=("surface",))
    noisy = evidence.rgba.copy()
    generator = np.random.default_rng(47)
    noisy[8:56, 8:56, :3] += generator.normal(0, 3 / 255, (48, 48, 3))
    np.clip(noisy, 0, 1, out=noisy)
    noisy.flags.writeable = False
    evidence = replace(evidence, rgba=noisy)
    frontier = Frontier(Policy(noisy, weights=Weights(edges=0, features=0)))
    assert frontier.add(INITIAL, "Initial", {"geometry_constraints": ("surface",)})
    frontier.freeze_normalizer()
    refine(frontier, evidence, options, Work.start(10))
    assert frontier.select(50).metrics["gradients"] == 0


def test_shared_simplification_keeps_corners_junctions_and_both_edge_copies():
    svg = (
        '<svg width="64" height="64">'
        '<path id="a" fill="red" d="M8 8L32 8L32 24L32 40L8 40Z"/>'
        '<path id="b" fill="blue" d="M32 8L56 8L56 40L32 40L32 24Z"/>'
        "</svg>"
    )
    document = import_svg(svg)
    links = shared.links(document, ["a"], ["b"])
    image = Image.new("RGB", (64, 64), "white")
    proposed = _geometry_proposal(
        document, "a", image, Options(), Work.start(10), frozenset(), move=False
    )
    assert proposed is not None
    assert all(shared.intact(proposed, link) for link in links)
    assert svg_metrics(svg)["nodes"] > sum(
        len(s.nodes) for oid in ("a", "b") for s in proposed.geometry_for(oid).subpaths
    )
    for oid in ("a", "b"):
        points = {
            node.endpoint
            for s in proposed.geometry_for(oid).subpaths
            for node in s.nodes
        }
        assert {(32, 8), (32, 40)} <= points


def test_shared_geometry_cannot_move_a_constrained_neighbour():
    svg = (
        '<svg width="64" height="64">'
        '<path id="a" fill="red" d="M8 8L32 8L32 24L32 40L8 40Z"/>'
        '<path id="b" fill="blue" d="M32 8L56 8L56 40L32 40L32 24Z"/>'
        "</svg>"
    )
    assert (
        _geometry_proposal(
            import_svg(svg),
            "a",
            Image.new("RGB", (64, 64)),
            Options(),
            Work.start(10),
            frozenset({"b"}),
            move=False,
        )
        is None
    )


def test_explicit_line_width_survives_automatic_refinement():
    svg = (
        '<svg width="64" height="64"><path id="surface" d="M8 32H56" '
        'fill="none" stroke="#303030" stroke-width="4"/></svg>'
    )
    frontier, evidence, _options = setup(target=svg, initial=svg)
    refine(
        frontier,
        evidence,
        Options(quality="fast", gradients=False, line_width=4),
        Work.start(10),
    )
    for entry in frontier.entries:
        assert (
            dict(import_svg(entry.svg).element("surface").attributes)["stroke-width"]
            == "4"
        )
    assert not any("CPU width" in d["candidate"] for d in frontier.decisions)
