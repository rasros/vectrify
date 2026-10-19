"""Selected fitting separates proposal mechanics from exact acceptance state."""

import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from vectrify.document import Selection, import_svg
from vectrify.refine.selected import FitContext, FitOptions, fit_selected_path
from vectrify.refine.selected_fit.objective import ExactObjective
from vectrify.refine.selected_fit.policy import Candidate, RetainedFit
from vectrify.svg_render import render_image

SVG = (
    '<svg width="64" height="64"><path id="p" fill="black" '
    'd="M12 16 L48 16 L48 48 L12 48 L12 16 Z"/></svg>'
)
SELECTION = Selection(object_ids=frozenset({"p"}))


def retained(*, cleanup=False, crossed=False):
    options = FitOptions(cleanup=cleanup)
    source = (
        SVG.replace("L48 16 L48 48 L12 48", "L48 48 L12 48 L48 16") if crossed else SVG
    )
    context = FitContext(import_svg(source), SELECTION, render_image(source), options)
    objective = Mock(spec=ExactObjective)
    objective.score.return_value = 1.0
    objective.quality.side_effect = lambda _geometry, error, _smoothing=1: error
    return RetainedFit(context, options, objective)


def candidate(owner, error, quality, *, count=0):
    return Candidate(
        owner.context.geometry,
        {},
        owner.context.before_image,
        error,
        quality,
        count,
    )


def test_rejected_exact_proposals_leave_every_published_field_unchanged():
    owner = retained()
    best = candidate(owner, 0.8, 0.8)
    assert owner.consider(best)
    for error in (0.8, 0.9, 1.0):
        assert not owner.consider(candidate(owner, error, 0.1))
        assert owner.best is best
        assert owner.least_difference == 0.8


def test_cleanup_error_allowance_stays_anchored_to_the_best_exact_match():
    owner = retained(cleanup=True)
    assert owner.consider(candidate(owner, 0.8, 0.8))
    assert owner.consider(candidate(owner, 0.809, 0.7))
    assert owner.best.error == 0.809
    assert owner.least_difference == 0.8
    best = owner.best
    # Returning a little error for a smoother outline never compounds the cap.
    assert not owner.consider(candidate(owner, 0.811, 0.6))
    assert not owner.consider(candidate(owner, 1.0, 0.1))
    assert owner.best is best


def test_cleanup_prefers_fewer_crossings_within_the_same_exact_error_cap():
    owner = retained(cleanup=True, crossed=True)
    assert owner.consider(candidate(owner, 0.8, 0.3, count=1))
    assert owner.consider(candidate(owner, 0.809, 0.9, count=0))
    assert owner.best.crossings == 0
    assert not owner.consider(candidate(owner, 0.7, 0.1, count=1))


def test_bridge_repair_requires_explicit_error_permission():
    owner = retained(cleanup=True)
    assert owner.consider(candidate(owner, 0.8, 0.8))
    bridge = candidate(owner, 0.805, 0.7)
    assert not owner.consider_bridge(bridge, 0, allow_error=False)
    assert owner.consider_bridge(bridge, 0, allow_error=True)
    assert owner.best is bridge
    assert owner.least_difference == 0.8
    best = owner.best
    assert not owner.consider_bridge(
        candidate(owner, 0.7, 0.1, count=1), 0, allow_error=True
    )
    assert not owner.consider_bridge(candidate(owner, 0.82, 0.1), 0, allow_error=True)
    assert owner.best is best


def test_selected_helpers_keep_optional_torch_lazy():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class NoTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise AssertionError("selected helpers loaded Torch eagerly")
                return None

        sys.meta_path.insert(0, NoTorch())
        from vectrify.refine import selected
        from vectrify.refine.selected_fit import (
            controls, objective, policy, observe, cleanup, stroke
        )
        assert "torch" not in sys.modules
        """
    )
    subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, PYTHONPATH="src:."),
        check=True,
        capture_output=True,
    )


def test_an_exact_edge_seed_does_not_start_or_stall_gradient_descent(monkeypatch):
    torch = pytest.importorskip("torch")
    from vectrify.refine.frozen import Paths
    from vectrify.refine.paths import parse_filled_cubics

    document = import_svg(SVG)
    geometry = document.geometry_for("p")
    shifted = replace(
        geometry,
        subpaths=tuple(
            replace(
                subpath,
                nodes=tuple(
                    replace(
                        node,
                        values=tuple(
                            value + 1.5 if index % 2 == 0 else value
                            for index, value in enumerate(node.values)
                        ),
                    )
                    for node in subpath.nodes
                ),
            )
            for subpath in geometry.subpaths
        ),
    )
    target = render_image(SVG.replace('id="p"', 'id="p" transform="translate(1.5 0)"'))
    options = FitOptions(steps=60, stall=0.001, snap=True, resolution=64)
    context = FitContext(document, SELECTION, target, options)
    seen = []

    def fitting(work, _target, **hooks):
        data = ET.fromstring(work)[0].get("d")
        assert data is not None
        paths = [
            [torch.tensor(c, dtype=torch.float32) for c in parse_filled_cubics(data)]
        ]
        assert float(paths[0][0][0, 0, 0]) == pytest.approx(
            (12 - context.crop[0]) * context.scale[0]
        )
        colors = torch.zeros((1, 3))
        for step, shift in ((10, 0.5), (20, 0.5), (30, 0)):
            for contour in paths[0]:
                contour[..., 0] += shift * context.scale[0]
            hooks["project_controls"](paths)
            seen.append(step)
            if not hooks["observe"](step, paths, colors):
                break
        return work

    monkeypatch.setattr("vectrify.refine.selected.fit_device", lambda: "cpu")
    monkeypatch.setattr(
        "vectrify.refine.snap.snap", lambda *_a, **_k: Paths({"p": shifted})
    )
    monkeypatch.setattr("vectrify.refine.paths.fit_filled_svg", fitting)
    result = fit_selected_path(document, SELECTION, target, options)
    assert seen == [10, 20, 30]
    assert result.after < result.before / 10
    assert result.steps == 30


@pytest.mark.parametrize("always_crossed", [False, True])
def test_crossing_rollback_keeps_coupled_knots_and_the_full_rollback_budget(
    always_crossed, monkeypatch
):
    pytest.importorskip("torch")
    from vectrify.refine.selected_fit.controls import FitControls
    from vectrify.refine.selected_fit.observe import CrossingRollback

    context = FitContext(import_svg(SVG), SELECTION, render_image(SVG), FitOptions())
    controls = FitControls(context, SELECTION, FitOptions(), "cpu", frozenset())
    rollback = CrossingRollback(context, controls, 2)
    groups = controls.parameters.coupled.cpu().numpy()
    closing = groups == groups[0]
    assert closing.sum() == 2
    coordinates = context.local + np.array([4, 0])
    seen = []

    def crossed(_geometry):
        seen.append(True)
        return (
            (1, {context.nodes[0].id})
            if always_crossed or len(seen) == 1
            else (0, set())
        )

    monkeypatch.setattr("vectrify.refine.selected_fit.observe.crossed_nodes", crossed)
    repaired, values, geometry = rollback.repair(coordinates)
    if always_crossed:
        assert len(seen) == 4
        assert np.array_equal(repaired, context.local)
        assert not values
    else:
        assert len(seen) == 2
        assert np.array_equal(
            repaired[closing], context.local[closing] + np.array([2, 0])
        )
        assert np.array_equal(repaired[~closing], coordinates[~closing])
    assert geometry.id == context.geometry.id
    assert [node.id for sub in geometry.subpaths for node in sub.nodes] == [
        node.id for node in context.nodes
    ]


def test_cancellation_observes_the_last_step_and_publishes_the_retained_candidate(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from threading import Event

    from vectrify.refine.paths import parse_filled_cubics

    stop = Event()
    target = render_image(SVG.replace('id="p"', 'id="p" transform="translate(1.5 0)"'))
    options = FitOptions(steps=60, snap=False, resolution=64, displacement=3)
    document = import_svg(SVG)
    context = FitContext(document, SELECTION, target, options)
    steps = []

    def fitting(work, _target, **hooks):
        data = ET.fromstring(work)[0].get("d")
        assert data is not None
        paths = [
            [torch.tensor(c, dtype=torch.float32) for c in parse_filled_cubics(data)]
        ]
        colors = torch.zeros((1, 3))
        for contour in paths[0]:
            contour[..., 0] += context.scale[0]
        hooks["project_controls"](paths)
        assert hooks["observe"](10, paths, colors)
        stop.set()
        for contour in paths[0]:
            contour[..., 1] += 2 * context.scale[1]
        hooks["project_controls"](paths)
        assert not hooks["observe"](11, paths, colors)
        return work

    monkeypatch.setattr("vectrify.refine.selected.fit_device", lambda: "cpu")
    monkeypatch.setattr("vectrify.refine.paths.fit_filled_svg", fitting)
    result = fit_selected_path(
        document,
        SELECTION,
        target,
        options,
        stop=stop,
        progress=lambda step, _message: steps.append(step),
    )
    assert steps == [0, 10, 11]
    assert result.steps == 11
    assert result.after < result.before
    for node in context.nodes:
        values = result.values.get(node.id, node.values)
        assert values[1::2] == node.values[1::2]
        assert values[::2] == tuple(value + 1 for value in node.values[::2])
