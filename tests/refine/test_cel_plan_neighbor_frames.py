"""Paint continuation retains its own frame under differently transformed ink."""

from dataclasses import replace

import numpy as np
import pytest

from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_ink_contours import rims, ring
from tests.refine.test_cel_plan_ink_replace import source
from vectrify.document import Editor, Selection, export_svg, load_project, save_project
from vectrify.document.join import transformed_geometry
from vectrify.document.model import paint_server
from vectrify.document.paint import GradientStop, LinearGradient
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.ink_replace import InkReplacement
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.score import render

FRAMES = [(1, 0, 0, 1, 14, -9), (1.2, 0.15, 0.2, 1, 6, -10), (-1, 0, 0.2, 1, 100, 0)]


def refreshed(frontier, state, document):
    svg = export_svg(document)
    return replace(
        state,
        document=document,
        svg=svg,
        snapshot=LocalPolicy(frontier.policy).start(svg, frontier.policy.evaluate(svg)),
    )


def reexpress(frontier, state, oid, matrix):
    editor = Editor(state.document, selection=Selection(whole_document=True))
    inverse = inverse_matrix(matrix)
    with editor.transaction("Reexpress one material's private frame") as tx:
        tx.replace_geometry(
            oid, transformed_geometry(state.document.geometry_for(oid), inverse)
        )
        tx.set_attributes(oid, {"transform": f"matrix({' '.join(map(str, matrix))})"})
        server = paint_server(state.document.element(oid).get("fill"))
        if server:
            tx.set_attributes(
                server, {"gradientTransform": f"matrix({' '.join(map(str, inverse))})"}
            )
    result = refreshed(frontier, state, editor.snapshot.document)
    np.testing.assert_allclose(
        render(result.svg, frontier.policy.truth.shape[1::-1]),
        render(state.svg, frontier.policy.truth.shape[1::-1]),
        atol=1 / 255,
        rtol=0,
    )
    return result


@pytest.mark.parametrize("matrix", FRAMES)
@pytest.mark.parametrize("oid", ["cel-fill-0", "cel-fill-1"])
def test_closed_rim_and_compact_underpaint_keep_independent_neighbor_frames(
    matrix, oid
):
    evidence = ring(alpha=128)
    frontier, state, options = prepared(evidence, layers=True)
    baseline = rims(InkReplacement(evidence, build(evidence), options), state)[0]
    state = reexpress(frontier, state, oid, matrix)
    factory = InkReplacement(evidence, build(evidence), options)
    edits = rims(factory, state)
    assert edits, factory.restoration_rejections
    edit = edits[0]
    assert edit.details["ink_replacement"]["underpaint_model"] == "ellipse"
    assert edit.document.element(oid).get("transform") == state.document.element(
        oid
    ).get("transform")
    actual = render(export_svg(edit.document), evidence.source_size)
    expected = render(export_svg(baseline.document), evidence.source_size)
    np.testing.assert_allclose(actual, expected, atol=1 / 255, rtol=0)
    np.testing.assert_array_equal(actual[..., 3], expected[..., 3])
    full = frontier.policy.evaluate(export_svg(edit.document))
    assert full.valid
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, export_svg(edit.document), edit.bounds, full.structure
    )
    assert local.canvas.matches(actual)
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    assert edit.partition.follows(state.partition)
    restored, _ = load_project(save_project(edit.document))
    np.testing.assert_array_equal(
        render(export_svg(restored), evidence.source_size), actual
    )


@pytest.mark.parametrize("matrix", FRAMES)
def test_stroke_underlays_preserve_transformed_gradient_paint_coordinates(matrix):
    evidence = source(alpha=64)
    frontier, state, options = prepared(evidence, layers=True)
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Private neighboring gradient") as tx:
        tx.set_fill(
            "cel-fill-0",
            LinearGradient(
                (0, 0),
                (96, 0),
                (GradientStop(0, "#d8b464"), GradientStop(1, "#e0b464")),
            ),
        )
    state = refreshed(frontier, state, editor.snapshot.document)
    baseline = next(
        InkReplacement(evidence, build(evidence), options)(state, Work.start(10))
    )
    assert baseline.parameters[0] == "stroke"
    state = reexpress(frontier, state, "cel-fill-0", matrix)
    factory = InkReplacement(evidence, build(evidence), options)
    edit = next(factory(state, Work.start(10)))
    assert edit.parameters[0] == "stroke"
    server = paint_server(state.document.element("cel-fill-0").get("fill"))
    assert server is not None
    assert edit.document.element(server) == state.document.element(server)
    actual = render(export_svg(edit.document), evidence.source_size)
    expected = render(export_svg(baseline.document), evidence.source_size)
    np.testing.assert_allclose(actual, expected, atol=1 / 255, rtol=0)
    np.testing.assert_array_equal(actual[..., 3], expected[..., 3])
    full = frontier.policy.evaluate(export_svg(edit.document))
    assert full.valid
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, export_svg(edit.document), edit.bounds, full.structure
    )
    assert local.canvas.matches(actual)
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    assert edit.partition.follows(state.partition)
