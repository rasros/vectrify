"""Shared shade edges fit subpixel geometry and real paint compositing."""

from dataclasses import replace

import numpy as np
import pathops
import pytest

from tests.refine.test_cel_plan_families import prepared
from tests.refine.test_cel_plan_piecewise_surfaces import indented_family, step
from vectrify.document import Editor, Selection, export_svg, load_project, save_project
from vectrify.document.join import curve_path, path_geometry
from vectrify.document.svg import parse_path
from vectrify.refine.cel_plan import shade_edges
from vectrify.refine.cel_plan.families import Families
from vectrify.refine.cel_plan.graph import build
from vectrify.refine.cel_plan.local import LocalPolicy
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.ownership import Partition
from vectrify.refine.cel_plan.piecewise_surfaces import PiecewiseSurfaces
from vectrify.refine.cel_plan.proposals import Operators
from vectrify.refine.cel_plan.score import render


@pytest.mark.parametrize("angle", [0, 0.07, 0.21, 0.67, 1.91, 2.72])
def test_pixel_coverage_matches_independent_square_supersampling(angle):
    normal = np.array((np.cos(angle), np.sin(angle)))
    centers = np.column_stack((np.linspace(-0.8, 0.8, 7), np.zeros(7)))
    y, x = np.mgrid[:400, :400]
    square = np.column_stack(
        ((x.ravel() + 0.5) / 400 - 0.5, (y.ravel() + 0.5) / 400 - 0.5)
    )
    sampled = np.array([np.mean((square + p) @ normal < 0) for p in centers])
    actual = shade_edges.coverage(centers, normal, 0)
    np.testing.assert_allclose(actual, sampled, atol=0.0015)
    # The same physical line in a translated/scaled algebraic frame agrees.
    np.testing.assert_allclose(
        shade_edges.coverage(
            centers + np.array((1400, -900)),
            normal * 3,
            float(np.array((1400, -900)) @ normal) * 3,
        ),
        actual,
        atol=1e-11,
    )


def raster_edge(alpha, angle, *, hole):
    """Independent SVG render supplies colors, alpha and mixed edge samples."""
    evidence = step(alpha, hole=hole)
    normal = np.array((np.cos(angle), np.sin(angle)))
    tangent = np.array((-normal[1], normal[0]))
    rho = 45.37
    origin = normal * rho
    points = [
        origin - tangent * 200,
        origin + tangent * 200,
        origin + tangent * 200 + normal * 200,
        origin - tangent * 200 + normal * 200,
    ]
    shade = "M" + "L".join(f"{x} {y}" for x, y in points) + "Z"
    exterior = "M8 8H88V56H8Z" + ("M40 24V40H56V24Z" if hole else "")
    shade = path_geometry(
        pathops.op(
            curve_path(parse_path(shade)),
            curve_path(parse_path(exterior)),
            pathops.PathOp.INTERSECTION,
        )
    ).path_data()
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="64" '
        f'viewBox="0 0 96 64"><g opacity="{alpha / 255}">'
        f'<path d="{exterior}" fill="#20643c" fill-rule="nonzero"/>'
        f'<path d="{shade}" fill="#b4643c"/></g></svg>'
    )
    rgba = render(svg, (384, 256)).reshape(64, 4, 96, 4, 4).mean(axis=(1, 3))
    target = np.where(evidence.empty[..., None], 255, rgba[..., :3] * 255)
    return (
        replace(
            evidence,
            rgba=rgba,
            target=target,
            smooth=target,
            coarse=target,
            opacity=rgba[..., 3],
        ),
        normal,
        rho,
    )


@pytest.mark.parametrize("alpha", [255, 128])
@pytest.mark.parametrize("hole", [False, True])
@pytest.mark.parametrize("angle", [0.07, 0.21, 0.67])
def test_source_fit_compacts_fragments_and_avoids_shared_edge_core_seam(
    alpha, hole, angle
):
    evidence, expected_normal, expected_rho = raster_edge(alpha, angle, hole=hole)
    frontier, state, options = prepared(evidence, layers=True)
    graph = build(evidence)
    factory = PiecewiseSurfaces(Families(evidence, graph, options), options)
    edits = list(factory(state, Work.start(10)))
    candidates = [p for p in edits if p.parameters[-1] == "base-shade"]
    assert candidates
    edit = min(
        candidates,
        key=lambda p: frontier.policy.evaluate(export_svg(p.document)).visual,
    )
    np.testing.assert_allclose(edit.parameters[1], expected_normal, atol=0.005)
    assert edit.parameters[2] == pytest.approx(expected_rho, abs=0.15)
    full_svg = export_svg(edit.document)
    full = frontier.policy.evaluate(full_svg)
    assert full.valid
    assert full.structure["nodes"] < state.snapshot.evaluation.structure["nodes"]
    assert full.structure["nodes"] <= (30 if hole else 18)
    assert len([s for s in edit.partition.surfaces if s.role != "underlay"]) == 2
    assert full.terms["color"] < state.snapshot.evaluation.terms["color"]
    actual = render(full_svg, evidence.source_size)
    np.testing.assert_array_equal(
        actual[..., 3], render(state.svg, evidence.source_size)[..., 3]
    )
    if hole:
        assert actual[25:39, 41:55, 3].max() == 0
    Operators(evidence, graph, options).validate_partition(
        edit.partition, Work.start(10)
    )
    base = next(
        s for s in edit.partition.surfaces if s.role != "underlay" and s.covered
    )
    shade = next(
        s for s in edit.partition.surfaces if s.role != "underlay" and s.id != base.id
    )
    assert base.covered == shade.members
    assert not set(base.members).intersection(shade.members)
    # Remove just the hidden base support. Adjoining opaque fills expose an
    # unrelated core color at the antialiased shared edge despite exact union.
    adjoining = path_geometry(
        pathops.op(
            curve_path(edit.document.geometry_for(base.id)),
            curve_path(edit.document.geometry_for(shade.id)),
            pathops.PathOp.DIFFERENCE,
        )
    )
    editor = Editor(edit.document, selection=Selection(whole_document=True))
    with editor.transaction("Compare adjoining shade fills") as tx:
        tx.replace_geometry(base.id, adjoining)
    separated = frontier.policy.evaluate(export_svg(editor.snapshot.document))
    assert full.terms["color"] < separated.terms["color"] * 0.5
    local = LocalPolicy(frontier.policy).update(
        state.snapshot, full_svg, edit.bounds, full.structure
    )
    assert local.canvas.matches(actual)
    assert local.evaluation.terms == pytest.approx(full.terms, abs=2e-7)
    reloaded, _ = load_project(save_project(edit.document))
    Partition.from_metadata(edit.partition.metadata()).validate(reloaded)
    np.testing.assert_array_equal(
        render(export_svg(reloaded), evidence.source_size), actual
    )


def test_partial_actual_core_cannot_authorize_opaque_material_fit():
    evidence, _, _ = raster_edge(128, 0.21, hole=False)
    _, state, options = prepared(evidence, layers=True)
    core = next(s for s in state.partition.surfaces if s.role == "underlay")
    editor = Editor(state.document, selection=Selection(whole_document=True))
    with editor.transaction("Make actual core translucent") as tx:
        tx.set_attributes(core.id, {"fill-opacity": "0.5"})
    changed = replace(state, document=editor.snapshot.document)
    factory = PiecewiseSurfaces(Families(evidence, build(evidence), options), options)
    assert not any(
        p.parameters[-1] == "base-shade" for p in factory(changed, Work.start(10))
    )
    assert factory.diagnostics["shade_core_exclusions"] > 0


@pytest.mark.parametrize("alpha", [255, 128])
def test_independent_exterior_screen_accepts_a_valid_mixed_shared_edge(alpha):
    evidence, _, _ = raster_edge(alpha, 0.21, hole=False)
    frontier, state, options = prepared(evidence, layers=True)
    state, _ = indented_family(state)
    factory = PiecewiseSurfaces(Families(evidence, build(evidence), options), options)
    candidates = [
        p
        for p in factory(state, Work.start(10))
        if p.parameters[-1] == "base-shade" and p.parameters[0].startswith("supported-")
    ]
    assert candidates
    edit = candidates[0]
    assert frontier.policy.evaluate(export_svg(edit.document)).valid
    assert factory.diagnostics["supported_boundaries"] > 0
    assert factory.diagnostics["supported_boundary_pixels"] > 0
    assert edit.partition.follows(state.partition)
    Operators(evidence, build(evidence), options).validate_partition(
        edit.partition, Work.start(10)
    )
    np.testing.assert_array_equal(
        render(export_svg(edit.document), evidence.source_size)[..., 3],
        render(export_svg(state.document), evidence.source_size)[..., 3],
    )


def test_stop_during_subpixel_solve_does_not_publish_a_model(monkeypatch):
    evidence, normal, rho = raster_edge(128, 0.21, hole=False)
    y, x = np.nonzero(~evidence.empty)
    xy = np.column_stack((x + 0.5, y + 0.5))
    work = Work.start(10)
    original = shade_edges.least_squares

    def stop(*args, **kwargs):
        work.stop.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(shade_edges, "least_squares", stop)
    diagnostics = {
        "shade_fit_evaluations": 0,
        "shade_fit_exclusions": 0,
        "shade_fits": 0,
    }
    assert (
        shade_edges.refine(
            xy,
            evidence.rgba[y, x],
            normal,
            rho,
            work,
            gradients=True,
            diagnostics=diagnostics,
        )
        is None
    )
    assert diagnostics["shade_fits"] == 0
