"""Literal shared endpoints require original source topology and real ink."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from vectrify.document import Editor, Selection, export_svg, import_svg, save_project
from vectrify.document.join import transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.refine.cel_plan import source_junctions
from vectrify.refine.cel_plan.evidence import collect
from vectrify.refine.cel_plan.line_fidelity import SourceLineGuard, SourceProfile
from vectrify.refine.cel_plan.model import Options, StageInterruptedError, Work
from vectrify.refine.cel_plan.score import render
from vectrify.refine.cel_plan.source_junctions import SourceJunctions


def fixture(*, alpha=1, rotated=False, gap=False, crossing=False):
    frame = 'transform="matrix(.8 .6 -.6 .8 42 0)"' if rotated else ""
    source = (
        f'<svg width="96" height="96"><g opacity="{alpha}">'
        '<path id="bg" d="M0 0H96V96H0Z" fill="#ad8665"/>'
        f'<g {frame}><path id="ink" d="M28.5 20.5V76.5" fill="none" '
        'stroke="#121008" stroke-width="3" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
        + (
            '<path d="M20 51.5H40" fill="none" stroke="#121008" stroke-width="3"/>'
            if crossing
            else ""
        )
        + ('<path d="M24 46H33V51H24Z" fill="#ad8665"/>' if gap else "")
        + "</g></g></svg>"
    )
    pixels = render(source, (96, 96))
    evidence = collect(
        Image.fromarray(np.rint(pixels * 255).astype(np.uint8)),
        None,
        Options(),
        Work.start(10),
    )
    document = import_svg(
        source.replace(
            'id="ink" d="M28.5 20.5V76.5"', 'id="host" d="M28.5 20.5V48.5"'
        ).replace(
            "</g></g>",
            '<path id="tail" d="M29.25 48.5C29.25 58 29.25 68 28.5 76.5" '
            'fill="none" stroke="#121008" stroke-width="3" '
            'stroke-linecap="round" stroke-linejoin="round"/></g></g>',
        )
    )
    matrix = root_matrix(document, "host")
    a, b, c, d, e, f = matrix
    profiles = []
    for low, high in ((20.5, 48.5), (48.5, 76.5)):
        points = np.column_stack((np.full(29, 28.5), np.linspace(low, high, 29)))
        points = points @ np.array([[a, b], [c, d]]) + (e, f)
        profiles.append(replace(SourceProfile.at(points, 3), component=("source", 1)))
    guard = SourceLineGuard(evidence.rgba, profiles)
    return evidence, guard, document


@pytest.mark.parametrize("alpha", [1, 0.6])
@pytest.mark.parametrize("rotated", [False, True])
def test_shared_endpoint_preserves_nodes_controls_paint_frame_and_real_source(
    alpha, rotated
):
    evidence, guard, document = fixture(alpha=alpha, rotated=rotated)
    result = SourceJunctions(evidence, guard).connect(
        document, ("host", "tail"), (0, 0, 96, 96), Work.start(10)
    )
    assert result is not None
    candidate, witnesses = result
    assert len(witnesses) == 1
    witness = witnesses[0]
    assert witness["id"] == "tail"
    assert witness["raw_source_ink"]
    assert witness["native_body_gaps_preserved"]
    assert witness["holders"] == ("host",)
    old = document.geometry_for("tail")
    new = candidate.geometry_for("tail")
    assert old.subpaths[0].nodes[1:] == new.subpaths[0].nodes[1:]
    assert new.subpaths[0].nodes[0].id == old.subpaths[0].nodes[0].id
    assert candidate.element("tail") == document.element("tail")
    assert candidate.geometry_for("host") == document.geometry_for("host")
    assert candidate.geometry_for("bg") == document.geometry_for("bg")
    for oid in ("tail", "host"):
        geometry = transformed_geometry(
            candidate.geometry_for(oid), root_matrix(candidate, oid)
        )
        point = geometry.subpaths[0].nodes[0 if oid == "tail" else -1].endpoint
        np.testing.assert_allclose(point, witness["to"], rtol=0, atol=1e-8)
    comparison = guard.compare(
        render(export_svg(document), evidence.source_size),
        render(export_svg(candidate), evidence.source_size),
    )
    assert not comparison["rejections"]


@pytest.mark.parametrize(
    "kind", ["duplicate", "different_component", "gap", "no_holder", "paint", "pins"]
)
def test_missing_topology_or_real_ink_cannot_authorize_a_nearby_connection(kind):
    evidence, guard, document = fixture(gap=kind == "gap")
    profiles = guard.original_profiles()
    ids = ("host", "tail")
    if kind == "duplicate":
        guard = SourceLineGuard(evidence.rgba, (profiles[0], replace(profiles[0])))
    if kind == "different_component":
        guard = SourceLineGuard(
            evidence.rgba, (profiles[0], replace(profiles[1], component=("source", 2)))
        )
    if kind == "no_holder":
        ids = ("tail",)
    if kind in {"paint", "pins"}:
        editor = Editor(document, selection=Selection(whole_document=True))
        with editor.transaction("Excluded continuation") as tx:
            if kind == "paint":
                tx.set_attributes("tail", {"stroke": "#704020"})
            else:
                geometry = document.geometry_for("tail")
                sub = geometry.subpaths[0]
                tx.replace_geometry(
                    "tail",
                    replace(
                        geometry,
                        subpaths=(
                            replace(
                                sub,
                                nodes=(
                                    replace(sub.nodes[0], pinned=True),
                                    *sub.nodes[1:],
                                ),
                            ),
                        ),
                    ),
                )
        document = editor.snapshot.document
    assert (
        SourceJunctions(evidence, guard).connect(
            document, ids, (0, 0, 96, 96), Work.start(10)
        )
        is None
    )


def test_copied_source_topology_survives_mutated_identity_coordinates():
    evidence, guard, document = fixture()
    for profile in guard.original_profiles():
        profile.points.flags.writeable = True
        profile.points[:] += 10
    result = SourceJunctions(evidence, guard).connect(
        document, ("host", "tail"), (0, 0, 96, 96), Work.start(10)
    )
    assert result is not None


def test_an_original_physical_terminal_cannot_snap_to_another_nearby_point():
    evidence, guard, document = fixture()
    points = np.column_stack((np.full(23, 29.25), np.linspace(48.5, 70.5, 23)))
    profile = replace(SourceProfile.at(points, 3), component=("source", 1))
    guard = SourceLineGuard(evidence.rgba, (*guard.original_profiles(), profile))
    assert guard.source_breaks(profile) is not None
    assert (
        SourceJunctions(evidence, guard).connect(
            document, ("host", "tail"), (0, 0, 96, 96), Work.start(10)
        )
        is None
    )


@pytest.mark.parametrize(
    "bound", ["MAX_PROFILES", "MAX_PATHS", "MAX_NODES", "MAX_PORTS"]
)
def test_bounds_exclude_before_publishing(monkeypatch, bound):
    evidence, guard, document = fixture()
    monkeypatch.setattr(source_junctions, bound, 0)
    assert (
        SourceJunctions(evidence, guard).connect(
            document, ("host", "tail"), (0, 0, 96, 96), Work.start(10)
        )
        is None
    )


def test_local_source_interval_can_identify_a_continuation_beside_a_crossing():
    evidence, guard, document = fixture(crossing=True)
    observed = guard.source_breaks(guard.original_profiles()[1])
    at = int(np.linalg.norm(observed.points - (28.5, 51.5), axis=1).argmin())
    assert not observed.qualified[at]
    result = SourceJunctions(evidence, guard).connect(
        document, ("host", "tail"), (0, 0, 96, 96), Work.start(10)
    )
    assert result is not None
    assert result[1][0]["raw_source_ink"]


def test_ambiguous_outgoing_source_chains_cannot_authorize_a_snap():
    evidence, guard, document = fixture()
    points = np.column_stack((np.linspace(28.5, 29, 29), np.linspace(48.5, 76.5, 29)))
    other = replace(SourceProfile.at(points, 3), component=("source", 1))
    guard = SourceLineGuard(evidence.rgba, (*guard.original_profiles(), other))
    assert (
        SourceJunctions(evidence, guard).connect(
            document, ("host", "tail"), (0, 0, 96, 96), Work.start(10)
        )
        is None
    )


def test_native_body_stop_cannot_publish_a_partial_connection(monkeypatch):
    evidence, guard, document = fixture()
    saved = save_project(document)
    work = Work.start(10)
    original = source_junctions._native_raster

    def stopped(*args):
        raster = original(*args)
        work.stop.set()
        return raster

    monkeypatch.setattr(source_junctions, "_native_raster", stopped)
    with pytest.raises(StageInterruptedError):
        SourceJunctions(evidence, guard).connect(
            document, ("host", "tail"), (0, 0, 96, 96), work
        )
    assert save_project(document) == saved
