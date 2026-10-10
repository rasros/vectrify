"""Shared original face edges partition construction fields, never acceptance."""

from dataclasses import replace

import numpy as np
import pathops
import pytest

from vectrify.document import Editor, Geometry, Selection, import_svg
from vectrify.document.holes import reversed_subpath
from vectrify.document.join import curve_path, transformed_geometry
from vectrify.document.redraw import root_matrix
from vectrify.document.svg import parse_path
from vectrify.document.topology import inverse_matrix
from vectrify.refine.cel_plan import source_faces
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.ink_models import footprint
from vectrify.refine.cel_plan.ink_replace import identified
from vectrify.refine.cel_plan.line_fidelity import SourceBreaks
from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.source_spans import SourceSpan

OWNER = "M10 10 H24 V20 V48 L40 64 L56 48 V10 H60 V50 L40 70 L20 50 V20 H10 Z"
FIELD = "M20 20 V50 L40 70 L60 50 V10 H56 V48 L40 64 L24 48 V20 Z"
MATERIAL = "M56 48 L40 64 L40 40 L56 30 Z"
FACE = "M60 50 L40 70 L40 64 L56 48 Z"


def fixture(
    *,
    transform="matrix(1 0 0 1 0 0)",
    opacity=1,
    material=MATERIAL,
    fill_opacity=1,
    owner=OWNER,
    field=FIELD,
    points=None,
    reverse=False,
):
    document = import_svg(
        f'<svg width="160" height="160"><g transform="{transform}" '
        f'opacity="{opacity}"><path id="ink" d="{owner}" fill="#271b10"/>'
        '<path id="base" d="M0 0 H80 V80 H0 Z" fill="#b49150"/>'
        f'<path id="face" d="{material}" fill="#584022" '
        f'fill-opacity="{fill_opacity}"/></g></svg>'
    )
    if reverse:
        geometry = document.geometry_for("ink")
        editor = Editor(document, selection=Selection(whole_document=True))
        with editor.transaction("Reverse original owner commands") as tx:
            tx.replace_geometry(
                "ink",
                identified(
                    replace(
                        geometry,
                        subpaths=tuple(
                            reversed_subpath(sub) for sub in geometry.subpaths
                        ),
                    ),
                    "ink",
                ),
            )
        document = editor.snapshot.document
    if points is None:
        points = np.linspace((54.4, 52.6), (47.2, 59.8), 12)
    if transform != "matrix(1 0 0 1 0 0)":
        points = points @ np.array(((0.8, 0.6), (-0.6, 0.8))) + (60, 20)
    observed = SourceBreaks(
        points, np.ones(len(points), bool), np.zeros(len(points), bool), points
    )
    full = parse_path(field)
    span = SourceSpan(full, document.geometry_for("ink"), points)
    former = transformed_geometry(
        footprint(Geometry("former", (fitted(points, 0.25).contour,)), 0.5, cap="butt"),
        inverse_matrix(root_matrix(document, "ink")),
    )
    return document, span, observed, former


@pytest.mark.parametrize(
    ("transform", "opacity"),
    [("matrix(1 0 0 1 0 0)", 1), ("matrix(.8 .6 -.6 .8 60 20)", 0.6)],
)
def test_shared_original_edge_selects_complete_face_without_rewriting_field_prefix(
    transform, opacity
):
    document, span, observed, former = fixture(transform=transform, opacity=opacity)
    old = document.geometry_for("face")
    results = source_faces.discover(
        document, "ink", span, ((observed, former),), ("face",), "base", Work.start(10)
    )
    expected = curve_path(parse_path(FACE))
    matches = [
        result
        for result in results
        if not pathops.op(curve_path(result.field), expected, pathops.PathOp.XOR).area
    ]
    assert matches
    result = matches[0]
    assert result.material == "face"
    assert result.source_index == 0
    assert result.shared_edges
    assert result.remainder.subpaths[: len(span.field.subpaths)] == span.field.subpaths
    assert not pathops.op(
        curve_path(result.field), curve_path(old), pathops.PathOp.INTERSECTION
    ).area
    assert not pathops.op(
        curve_path(result.field),
        curve_path(result.remainder),
        pathops.PathOp.INTERSECTION,
    ).area
    combined = pathops.op(
        curve_path(result.field), curve_path(result.remainder), pathops.PathOp.UNION
    )
    assert not pathops.op(combined, curve_path(span.field), pathops.PathOp.XOR).area
    assert document.geometry_for("face") == old


@pytest.mark.parametrize(
    ("material", "fill_opacity"),
    [
        ("M55 47 L39 63 L39 39 L55 29 Z", 1),
        ("M60 50 L40 70 L40 64 L56 48 Z", 1),
        (MATERIAL, 0.6),
    ],
)
def test_nearby_edge_overlap_and_unsupported_paint_do_not_supply_shared_face(
    material, fill_opacity
):
    document, span, observed, former = fixture(
        material=material, fill_opacity=fill_opacity
    )
    assert not source_faces.discover(
        document, "ink", span, ((observed, former),), ("face",), "base", Work.start(10)
    )


def test_source_gaps_and_bounded_inputs_remain_excluded(monkeypatch):
    document, span, observed, former = fixture()
    gaps = observed.gaps.copy()
    gaps[5] = True
    assert not source_faces.discover(
        document,
        "ink",
        span,
        ((replace(observed, gaps=gaps), former),),
        ("face",),
        "base",
        Work.start(10),
    )
    with pytest.raises(StageInterruptedError):
        source_faces.discover(
            document,
            "ink",
            span,
            ((observed, former),),
            ("face",),
            "base",
            Work.start(0),
        )
    monkeypatch.setattr(source_faces, "MAX_OBSERVATIONS", 0)
    assert not source_faces.discover(
        document, "ink", span, ((observed, former),), ("face",), "base", Work.start(10)
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "transform", ["matrix(1 0 0 1 0 0)", "matrix(.8 .6 -.6 .8 60 20)"]
)
def test_curved_adjoining_face_copies_complete_original_cubics(reverse, transform):
    owner = (
        "M10 10 H24 V20 V48 L40 64 C44 62 52 54 56 48 V10 H60 V50 "
        "C55 58 46 66 40 70 L20 50 V20 H10 Z"
    )
    full = (
        "M20 20 V50 L40 70 C46 66 55 58 60 50 V10 H56 V48 "
        "C52 54 44 62 40 64 L24 48 V20 Z"
    )
    material = "M56 48 C52 54 44 62 40 64 L40 40 L56 30 Z"
    expected = parse_path("M60 50 C55 58 46 66 40 70 L40 64 C44 62 52 54 56 48 Z")
    t = np.linspace(0.2, 0.6, 12)[:, None]
    points = (
        (1 - t) ** 3 * [58, 49]
        + 3 * (1 - t) ** 2 * t * [53.5, 56]
        + 3 * (1 - t) * t**2 * [45, 64]
        + t**3 * [40, 67]
    )
    document, span, observed, former = fixture(
        owner=owner,
        field=full,
        material=material,
        points=points,
        reverse=reverse,
        transform=transform,
    )
    results = source_faces.discover(
        document, "ink", span, ((observed, former),), ("face",), "base", Work.start(10)
    )
    matches = [
        result
        for result in results
        if not pathops.op(
            curve_path(result.field), curve_path(expected), pathops.PathOp.XOR
        ).area
    ]
    assert matches
    assert source_faces._edges(matches[0].field) == source_faces._edges(expected)
    assert (
        matches[0].remainder.subpaths[: len(span.field.subpaths)] == span.field.subpaths
    )


def test_face_results_are_bounded_and_deterministic(monkeypatch):
    document, span, observed, former = fixture()
    monkeypatch.setattr(source_faces, "MAX_FACES", 1)
    first = source_faces.discover(
        document,
        "ink",
        span,
        ((observed, former),),
        ("face", "face"),
        "base",
        Work.start(10),
    )
    second = source_faces.discover(
        document, "ink", span, ((observed, former),), ("face",), "base", Work.start(10)
    )
    assert len(first) == len(second) == 1
    assert first[0] == second[0]
    monkeypatch.setattr(source_faces, "MAX_MATERIALS", 0)
    assert not source_faces.discover(
        document, "ink", span, ((observed, former),), ("face",), "base", Work.start(10)
    )
