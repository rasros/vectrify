"""Complete curve primitives supply bounded fields, never source acceptance."""

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
from vectrify.refine.cel_plan import source_spans
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.ink_models import footprint
from vectrify.refine.cel_plan.ink_replace import identified
from vectrify.refine.cel_plan.line_fidelity import SourceBreaks
from vectrify.refine.cel_plan.model import StageInterruptedError, Work

OWNER = "M10 10 C25 10 40 25 50 30 H90 V70 H50 V34 C40 29 25 14 10 14 Z"
FIELD = "M10 10 C25 10 40 25 50 30 V34 C40 29 25 14 10 14 Z"


def fixture(*, reverse=False, rotated=False, transform="matrix(1 0 0 1 0 0)", alpha=1):
    document = import_svg(
        '<svg width="192" height="192"><g '
        f'opacity="{alpha}"><path id="ink" d="{OWNER}" '
        f'fill="#121008" transform="{transform}"/></g></svg>'
    )
    original = document.geometry_for("ink")
    sub = original.subpaths[0]
    if reverse:
        sub = reversed_subpath(sub)
    if rotated:
        # Move the SVG start into the opposite rail; its old Z becomes an
        # explicit edge and a rail now crosses that serialization boundary.
        nodes = sub.nodes
        edges = (*nodes[1:], replace(nodes[0], command="L"))
        at = 2
        ordered = (*edges[at:], *edges[:at])
        sub = replace(
            sub,
            nodes=(
                replace(nodes[at], command="M", values=nodes[at].endpoint),
                *ordered,
            ),
        )
    editor = Editor(document, selection=Selection(whole_document=True))
    with editor.transaction("Prepare primitive discovery fixture") as tx:
        tx.replace_geometry(
            "ink", identified(replace(original, subpaths=(sub,)), "fixture-ink")
        )
    document = editor.snapshot.document
    t = np.linspace(0.2, 0.8, 25)[:, None]
    points = (
        (1 - t) ** 3 * [10, 12]
        + 3 * (1 - t) ** 2 * t * [25, 12]
        + 3 * (1 - t) * t**2 * [40, 27]
        + t**3 * [50, 32]
    )
    a, b, c, d, e, f = root_matrix(document, "ink")
    points = points @ np.array([[a, b], [c, d]]) + [e, f]
    observed = SourceBreaks(
        points, np.ones(len(points), bool), np.zeros(len(points), bool), points
    )
    former = transformed_geometry(
        footprint(Geometry("former", (fitted(points, 0.25).contour,)), 0.5, cap="butt"),
        inverse_matrix(root_matrix(document, "ink")),
    )
    return document, observed, former


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize(
    ("transform", "alpha"),
    [("matrix(1 0 0 1 0 0)", 1), ("matrix(.8 .6 -.6 .8 100 20)", 0.6)],
)
def test_discovery_copies_complete_original_curves_across_frames_and_svg_starts(
    reverse, rotated, transform, alpha
):
    document, observed, former = fixture(
        reverse=reverse, rotated=rotated, transform=transform, alpha=alpha
    )
    spans = source_spans.discover(document, "ink", observed, former, Work.start(10))
    expected = curve_path(parse_path(FIELD))
    matches = [
        span
        for span in spans
        if not pathops.op(curve_path(span.field), expected, pathops.PathOp.XOR).area
    ]
    assert matches
    old = document.geometry_for("ink")
    original_curves = {
        node.values for sub in old.subpaths for node in sub.nodes if node.command == "C"
    }
    for span in matches:
        assert span.retained.subpaths[: len(old.subpaths)] == old.subpaths
        assert {
            n.values for n in span.field.subpaths[0].nodes if n.command == "C"
        } == original_curves
        assert not pathops.op(
            curve_path(former), curve_path(span.field), pathops.PathOp.DIFFERENCE
        ).area
        difference = pathops.op(
            curve_path(old), curve_path(span.field), pathops.PathOp.DIFFERENCE
        )
        assert not pathops.op(
            curve_path(span.retained), difference, pathops.PathOp.XOR
        ).area
        assert not pathops.op(
            curve_path(span.retained),
            curve_path(span.field),
            pathops.PathOp.INTERSECTION,
        ).area
        assert span.guide.shape == (128, 2)
        assert np.isfinite(span.guide).all()
        assert not span.guide.flags.writeable


def test_gap_fragment_low_support_and_broad_source_field_remain_excluded():
    document, observed, field = fixture()
    gaps = observed.gaps.copy()
    gaps[len(gaps) // 2] = True
    assert not source_spans.discover(
        document, "ink", replace(observed, gaps=gaps), field, Work.start(10)
    )
    assert not source_spans.discover(
        document, "ink", replace(observed, anchors=None), field, Work.start(10)
    )
    assert not source_spans.discover(
        document,
        "ink",
        replace(observed, qualified=np.zeros(len(gaps), bool)),
        field,
        Work.start(10),
    )
    broad = parse_path("M55 35 H85 V65 H55 Z")
    assert not source_spans.discover(document, "ink", observed, broad, Work.start(10))
    outside = parse_path("M0 0 H5 V5 H0 Z")
    assert not source_spans.discover(document, "ink", observed, outside, Work.start(10))


def test_discovery_has_bounded_chords_outputs_and_interruptible_work(monkeypatch):
    document, observed, field = fixture()
    with pytest.raises(StageInterruptedError):
        source_spans.discover(document, "ink", observed, field, Work.start(0))
    monkeypatch.setattr(source_spans, "MAX_FIELDS", 1)
    first = source_spans.discover(document, "ink", observed, field, Work.start(10))
    second = source_spans.discover(document, "ink", observed, field, Work.start(10))
    assert len(first) == len(second) == 1
    assert first[0].field == second[0].field
    monkeypatch.setattr(source_spans, "MAX_CHORDS", 0)
    assert not source_spans.discover(document, "ink", observed, field, Work.start(10))


def test_discovery_follows_both_sides_of_a_u_attached_to_a_broad_pad():
    owner = "M10 10 H24 V20 V48 L40 64 L56 48 V10 H60 V50 L40 70 L20 50 V20 H10 Z"
    field = "M20 20 V50 L40 70 L60 50 V10 H56 V48 L40 64 L24 48 V20 Z"
    document = import_svg(
        f'<svg width="80" height="80"><path id="ink" d="{owner}"/></svg>'
    )
    points = np.column_stack((np.full(12, 22), np.linspace(30, 40, 12)))
    observed = SourceBreaks(
        points, np.ones(len(points), bool), np.zeros(len(points), bool), points
    )
    former = parse_path("M21.5 30 H22.5 V40 H21.5 Z")
    expected = curve_path(parse_path(field))
    spans = source_spans.discover(document, "ink", observed, former, Work.start(10))
    matches = [
        span
        for span in spans
        if not pathops.op(curve_path(span.field), expected, pathops.PathOp.XOR).area
    ]
    assert matches
    guide = matches[0].guide
    assert guide[:, 0].min() < 25
    assert guide[:, 0].max() > 55
    assert guide[:, 1].max() > 65
    assert abs(guide[0, 0] - guide[-1, 0]) > 30
