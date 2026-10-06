"""A shared native order proof covers all moved marks without hiding overlaps."""

import numpy as np
import pytest

from vectrify.document import export_svg, import_svg
from vectrify.refine.cel_plan import layer_order
from vectrify.refine.cel_plan.layer_order import ordered
from vectrify.refine.cel_plan.model import Work
from vectrify.refine.cel_plan.score import render


def scene(*, overlap=False, outside_mark=False):
    marks = tuple(f"mark-{i}" for i in range(20))
    drawing = (
        '<svg width="110" height="110"><g transform="translate(3 4)" opacity="0.5">'
    )
    drawing += '<path id="outer" fill="#204080" d="M30 30H70V70H30Z"/>'
    for i, oid in enumerate(marks):
        x, y = (25, 25) if outside_mark and i == 19 else (40, 32 + i)
        drawing += f'<path id="{oid}" fill="#208040" d="M{x} {y}h2v1h-2Z"/>'
    crossing = " M44 44h1v1h-1Z" if overlap else ""
    drawing += (
        '<path id="ring" fill="#000" fill-rule="evenodd" '
        f'd="M20 20H80V80H20Z M29 29H71V71H29Z{crossing}"/>'
        '<path id="base" fill="#a04020" d="M5 5H15V15H5Z"/>'
        "</g></svg>"
    )
    return import_svg(drawing), marks


def diagnostics():
    return {"order_proofs": 0, "order_proof_limits": 0}


def test_one_group_proof_is_reused_past_the_budget_without_changing_pixels(monkeypatch):
    document, marks = scene()
    monkeypatch.setattr(layer_order, "MAX_PROOFS", 1)
    monkeypatch.setattr(layer_order, "MAX_GROUP_PROOFS", 1)
    before = export_svg(document)
    measured = diagnostics()
    proposed = ordered(
        document,
        "outer",
        {"base"},
        marks,
        document.geometry_for("outer"),
        Work.start(10),
        measured,
    )
    assert proposed is not None
    assert measured["order_proofs"] == 1
    assert measured["order_proof_reuses"] == 20
    assert measured["order_proof_limits"] == 0
    assert proposed.geometry_for("ring") == document.geometry_for("ring")
    np.testing.assert_array_equal(
        render(before, (110, 110)), render(export_svg(proposed), (110, 110))
    )


@pytest.mark.parametrize("kind", ["overlap", "outside_mark"])
def test_group_overlap_cannot_be_waived_as_a_shared_disjoint_proof(kind, monkeypatch):
    monkeypatch.setattr(layer_order, "MAX_PROOFS", 64)
    document, marks = scene(**{kind: True})
    measured = diagnostics()
    assert (
        ordered(
            document,
            "outer",
            {"base"},
            marks,
            document.geometry_for("outer"),
            Work.start(10),
            measured,
        )
        is None
    )
    assert measured["order_proofs"] >= 2
    assert measured["order_proof_limits"] == 0


def test_stationary_enclosing_surface_is_excluded_from_inner_mark_order_proof(
    monkeypatch,
):
    monkeypatch.setattr(layer_order, "MAX_GROUP_PROOFS", 1)
    monkeypatch.setattr(layer_order, "MAX_PROOFS", 0)
    marks = tuple(f"mark-{i}" for i in range(20))
    # The ring overlaps the continuing outer surface, whose order is unchanged.
    # Only the inner marks cross the ring when collected above their surface.
    drawing = (
        '<svg width="110" height="110"><g transform="translate(3 4)" opacity="0.5">'
        '<path id="outer" fill="#204080" d="M30 30H70V70H30Z"/>'
        '<path id="ring" fill="#000" fill-rule="evenodd" '
        'd="M35 30H65V60H35Z M39 31H43V53H39Z"/>'
    )
    drawing += "".join(
        f'<path id="{oid}" fill="#208040" d="M40 {32 + i}h2v1h-2Z"/>'
        for i, oid in enumerate(marks)
    )
    document = import_svg(drawing + "</g></svg>")
    measured = diagnostics()
    proposed = ordered(
        document,
        "outer",
        set(),
        marks,
        document.geometry_for("outer"),
        Work.start(10),
        measured,
    )
    assert proposed is not None
    assert measured["order_group_proofs"] == 1
    assert measured.get("order_pair_proofs", 0) == 0
    np.testing.assert_array_equal(
        render(export_svg(document), (110, 110)),
        render(export_svg(proposed), (110, 110)),
    )


def test_moving_geometry_bound_and_stop_keep_the_original_order(monkeypatch):
    document, marks = scene()
    monkeypatch.setattr(layer_order, "MAX_NODES", 1)
    work = Work.start(10)
    measured = diagnostics()
    assert (
        ordered(
            document,
            "outer",
            {"base"},
            marks,
            document.geometry_for("outer"),
            work,
            measured,
        )
        is None
    )
    assert measured["order_proof_limits"] == 1
    work.stop.set()
    assert (
        ordered(
            document,
            "outer",
            {"base"},
            marks,
            document.geometry_for("outer"),
            work,
            diagnostics(),
        )
        is None
    )
