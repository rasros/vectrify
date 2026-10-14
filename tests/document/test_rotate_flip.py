"""Rotating and mirroring objects about a pivot on the page."""

import numpy as np
import pytest

from tests.document.test_document import render, select
from vectrify.document import Editor, EditRejectedError, export_svg, import_svg
from vectrify.document.hit_test import HitIndex, multiply
from vectrify.document.transforms import REFLECTIONS, about, root_matrix, rotation

# A rotated group holding a scaled group, a stroked arrow, a clipped shape and
# a skewed gradient fill: everything whose look must follow the object.
SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"
viewBox="0 0 100 100">
<defs><clipPath id="window"><circle cx="30" cy="70" r="14"/></clipPath>
<linearGradient id="ramp" gradientUnits="userSpaceOnUse" x1="60" y1="0" x2="95"
y2="0"><stop offset="0" stop-color="red"/><stop offset="1" stop-color="blue"/>
</linearGradient></defs>
<g id="outer" transform="translate(10 5) rotate(20 20 20)" stroke="black"
stroke-width="3">
<g id="inner" transform="scale(1.5 1)">
<rect id="box" x="2" y="2" width="16" height="10" fill="orange"/></g>
<path id="arrow" d="M5 30 L35 25" fill="none" data-vectrify-arrow-end="arrow"/>
</g>
<rect id="clipped" x="10" y="55" width="40" height="40" fill="green"
clip-path="url(#window)"/>
<rect id="shaded" x="60" y="10" width="35" height="30" fill="url(#ramp)"
transform="skewX(10)"/>
</svg>"""
TOP = ("outer", "clipped", "shaded")


def turn(editor, ids, pivot, linear, verb="transform"):
    with editor.transaction("Turn") as tx:
        tx.transform_objects(frozenset(ids), pivot, linear, verb=verb)


def page(editor, oid):
    return root_matrix(editor.snapshot.document, oid)


def assert_close(actual, expected):
    assert actual == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("linear", "expected"),
    [
        (REFLECTIONS["horizontal"], np.fliplr),
        (REFLECTIONS["vertical"], np.flipud),
        (rotation(90), lambda image: np.rot90(image, k=-1)),
    ],
    ids=["horizontal", "vertical", "quarter-turn"],
)
def test_the_drawing_looks_mirrored_or_turned_about_its_centre(linear, expected):
    # Strokes, the arrowhead, the clip and the gradient all follow: the whole
    # drawing turned about the canvas centre renders as the turned picture.
    document = import_svg(SVG)
    before = render(export_svg(document))
    editor = Editor(document, selection=select(*TOP))
    turn(editor, TOP, (50, 50), linear)
    after = render(export_svg(editor.snapshot.document))
    difference = np.abs(after.astype(int) - expected(before).astype(int))
    # Only antialiasing along a few edges differs.
    assert difference.mean() < 0.1
    assert (difference > 64).sum() <= 2


def test_rotation_is_composed_in_the_parents_frame():
    editor = Editor(import_svg(SVG), selection=select("box"))
    old = page(editor, "box")
    turn(editor, {"box"}, (12, 34), rotation(30), "rotate")
    assert_close(page(editor, "box"), multiply(about((12, 34), rotation(30)), old))
    # The groups' own transforms stay; the box's undoes their frame.
    document = editor.snapshot.document
    assert document.element("outer").get("transform") == (
        "translate(10 5) rotate(20 20 20)"
    )
    assert document.element("inner").get("transform") == "scale(1.5 1)"


def test_a_selected_group_takes_its_selected_descendants_along_once():
    editor = Editor(import_svg(SVG), selection=select("outer", "box", "arrow"))
    old = {oid: page(editor, oid) for oid in ("outer", "box", "arrow")}
    flip = about((40, 20), REFLECTIONS["vertical"])
    turn(editor, {"outer", "box", "arrow"}, (40, 20), REFLECTIONS["vertical"])
    for oid, matrix in old.items():
        assert_close(page(editor, oid), multiply(flip, matrix))
    document = editor.snapshot.document
    assert document.element("box").get("transform") is None
    assert document.element("arrow").get("transform") is None


def test_a_flip_is_a_mirroring_transform_not_a_negative_resize():
    editor = Editor(import_svg(SVG), selection=select("clipped"))
    before = HitIndex(editor.snapshot.document).bounds(frozenset({"clipped"}))
    turn(editor, {"clipped"}, (10, 0), REFLECTIONS["horizontal"], "flip")
    element = editor.snapshot.document.element("clipped")
    assert element.get("transform") == "translate(20 0) scale(-1 1)"
    # The clip lives in the object's frame, so it stays on the shape.
    assert element.get("clip-path") == "url(#window)"
    after = HitIndex(editor.snapshot.document).bounds(frozenset({"clipped"}))
    assert before is not None
    assert after is not None
    assert after == pytest.approx(
        (20 - before[2], before[1], 20 - before[0], before[3])
    )


def test_strokes_and_arrowheads_keep_their_attributes():
    editor = Editor(import_svg(SVG), selection=select("outer", "arrow"))
    before = {oid: editor.snapshot.document.element(oid) for oid in ("outer", "arrow")}
    turn(editor, {"outer", "arrow"}, (0, 0), rotation(-45))
    document = editor.snapshot.document
    assert document.element("outer").get("stroke-width") == "3"
    assert document.element("arrow") == before["arrow"]
    assert document.element("arrow").get("data-vectrify-arrow-end") == "arrow"


def test_four_quarter_turns_give_the_original_transform_back():
    editor = Editor(import_svg(SVG), selection=select("clipped", "shaded"))
    for _ in range(4):
        turn(editor, {"clipped", "shaded"}, (33, 71), rotation(90))
    document = editor.snapshot.document
    assert document.element("clipped").get("transform") is None
    assert (
        document.element("shaded").get("transform")
        == "matrix(1 0 0.176326980708 1 0 0)"
    )


def test_one_turn_is_one_undo_step():
    editor = Editor(import_svg(SVG), selection=select(*TOP))
    before = editor.snapshot.document
    with editor.transaction("Rotate") as tx:
        tx.transform_objects(frozenset(TOP), (50, 50), rotation(15), verb="rotate")
    assert editor.undo_labels == ("Rotate",)
    editor.undo()
    assert editor.snapshot.document == before


@pytest.mark.parametrize(
    ("locked", "lock", "match"),
    [
        ("clipped", "transform", "position is locked; unlock it to rotate"),
        ("clipped", "geometry", "geometry is locked; unlock it to rotate"),
        ("outer", "transform", "position is locked; unlock it to rotate"),
    ],
)
def test_locked_position_or_geometry_is_refused(locked, lock, match):
    target = "box" if locked == "outer" else "clipped"
    editor = Editor(import_svg(SVG), selection=select(target))
    editor.set_locks(locked, frozenset({lock}))
    before = editor.snapshot.document
    with pytest.raises(EditRejectedError, match=match):
        turn(editor, {target}, (0, 0), rotation(10), "rotate")
    assert editor.snapshot.document == before


@pytest.mark.parametrize(
    "linear",
    [(0, 0, 0, 0), (1, 2, 2, 4), (float("nan"), 0, 0, 1), (float("inf"), 0, 0, 1)],
)
def test_degenerate_maps_are_refused(linear):
    editor = Editor(import_svg(SVG), selection=select("clipped"))
    with pytest.raises(EditRejectedError, match=r"Cannot flip: .*invertible"):
        turn(editor, {"clipped"}, (0, 0), linear, "flip")


def test_definitions_and_clipping_boundaries_are_refused():
    editor = Editor(import_svg(SVG), selection=select("ramp"))
    with pytest.raises(EditRejectedError, match="cannot be flipped"):
        turn(editor, {"ramp"}, (0, 0), REFLECTIONS["vertical"], "flip")


def test_resize_still_refuses_negative_scales():
    # Mirroring is transform_objects' alone; a resize never turns inside out.
    editor = Editor(import_svg(SVG), selection=select("clipped"))
    with (
        pytest.raises(EditRejectedError, match="positive"),
        editor.transaction("Resize") as tx,
    ):
        tx.scale_objects(frozenset({"clipped"}), (0, 0), (-1, 1))
