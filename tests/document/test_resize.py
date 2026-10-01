"""Resizing objects scales them about an anchor on the page."""

import pytest

from tests.document.test_document import select
from vectrify.document import Editor, EditRejectedError, import_svg
from vectrify.document.hit_test import HitIndex

SVG = """<svg width="200" height="200" fill="red">
<g id="layer" transform="translate(40 40) rotate(90) scale(2)">
<rect id="a" width="10" height="10"/>
<rect id="b" x="12" width="5" height="5" fill="blue"/>
</g>
<rect id="c" x="100" y="100" width="20" height="10" fill="green"/>
<rect id="d" x="150" y="100" width="10" height="10" transform="translate(5 5)"/>
<rect id="lined" x="10" y="150" width="20" height="20" stroke="black"
 stroke-width="2"/>
<g id="outlined" stroke="black" stroke-width="4">
<rect id="inner" x="50" y="150" width="10" height="10"/>
<rect id="thin" x="70" y="150" width="10" height="10" stroke-width="1"/>
</g>
</svg>"""


def resize(editor, ids, anchor, scale):
    with editor.transaction("Resize") as tx:
        tx.scale_objects(frozenset(ids), anchor, scale)


def bounds(editor, *ids):
    box = HitIndex(editor.snapshot.document).bounds(frozenset(ids))
    assert box is not None
    return box


def scaled(box, anchor, scale):
    (ax, ay), (sx, sy) = anchor, scale
    left, top, right, bottom = box
    return (
        ax + (left - ax) * sx,
        ay + (top - ay) * sy,
        ax + (right - ax) * sx,
        ay + (bottom - ay) * sy,
    )


def test_scales_about_the_anchor_on_the_transform():
    editor = Editor(import_svg(SVG), selection=select("c"))
    resize(editor, {"c"}, (100, 100), (2, 0.5))
    assert bounds(editor, "c") == pytest.approx((100, 100, 140, 105))
    element = editor.snapshot.document.element("c")
    assert element.get("transform") == "translate(-100 50) scale(2 0.5)"
    # The geometry is untouched; only the transform changes.
    assert element.get("width") == "20"


def test_composes_with_the_objects_own_transform():
    editor = Editor(import_svg(SVG), selection=select("d"))
    before = bounds(editor, "d")
    resize(editor, {"d"}, (155, 105), (3, 3))
    assert bounds(editor, "d") == pytest.approx(scaled(before, (155, 105), (3, 3)))


def test_scales_several_objects_about_one_anchor():
    editor = Editor(import_svg(SVG), selection=select("c", "d"))
    old = {oid: bounds(editor, oid) for oid in ("c", "d")}
    resize(editor, {"c", "d"}, (100, 100), (0.5, 1.5))
    for oid, box in old.items():
        assert bounds(editor, oid) == pytest.approx(scaled(box, (100, 100), (0.5, 1.5)))


def test_objects_in_a_transformed_group_scale_on_the_page():
    editor = Editor(import_svg(SVG), selection=select("a"))
    before = bounds(editor, "a")
    anchor = (before[0], before[1])
    resize(editor, {"a"}, anchor, (1.5, 3))
    assert bounds(editor, "a") == pytest.approx(scaled(before, anchor, (1.5, 3)))
    # The group's rotation stays outside: the child's transform undoes it.
    assert editor.snapshot.document.element("layer").get("transform") == (
        "translate(40 40) rotate(90) scale(2)"
    )


def test_a_selected_group_takes_its_selected_children_along_once():
    editor = Editor(import_svg(SVG), selection=select("layer", "a"))
    old = {oid: bounds(editor, oid) for oid in ("a", "b")}
    resize(editor, {"layer", "a"}, (0, 0), (2, 2))
    for oid, box in old.items():
        assert bounds(editor, oid) == pytest.approx(scaled(box, (0, 0), (2, 2)))
    assert editor.snapshot.document.element("a").get("transform") is None


def test_strokes_keep_their_width():
    editor = Editor(import_svg(SVG), selection=select("lined", "outlined"))
    resize(editor, {"lined", "outlined"}, (0, 0), (2, 2))
    document = editor.snapshot.document
    assert document.element("lined").get("stroke-width") == "1"
    # The group's width is divided; a child's own width too.
    assert document.element("outlined").get("stroke-width") == "2"
    assert document.element("thin").get("stroke-width") == "0.5"
    assert document.element("inner").get("stroke-width") is None


def test_unstroked_objects_get_no_stroke_width():
    editor = Editor(import_svg(SVG), selection=select("c"))
    resize(editor, {"c"}, (0, 0), (2, 2))
    assert editor.snapshot.document.element("c").get("stroke-width") is None


@pytest.mark.parametrize(
    ("locked", "lock", "match"),
    [
        ("c", "transform", "position is locked"),
        ("c", "geometry", "geometry is locked"),
        ("layer", "transform", "position is locked"),
    ],
)
def test_locked_position_or_geometry_is_refused(locked, lock, match):
    target = "a" if locked == "layer" else "c"
    editor = Editor(import_svg(SVG), selection=select(target))
    editor.set_locks(locked, frozenset({lock}))
    before = editor.snapshot.document
    with pytest.raises(EditRejectedError, match=match):
        resize(editor, {target}, (0, 0), (2, 2))
    assert editor.snapshot.document == before


@pytest.mark.parametrize("scale", [(0, 1), (-1, 1), (float("inf"), 1)])
def test_degenerate_scales_are_refused(scale):
    editor = Editor(import_svg(SVG), selection=select("c"))
    with pytest.raises(EditRejectedError, match="positive"):
        resize(editor, {"c"}, (0, 0), scale)


def test_a_resize_is_one_undo_step():
    editor = Editor(import_svg(SVG), selection=select("c", "lined"))
    before = editor.snapshot.document
    resize(editor, {"c", "lined"}, (0, 0), (2, 3))
    assert editor.undo_labels == ("Resize",)
    editor.undo()
    assert editor.snapshot.document == before
