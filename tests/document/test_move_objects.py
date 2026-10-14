"""Object moves keep placement and paint, and adopt their new group's clipping."""

import numpy as np
import pytest

from tests.document.helpers import render, select
from vectrify.document import Editor, EditRejectedError, export_svg, import_svg
from vectrify.document.hit_test import HitIndex

# Nothing overlaps, so restacking alone never changes the rendering.
SVG = """<svg width="100" height="100" fill="red">
<g id="layer" transform="translate(40 40) scale(2)" stroke="black">
<rect id="a" width="10" height="10"/>
<rect id="b" x="12" width="5" height="5" fill="blue"/>
</g>
<rect id="c" width="10" height="10" fill="green"/>
<rect id="d" y="80" width="10" height="10"/>
<g id="faded" opacity="0.5"><rect id="e" x="80" width="5" height="5"/></g>
</svg>"""


def children(editor, parent):
    return [c.id for c in editor.snapshot.document.element(parent).children]


def move(editor, ids, parent, index):
    with editor.transaction("Move") as tx:
        tx.move_objects(frozenset(ids), parent, index)


def assert_same_look(before, after, ids):
    assert (
        np.abs(render(export_svg(before)).astype(int) - render(export_svg(after))).max()
        <= 2
    )
    old, new = HitIndex(before), HitIndex(after)
    for oid in ids:
        assert new.bounds(frozenset({oid})) == pytest.approx(
            old.bounds(frozenset({oid}))
        )


def test_move_within_a_container_to_back_front_and_index():
    editor = Editor(import_svg(SVG), selection=select("c"))
    root = editor.snapshot.document.root.id
    move(editor, {"c"}, root, 0)
    assert children(editor, root) == ["c", "layer", "d", "faded"]
    move(editor, {"c"}, root, 3)
    assert children(editor, root) == ["layer", "d", "faded", "c"]
    move(editor, {"c"}, root, 1)
    assert children(editor, root) == ["layer", "c", "d", "faded"]
    assert editor.snapshot.document.element("c") == import_svg(SVG).element("c")


def test_moving_into_a_transformed_group_keeps_position_and_paint():
    editor = Editor(import_svg(SVG), selection=select("c", "d"))
    before = editor.snapshot.document
    move(editor, {"c", "d"}, "layer", 2)
    after = editor.snapshot.document
    assert children(editor, "layer") == ["a", "b", "c", "d"]
    # The group's stroke would otherwise outline them.
    assert after.element("c").get("stroke") == "none"
    assert after.element("c").get("transform")
    assert_same_look(before, after, {"c", "d"})
    assert editor.undo_labels == ("Move",)
    assert editor.undo().document == before


def test_moving_out_of_a_transformed_group_keeps_position_and_paint():
    editor = Editor(import_svg(SVG), selection=select("a"))
    before = editor.snapshot.document
    root = before.root.id
    move(editor, {"a"}, root, 4)
    after = editor.snapshot.document
    assert children(editor, root)[-1] == "a"
    assert after.element("a").get("stroke") == "black"
    assert_same_look(before, after, {"a"})


def test_multi_object_move_keeps_paint_order_and_takes_children_along():
    editor = Editor(import_svg(SVG), selection=select("d", "b", "a", "layer"))
    root = editor.snapshot.document.root.id
    move(editor, {"d", "b", "a"}, root, 0)
    assert children(editor, root) == ["a", "b", "d", "layer", "c", "faded"]
    editor.undo()
    move(editor, {"layer", "a"}, root, 3)
    assert children(editor, root) == ["c", "d", "faded", "layer"]
    assert children(editor, "layer") == ["a", "b"]


@pytest.mark.parametrize(
    ("ids", "parent", "match"),
    [
        ({"layer"}, "layer", "into itself"),
        ({"layer", "a"}, "a", "group or the drawing"),
        ({"c"}, "layer", "outside the container"),
    ],
)
def test_moves_that_cannot_keep_their_look_or_structure_are_refused(ids, parent, match):
    editor = Editor(import_svg(SVG), selection=select("layer", "c", "faded"))
    before = editor.snapshot.document
    index = 9 if match == "outside the container" else 0
    with pytest.raises(EditRejectedError, match=match):
        move(editor, ids, parent, index)
    assert editor.snapshot.document == before


def test_root_and_definitions_are_refused():
    source = SVG.replace(
        '<g id="faded" opacity="0.5">',
        '<defs id="defs"><clipPath id="clip"><rect id="r" width="50" height="50"/>'
        '</clipPath></defs><g id="faded" clip-path="url(#clip)">',
    )
    editor = Editor(import_svg(source), selection=select("c", "e", "defs"))
    root = editor.snapshot.document.root.id
    with pytest.raises(EditRejectedError, match="root"):
        move(editor, {root}, root, 0)
    with pytest.raises(EditRejectedError, match="group or the drawing"):
        move(editor, {"c"}, "clip", 0)
    with pytest.raises(EditRejectedError, match="Definitions"):
        move(editor, {"r"}, root, 0)


@pytest.mark.parametrize(
    ("source", "target"),
    [("root", "clipped"), ("root", "nested"), ("old", "clipped"), ("old", "root")],
)
def test_moves_apply_the_destination_clipping_and_keep_placement(source, target):
    shape = '<rect id="moving" x="30" y="30" width="30" height="30" fill="red"/>'
    svg = f"""<svg width="100" height="100">
    <defs>
      <clipPath id="old-clip"><rect width="40" height="40"/></clipPath>
      <clipPath id="new-clip"><rect width="15" height="15"/></clipPath>
    </defs>
    <g id="old" transform="translate(5 5)" clip-path="url(#old-clip)">
      {shape if source == "old" else ""}
    </g>
    <g id="clipped" transform="translate(20 20) scale(2)" fill="blue"
       clip-path="url(#new-clip)"><g id="nested"/></g>
    {shape if source == "root" else ""}
    </svg>"""
    editor = Editor(import_svg(svg), selection=select("moving"))
    before = editor.snapshot.document
    parent = before.root.id if target == "root" else target
    move(editor, {"moving"}, parent, 0)
    after = editor.snapshot.document
    assert after.ancestry("moving")[-2].id == parent
    assert after.element("moving").get("clip-path") is None
    # Compare the rendered result with the same geometry in page coordinates.
    position = 35 if source == "old" else 30
    clip = 'clip-path="url(#expected-clip)"' if target != "root" else ""
    expected = f"""<svg width="100" height="100">
    <defs><clipPath id="expected-clip">
      <rect x="20" y="20" width="30" height="30"/>
    </clipPath></defs>
    <rect id="moving" x="{position}" y="{position}" width="30" height="30"
          fill="red" {clip}/>
    </svg>"""
    assert np.array_equal(render(export_svg(after)), render(expected))
    assert HitIndex(after).bounds(frozenset({"moving"})) == pytest.approx(
        HitIndex(import_svg(expected)).bounds(frozenset({"moving"}))
    )
    assert editor.undo().document == before
    assert editor.redo().document == after


def test_locked_objects_and_groups_are_refused():
    editor = Editor(import_svg(SVG), selection=select("c"))
    editor.set_locks("layer", frozenset({"structure"}))
    with pytest.raises(EditRejectedError, match="locked"):
        move(editor, {"c"}, "layer", 0)
    editor.set_locks("layer", frozenset())
    editor.set_locks("c", frozenset({"transform"}))
    with pytest.raises(EditRejectedError, match="locked"):
        move(editor, {"c"}, "layer", 0)
    # Restacking in place leaves the locked transform alone.
    move(editor, {"c"}, editor.snapshot.document.root.id, 0)


def test_instances_follow_a_changed_transform_and_the_move_is_undoable():
    source = SVG.replace("</svg>", '<use id="copy" href="#c" x="50"/></svg>')
    editor = Editor(import_svg(source), selection=select("c", "copy"))
    before = editor.snapshot.document
    move(editor, {"c"}, "layer", 0)
    after = editor.snapshot.document
    after.validate()
    assert after.element("copy").get("href") == "#c"
    assert not np.array_equal(render(source), render(export_svg(after)))
    assert editor.undo().document == before
    assert editor.redo().document == after


@pytest.mark.parametrize("entering", [False, True])
def test_moves_apply_the_destination_opacity(entering):
    editor = Editor(import_svg(SVG), selection=select("c", "e"))
    before = editor.snapshot.document
    oid, parent = ("c", "faded") if entering else ("e", before.root.id)
    move(editor, {oid}, parent, 0)
    after = editor.snapshot.document
    x, y = (5, 5) if entering else (82, 2)
    assert render(export_svg(after))[y, x, 3] == (128 if entering else 255)
    assert after.element(oid).get("opacity") is None
    assert editor.undo().document == before
    assert editor.redo().document == after
