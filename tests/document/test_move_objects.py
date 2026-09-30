"""Moving objects in paint order and between groups keeps their appearance."""

import numpy as np
import pytest

from tests.document.test_document import render, select
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
        ({"e"}, "layer", "opacity or clipping"),
        ({"c"}, "faded", "opacity or clipping"),
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


def test_root_clipped_groups_and_definitions_are_refused():
    source = SVG.replace(
        '<g id="faded" opacity="0.5">',
        '<defs id="defs"><clipPath id="clip"><rect id="r" width="50" height="50"/>'
        '</clipPath></defs><g id="faded" clip-path="url(#clip)">',
    )
    editor = Editor(import_svg(source), selection=select("c", "e", "defs"))
    root = editor.snapshot.document.root.id
    with pytest.raises(EditRejectedError, match="root"):
        move(editor, {root}, root, 0)
    with pytest.raises(EditRejectedError, match="opacity or clipping"):
        move(editor, {"e"}, root, 0)
    with pytest.raises(EditRejectedError, match="group or the drawing"):
        move(editor, {"c"}, "clip", 0)
    with pytest.raises(EditRejectedError, match="Definitions"):
        move(editor, {"r"}, root, 0)


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


def test_instances_and_shared_edges_block_a_changed_transform():
    source = SVG.replace("</svg>", '<use id="copy" href="#c" x="50"/></svg>')
    editor = Editor(import_svg(source), selection=select("c", "copy"))
    with pytest.raises(EditRejectedError, match="instances"):
        move(editor, {"c"}, "layer", 0)
    shared = SVG.replace(
        '<rect id="d" y="80" width="10" height="10"/>',
        '<path id="p" d="M0 60 L10 60 L10 70 L0 70 Z"/>'
        '<path id="q" d="M10 60 L20 60 L20 70 L10 70 Z"/>',
    )
    editor = Editor(import_svg(shared), selection=select("p", "q"))
    with editor.transaction("Link") as tx:
        assert tx.share_boundaries(0.01) == 1
    with pytest.raises(EditRejectedError, match="Unlink"):
        move(editor, {"p"}, "layer", 0)
    root = editor.snapshot.document.root.id
    move(editor, {"p"}, root, 0)
    assert children(editor, root)[0] == "p"
