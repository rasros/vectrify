"""Split compound paths without losing holes, paint, identity or edit constraints."""

from dataclasses import replace

import numpy as np
import pytest

from tests.document.test_document import render, select
from vectrify.document import Editor, EditRejectedError, export_svg, import_svg

RINGS = "M0 0H30V30H0Z M5 5V25H25V5Z M60 0H90V30H60Z"


def drawing(data=RINGS, attrs=""):
    return import_svg(
        '<svg width="120" height="80"><g fill="red" transform="translate(5 8)">'
        f'<path id="p" d="{data}" {attrs}/></g></svg>'
    )


@pytest.mark.parametrize("rule", ["evenodd", "nonzero"])
def test_holes_stay_with_outer_ring_and_pixels_and_ids_survive(rule):
    doc = drawing(
        attrs=(f'fill-rule="{rule}" opacity=".4" stroke="blue" stroke-width=".5"')
    )
    editor = Editor(doc, selection=select("p"))
    old = doc.geometry_for("p")
    node = old.subpaths[0].nodes[0]
    editor.pin_node("p", node.id, pinned=True)
    before = editor.snapshot
    with editor.transaction("Split") as tx:
        ids = tx.split_disconnected("p")
    after = editor.snapshot
    assert len(ids) == 2
    assert after.selection.object_ids == frozenset(ids)
    assert sorted(len(after.document.geometry_for(i).subpaths) for i in ids) == [1, 2]
    nodes = [
        n for i in ids for s in after.document.geometry_for(i).subpaths for n in s.nodes
    ]
    assert {n.id for n in nodes} == {n.id for s in old.subpaths for n in s.nodes}
    assert next(n for n in nodes if n.id == node.id).pinned
    np.testing.assert_array_equal(
        render(export_svg(before.document)), render(export_svg(after.document))
    )
    assert editor.undo().document == before.document
    assert editor.redo().document == after.document
    assert editor.snapshot.selection == after.selection


def test_touching_overlapping_and_nested_rings_stay_connected():
    for data in [
        "M0 0H30V30H0Z M30 0H60V30H30Z",
        "M0 0H30V30H0Z M20 0H50V30H20Z",
        "M0 0H30V30H0Z M5 5H25V25H5Z",
    ]:
        doc = drawing(data)
        editor = Editor(doc, selection=select("p"))
        with editor.transaction("Split") as tx:
            assert tx.split_disconnected("p") == ("p",)
        assert editor.snapshot.document == doc
        assert not editor.undo_labels


def test_curves_are_kept_exact_and_open_strokes_can_split():
    doc = drawing(
        "M0 0C10 20 20 20 30 0 M60 0C70 20 80 20 90 0", 'fill="none" stroke="black"'
    )
    editor = Editor(doc, selection=select("p"))
    with editor.transaction("Split") as tx:
        ids = tx.split_disconnected("p")
    assert len(ids) == 2
    assert (
        tuple(
            s
            for oid in ids
            for s in editor.snapshot.document.geometry_for(oid).subpaths
        )
        == doc.geometry_for("p").subpaths
    )
    np.testing.assert_array_equal(
        render(export_svg(doc)), render(export_svg(editor.snapshot.document))
    )


@pytest.mark.parametrize("cap", ["butt", "round", "square"])
def test_stroke_paint_does_not_connect_separate_paths(cap):
    doc = drawing(
        "M0 0L10 0 M11 0L20 0",
        f'fill="none" stroke="black" stroke-width="4" stroke-linecap="{cap}"',
    )
    editor = Editor(doc, selection=select("p"))
    with editor.transaction("Split") as tx:
        assert len(tx.split_disconnected("p")) == 2


@pytest.mark.parametrize("gap", [0.001, 0.01, 1, 10])
@pytest.mark.parametrize("join", ["round", "bevel", "miter"])
def test_positive_gaps_split_regardless_of_stroke_style(gap, join):
    doc = drawing(
        f"M0 0H10V10H0Z M{10 + gap} 0H30V10H{10 + gap}Z",
        f'stroke="black" stroke-width="3" stroke-linejoin="{join}" '
        'stroke-miterlimit="100"',
    )
    editor = Editor(doc, selection=select("p"))
    with editor.transaction("Split") as tx:
        assert len(tx.split_disconnected("p")) == 2


def test_open_paths_with_exact_endpoint_contact_stay_together():
    doc = drawing("M0 0L10 0 M10 0L20 10", 'fill="none" stroke="black"')
    editor = Editor(doc, selection=select("p"))
    with editor.transaction("Split") as tx:
        assert tx.split_disconnected("p") == ("p",)


@pytest.mark.parametrize("lock", ["structure", "geometry"])
def test_split_respects_locks(lock):
    editor = Editor(drawing(), selection=select("p"))
    editor.set_locks("p", frozenset({lock}))
    before = editor.snapshot
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Split") as tx,
    ):
        tx.split_disconnected("p")
    assert editor.snapshot == before


def test_split_checks_scope_and_shared_geometry():
    doc = drawing()
    with (
        pytest.raises(EditRejectedError, match="unselected"),
        Editor(doc).transaction("Split") as tx,
    ):
        tx.split_disconnected("p")
    path = doc.element("p")
    parent = doc.ancestry("p")[-2]
    doc = doc.replace_element(
        replace(parent, children=(*parent.children, replace(path, id="q")))
    )
    editor = Editor(doc, selection=select("p"))
    with (
        pytest.raises(EditRejectedError, match="Detach shared"),
        editor.transaction("Split") as tx,
    ):
        tx.split_disconnected("p")


def test_object_bounding_box_clip_keeps_original_combined_bounds():
    doc = import_svg(
        '<svg width="100" height="100"><defs><clipPath id="c" '
        'clipPathUnits="objectBoundingBox"><rect width=".5" height="1"/>'
        '</clipPath></defs><path id="p" clip-path="url(#c)" '
        'd="M0 0H10V10H0Z M60 0H90V30H60Z"/></svg>'
    )
    editor = Editor(doc, selection=select("p"))
    with editor.transaction("Split") as tx:
        assert len(tx.split_disconnected("p")) == 2
    from tests.document.test_hit_test import hits
    from vectrify.document import HitIndex

    before, after = HitIndex(doc), HitIndex(editor.snapshot.document)
    assert hits(before, 2, 2, 2, 2)
    assert hits(after, 2, 2, 2, 2)
    assert not hits(before, 65, 5, 5, 5)
    assert not hits(after, 65, 5, 5, 5)
