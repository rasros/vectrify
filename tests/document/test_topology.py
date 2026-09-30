"""Exact subdivision, node deletion and detaching across edits and history."""

import json

import numpy as np
import pytest

from tests.document.test_document import SHARED, SVG, render, select
from vectrify.document import (
    DocumentError,
    Editor,
    EditRejectedError,
    Selection,
    StaleRevisionError,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.topology import EdgeRef, edge

CONTOURS = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">
<path id="fill" fill="red" d="M8 8 C16 4 40 4 48 8 L48 40 L8 40 Z"/>
<path id="outline" fill="none" stroke="black" d="M48 8 C40 4 16 4 8 8"/>
<path id="other" fill="none" stroke="blue" d="M48 40 L48 8"/>
</svg>"""


def ref(document, object_id, index, *, reverse=False):
    geometry = document.geometry_for(object_id)
    return EdgeRef(geometry.id, geometry.subpaths[0].nodes[index].id, reverse)


def contour_editor():
    return Editor(import_svg(CONTOURS), selection=select("fill", "outline"))


def evaluate(points, t):
    points = np.array(points)
    while len(points) > 1:
        points = points[:-1] * (1 - t) + points[1:] * t
    return points[0]


@pytest.mark.parametrize("object_id", ["fill", "outline"])
@pytest.mark.parametrize("t", [0.125, 0.37, 0.8])
def test_split_cubic_preserves_shape_identity_and_pins(object_id, t):
    editor = contour_editor()
    before = editor.snapshot.document
    original = edge(before, ref(before, object_id, 1))
    editor.pin_node(object_id, original.end.id)
    before = editor.snapshot.document
    with editor.transaction("Subdivide contour") as tx:
        added = tx.split_edge(object_id, original.end.id, t)
    after = editor.snapshot.document
    assert len(added) == 1
    after.validate()
    old = before.geometry_for(object_id)
    new = after.geometry_for(object_id)
    assert old.id == new.id
    assert old.subpaths[0].id == new.subpaths[0].id
    assert {n.id for n in old.subpaths[0].nodes} <= {
        n.id for n in new.subpaths[0].nodes
    }
    # Only the split path changes.
    for name in {"fill", "outline"} - {object_id}:
        assert after.geometry_for(name) == before.geometry_for(name)
    assert after.geometry_for(object_id).node(original.end.id).pinned
    nodes = after.geometry_for(object_id).subpaths[0].nodes
    left = edge(after, ref(after, object_id, 1)).points
    right = edge(after, ref(after, object_id, 2)).points
    assert nodes[2].id == original.end.id
    for u in np.linspace(0, 1, 11):
        np.testing.assert_allclose(
            evaluate(left, u), evaluate(original.points, t * u), atol=1e-12
        )
        np.testing.assert_allclose(
            evaluate(right, u), evaluate(original.points, t + (1 - t) * u), atol=1e-12
        )
    assert editor.undo().document == before


def test_split_reversed_edge_follows_its_traversal():
    doc = import_svg(CONTOURS)
    forward = edge(doc, ref(doc, "outline", 1))
    from vectrify.document.topology import split_edges

    after, (added,) = split_edges(doc, ref(doc, "outline", 1, reverse=True), 0.25)
    np.testing.assert_allclose(
        after.geometry_for("outline").node(added).endpoint,
        evaluate(forward.points, 0.75),
        atol=1e-12,
    )


def test_closing_line_splits_without_opening_path():
    source = """<svg width="64" height="64">
    <path id="a" fill="red" d="M8 8 L48 8 L48 40 L8 40 Z"/></svg>"""
    editor = Editor(import_svg(source), selection=Selection.all())
    doc = editor.snapshot.document
    with editor.transaction("Split closure") as tx:
        tx.split_edge("a", ref(doc, "a", 0).node_id, 0.25)
    after = editor.snapshot.document
    nodes = after.geometry_for("a").subpaths[0].nodes
    assert nodes[-1].endpoint == (8, 32)
    assert after.geometry_for("a").subpaths[0].closed
    assert np.array_equal(render(source), render(export_svg(after)))


@pytest.mark.parametrize("t", [-1, 0, 1, 2, float("nan"), float("inf")])
def test_invalid_split_parameter_is_atomic(t):
    editor = contour_editor()
    before = editor.snapshot
    with pytest.raises(EditRejectedError), editor.transaction("Invalid split") as tx:
        tx.split_edge("fill", ref(before.document, "fill", 1).node_id, t)
    assert editor.snapshot == before


def test_splitting_checks_all_consumers_structure_permission_and_node_filter():
    editor = Editor(import_svg(SHARED), selection=select("first"))
    doc = editor.snapshot.document
    node = ref(doc, "first", 1).node_id
    with (
        pytest.raises(EditRejectedError, match="unselected"),
        editor.transaction("Split") as tx,
    ):
        tx.split_edge("first", node)
    editor = contour_editor()
    doc = editor.snapshot.document
    node = ref(doc, "fill", 1).node_id
    other = ref(doc, "fill", 2).node_id
    editor.select(select("fill", nodes=[other]))
    with (
        pytest.raises(EditRejectedError, match="selected nodes"),
        editor.transaction("Split") as tx,
    ):
        tx.split_edge("fill", node)
    editor.select(select("fill", nodes=[node]))
    with (
        pytest.raises(EditRejectedError, match="not permitted"),
        editor.transaction("Split", allowed=frozenset({"geometry"})) as tx,
    ):
        tx.split_edge("fill", node)
    with editor.transaction("Split selected edge") as tx:
        tx.split_edge("fill", node)
    assert editor.snapshot.selection.node_ids == {node}


def test_delete_selected_node_clears_filter_safely_and_undo_restores_selection():
    doc = import_svg(SVG)
    node = ref(doc, "a", 1).node_id
    selection = select("a", nodes=[node])
    editor = Editor(doc, selection=selection)
    with editor.transaction("Delete corner") as tx:
        tx.delete_node("a", node)
        assert tx.preview_selection == Selection()
        assert tx.node_remapping == ((node, ()),)
    assert editor.snapshot.selection == Selection()
    assert editor.undo().selection == selection
    assert editor.snapshot.document == doc
    assert not editor.redo().selection.object_ids


def test_delete_leaves_surviving_ids_and_selection_intact():
    doc = import_svg(SVG)
    a, b = ref(doc, "a", 1).node_id, ref(doc, "a", 2).node_id
    editor = Editor(doc, selection=select("a", nodes=[a, b]))
    with editor.transaction("Delete one") as tx:
        tx.delete_node("a", a)
    assert editor.snapshot.selection.node_ids == {b}
    assert editor.snapshot.document.geometry_for("a").node(b) == doc.geometry_for(
        "a"
    ).node(b)


def test_delete_checks_pins():
    editor = contour_editor()
    node = ref(editor.snapshot.document, "fill", 1).node_id
    editor.pin_node("fill", node)
    with (
        pytest.raises(EditRejectedError, match="pinned"),
        editor.transaction("Delete") as tx,
    ):
        tx.delete_node("fill", node)


def test_detach_remaps_live_selection_and_preserves_other_consumer_nodes():
    editor = Editor(import_svg(SHARED), selection=select("first"))
    doc = editor.snapshot.document
    node = ref(doc, "first", 1).node_id
    tx = editor.transaction("Detach instance")
    tx.detach_geometry("first")
    new = ref(tx.preview, "first", 1).node_id
    editor.select(select("first", "second", nodes=[node]))
    tx.commit()
    assert editor.snapshot.selection.node_ids == {node, new}
    assert editor.undo().selection.node_ids == {node}
    assert editor.redo().selection.node_ids == {node, new}


def test_repeated_detach_remaps_final_ids_and_keeps_changed_ui_selection():
    editor = Editor(import_svg(SVG), selection=select("a"))
    doc = editor.snapshot.document
    node = ref(doc, "a", 1).node_id
    tx = editor.transaction("Detach twice")
    tx.detach_geometry("a")
    tx.detach_geometry("a")
    new = ref(tx.preview, "a", 1).node_id
    editor.select(select("a", nodes=[node]))
    tx.commit()
    assert editor.snapshot.selection.node_ids == {new}
    tx = editor.transaction("Delete", selection=select("a"))
    tx.delete_node("a", new)
    editor.select(select("b", nodes=[ref(doc, "b", 1).node_id]))
    selection = editor.snapshot.selection
    tx.commit()
    assert editor.snapshot.selection == selection


def test_topology_preview_cannot_commit_after_undo_even_with_matching_document():
    editor = contour_editor()
    doc = editor.snapshot.document
    tx = editor.transaction("Split")
    tx.split_edge("fill", ref(doc, "fill", 1).node_id)
    with editor.transaction("Colour") as other:
        other.set_attributes("fill", {"fill": "blue"})
    editor.undo()
    with pytest.raises(StaleRevisionError):
        tx.commit()
    assert editor.snapshot.document == doc


def test_project_reads_older_versions_and_drops_stored_boundaries():
    doc = import_svg(CONTOURS)
    data = json.loads(save_project(doc))
    assert data["version"] == 3
    assert "boundaries" not in data
    data["version"] = 1
    assert load_project(json.dumps(data))[0] == doc
    # Version 2 projects linked edges into shared boundaries. They load with
    # the links dropped, even ones that no longer match the contours.
    fill, outline = doc.geometry_for("fill"), doc.geometry_for("outline")
    data["version"] = 2
    data["boundaries"] = [
        {
            "id": "boundary_1",
            "members": [
                {
                    "geometry_id": fill.id,
                    "node_id": fill.subpaths[0].nodes[1].id,
                    "reversed": False,
                    "matrix": [1, 0, 0, 1, 0, 0],
                },
                {
                    "geometry_id": outline.id,
                    "node_id": outline.subpaths[0].nodes[1].id,
                    "reversed": False,
                    "matrix": [1, 0, 0, 1, 0, 0],
                },
            ],
        }
    ]
    loaded, _ = load_project(json.dumps(data))
    assert loaded == doc
    data["version"] = 4
    with pytest.raises(DocumentError, match="version"):
        load_project(json.dumps(data))


def test_detaching_two_instances_retains_both_node_replacements():
    editor = Editor(import_svg(SHARED), selection=select("first", "second"))
    original = ref(editor.snapshot.document, "first", 1).node_id
    tx = editor.transaction("Detach both")
    tx.detach_geometry("first")
    tx.detach_geometry("second")
    replacements = {ref(tx.preview, name, 1).node_id for name in ("first", "second")}
    editor.select(select("first", "second", nodes=[original]))
    tx.commit()
    assert editor.snapshot.selection.node_ids == replacements
    assert set(dict(tx.node_remapping)[original]) == replacements


def test_editing_one_path_never_moves_a_coincident_neighbour():
    editor = contour_editor()
    doc = editor.snapshot.document
    with editor.transaction("Move fill corner") as tx:
        tx.update_node("fill", ref(doc, "fill", 1).node_id, (17, 4, 40, 4, 50, 10))
    after = editor.snapshot.document
    assert after.geometry_for("outline") == doc.geometry_for("outline")
    assert after.geometry_for("other") == doc.geometry_for("other")


def test_split_preview_rolls_back_on_a_later_caught_edit_failure():
    editor = contour_editor()
    before = editor.snapshot
    tx = editor.transaction("Split and fail")
    tx.split_edge("fill", ref(before.document, "fill", 1).node_id)
    with pytest.raises(EditRejectedError):
        tx.set_attributes("other", {"stroke": "red"})
    with pytest.raises(EditRejectedError, match="failed"):
        tx.commit()
    assert editor.snapshot == before
