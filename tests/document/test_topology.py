"""Shared contours must stay coincident across edits, subdivision and history."""

import json
from dataclasses import replace

import numpy as np
import pytest

from tests.document.test_document import SHARED, SVG, render, select
from vectrify.document import (
    DocumentError,
    EdgeRef,
    Editor,
    EditRejectedError,
    Rect,
    Selection,
    StaleRevisionError,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.topology import edge

CONTOURS = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">
<path id="fill" fill="red" d="M8 8 C16 4 40 4 48 8 L48 40 L8 40 Z"/>
<path id="outline" fill="none" stroke="black" d="M48 8 C40 4 16 4 8 8"/>
<path id="other" fill="none" stroke="blue" d="M48 40 L48 8"/>
</svg>"""


def ref(document, object_id, index, *, reverse=False):
    geometry = document.geometry_for(object_id)
    return EdgeRef(geometry.id, geometry.subpaths[0].nodes[index].id, reverse)


def linked_editor():
    editor = Editor(import_svg(CONTOURS), selection=select("fill", "outline"))
    doc = editor.snapshot.document
    with editor.transaction("Link contour") as tx:
        tx.link_boundary((ref(doc, "fill", 1), ref(doc, "outline", 1, reverse=True)))
    return editor


def test_link_does_not_change_rendering_and_survives_project_roundtrip():
    editor = linked_editor()
    doc = editor.snapshot.document
    assert np.array_equal(render(CONTOURS), render(export_svg(doc)))
    assert load_project(save_project(doc, editor.snapshot.selection)) == (
        doc,
        editor.snapshot.selection,
    )
    assert not editor.undo().document.boundaries
    assert editor.redo().document == doc


def test_reversed_contour_propagates_handles_and_endpoints():
    editor = linked_editor()
    doc = editor.snapshot.document
    target = ref(doc, "fill", 1)
    with editor.transaction("Refine tip") as tx:
        tx.update_node("fill", target.node_id, (17, 3, 41, 5, 50, 9))
    after = editor.snapshot.document
    assert after.geometry_for("outline").subpaths[0].nodes[0].endpoint == (50, 9)
    assert after.geometry_for("outline").subpaths[0].nodes[1].values == (
        41,
        5,
        17,
        3,
        8,
        8,
    )
    assert after.geometry_for("other") == doc.geometry_for("other")
    assert after.boundaries == doc.boundaries
    assert editor.undo().document == doc
    assert editor.redo().document == after


def test_moving_moveto_propagates_to_reversed_endpoint():
    editor = linked_editor()
    node = ref(editor.snapshot.document, "fill", 0).node_id
    with editor.transaction("Move start") as tx:
        tx.update_node("fill", node, (9, 10))
    assert editor.snapshot.document.geometry_for("outline").subpaths[0].nodes[
        1
    ].endpoint == (9, 10)


@pytest.mark.parametrize(
    "constraint", ["selection", "pin", "lock", "nodes", "permission"]
)
def test_propagation_cannot_bypass_constraints_on_another_consumer(constraint):
    editor = linked_editor()
    doc = editor.snapshot.document
    target = ref(doc, "fill", 1)
    peer = ref(doc, "outline", 0)
    allowed = frozenset({"geometry"})
    if constraint == "selection":
        editor.select(select("fill"))
    elif constraint == "pin":
        editor.pin_node("outline", peer.node_id)
    elif constraint == "lock":
        editor.set_locks("outline", frozenset({"geometry"}))
    elif constraint == "nodes":
        editor.select(select("fill", "outline", nodes=[target.node_id]))
    else:
        allowed = frozenset({"paint"})
    before = editor.snapshot
    with (
        pytest.raises(EditRejectedError),
        editor.transaction("Blocked", allowed=allowed) as tx,
    ):
        tx.update_node("fill", target.node_id, (16, 4, 40, 4, 49, 8))
    assert editor.snapshot == before


def test_pin_does_not_block_linked_handle_edit():
    editor = linked_editor()
    doc = editor.snapshot.document
    editor.pin_node("outline", ref(doc, "outline", 1).node_id)
    with editor.transaction("Handle") as tx:
        tx.update_node("fill", ref(doc, "fill", 1).node_id, (17, 4, 40, 4, 48, 8))
    assert (
        editor.snapshot.document.geometry_for("outline").subpaths[0].nodes[1].values[2]
        == 17
    )


def test_shared_vertex_propagates_transitively_between_adjacent_boundaries():
    editor = linked_editor()
    editor.select(Selection.all())
    doc = editor.snapshot.document
    with editor.transaction("Link next edge") as tx:
        tx.link_boundary((ref(doc, "fill", 2), ref(doc, "other", 1, reverse=True)))
    with editor.transaction("Move junction") as tx:
        tx.update_node("fill", ref(doc, "fill", 1).node_id, (16, 4, 40, 4, 50, 10))
    doc = editor.snapshot.document
    assert doc.geometry_for("outline").subpaths[0].nodes[0].endpoint == (50, 10)
    assert doc.geometry_for("other").subpaths[0].nodes[1].endpoint == (50, 10)


@pytest.mark.parametrize(
    "bad", ["orientation", "unknown", "duplicate", "open_moveto", "arity"]
)
def test_link_rejects_inconsistent_or_invalid_edges_atomically(bad):
    editor = Editor(import_svg(CONTOURS), selection=Selection.all())
    doc = editor.snapshot.document
    first = ref(doc, "fill", 1)
    second = ref(doc, "outline", 1, reverse=True)
    if bad == "orientation":
        second = replace(second, reversed=False)
    elif bad == "unknown":
        second = replace(second, node_id="missing")
    elif bad == "duplicate":
        second = first
    elif bad == "open_moveto":
        second = ref(doc, "outline", 0)
    else:
        second = ref(doc, "other", 1)
    with pytest.raises(DocumentError), editor.transaction("Invalid link") as tx:
        tx.link_boundary((first, second))
    assert editor.snapshot.document == doc
    assert not editor.undo_labels


def test_existing_membership_must_be_detached_before_relinking():
    editor = linked_editor()
    doc = editor.snapshot.document
    with (
        pytest.raises(DocumentError, match="only one"),
        editor.transaction("Duplicate") as tx,
    ):
        tx.link_boundary(doc.boundaries[0].members)


def test_detach_one_edge_allows_independent_edit_and_undo_restores_link():
    editor = linked_editor()
    editor.select(select("fill"))
    before = editor.snapshot.document
    target = ref(before, "fill", 1)
    with editor.transaction("Detach and edit") as tx:
        tx.detach_boundary(target)
        tx.update_node("fill", target.node_id, (17, 4, 40, 4, 48, 8))
    assert not editor.snapshot.document.boundaries
    assert editor.snapshot.document.geometry_for("outline") == before.geometry_for(
        "outline"
    )
    assert editor.undo().document == before


def evaluate(points, t):
    points = np.array(points)
    while len(points) > 1:
        points = points[:-1] * (1 - t) + points[1:] * t
    return points[0]


@pytest.mark.parametrize("object_id", ["fill", "outline"])
@pytest.mark.parametrize("t", [0.125, 0.37, 0.8])
def test_split_linked_cubic_preserves_shape_identity_pins_and_opposite_direction(
    object_id, t
):
    editor = linked_editor()
    before = editor.snapshot.document
    original = edge(before, ref(before, object_id, 1))
    editor.pin_node(object_id, original.end.id)
    before = editor.snapshot.document
    with editor.transaction("Subdivide contour") as tx:
        added = tx.split_edge(object_id, original.end.id, t)
    after = editor.snapshot.document
    assert len(added) == 2
    assert len(after.boundaries) == 2
    after.validate()
    for name in ("fill", "outline"):
        old = before.geometry_for(name)
        new = after.geometry_for(name)
        assert old.id == new.id
        assert old.subpaths[0].id == new.subpaths[0].id
        assert {n.id for n in old.subpaths[0].nodes} <= {
            n.id for n in new.subpaths[0].nodes
        }
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
    with editor.transaction("Edit new linked tip") as tx:
        value = nodes[1].values
        tx.update_node(object_id, nodes[1].id, (*value[:-2], value[-2] + 1, value[-1]))
    editor.snapshot.document.validate()
    editor.undo()
    assert editor.undo().document == before


def test_closing_line_links_splits_and_propagates_without_opening_path():
    source = """<svg width="64" height="64">
    <path id="a" fill="red" d="M8 8 L48 8 L48 40 L8 40 Z"/>
    <path id="b" stroke="black" d="M8 8 L8 40"/></svg>"""
    editor = Editor(import_svg(source), selection=Selection.all())
    doc = editor.snapshot.document
    with editor.transaction("Link closure") as tx:
        tx.link_boundary((ref(doc, "a", 0), ref(doc, "b", 1, reverse=True)))
    with editor.transaction("Split closure") as tx:
        tx.split_edge("a", ref(doc, "a", 0).node_id, 0.25)
    after = editor.snapshot.document
    nodes = after.geometry_for("a").subpaths[0].nodes
    assert nodes[-1].endpoint == (8, 32)
    assert after.geometry_for("a").subpaths[0].closed
    assert np.array_equal(render(source), render(export_svg(after)))
    with editor.transaction("Move closure start") as tx:
        tx.update_node("a", nodes[0].id, (9, 9))
    assert editor.snapshot.document.geometry_for("b").subpaths[0].nodes[0].endpoint == (
        9,
        9,
    )


@pytest.mark.parametrize("t", [-1, 0, 1, 2, float("nan"), float("inf")])
def test_invalid_split_parameter_is_atomic(t):
    editor = linked_editor()
    before = editor.snapshot
    with pytest.raises(EditRejectedError), editor.transaction("Invalid split") as tx:
        tx.split_edge("fill", ref(before.document, "fill", 1).node_id, t)
    assert editor.snapshot == before


def test_splitting_checks_all_consumers_structure_permission_and_node_filter():
    editor = linked_editor()
    doc = editor.snapshot.document
    node = ref(doc, "fill", 1).node_id
    editor.select(select("fill"))
    with (
        pytest.raises(EditRejectedError, match="unselected"),
        editor.transaction("Split") as tx,
    ):
        tx.split_edge("fill", node)
    editor.select(select("fill", "outline", nodes=[node]))
    with (
        pytest.raises(EditRejectedError, match="selected nodes"),
        editor.transaction("Split") as tx,
    ):
        tx.split_edge("fill", node)
    peer = ref(doc, "outline", 1).node_id
    editor.select(select("fill", "outline", nodes=[node, peer]))
    with (
        pytest.raises(EditRejectedError, match="not permitted"),
        editor.transaction("Split", allowed=frozenset({"geometry"})) as tx,
    ):
        tx.split_edge("fill", node)
    with editor.transaction("Split selected edges") as tx:
        tx.split_edge("fill", node)
    assert editor.snapshot.selection.node_ids == {node, peer}


def test_delete_selected_node_clears_filter_safely_and_undo_restores_selection():
    doc = import_svg(SVG)
    node = ref(doc, "a", 1).node_id
    selection = select("a", nodes=[node], focus=Rect(0, 0, 32, 32))
    editor = Editor(doc, selection=selection)
    with editor.transaction("Delete corner") as tx:
        tx.delete_node("a", node)
        assert tx.preview_selection == Selection(focus=selection.focus)
        assert tx.node_remapping == ((node, ()),)
    assert editor.snapshot.selection == Selection(focus=selection.focus)
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


def test_delete_checks_pins_and_adjacent_boundary():
    editor = linked_editor()
    doc = editor.snapshot.document
    node = ref(doc, "fill", 1).node_id
    with (
        pytest.raises(EditRejectedError, match="Detach adjacent"),
        editor.transaction("Delete") as tx,
    ):
        tx.delete_node("fill", node)
    with editor.transaction("Detach") as tx:
        tx.detach_boundary(ref(doc, "fill", 1))
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
    editor = linked_editor()
    doc = editor.snapshot.document
    tx = editor.transaction("Split")
    tx.split_edge("fill", ref(doc, "fill", 1).node_id)
    with editor.transaction("Colour") as other:
        other.set_attributes("fill", {"fill": "blue"})
    editor.undo()
    with pytest.raises(StaleRevisionError):
        tx.commit()
    assert editor.snapshot.document == doc


def test_project_reads_version_one_and_rejects_corrupted_boundary_metadata():
    editor = linked_editor()
    data = json.loads(save_project(editor.snapshot.document))
    data["boundaries"][0]["members"][1]["reversed"] = False
    with pytest.raises(DocumentError, match="identical"):
        load_project(json.dumps(data))
    data = json.loads(save_project(import_svg(SVG)))
    data["version"] = 1
    del data["boundaries"]
    doc, _ = load_project(json.dumps(data))
    assert not doc.boundaries
    data["version"] = 2
    with pytest.raises(DocumentError, match="boundaries"):
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


def test_whole_asset_detach_removes_unused_boundary_constraints():
    editor = linked_editor()
    doc = editor.snapshot.document
    node = ref(doc, "fill", 1).node_id
    editor.pin_node("fill", node)
    editor.select(select("fill"))
    before = editor.snapshot.document
    with editor.transaction("Detach entire contour") as tx:
        tx.detach_geometry("fill")
    assert not editor.snapshot.document.boundaries
    editor.select(select("outline"))
    with editor.transaction("Move independent endpoint") as tx:
        tx.update_node("outline", ref(doc, "outline", 0).node_id, (50, 10))
    assert editor.snapshot.document.geometry_for("fill").subpaths[0].nodes[
        1
    ].endpoint == (48, 8)
    editor.undo()
    assert editor.undo().document == before


def test_detaching_one_of_three_members_retains_other_members_link():
    doc = import_svg(
        CONTOURS.replace("</svg>", '<path id="copy" d="M8 8 C16 4 40 4 48 8"/></svg>')
    )
    editor = Editor(doc, selection=Selection.all())
    with editor.transaction("Link three") as tx:
        tx.link_boundary(
            (
                ref(doc, "fill", 1),
                ref(doc, "outline", 1, reverse=True),
                ref(doc, "copy", 1),
            )
        )
    with editor.transaction("Unlink copy") as tx:
        tx.detach_boundary(ref(doc, "copy", 1))
    editor.select(select("fill", "outline"))
    with editor.transaction("Move linked") as tx:
        tx.update_node("fill", ref(doc, "fill", 1).node_id, (17, 4, 40, 4, 48, 8))
    after = editor.snapshot.document
    assert len(after.boundaries[0].members) == 2
    assert after.geometry_for("copy") == doc.geometry_for("copy")


def test_shared_adjacent_edges_in_one_closed_path_propagate_and_split():
    doc = import_svg(
        '<svg width="64" height="64"><path id="a" d="M8 8 L48 8 Z"/></svg>'
    )
    editor = Editor(doc, selection=Selection.all())
    with editor.transaction("Link retraced edge") as tx:
        tx.link_boundary((ref(doc, "a", 1), ref(doc, "a", 0, reverse=True)))
    with editor.transaction("Split both traversals") as tx:
        tx.split_edge("a", ref(doc, "a", 1).node_id, 0.25)
    after = editor.snapshot.document
    nodes = after.geometry_for("a").subpaths[0].nodes
    assert len(nodes) == 4
    assert nodes[1].endpoint == nodes[3].endpoint == (18, 8)
    with editor.transaction("Move both split points") as tx:
        tx.update_node("a", nodes[1].id, (18, 10))
    after = editor.snapshot.document
    assert after.geometry_for("a").node(nodes[3].id).endpoint == (18, 10)
    after.validate()


def test_split_preview_rolls_back_on_a_later_caught_edit_failure():
    editor = linked_editor()
    before = editor.snapshot
    tx = editor.transaction("Split and fail")
    tx.split_edge("fill", ref(before.document, "fill", 1).node_id)
    with pytest.raises(EditRejectedError):
        tx.set_attributes("other", {"stroke": "red"})
    with pytest.raises(EditRejectedError, match="failed"):
        tx.commit()
    assert editor.snapshot == before


def test_shared_boundary_definition_users_and_ancestor_locks_are_checked():
    source = """<svg width="64" height="64"><defs><g id="library">
    <path id="a" d="M8 8 L48 8"/></g></defs>
    <use id="visible" href="#a"/>
    <path id="b" d="M48 8 L8 8"/></svg>"""
    doc = import_svg(source)
    editor = Editor(doc, selection=select("visible", "b"))
    with editor.transaction("Link definition") as tx:
        tx.link_boundary((ref(doc, "visible", 1), ref(doc, "b", 1, reverse=True)))
    editor.set_locks("library", frozenset({"geometry"}))
    before = editor.snapshot
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Edit through peer") as tx,
    ):
        tx.update_node("b", ref(doc, "b", 1).node_id, (9, 9))
    assert editor.snapshot == before
