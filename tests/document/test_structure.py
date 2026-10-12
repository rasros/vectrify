"""Structural commands preserve identity, context and authorization."""

from dataclasses import replace

import numpy as np
import pytest

from tests.document.test_document import SHARED, SVG, render, select
from vectrify.document import (
    DocumentError,
    Editor,
    EditRejectedError,
    Element,
    Selection,
    export_svg,
    import_svg,
    load_project,
    save_project,
)
from vectrify.document.svg import parse_path

OBJECTS = """<svg width="100"
height="100"
fill="red">
<g id="layer"
transform="translate(5 5)">
<rect id="a"
width="20"
height="20"/>
<rect id="b"
x="10"
y="10"
width="20"
height="20"
fill="blue"/>
<rect id="c"
x="20"
y="20"
width="20"
height="20"
fill="green"/>
</g>
</svg>"""


def test_insert_into_selected_container_then_edit_new_object_in_same_transaction():
    editor = Editor(import_svg(OBJECTS), selection=select("layer"))
    before = editor.snapshot
    geometry = parse_path("M50 50 L60 50 L60 60 Z")
    with editor.transaction("Add path") as tx:
        tx.insert_object(
            "layer",
            Element("new", "path", geometry_id=geometry.id),
            geometries=(geometry,),
            index=1,
        )
        tx.set_attributes("new", {"fill": "orange"})
    assert [e.id for e in editor.snapshot.document.element("layer").children] == [
        "a",
        "new",
        "b",
        "c",
    ]
    assert editor.snapshot.document.geometry_for("new") == geometry
    assert editor.undo().document == before.document
    assert editor.redo().document.element("new").get("fill") == "orange"


@pytest.mark.parametrize(
    "failure", ["scope", "nodes", "duplicate", "permission", "lock", "reference"]
)
def test_invalid_insert_is_atomic(failure):
    editor = Editor(import_svg(SVG), selection=select("layer"))
    new = Element("new", "rect", (("width", "3"), ("height", "3")))
    allowed = frozenset({"structure"})
    if failure == "scope":
        editor.select(select("a"))
    elif failure == "nodes":
        node = editor.snapshot.document.geometry_for("a").subpaths[0].nodes[0]
        editor.select(select("layer", nodes=[node.id]))
    elif failure == "duplicate":
        new = replace(new, id="a")
    elif failure == "permission":
        allowed = frozenset({"geometry"})
    elif failure == "lock":
        editor.set_locks("layer", frozenset({"structure"}))
    else:
        new = Element("new", "use", (("href", "#missing"),))
    before = editor.snapshot
    with (
        pytest.raises(DocumentError),
        editor.transaction("Invalid add", allowed=allowed) as tx,
    ):
        tx.insert_object("layer", new)
    assert editor.snapshot == before


def test_group_and_ungroup_keep_rendering_ids_and_paint_order():
    editor = Editor(import_svg(OBJECTS), selection=select("a", "b"))
    before = editor.snapshot.document
    with editor.transaction("Group") as tx:
        group = tx.group_objects(frozenset({"a", "b"}))
    grouped = editor.snapshot.document
    assert np.array_equal(render(OBJECTS), render(export_svg(grouped)))
    assert grouped.element("a") == before.element("a")
    editor.select(select(group))
    selection = editor.snapshot.selection
    with editor.transaction("Ungroup") as tx:
        assert tx.ungroup_object(group) == ("a", "b")
        assert tx.preview_selection.object_ids == {"a", "b"}
    assert editor.snapshot.document == before
    assert editor.snapshot.selection == select("a", "b")
    assert editor.undo().selection == selection
    assert editor.redo().selection.object_ids == {"a", "b"}


def test_ungroup_transfers_transform_and_inherited_paint_without_changing_pixels():
    source = OBJECTS.replace(
        'id="layer"', 'id="layer" stroke="black" stroke-width="2" fill-opacity="0.6"'
    ).replace('id="a"', 'id="a" transform="scale(0.5)"')
    editor = Editor(import_svg(source), selection=select("layer"))
    with editor.transaction("Ungroup transformed layer") as tx:
        tx.ungroup_object("layer")
    after = editor.snapshot.document
    assert after.element("a").get("transform") == "translate(5 5) scale(0.5)"
    assert after.element("b").get("fill") == "blue"
    assert np.array_equal(render(source), render(export_svg(after)))


@pytest.mark.parametrize("attribute", ['opacity="0.5"', 'clip-path="url(#clip)"'])
def test_ungroup_distributes_group_compositing_and_is_undoable(attribute):
    source = OBJECTS.replace('id="layer"', f'id="layer" {attribute}').replace(
        "</svg>",
        """<defs>
<clipPath id="clip">
<rect width="20"
height="20"/>
</clipPath>
</defs>
</svg>""",
    )
    editor = Editor(import_svg(source), selection=select("layer"))
    before = editor.snapshot.document
    with editor.transaction("Ungroup") as tx:
        ids = tx.ungroup_object("layer")
    after = editor.snapshot.document
    if attribute.startswith("opacity"):
        assert all(after.element(oid).get("opacity") == "0.5" for oid in ids)
        assert render(export_svg(after))[20, 20, 3] > render(source)[20, 20, 3]
    else:
        assert all(after.element(oid).get("clip-path") for oid in ids)
        np.testing.assert_array_equal(render(source), render(export_svg(after)))
    assert editor.undo().document == before
    assert editor.redo().document == after


def test_grouping_nonconsecutive_objects_gathers_at_the_frontmost_position():
    editor = Editor(import_svg(OBJECTS), selection=select("a", "c"))
    before = editor.snapshot.document
    with editor.transaction("Group") as tx:
        group = tx.group_objects(frozenset({"a", "c"}))
    assert [e.id for e in editor.snapshot.document.element(group).children] == [
        "a",
        "c",
    ]
    after = editor.snapshot.document
    assert [e.id for e in after.element("layer").children] == ["b", group]
    assert after.element("b") == before.element("b")
    assert editor.undo().document == before
    assert editor.redo().document == after


def test_grouping_across_containers_keeps_placement_and_counts_descendants_once():
    source = """<svg width="100" height="100">
    <g id="back" transform="translate(10 10)" fill="red" opacity=".5">
      <rect id="a" width="10" height="10"/>
      <rect id="keep" y="30" width="10" height="10"/>
    </g>
    <g id="front" transform="translate(50 10) scale(2)" fill="blue">
      <rect id="b" width="10" height="10"/>
    </g></svg>"""
    editor = Editor(import_svg(source), selection=select("a", "front", "b"))
    before = editor.snapshot.document
    with editor.transaction("Group") as tx:
        oid = tx.group_objects(frozenset({"a", "front", "b"}))
    after = editor.snapshot.document
    assert [e.id for e in after.root.children] == ["back", oid]
    assert [e.id for e in after.element(oid).children] == ["a", "front"]
    assert after.element("a").get("fill") == "red"
    assert after.element("a").get("transform") == "translate(10 10)"
    assert after.element("keep") == before.element("keep")
    # The object left the faded group; grouping is allowed to change its alpha.
    image = render(export_svg(after))
    assert image[15, 15].tolist() == [255, 0, 0, 255]
    assert image[15, 55].tolist() == [0, 0, 255, 255]
    assert image[45, 15, 3] == 128
    assert editor.undo().document == before
    assert editor.redo().document == after


@pytest.mark.parametrize("bbox", [False, True])
def test_ungroup_preserves_nested_clips_with_child_transforms_and_opacity(bbox):
    units = 'clipPathUnits="objectBoundingBox"' if bbox else ""
    size = 0.75 if bbox else 30
    source = f"""<svg width="100" height="100">
    <defs>
      <clipPath id="outer" {units}><rect width="{size}" height="{size}"/></clipPath>
      <clipPath id="inner"><rect x="5" width="20" height="30"/></clipPath>
    </defs>
    <g id="g" transform="translate(10 10)" opacity=".5" clip-path="url(#outer)">
      <path id="a" transform="scale(2)" clip-path="url(#inner)" opacity=".4"
            fill="red" d="M0 0H20V20H0Z"/>
    </g></svg>"""
    editor = Editor(import_svg(source), selection=select("g"))
    before = editor.snapshot.document
    with editor.transaction("Ungroup") as tx:
        assert tx.ungroup_object("g") == ("a",)
    after = editor.snapshot.document
    assert float(after.element("a").get("opacity", "1") or "1") == pytest.approx(0.2)
    # Cairo cannot render objectBoundingBox clips on groups; compare with
    # the equivalent clip in the group's 40 by 40 user-space bounds.
    expected = (
        source.replace(units, "").replace(
            f'width="{size}" height="{size}"', 'width="30" height="30"'
        )
        if bbox
        else source
    )
    np.testing.assert_array_equal(render(expected), render(export_svg(after)))
    assert editor.undo().document == before
    assert editor.redo().document == after


def test_reorder_selected_object_preserves_other_objects_and_supports_undo():
    editor = Editor(import_svg(OBJECTS), selection=select("a"))
    before = editor.snapshot.document
    with editor.transaction("Bring to front") as tx:
        tx.reorder_object("a", 2)
    after = editor.snapshot.document
    assert [e.id for e in after.element("layer").children] == ["b", "c", "a"]
    assert after.element("b") == before.element("b")
    assert not np.array_equal(render(export_svg(after)), render(OBJECTS))
    assert editor.undo().document == before


def test_delete_group_remaps_descendant_selection_changed_during_preview():
    editor = Editor(import_svg(OBJECTS), selection=select("layer"))
    before = editor.snapshot.document
    tx = editor.transaction("Delete layer")
    tx.delete_objects(frozenset({"layer", "a"}))
    editor.select(select("b"))
    tx.commit()
    assert editor.snapshot.selection == Selection()
    assert not editor.snapshot.document.root.children
    assert editor.undo().selection == select("b")
    assert editor.snapshot.document == before
    assert editor.redo().selection == Selection()


def test_delete_with_node_selection_chosen_during_preview_clears_scope():
    editor = Editor(import_svg(SVG), selection=select("a"))
    node = editor.snapshot.document.geometry_for("a").subpaths[0].nodes[0]
    tx = editor.transaction("Delete")
    tx.delete_objects(frozenset({"a"}))
    editor.select(select("a", nodes=[node.id]))
    tx.commit()
    assert editor.snapshot.selection == Selection()


def test_ungroup_selected_group_keeps_node_filter_on_surviving_child():
    editor = Editor(import_svg(SVG), selection=select("layer"))
    node = editor.snapshot.document.geometry_for("a").subpaths[0].nodes[0]
    tx = editor.transaction("Ungroup")
    tx.ungroup_object("layer")
    editor.select(select("layer", nodes=[node.id]))
    tx.commit()
    assert editor.snapshot.selection == select("a", nodes=[node.id])


def test_delete_referenced_definition_requires_explicit_consumer_deletion():
    editor = Editor(import_svg(SHARED), selection=Selection.all())
    with (
        pytest.raises(EditRejectedError, match="references"),
        editor.transaction("Delete definition") as tx,
    ):
        tx.delete_objects(frozenset({"shape"}))
    with editor.transaction("Delete whole referenced graph") as tx:
        tx.delete_objects(frozenset({"library", "first", "second", "clipped"}))
    assert not editor.snapshot.document.root.children


def test_delete_pinned_use_instance_is_rejected():
    editor = Editor(import_svg(SHARED), selection=select("first"))
    node = editor.snapshot.document.geometry_for("first").subpaths[0].nodes[0]
    editor.pin_node("first", node.id)
    with (
        pytest.raises(EditRejectedError, match="pinned"),
        editor.transaction("Delete") as tx,
    ):
        tx.delete_objects(frozenset({"first"}))


@pytest.mark.parametrize("object_id", ["a", "layer"])
def test_confirmed_delete_pinned_path_or_group_is_one_undoable_edit(object_id):
    editor = Editor(import_svg(SVG), selection=select(object_id))
    node = editor.snapshot.document.geometry_for("a").subpaths[0].nodes[0]
    editor.pin_node("a", node.id)
    before = editor.snapshot.document
    with (
        pytest.raises(EditRejectedError, match="pinned"),
        editor.transaction("Delete") as tx,
    ):
        tx.delete_objects(frozenset({object_id}))
    assert editor.snapshot.document == before
    with editor.transaction("Delete") as tx:
        tx.delete_objects(frozenset({object_id}), allow_pinned=True)
    assert object_id not in {e.id for e in editor.snapshot.document.elements()}
    assert editor.undo().document == before
    assert editor.snapshot.document.geometry_for("a").node(node.id).pinned
    editor.redo()
    assert object_id not in {e.id for e in editor.snapshot.document.elements()}


def test_confirmed_delete_instance_keeps_shared_pins_and_reference_checks():
    editor = Editor(import_svg(SHARED), selection=Selection.all())
    node = editor.snapshot.document.geometry_for("first").subpaths[0].nodes[0]
    editor.pin_node("first", node.id)
    before = editor.snapshot.document
    with (
        pytest.raises(EditRejectedError, match="references"),
        editor.transaction("Delete definition") as tx,
    ):
        tx.delete_objects(frozenset({"shape"}), allow_pinned=True)
    assert editor.snapshot.document == before
    editor.set_locks("first", frozenset({"structure"}))
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Delete locked instance") as tx,
    ):
        tx.delete_objects(frozenset({"first"}), allow_pinned=True)
    editor.set_locks("first", frozenset())
    with editor.transaction("Delete instance") as tx:
        tx.delete_objects(frozenset({"first"}), allow_pinned=True)
    assert editor.snapshot.document.geometry_for("second").node(node.id).pinned


def test_inherited_locks_and_external_group_instances_block_structure_changes():
    source = OBJECTS.replace("</svg>", '<use id="copy" href="#layer" x="50"/></svg>')
    editor = Editor(import_svg(source), selection=select("a"))
    with (
        pytest.raises(EditRejectedError, match="unselected"),
        editor.transaction("Reorder") as tx,
    ):
        tx.reorder_object("a", 2)
    editor.select(select("a", "copy"))
    editor.set_locks("copy", frozenset({"structure"}))
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Reorder") as tx,
    ):
        tx.reorder_object("a", 2)


def test_ungroup_then_delete_composes_mapping_and_roundtrips_project():
    editor = Editor(import_svg(OBJECTS), selection=select("layer"))
    with editor.transaction("Ungroup then remove") as tx:
        tx.ungroup_object("layer")
        tx.delete_objects(frozenset({"a", "c"}))
        assert dict(tx.object_remapping)["layer"] == ("b",)
    doc = editor.snapshot.document
    assert editor.snapshot.selection.object_ids == {"b"}
    assert load_project(save_project(doc, editor.snapshot.selection)) == (
        doc,
        editor.snapshot.selection,
    )


def test_invalid_late_structure_command_poisons_batch():
    editor = Editor(import_svg(OBJECTS), selection=select("a", "b"))
    before = editor.snapshot
    tx = editor.transaction("Group and fail")
    tx.group_objects(frozenset({"a", "b"}))
    with pytest.raises(EditRejectedError):
        tx.delete_objects(frozenset({"c"}))
    with pytest.raises(EditRejectedError, match="failed"):
        tx.commit()
    assert editor.snapshot == before


@pytest.mark.parametrize("internal", [False, True])
def test_ungroup_distributes_attributes_into_referenced_children(internal):
    instance = '<use id="copy" href="#a" x="40"/>'
    source = (
        OBJECTS.replace("</g>", instance + "</g>")
        if internal
        else OBJECTS.replace("</svg>", instance + "</svg>")
    )
    editor = Editor(import_svg(source), selection=Selection.all())
    before = editor.snapshot.document
    with editor.transaction("Ungroup") as tx:
        tx.ungroup_object("layer")
    after = editor.snapshot.document
    after.validate()
    assert after.element("copy").get("href") == "#a"
    assert after.element("a").get("transform") == "translate(5 5)"
    assert editor.undo().document == before
    assert editor.redo().document == after


def test_ungroup_does_not_erase_a_property_lock():
    editor = Editor(import_svg(OBJECTS), selection=select("layer"))
    editor.set_locks("layer", frozenset({"fill"}))
    with (
        pytest.raises(EditRejectedError, match="Unlock"),
        editor.transaction("Ungroup") as tx,
    ):
        tx.ungroup_object("layer")
