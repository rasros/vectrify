"""The UI adapter uses backend transactions, identities and constraints."""

import base64
import io
import json

import pytest
from PIL import Image

from vectrify.document import DocumentError, StaleRevisionError, import_svg
from vectrify.ui.session import Session

SVG = """<svg width="100" height="100"><g id="layer">
<path id="a" fill="red" d="M0 0 L20 0 L20 20 Z"/>
<rect id="b" x="30" width="20" height="20" fill="blue"/>
</g></svg>"""


def send(session, command, **data):
    return session.action(
        {
            "command": command,
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            **data,
        }
    )


def test_ui_paint_and_history_leave_original_svg_unchanged_until_applied():
    session = Session(import_svg(SVG))
    before = session.state()["svg"]
    selection = send(session, "select", objects=["a"])
    assert "svg" not in selection
    result = send(session, "paint", changes={"fill": "green"})
    assert 'fill="green"' in result["svg"]
    assert result["undo"] == ["Change paint"]
    assert send(session, "undo")["svg"] == before
    assert 'fill="green"' in send(session, "redo")["svg"]


def test_ui_node_edit_pin_and_lock_are_enforced():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["a"])
    node = session.nodes("a")["geometry"]["subpaths"][0]["nodes"][1]
    send(session, "node", object="a", node=node["id"], values=[22, 0])
    send(session, "pin", object="a", node=node["id"], pinned=True)
    with pytest.raises(DocumentError, match="pinned"):
        send(session, "node", object="a", node=node["id"], values=[25, 0])
    send(session, "locks", object="a", locks=["paint"])
    with pytest.raises(DocumentError, match="locked"):
        send(session, "paint", changes={"fill": "black"})
    assert session.editor.snapshot.document.geometry_for("a").node(
        node["id"]
    ).endpoint == (22, 0)


def test_stale_edit_is_rejected_after_reopen_even_at_same_revision():
    session = Session(import_svg(SVG))
    old = {"epoch": session.epoch, "revision": 0, "command": "select", "objects": ["a"]}
    send(session, "open", source=SVG, name="new.svg")
    with pytest.raises(StaleRevisionError):
        session.action(old)


def test_open_invalid_document_preserves_current_edits():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["a"])
    send(session, "paint", changes={"fill": "green"})
    before = session.state()
    with pytest.raises(DocumentError):
        send(session, "open", source="<svg><script/></svg>", name="bad.svg")
    assert session.state() == before


def test_project_roundtrip_preserves_reference_and_constraints():
    session = Session(import_svg(SVG))
    stream = io.BytesIO()
    Image.new("RGB", (10, 10), "red").save(stream, format="PNG")
    reference = {
        "name": "original.png",
        "data_url": "data:image/png;base64,"
        + base64.b64encode(stream.getvalue()).decode(),
        "opacity": 0.35,
    }
    send(session, "reference", reference=reference)
    send(session, "select", objects=["a"])
    send(session, "locks", object="a", locks=["geometry"])
    saved = session.project()
    restored = Session(import_svg(SVG))
    send(restored, "open", source=saved, name="saved.vectrify")
    assert restored.reference == reference
    assert restored.editor.snapshot.document == session.editor.snapshot.document
    assert restored.editor.snapshot.selection == session.editor.snapshot.selection
    assert json.loads(saved)["vectrify_editor"] == 1


def test_group_selects_result_and_ungroup_restores_child_selection():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["a", "b"])
    state = send(session, "group")
    assert len(state["selection"]["objects"]) == 1
    group = state["selection"]["objects"][0]
    assert session.editor.snapshot.document.element(group).tag == "g"
    state = send(session, "ungroup")
    assert set(state["selection"]["objects"]) == {"a", "b"}


@pytest.mark.parametrize(
    "data", ["M60 60 L80 60 L80 70 Z", "M60 60 C60 40 90 40 90 60"]
)
def test_draw_path_is_editable_selected_and_undoable(data):
    session = Session(import_svg(SVG))
    before = session.editor.snapshot.document
    state = send(session, "add_path", d=data, stroke_width=3)
    oid = state["selection"]["objects"][0]
    document = session.editor.snapshot.document
    element = document.element(oid)
    geometry = document.geometry_for(oid)
    assert element.tag == "path"
    assert geometry.subpaths[0].closed == data.endswith("Z")
    assert element.get("fill") == ("#83b899" if data.endswith("Z") else "none")
    assert element.get("stroke") == ("none" if data.endswith("Z") else "#83b899")
    assert state["undo"] == ["Draw path"]
    if "C" in data:
        assert geometry.subpaths[0].nodes[1].command == "C"
    assert send(session, "undo")["selection"]["objects"] == []
    assert session.editor.snapshot.document == before
    send(session, "redo")
    assert session.editor.snapshot.document == document


@pytest.mark.parametrize("data", ["", "M1 2", "M1 2 L3 4 M5 6 L7 8", "M1 2 Lnan 3"])
def test_invalid_draw_path_does_not_change_drawing(data):
    session = Session(import_svg(SVG))
    before = session.editor.snapshot.document
    with pytest.raises(DocumentError):
        send(session, "add_path", d=data)
    assert session.editor.snapshot.document == before
    assert session.editor.undo_labels == ()


def test_multiselection_drag_uses_each_parent_coordinate_frame_and_skips_children():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["a", "b"])
    send(session, "move", dx=0, dy=0, offsets={"a": [5, 0], "b": [10, 0]})
    assert (
        session.editor.snapshot.document.element("a").get("transform")
        == "translate(5.0 0.0)"
    )
    assert (
        session.editor.snapshot.document.element("b").get("transform")
        == "translate(10.0 0.0)"
    )
    send(session, "select", objects=["layer", "a"])
    send(session, "move", dx=5, dy=2)
    assert (
        session.editor.snapshot.document.element("a").get("transform")
        == "translate(5.0 0.0)"
    )
    assert (
        session.editor.snapshot.document.element("layer").get("transform")
        == "translate(5.0 2.0)"
    )


def test_rectangle_selection_and_empty_paint_scope():
    session = Session(import_svg(SVG))
    assert send(session, "rectangle", rectangle=[29, 0, 22, 21])["selection"][
        "objects"
    ] == ["b"]
    assert (
        send(session, "rectangle", rectangle=[90, 90, 2, 2])["selection"]["objects"]
        == []
    )
    with pytest.raises(DocumentError, match="Select"):
        send(session, "paint", changes={"fill": "green"})


def test_ui_cannot_inject_attributes_or_pin_unselected_object():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["b"])
    with pytest.raises(DocumentError, match="Unsupported paint"):
        send(session, "paint", changes={"onclick": "alert(1)"})
    with pytest.raises(DocumentError, match="Select"):
        send(session, "pin", object="a", node="missing", pinned=True)


def test_split_disconnected_selects_parts_and_undo_restores_compound_path():
    session = Session(
        import_svg(
            '<svg width="100" height="100"><path id="p" '
            'd="M0 0H10V10H0Z M30 30H40V40H30Z"/></svg>'
        )
    )
    send(session, "select", objects=["p"])
    result = send(session, "split_disconnected")
    assert len(result["selection"]["objects"]) == 2
    assert result["undo"] == ["Split disconnected parts"]
    assert all(
        session.editor.snapshot.document.element(oid).tag == "path"
        for oid in result["selection"]["objects"]
    )
    assert send(session, "undo")["selection"]["objects"] == ["p"]
    assert len(send(session, "redo")["selection"]["objects"]) == 2


def test_join_button_command_combines_split_parts_and_undo_restores_selection():
    session = Session(
        import_svg(
            '<svg width="100" height="100"><path id="p" '
            'd="M0 0H10V10H0Z M30 30H40V40H30Z"/></svg>'
        )
    )
    send(session, "select", objects=["p"])
    split = send(session, "split_disconnected")
    joined = send(session, "join_paths")
    assert len(joined["selection"]["objects"]) == 1
    assert joined["undo"][-1] == "Join outlines"
    oid = joined["selection"]["objects"][0]
    assert len(session.editor.snapshot.document.geometry_for(oid).subpaths) == 2
    assert send(session, "undo")["selection"] == split["selection"]


def test_join_nonadjacent_regions_averages_by_area_at_frontmost_position():
    session = Session(
        import_svg(
            '<svg width="100" height="100">'
            '<path id="a" fill="red" d="M0 0H10V10H0Z"/>'
            '<rect id="middle" width="5" height="5"/>'
            '<path id="b" fill="blue" d="M40 0H70V10H40Z"/></svg>'
        )
    )
    send(session, "select", objects=["b", "a"])
    result = send(session, "join_paths")
    doc = session.editor.snapshot.document
    oid = result["selection"]["objects"][0]
    assert [e.id for e in doc.root.children] == ["middle", oid]
    assert doc.element(oid).get("fill") == "#4000bf"
    assert send(session, "undo")["selection"]["objects"] == ["a", "b"]


def test_rectangle_defaults_to_containing_all_parts_of_a_compound_path():
    session = Session(
        import_svg(
            '<svg width="100" height="100"><g id="g">'
            '<path id="islands" d="M10 10H20V20H10Z M70 70H80V80H70Z"/>'
            '<rect id="small" x="12" y="12" width="2" height="2"/>'
            "</g></svg>"
        )
    )
    assert send(session, "rectangle", rectangle=[9, 9, 12, 12])["selection"][
        "objects"
    ] == ["small"]
    assert send(session, "rectangle", rectangle=[9, 9, 72, 72])["selection"][
        "objects"
    ] == ["islands", "small"]
    assert send(session, "rectangle", rectangle=[9, 9, 12, 12], mode="intersect")[
        "selection"
    ]["objects"] == ["islands", "small"]


def test_join_options_choose_source_and_validate_before_applying():
    session = Session(
        import_svg("""<svg width="100" height="100"><g fill="red">
    <path id="a" d="M0 0H10V10H0Z"/><path id="b" fill="blue" d="M40 0H70V10H40Z"/>
    </g></svg>""")
    )
    send(session, "select", objects=["a", "b"])
    before = session.editor.snapshot
    for options in [
        [],
        {"colors": {}},
        {"colors": "unknown"},
        {"colors": "source"},
        {"colors": "mix", "color_source": "a"},
        {"extra": True},
        {"colors": "source", "color_source": "outside"},
    ]:
        with pytest.raises(DocumentError):
            send(session, "join_paths", options=options)
        assert session.editor.snapshot == before
    result = send(
        session, "join_paths", options={"colors": "source", "color_source": "a"}
    )
    oid = result["selection"]["objects"][0]
    assert session.editor.snapshot.document.element(oid).get("fill") == "red"
    assert result["undo"] == ["Join outlines"]
    send(session, "undo")
    assert session.editor.snapshot.document == before.document


def test_ui_join_mixed_group_and_path_uses_descendant_color_and_single_undo():
    session = Session(
        import_svg("""<svg width="100" height="100">
      <g id="group" fill="red"><g id="nested">
        <path id="a" d="M0 0H10V10H0Z"/></g></g>
      <path id="b" fill="blue" d="M20 0H30V10H20Z"/></svg>""")
    )
    send(session, "select", objects=["group", "b"])
    before = session.editor.snapshot
    result = send(
        session, "join_paths", options={"colors": "source", "color_source": "a"}
    )
    assert len(result["selection"]["objects"]) == 1
    oid = result["selection"]["objects"][0]
    assert session.editor.snapshot.document.element(oid).get("fill") == "red"
    assert result["undo"] == ["Join outlines"]
    send(session, "undo")
    assert session.editor.snapshot.document == before.document
    assert session.editor.snapshot.selection == before.selection


def test_detach_instance_enables_split_and_keeps_other_instances_unchanged():
    session = Session(
        import_svg(
            '<svg width="100" height="100"><defs>'
            '<path id="shape" '
            'd="M0 0H10V10H0Z M30 30H40V40H30Z"/></defs>'
            '<use id="first" href="#shape" fill="red"/>'
            '<use id="second" href="#shape" fill="blue" x="50"/></svg>'
        )
    )
    original = session.editor.snapshot.document
    send(session, "select", objects=["first"])
    detached = send(session, "detach")
    assert detached["selection"]["objects"] == ["first"]
    assert session.editor.snapshot.document.element("first").tag == "path"
    split = send(session, "split_disconnected")
    assert len(split["selection"]["objects"]) == 2
    document = session.editor.snapshot.document
    assert document.geometry_for("second") == original.geometry_for("second")
    assert document.element("shape") == original.element("shape")
    send(session, "undo")
    send(session, "undo")
    assert session.editor.snapshot.document == original


def test_rename_preserves_references_and_roundtrips_with_history():
    from vectrify.document import export_svg
    from vectrify.document.project import load_project, save_project

    session = Session(
        import_svg(
            '<svg width="100" height="100"><defs>'
            '<path id="shape" d="M0 0 L20 0 L20 20Z"/></defs>'
            '<use id="copy" href="#shape"/></svg>'
        )
    )
    send(session, "select", objects=["shape"])
    before = session.editor.snapshot.document
    result = send(session, "rename", object="shape", name="  Tree & <branches> 🌳  ")
    renamed = session.editor.snapshot.document
    assert (
        next(o for o in result["objects"] if o["id"] == "shape")["label"]
        == "Tree & <branches> 🌳"
    )
    assert renamed.element("copy").get("href") == "#shape"
    assert renamed.geometries == before.geometries
    assert load_project(save_project(renamed))[0] == renamed
    assert (
        import_svg(export_svg(renamed)).element("shape").name == "Tree & <branches> 🌳"
    )
    send(session, "undo")
    assert session.editor.snapshot.document == before
    send(session, "redo")
    assert session.editor.snapshot.document == renamed
    send(session, "rename", object="shape", name="")
    assert session.editor.snapshot.document.element("shape").name == ""
    assert (
        next(o for o in session.state()["objects"] if o["id"] == "shape")["label"]
        == "shape"
    )


def test_rename_rejects_invalid_names_and_unselected_targets():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["a"])
    before = session.editor.snapshot
    for name in (None, "x" * 201, "bad\x00name"):
        with pytest.raises(DocumentError):
            send(session, "rename", object="a", name=name)
    with pytest.raises(DocumentError):
        send(session, "rename", object="b", name="Other")
    assert session.editor.snapshot == before


def test_contact_preview_apply_single_path_edit_and_unlink():
    from vectrify.document.topology import edge

    session = Session(
        import_svg(
            '<svg width="30" height="30">'
            '<path id="a" d="M0 0L10 0L10 20L0 20Z"/>'
            '<path id="b" d="M10 0L20 0L20 20L10 20Z"/></svg>'
        )
    )
    send(session, "select", objects=["a", "b"])
    before = session.editor.snapshot.document
    job = session.operation(
        {
            "command": "start",
            "action": "link",
            "method": "boundaries",
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            "permissions": {"geometry": True, "structure": True},
            "settings": {"tolerance": 0.01},
        }
    )
    assert job["status"] == "ready"
    assert job["result"]["metrics"]["edges"] == 1
    assert job["result"]["previews"]["after"].startswith("data:image/png;base64,")
    assert session.editor.snapshot.document == before
    session.operation({"command": "apply", "job": job["id"]})
    send(session, "select", objects=["b"])
    doc = session.editor.snapshot.document
    member = doc.boundaries[0].members[0]
    node = doc.geometry(member.geometry_id).node(member.node_id)
    send(
        session,
        "node",
        object="b",
        node=node.id,
        values=[node.values[0] + 1, node.values[1]],
    )
    doc = session.editor.snapshot.document
    assert (
        edge(doc, doc.boundaries[0].members[0]).points
        == edge(doc, doc.boundaries[0].members[1]).points
    )
    assert session.editor.snapshot.selection.object_ids == {"b"}
    with pytest.raises(DocumentError, match="Unlink"):
        send(session, "move", dx=1, dy=0)
    send(session, "unlink_boundaries")
    assert not session.editor.snapshot.document.boundaries
    send(session, "undo")
    assert session.editor.snapshot.document.boundaries
