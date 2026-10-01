"""The UI adapter uses backend transactions, identities and constraints."""

import base64
import io
import json

import pytest
from PIL import Image

from vectrify.document import (
    DocumentError,
    Selection,
    StaleRevisionError,
    import_svg,
)
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


def test_paint_needs_a_selection():
    session = Session(import_svg(SVG))
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


def snap_job(session, tolerance):
    return session.operation(
        {
            "command": "start",
            "action": "snap",
            "method": "edges",
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            "permissions": {"geometry": True, "structure": True},
            "settings": {"tolerance": tolerance},
        }
    )


def test_snap_edges_preview_apply_and_independent_edits():
    session = Session(
        import_svg(
            '<svg width="30" height="30">'
            '<path id="a" d="M0 0L10 0L10 20L0 20Z"/>'
            '<path id="b" d="M10.2 0L20 0L20 20L10.2 20Z"/></svg>'
        )
    )
    send(session, "select", objects=["a", "b"])
    before = session.editor.snapshot.document
    job = snap_job(session, 0.5)
    assert job["status"] == "ready"
    assert job["result"]["changed"]
    assert job["result"]["metrics"]["edges"] == 1
    assert job["result"]["previews"]["after"].startswith("data:image/png;base64,")
    assert session.editor.snapshot.document == before
    session.operation({"command": "apply", "job": job["id"]})
    assert session.state()["undo"] == ["Snap edges"]
    doc = session.editor.snapshot.document
    assert {n.endpoint[0] for n in doc.geometry_for("a").subpaths[0].nodes} == {
        0,
        10.2,
    }
    assert "shared_edges" not in session.state()["objects"][0]
    # The paths stay independent: an edit or a move of one leaves the other.
    send(session, "select", objects=["b"])
    node = doc.geometry_for("b").subpaths[0].nodes[0]
    send(session, "node", object="b", node=node.id, values=[11, 0])
    send(session, "move", dx=1, dy=0)
    assert session.editor.snapshot.document.geometry_for("a") == doc.geometry_for("a")
    send(session, "undo")
    send(session, "undo")
    send(session, "undo")
    assert session.editor.snapshot.document == before


def test_snapping_edges_that_already_meet_proposes_no_change():
    session = Session(
        import_svg(
            '<svg width="30" height="30">'
            '<path id="a" d="M0 0L10 0L10 20L0 20Z"/>'
            '<path id="b" d="M10 0L20 0L20 20L10 20Z"/></svg>'
        )
    )
    send(session, "select", objects=["a", "b"])
    job = snap_job(session, 0.01)
    assert job["result"]["metrics"]["edges"] == 1
    assert not job["result"]["changed"]


def test_knife_cuts_selected_paths_into_pieces_that_meet_and_undoes_in_one_step():
    session = Session(
        import_svg(
            '<svg width="100" height="100"><g transform="translate(10 0)">'
            '<path id="p" fill="red" d="M0 0H20V20H0Z"/></g></svg>'
        )
    )
    send(session, "select", objects=["p"])
    result = send(session, "knife", start=[15, -5], end=[15, 30])
    pieces = result["selection"]["objects"]
    assert result["undo"] == ["Cut with knife"]
    assert len(pieces) == 2
    assert "p" in pieces
    doc = session.editor.snapshot.document
    seams = [
        {n.endpoint for s in doc.geometry_for(oid).subpaths for n in s.nodes}
        & {(5.0, 0.0), (5.0, 20.0)}
        for oid in pieces
    ]
    assert seams == [{(5.0, 0.0), (5.0, 20.0)}] * 2
    # Moving a seam node in one piece leaves the other piece alone.
    oid, other = pieces[1], pieces[0]
    send(session, "select", objects=[oid])
    node = next(
        n
        for s in doc.geometry_for(oid).subpaths
        for n in s.nodes
        if n.endpoint == (5.0, 0.0)
    )
    send(session, "node", object=oid, node=node.id, values=[7, 0])
    after = session.editor.snapshot.document
    assert after.geometry_for(other) == doc.geometry_for(other)
    send(session, "undo")
    assert send(session, "undo")["selection"]["objects"] == ["p"]
    assert len(session.editor.snapshot.document.geometries) == 1


def test_knife_line_that_ends_inside_changes_nothing():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["a"])
    before = session.editor.snapshot.document
    with pytest.raises(DocumentError, match="Drag the knife across"):
        send(session, "knife", start=[10, -5], end=[10, 5])
    assert session.editor.snapshot.document == before


def test_dragging_a_node_moves_both_handles_and_a_handle_drag_only_itself():
    session = Session(
        import_svg(
            '<svg width="40" height="40"><path id="p" fill="none" '
            'd="M0 0 C0 10 10 10 10 0 C10 -10 20 -10 20 0"/></svg>'
        )
    )
    send(session, "select", objects=["p"])
    _, point, end = session.nodes("p")["geometry"]["subpaths"][0]["nodes"]
    send(session, "node", object="p", node=point["id"], values=[0, 10, 10, 10, 12, 3])
    _, point, end = session.nodes("p")["geometry"]["subpaths"][0]["nodes"]
    assert point["values"] == (0, 10, 12, 13, 12, 3)
    assert end["values"] == (12, -7, 20, -10, 20, 0)
    send(session, "node", object="p", node=end["id"], values=[14, -9, 20, -10, 20, 0])
    _, point, end = session.nodes("p")["geometry"]["subpaths"][0]["nodes"]
    assert point["values"] == (0, 10, 12, 13, 12, 3)
    assert end["values"] == (14, -9, 20, -10, 20, 0)


def test_nodes_payload_is_the_geometry_alone():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["a"])
    payload = session.nodes("a")
    assert set(payload) == {"epoch", "revision", "object", "geometry"}
    json.dumps(payload)


def test_deleting_a_start_point_and_a_contour_from_the_editor():
    session = Session(
        import_svg(
            '<svg width="40" height="40"><path id="p" '
            'd="M0 0L30 0L30 30L0 30Z M10 10L11 10L11 11Z"/></svg>'
        )
    )
    send(session, "select", objects=["p"])
    subpaths = session.nodes("p")["geometry"]["subpaths"]
    start = subpaths[0]["nodes"][0]
    send(session, "delete_node", object="p", node=start["id"])
    subpaths = session.nodes("p")["geometry"]["subpaths"]
    assert [n["values"] for n in subpaths[0]["nodes"]] == [(30, 0), (30, 30), (0, 30)]
    speck = subpaths[1]["nodes"][1]
    result = send(session, "delete_contour", object="p", node=speck["id"])
    assert result["undo"][-1] == "Delete contour"
    assert len(session.nodes("p")["geometry"]["subpaths"]) == 1
    last = session.nodes("p")["geometry"]["subpaths"][0]["nodes"][0]
    send(session, "delete_contour", object="p", node=last["id"])
    assert "p" not in {o["id"] for o in session.state()["objects"]}


HOLES = (
    '<svg width="100" height="100">'
    '<path id="outer" fill="#336699" d="M0 0H100V100H0Z M10 10V30H30V10Z"/>'
    '<path id="inner" fill="red" d="M50 50H80V80H50Z"/></svg>'
)


def hole_ids(session, oid):
    send(session, "select", objects=[oid])
    revision = session.editor.snapshot.revision
    payload = {"object": oid, "epoch": session.epoch, "revision": revision}
    return [h["id"] for h in session.holes(payload)["holes"]]


def test_cut_hole_command_makes_one_path_and_one_undo_step():
    session = Session(import_svg(HOLES))
    send(session, "select", objects=["outer", "inner"])
    result = send(session, "cut_hole")
    assert result["undo"] == ["Cut out as hole"]
    assert result["selection"]["objects"] == ["outer"]
    assert [o["id"] for o in result["objects"] if o["tag"] == "path"] == ["outer"]
    assert len(hole_ids(session, "outer")) == 2
    undone = send(session, "undo")
    assert {o["id"] for o in undone["objects"]} >= {"outer", "inner"}
    send(session, "select", objects=["outer"])
    with pytest.raises(DocumentError, match="two paths"):
        send(session, "cut_hole")


def test_fill_hole_and_hole_to_shape_commands():
    session = Session(import_svg(HOLES))
    (hole,) = hole_ids(session, "outer")
    result = send(session, "holes_to_shapes", object="outer", holes=[hole])
    (shape,) = result["selection"]["objects"]
    paths = [o for o in result["objects"] if o["tag"] == "path"]
    assert [o["id"] for o in paths] == ["outer", shape, "inner"]
    assert paths[1]["attributes"]["fill"] == "#336699"
    assert not hole_ids(session, "outer")
    send(session, "undo")
    (hole,) = hole_ids(session, "outer")
    result = send(session, "fill_holes", object="outer", holes=[hole])
    assert result["undo"][-1] == "Fill holes"
    assert not hole_ids(session, "outer")


def test_snapping_a_grid_of_touching_squares_is_one_undoable_edit():
    squares = "".join(
        f'<path id="p{i}{j}" d="M{i * 10} {j * 10} h10 v10 h-10 Z"/>'
        for i in range(5)
        for j in range(5)
    )
    session = Session(import_svg(f'<svg width="50" height="50">{squares}</svg>'))
    everything = frozenset(
        e.id for e in session.editor.snapshot.document.elements() if e.tag == "path"
    )
    session.editor.select(Selection(object_ids=everything))
    before = session.editor.snapshot.document
    with session.editor.transaction(
        "Snap edges", selection=Selection(object_ids=everything)
    ) as tx:
        assert len(tx.snap_edges(0.5)) == 40
    send(session, "undo")
    assert session.editor.snapshot.document == before


def test_tree_drag_moves_objects_into_a_group_as_one_undoable_edit():
    session = Session(
        import_svg(SVG.replace("<g id", '<g transform="translate(5 5)" id'))
    )
    before = session.editor.snapshot.document
    root = before.root.id
    state = send(session, "move_objects", objects=["b"], parent=root, index=1)
    assert state["undo"] == ["Move into group"]
    assert state["selection"]["objects"] == ["b"]
    document = session.editor.snapshot.document
    assert [c.id for c in document.root.children] == ["layer", "b"]
    assert document.element("b").get("transform") == "translate(5 5)"
    state = send(session, "move_objects", objects=["b"], parent=root, index=0)
    assert state["undo"][-1] == "Change stacking"
    with pytest.raises(DocumentError, match="into itself"):
        send(session, "move_objects", objects=["layer"], parent="layer", index=0)
    send(session, "undo")
    send(session, "undo")
    assert session.editor.snapshot.document == before


def test_bring_to_front_and_send_to_back_move_the_selection_together():
    session = Session(
        import_svg(
            SVG.replace("</g>", '<rect id="c" y="30" width="5" height="5"/></g>')
        )
    )
    send(session, "select", objects=["a", "b"])
    state = send(session, "reorder", to="front")
    assert state["undo"] == ["Bring to front"]
    layer = session.editor.snapshot.document.element("layer")
    assert [c.id for c in layer.children] == ["c", "a", "b"]
    send(session, "select", objects=["b"])
    assert send(session, "reorder", to="back")["undo"][-1] == "Send to back"
    layer = session.editor.snapshot.document.element("layer")
    assert [c.id for c in layer.children] == ["b", "c", "a"]


TWO_PATHS = """<svg width="100" height="100">
<path id="p" d="M0 0 L20 0 L20 20 L0 20 Z"/>
<g id="g"><path id="q" d="M50 50 L70 50 L70 70 L50 70 Z"/></g></svg>"""


def node_ids(session, oid):
    return [
        n["id"] for s in session.nodes(oid)["geometry"]["subpaths"] for n in s["nodes"]
    ]


def test_points_selected_across_paths_are_edited_as_one_step():
    session = Session(import_svg(TWO_PATHS))
    p, q = node_ids(session, "p"), node_ids(session, "q")
    state = send(session, "select", objects=["p", "q"], nodes=[p[1], q[2]])
    assert state["selection"] == {"objects": ["p", "q"], "nodes": sorted([p[1], q[2]])}
    assert set(session.geometries(["p", "q"])["geometries"]) == {"p", "q"}
    points = [["p", p[1]], ["q", q[2]]]
    send(session, "node_handles", points=points, count=2)
    assert session.state()["undo"] == ["Change handles"]
    send(session, "pin", points=points, pinned=True)
    assert session.state()["undo"][-1] == "Pin nodes"
    document = session.editor.snapshot.document
    assert document.geometry_for("p").node(p[1]).pinned
    assert document.geometry_for("q").node(q[2]).pinned
    send(session, "pin", points=points, pinned=False)
    p_values = document.geometry_for("p").node(p[1]).values
    q_values = document.geometry_for("q").node(q[2]).values
    moved = {
        "p": {p[1]: [*p_values[:-2], 25, 0]},
        "q": {q[2]: [*q_values[:-2], 75, 75]},
    }
    send(session, "move_nodes", changes=moved)
    document = session.editor.snapshot.document
    assert document.geometry_for("p").node(p[1]).endpoint == (25, 0)
    assert document.geometry_for("q").node(q[2]).endpoint == (75, 75)
    assert session.state()["undo"][-1] == "Move points"
    # Whole-object edits still apply while points are selected, and keep them.
    send(session, "paint", changes={"fill": "green"})
    send(session, "move", dx=1, dy=0)
    assert session.state()["selection"]["nodes"] == sorted([p[1], q[2]])


def test_deleting_selected_points_in_two_paths_keeps_the_paths_selected():
    session = Session(import_svg(TWO_PATHS))
    p, q = node_ids(session, "p"), node_ids(session, "q")
    send(session, "select", objects=["p", "q"], nodes=[p[1], q[1]])
    send(session, "delete_node", points=[["p", p[1]], ["q", q[1]]])
    assert len(node_ids(session, "p")) == len(p) - 1
    assert len(node_ids(session, "q")) == len(q) - 1
    assert session.state()["selection"] == {"objects": ["p", "q"], "nodes": []}
    assert session.state()["undo"] == ["Delete node"]
    send(session, "undo")
    assert len(node_ids(session, "p")) == len(p)


def test_deleting_the_contours_of_points_takes_each_contour_once():
    session = Session(import_svg(TWO_PATHS))
    p, q = node_ids(session, "p"), node_ids(session, "q")
    send(session, "select", objects=["p", "q"])
    # Two points of p's only contour delete it, and p with it, once.
    send(session, "delete_contour", points=[["p", p[1]], ["p", p[2]], ["q", q[0]]])
    assert {o["id"] for o in session.state()["objects"]} == {"g"}
    assert session.state()["undo"] == ["Delete contour"]


def test_point_commands_need_their_paths_selected():
    session = Session(import_svg(TWO_PATHS))
    q = node_ids(session, "q")
    send(session, "select", objects=["p"])
    with pytest.raises(DocumentError, match="Select the path"):
        send(session, "delete_node", points=[["q", q[0]]])
    with pytest.raises(DocumentError, match="Select the path"):
        send(session, "move_nodes", changes={"q": {q[0]: [1, 1]}})


def test_points_of_paths_inside_a_selected_group_can_be_edited():
    session = Session(import_svg(TWO_PATHS))
    q = node_ids(session, "q")
    # The group is selected; its path's points are selected with it.
    state = send(session, "select", objects=["g"], nodes=[q[1]])
    assert state["selection"] == {"objects": ["g"], "nodes": [q[1]]}
    send(session, "delete_node", points=[["q", q[1]]])
    assert len(node_ids(session, "q")) == len(q) - 1
    assert session.state()["selection"]["objects"] == ["g"]
    with pytest.raises(DocumentError, match="Select the path"):
        send(session, "delete_node", points=[["p", node_ids(session, "p")[0]]])


SHARED = """<svg width="100" height="100">
<path id="a" d="M0 0 L20 0 L20 20 L0 20 Z"/>
<path id="b" transform="translate(50 0)" d="M0 0 L10 0 L10 10 Z"/></svg>"""


def shared_session():
    session = Session(import_svg(SHARED))
    send(session, "select", objects=["a", "b"])
    both = Selection(object_ids=frozenset({"a", "b"}))
    with session.editor.transaction("Share", selection=both) as tx:
        tx.share_geometry("b", "a")
    return session


def test_shared_geometry_lists_its_users_and_takes_a_point_once():
    session = shared_session()
    geometries = session.geometries(["a", "b"])["geometries"]
    assert geometries["a"]["users"] == geometries["b"]["users"] == ["a", "b"]
    nodes = node_ids(session, "a")
    # The same node named in both paths is one point: split once, not twice.
    send(session, "split", points=[["a", nodes[1]], ["b", nodes[1]]])
    assert len(node_ids(session, "a")) == len(nodes) + 1
    assert node_ids(session, "b") == node_ids(session, "a")
    # Moved in both paths' frames, it moves once, as the first path has it.
    node = nodes[2]
    moved = {"a": {node: [21, 22]}, "b": {node: [-29, 22]}}
    send(session, "move_nodes", changes=moved)
    geometry = session.editor.snapshot.document.geometry_for("b")
    assert geometry.node(node).endpoint == (21, 22)


DONUTS = """<svg width="200" height="100"><g id="g">
<path id="left" fill="#336699" d="M0 0H90V90H0Z M10 10V30H30V10Z"/>
<path id="right" fill="#993366"
 d="M100 0H190V90H100Z M110 10V30H130V10Z M150 50V70H170V50Z"/>
</g></svg>"""


def test_holes_of_several_paths_are_filled_as_one_edit():
    session = Session(import_svg(DONUTS))
    (left,) = hole_ids(session, "left")
    right = hole_ids(session, "right")
    send(session, "select", objects=["g"])
    pairs = [["left", left], *(["right", hole] for hole in right)]
    result = send(session, "fill_holes", contours=pairs)
    assert result["undo"] == ["Fill holes"]
    assert result["selection"]["objects"] == ["g"]
    assert not hole_ids(session, "left")
    assert not hole_ids(session, "right")
    send(session, "undo")
    assert len(hole_ids(session, "left")) == 1
    assert len(hole_ids(session, "right")) == 2


def test_holes_of_several_paths_become_shapes_as_one_edit():
    session = Session(import_svg(DONUTS))
    (left,) = hole_ids(session, "left")
    right = hole_ids(session, "right")
    send(session, "select", objects=["left", "right"])
    pairs = [["left", left], ["right", right[0]]]
    result = send(session, "holes_to_shapes", contours=pairs)
    assert result["undo"] == ["Holes to shapes"]
    shapes = result["selection"]["objects"]
    assert len(shapes) == 2
    fills = {o["id"]: o["attributes"].get("fill") for o in result["objects"]}
    assert sorted(fills[s] for s in shapes) == ["#336699", "#993366"]
    assert len(hole_ids(session, "right")) == 1
    send(session, "select", objects=["left"])
    with pytest.raises(DocumentError, match="Select the path"):
        send(session, "fill_holes", contours=[["right", right[1]]])


def test_resize_scales_the_selection_about_the_anchor_as_one_edit():
    session = Session(import_svg(SVG))
    before = session.state()["svg"]
    send(session, "select", objects=["a", "b"])
    result = send(session, "resize", anchor=[0, 0], scale=[2, 0.5])
    assert result["undo"] == ["Resize"]
    document = session.editor.snapshot.document
    assert document.element("a").get("transform") == "scale(2 0.5)"
    assert document.element("b").get("transform") == "scale(2 0.5)"
    assert send(session, "undo")["svg"] == before
    with pytest.raises(DocumentError, match="positive"):
        send(session, "resize", anchor=[0, 0], scale=[0, 1])
