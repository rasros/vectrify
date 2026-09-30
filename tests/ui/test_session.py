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


def test_knife_cuts_selected_paths_links_the_seam_and_undoes_in_one_step():
    from vectrify.document.topology import edge

    session = Session(
        import_svg(
            '<svg width="100" height="100"><g transform="translate(10 0)">'
            '<path id="p" fill="red" d="M0 0H20V20H0Z"/></g></svg>'
        )
    )
    with pytest.raises(DocumentError, match="Select"):
        send(session, "knife", start=[15, -5], end=[15, 30])
    send(session, "select", objects=["p"])
    result = send(session, "knife", start=[15, -5], end=[15, 30])
    pieces = result["selection"]["objects"]
    assert result["undo"] == ["Cut with knife"]
    assert len(pieces) == 2
    assert "p" in pieces
    doc = session.editor.snapshot.document
    first, second = doc.boundaries[0].members
    assert edge(doc, first).points == edge(doc, second).points
    assert {p[0] for p in edge(doc, first).points} == {5.0}
    # Moving a seam node in one piece moves the other piece's seam with it.
    oid = pieces[1]
    send(session, "select", objects=[oid])
    member = first if doc.geometry_for(oid).id == first.geometry_id else second
    node = edge(doc, member).end
    send(session, "node", object=oid, node=node.id, values=[7, node.values[1]])
    doc = session.editor.snapshot.document
    assert edge(doc, first).points == edge(doc, second).points
    assert (7, node.values[1]) in edge(doc, first).points
    send(session, "unlink_boundaries")
    assert not session.editor.snapshot.document.boundaries
    send(session, "undo")
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


def shared_curve_session():
    # The right region's contour closes with a curve that ends on its moveto,
    # so its top seam vertex is two nodes: the moveto and the closing node.
    session = Session(
        import_svg(
            '<svg width="40" height="40">'
            '<path id="a" d="M0 0 C5 -3 15 -3 20 0 C22 5 22 15 20 20 L0 20Z"/>'
            '<path id="b" d="M20 0 C25 -3 35 -3 40 0 L40 20 L20 20 '
            'C22 15 22 5 20 0Z"/></svg>'
        )
    )
    send(session, "select", objects=["a", "b"])
    session.operation(
        {
            "command": "apply",
            "job": session.operation(
                {
                    "command": "start",
                    "action": "link",
                    "method": "boundaries",
                    "epoch": session.epoch,
                    "revision": session.editor.snapshot.revision,
                    "permissions": {"geometry": True, "structure": True},
                    "settings": {"tolerance": 0.01},
                }
            )["id"],
        }
    )
    assert len(session.editor.snapshot.document.boundaries) == 1
    return session


def contour(session, oid):
    return [n["values"] for n in session.nodes(oid)["geometry"]["subpaths"][0]["nodes"]]


@pytest.mark.parametrize("index", [0, 4])
def test_dragging_a_shared_seam_vertex_moves_both_regions_with_their_handles(index):
    from vectrify.document.topology import validate_boundaries

    session = shared_curve_session()
    send(session, "select", objects=["b"])
    node = session.nodes("b")["geometry"]["subpaths"][0]["nodes"][index]
    values = [*node["values"][:-2], 23, 2]
    send(session, "node", object="b", node=node["id"], values=values)
    assert contour(session, "b") == [
        (23, 2),
        (28, -1, 35, -3, 40, 0),
        (40, 20),
        (20, 20),
        (22, 15, 25, 7, 23, 2),
    ]
    assert contour(session, "a") == [
        (0, 0),
        (5, -3, 18, -1, 23, 2),
        (25, 7, 22, 15, 20, 20),
        (0, 20),
    ]
    document = session.editor.snapshot.document
    document.validate()
    validate_boundaries(document)


def test_nodes_payload_names_shared_edges_and_the_peers_a_drag_moves():
    session = shared_curve_session()
    send(session, "select", objects=["b"])
    payload = session.nodes("b")
    seam = payload["geometry"]["subpaths"][0]["nodes"][4]["id"]
    assert payload["shared"] == [seam]
    (peer,) = payload["peers"].values()
    assert peer["objects"] == ["a"]
    top = peer["geometry"]["subpaths"][0]["nodes"][1]["id"]
    assert payload["links"][f"{seam}/4"] == [
        [
            session.editor.snapshot.document.geometry_for("a").id,
            top,
            4,
            [1, 0, 0, 1, 0, 0],
        ]
    ]
    json.dumps(payload)


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


def test_unlinking_a_grid_of_shared_edges_is_one_undoable_edit():
    # A traced drawing links thousands of edges; they all go in one pass.
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
    with session.editor.transaction(
        "share", selection=Selection(object_ids=everything)
    ) as tx:
        linked = tx.share_boundaries(0.5)
    assert linked == 40
    send(session, "unlink_boundaries")
    assert not session.editor.snapshot.document.boundaries
    send(session, "undo")
    assert len(session.editor.snapshot.document.boundaries) == 40


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
