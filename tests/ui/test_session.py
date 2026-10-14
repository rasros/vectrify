"""The UI adapter uses backend transactions, identities and constraints."""

import base64
import io
import json
from unittest.mock import patch

import pytest
from PIL import Image

from vectrify.document import (
    Document,
    DocumentError,
    Selection,
    StaleRevisionError,
    import_svg,
)
from vectrify.ui.agent import Agent
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


def test_manual_history_skips_agent_edits_and_keeps_redo_while_agent_works():
    session = Session(import_svg(SVG))
    agent = Agent(session)
    send(session, "select", objects=["a"])
    send(session, "paint", changes={"fill": "green"})
    send(session, "select", objects=["b"])
    selected = session.editor.snapshot.selection
    moved = agent.call(
        "transform",
        {
            "seen": [session.epoch, 1],
            "ids": ["a"],
            "dx": 5,
            "dy": 0,
        },
    ).data
    assert session.editor.snapshot.selection == selected
    assert session.state()["undo"] == ["Change paint"]
    undone = send(session, "undo")
    assert undone["undo"] == []
    assert undone["redo"] == ["Change paint"]
    document = session.editor.snapshot.document
    assert document.element("a").get("fill") == "red"
    assert document.element("a").get("transform") == "translate(5.0 0.0)"
    unchanged = session.editor.snapshot
    send(session, "undo")
    assert session.editor.snapshot == unchanged
    agent.call(
        "properties",
        {
            "seen": [session.epoch, unchanged.revision],
            "ids": ["b"],
            "fill": "black",
        },
    )
    send(session, "redo")
    document = session.editor.snapshot.document
    assert document.element("a").get("fill") == "green"
    assert document.element("b").get("fill") == "black"
    agent.call(
        "undo",
        {
            "seen": [session.epoch, session.editor.snapshot.revision],
            "ids": [moved["edit_id"]],
        },
    )
    assert session.editor.snapshot.document.element("a").get("transform") is None
    assert session.editor.snapshot.document.element("a").get("fill") == "green"


def test_selection_accepts_a_concurrent_revision_but_requires_the_same_document():
    session = Session(import_svg(SVG))
    revision = session.editor.snapshot.revision
    send(session, "select", objects=["a"])
    send(session, "paint", changes={"fill": "green"})
    selected = session.action(
        {
            "command": "select",
            "objects": ["b"],
            "epoch": session.epoch,
            "revision": revision,
        }
    )
    assert selected["selection"]["objects"] == ["b"]
    assert selected["revision"] == 1
    assert 'fill="green"' in selected["svg"]
    with pytest.raises(StaleRevisionError):
        session.action(
            {
                "command": "select",
                "objects": ["a"],
                "epoch": "previous drawing",
                "revision": revision,
            }
        )


def test_large_drawing_selection_reuses_metadata_and_edits_refresh_it():
    svg = (
        '<svg width="1000" height="1000">'
        + "".join(
            f'<path id="p{i}" d="M{i} 0 L{i + 1} 0 L{i + 1} 1 Z"/>' for i in range(772)
        )
        + "</svg>"
    )
    session = Session(import_svg(svg))
    before = session.state()
    node = session.nodes("p771")["geometry"]["subpaths"][0]["nodes"][0]["id"]
    # A click must not traverse the entire tree again to rebuild unchanged
    # hierarchy, labels and locks, even when selecting points or a range.
    with patch.object(Document, "ancestry", side_effect=AssertionError("rebuilt tree")):
        for objects, nodes in [(["p0"], []), (["p771"], [node]), (["p0", "p771"], [])]:
            state = send(session, "select", objects=objects, nodes=nodes)
            assert state["objects"] == before["objects"]
            assert set(state["selection"]["objects"]) == set(objects)
            assert state["selection"]["nodes"] == nodes
            assert "svg" not in state

    send(session, "select", objects=["p0"])
    renamed = send(session, "rename", object="p0", name="First")
    assert renamed["objects"][0]["label"] == "First"
    assert send(session, "undo")["objects"] == before["objects"]
    assert send(session, "redo")["objects"] == renamed["objects"]
    locked = send(session, "locks", object="p0", locks=["paint"])
    assert locked["objects"][0]["inherited_locks"] == ["paint"]
    send(session, "select", objects=["p1"])
    painted = send(session, "paint", changes={"fill": "green"})
    assert painted["objects"][1]["attributes"]["fill"] == "green"
    reopened = send(session, "open", source=SVG)
    assert reopened["revision"] == 0
    assert {item["id"] for item in reopened["objects"]} == {"layer", "a", "b"}


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


@pytest.mark.parametrize("data", [b"\x1f\x8bbroken", b"\xff\xfe"])
def test_open_invalid_binary_file_preserves_current_edits(data):
    session = Session(import_svg(SVG))
    send(session, "select", objects=["a"])
    send(session, "paint", changes={"fill": "green"})
    before = session.state()
    with pytest.raises(DocumentError):
        send(
            session,
            "open",
            source=base64.b64encode(data).decode(),
            encoding="base64",
            name="bad.vectrify",
        )
    assert session.state() == before


@pytest.mark.parametrize("compressed", [False, True])
def test_project_roundtrip_preserves_reference_and_constraints(compressed):
    from vectrify.project_file import encode_project

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
    source = base64.b64encode(encode_project(saved)).decode() if compressed else saved
    send(
        restored,
        "open",
        source=source,
        encoding="base64" if compressed else None,
        name="saved.vectrify",
    )
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


@pytest.mark.parametrize("data", ["", "M1 2", "M1 2 L3 4 M5 6", "M1 2 Lnan 3"])
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
    assert joined["undo"][-1] == "Merge shapes"
    oid = joined["selection"]["objects"][0]
    assert len(session.editor.snapshot.document.geometry_for(oid).subpaths) == 2
    assert send(session, "undo")["selection"] == split["selection"]


def test_join_single_path_unions_its_contours_in_one_undoable_step():
    session = Session(
        import_svg(
            '<svg width="100" height="100"><path id="a" fill="red" '
            'd="M0 0H10V10H0Z M5 0H15V10H5Z"/></svg>'
        )
    )
    send(session, "select", objects=["a"])
    before = session.editor.snapshot
    result = send(session, "join_paths")
    assert result["selection"]["objects"] == ["a"]
    assert result["undo"] == ["Merge shapes"]
    assert len(session.editor.snapshot.document.geometry_for("a").subpaths) == 1
    after = session.editor.snapshot.document
    send(session, "undo")
    assert session.editor.snapshot.document == before.document
    assert session.editor.snapshot.selection == before.selection
    send(session, "redo")
    assert session.editor.snapshot.document == after


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
    assert result["undo"] == ["Merge shapes"]
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
    assert result["undo"] == ["Merge shapes"]
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


CONVERTIBLE = """<svg width="100" height="100"><defs>
<clipPath id="clip"><rect x="0" y="0" width="60" height="100"/></clipPath>
<path id="def" d="M0 0H10V10H0Z"/></defs>
<g id="layer" transform="translate(5 5)">
<rect id="r" x="10" y="10" width="40" height="30" rx="4" fill="red"
 stroke="navy" stroke-width="2" opacity=".8" transform="rotate(10 30 25)"
 clip-path="url(#clip)"/>
<circle id="c" cx="70" cy="70" r="15" fill="green"/>
<line id="l" x1="10" y1="90" x2="90" y2="60" stroke="black"/>
</g><use id="inst" href="#def" x="80"/></svg>"""


def test_convert_to_path_keeps_shapes_look_identity_and_is_one_undo():
    from tests.document.test_document import render

    session = Session(import_svg(CONVERTIBLE))
    before = session.editor.snapshot
    image = render(session.state()["svg"])
    send(session, "select", objects=["r", "c", "l"])
    result = send(session, "convert_to_path")
    document = session.editor.snapshot.document
    assert result["selection"]["objects"] == ["c", "l", "r"]
    assert result["undo"] == ["Convert to path"]
    assert {document.element(i).tag for i in ("r", "c", "l")} == {"path"}
    rect = document.element("r")
    for name, value in before.document.element("r").attributes:
        if name not in {"x", "y", "width", "height", "rx"}:
            assert rect.get(name) == value
    assert rect.get("x") is None
    assert document.element("l").get("fill") == "none"
    assert document.ancestry("r")[-2].id == "layer"
    # The same picture, give or take the curves' antialiasing.
    assert abs(render(session.state()["svg"]).astype(int) - image).max() < 64
    # Its points are now the path's own to edit in Nodes.
    first = node_ids(session, "c")[0]
    values = document.geometry_for("c").node(first).values
    send(session, "select", objects=["c"], nodes=[first])
    send(session, "move_nodes", changes={"c": {first: [*values[:-2], 90, 70]}})
    moved = session.editor.snapshot.document.geometry_for("c").node(first)
    assert moved.endpoint == (90, 70)
    send(session, "undo")
    send(session, "undo")
    assert session.editor.snapshot.document == before.document


@pytest.mark.parametrize(
    ("objects", "locked", "message"),
    [
        (["inst"], None, "Convert to path takes rectangles"),
        (["c", "def"], None, "detach instances and shared geometry"),
        (["c", "r"], "r", "r: structure is locked"),
    ],
)
def test_convert_to_path_refuses_without_converting_any(objects, locked, message):
    session = Session(import_svg(CONVERTIBLE))
    if locked:
        send(session, "select", objects=[locked])
        send(session, "locks", object=locked, locks=["geometry", "structure"])
    send(session, "select", objects=objects)
    before = session.editor.snapshot
    with pytest.raises(DocumentError, match=message):
        send(session, "convert_to_path")
    assert session.editor.snapshot == before
    assert session.editor.snapshot.document.element("c").tag == "circle"


def test_convert_to_path_refuses_a_shape_another_object_references():
    session = Session(
        import_svg(
            '<svg><rect id="r" width="10" height="10"/>'
            '<use id="u" href="#r" x="20"/></svg>'
        )
    )
    send(session, "select", objects=["r"])
    before = session.editor.snapshot
    with pytest.raises(DocumentError, match="Detach references to this shape"):
        send(session, "convert_to_path")
    assert session.editor.snapshot == before


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


def test_tree_drag_into_a_clipped_group_applies_clipping_and_is_undoable():
    session = Session(
        import_svg(
            SVG.replace(
                "</svg>",
                '<defs><clipPath id="clip"><rect x="35" width="10" height="20"/>'
                '</clipPath></defs><g id="clipped" clip-path="url(#clip)"/></svg>',
            )
        )
    )
    before = session.editor.snapshot.document
    state = send(session, "move_objects", objects=["b"], parent="clipped", index=0)
    after = session.editor.snapshot.document
    assert after.ancestry("b")[-2].id == "clipped"
    assert after.element("clipped").get("clip-path") == "url(#clip)"
    assert after.element("b") == before.element("b")
    assert state["undo"] == ["Move into group"]
    assert state["selection"]["objects"] == ["b"]
    send(session, "undo")
    assert session.editor.snapshot.document == before
    send(session, "redo")
    assert session.editor.snapshot.document == after


@pytest.mark.parametrize(
    ("command", "selected"),
    [("group", ["a", "b"]), ("ungroup", ["g"]), ("combine_paths", ["a", "b"])],
)
def test_object_commands_accept_clipped_faded_groups_and_undo(command, selected):
    session = Session(
        import_svg("""<svg width="100" height="100">
    <defs><clipPath id="clip"><rect width="20" height="20"/></clipPath></defs>
    <g id="g" transform="translate(10 10)" opacity=".5" clip-path="url(#clip)">
      <path id="a" d="M0 0H30V30H0Z"/>
    </g><path id="b" d="M50 0H60V10H50Z"/>
    </svg>""")
    )
    send(session, "select", objects=selected)
    before = session.editor.snapshot.document
    state = send(session, command)
    after = session.editor.snapshot.document
    after.validate()
    assert len(state["undo"]) == 1
    assert state["selection"]["objects"]
    send(session, "undo")
    assert session.editor.snapshot.document == before
    send(session, "redo")
    assert session.editor.snapshot.document == after


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


@pytest.mark.parametrize("position", [0, -1])
@pytest.mark.parametrize("segment", ["L20 0", "C5 10 15 10 20 0"])
def test_add_node_at_either_open_endpoint_preserves_the_curve_and_is_undoable(
    position, segment
):
    from tests.document.test_document import render

    session = Session(
        import_svg(
            f'<svg width="30" height="20"><path id="p" fill="none" '
            f'stroke="black" d="M0 0 {segment}"/></svg>'
        )
    )
    ids = node_ids(session, "p")
    send(session, "select", objects=["p"], nodes=[ids[position]])
    send(session, "pin", points=[["p", ids[position]]], pinned=True)
    before = session.editor.snapshot.document
    image = render(session.state()["svg"])
    send(session, "split", points=[["p", ids[position]]])
    after = session.editor.snapshot.document
    nodes = after.geometry_for("p").subpaths[0].nodes
    assert len(nodes) == 3
    assert [nodes[0].id, nodes[-1].id] == ids
    assert nodes[position].pinned
    assert nodes[1].endpoint == pytest.approx((10, 7.5 if segment[0] == "C" else 0))
    assert (render(session.state()["svg"]) == image).all()
    assert session.state()["undo"][-1] == "Add node"
    send(session, "undo")
    assert session.editor.snapshot.document == before
    send(session, "redo")
    assert session.editor.snapshot.document == after


@pytest.mark.parametrize("reverse", [False, True])
def test_add_node_splits_the_first_edge_once_when_both_ends_are_selected(reverse):
    session = Session(import_svg('<svg><path id="p" d="M0 0 L20 0 L40 0"/></svg>'))
    ids = node_ids(session, "p")[:2]
    if reverse:
        ids.reverse()
    send(session, "select", objects=["p"], nodes=ids)
    send(session, "split", points=[["p", nid] for nid in ids])
    nodes = session.editor.snapshot.document.geometry_for("p").subpaths[0].nodes
    assert [n.endpoint for n in nodes] == [(0, 0), (10, 0), (20, 0), (40, 0)]


def test_add_node_to_a_lone_point_reports_the_missing_edge_without_changing_it():
    session = Session(import_svg('<svg><path id="p" d="M0 0"/></svg>'))
    nid = node_ids(session, "p")[0]
    send(session, "select", objects=["p"], nodes=[nid])
    before = session.editor.snapshot
    with pytest.raises(DocumentError, match="no edge"):
        send(session, "split", points=[["p", nid]])
    assert session.editor.snapshot == before


def test_one_handle_switches_once_when_both_nodes_at_a_closed_join_are_selected():
    session = Session(
        import_svg('<svg><path id="p" d="M0 0 L20 0 L20 20 L0 0 Z"/></svg>')
    )
    ids = node_ids(session, "p")
    send(session, "select", objects=["p"], nodes=[ids[0], ids[-1]])
    points = [["p", ids[0]], ["p", ids[-1]]]
    before = session.editor.snapshot.document.geometry_for("p")
    send(session, "node_handles", points=points, count=1)
    nodes = session.editor.snapshot.document.geometry_for("p").subpaths[0].nodes
    assert (nodes[-1].command, nodes[1].command) == ("C", "L")
    assert session.state()["undo"] == ["Change handles"]
    send(session, "node_handles", points=points, count=1)
    nodes = session.editor.snapshot.document.geometry_for("p").subpaths[0].nodes
    assert (nodes[-1].command, nodes[1].command) == ("L", "C")
    send(session, "undo")
    send(session, "undo")
    assert session.editor.snapshot.document.geometry_for("p") == before


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


def test_the_state_marks_paths_that_share_their_geometry():
    session = Session(import_svg(SHARED))
    assert not any(o["shared"] for o in session.state(svg=False)["objects"])
    objects = shared_session().state(svg=False)["objects"]
    assert {o["id"] for o in objects if o["shared"]} == {"a", "b"}


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


NESTED = """<svg width="200" height="200">
<g id="turned" transform="translate(100 20) rotate(90) scale(2)">
<rect id="inner" width="10" height="5" stroke="black" stroke-width="2"/>
<rect id="other" x="20" width="5" height="5"/>
</g>
<rect id="plain" x="10" y="150" width="20" height="10"/>
</svg>"""


def page_box(session, *ids):
    from vectrify.document.hit_test import HitIndex

    box = HitIndex(session.editor.snapshot.document).bounds(frozenset(ids))
    assert box is not None
    return box


def test_numeric_move_is_in_document_units_inside_transformed_groups():
    session = Session(import_svg(NESTED))
    before = {oid: page_box(session, oid) for oid in ("inner", "plain")}
    send(session, "select", objects=["inner", "plain"])
    result = send(session, "move", dx=7, dy=-3)
    assert result["undo"] == ["Move selection"]
    # Both move as far on the page, so they keep their relative placement.
    for oid, (left, top, right, bottom) in before.items():
        assert page_box(session, oid) == pytest.approx(
            (left + 7, top - 3, right + 7, bottom - 3)
        )
    # The child takes the move in its rotated, scaled group's frame.
    assert session.editor.snapshot.document.element("inner").get("transform") == (
        "translate(-1.5 -3.5)"
    )
    send(session, "undo")
    assert page_box(session, "inner") == pytest.approx(before["inner"])


def test_numeric_move_and_resize_change_a_selected_group_and_child_once():
    session = Session(import_svg(NESTED))
    group = page_box(session, "turned")
    send(session, "select", objects=["turned", "other"])
    send(session, "move", dx=10, dy=0)
    assert session.editor.snapshot.document.element("other").get("transform") is None
    assert page_box(session, "turned") == pytest.approx(
        (group[0] + 10, group[1], group[2] + 10, group[3])
    )
    left, top = group[0] + 10, group[1]
    send(session, "resize", anchor=[left, top], scale=[2, 2])
    assert session.editor.snapshot.document.element("other").get("transform") is None
    assert session.editor.undo_labels[-2:] == ("Move selection", "Resize")


def test_numeric_move_keeps_the_position_lock():
    session = Session(import_svg(NESTED))
    send(session, "select", objects=["turned"])
    send(session, "locks", object="turned", locks=["transform"])
    send(session, "select", objects=["inner"])
    before = session.editor.snapshot.document
    with pytest.raises(DocumentError, match="locked"):
        send(session, "move", dx=1, dy=0)
    with pytest.raises(DocumentError, match="position is locked"):
        send(session, "resize", anchor=[0, 0], scale=[2, 1])
    assert session.editor.snapshot.document == before


def test_nudges_of_one_run_are_one_undo_step_and_one_translate():
    session = Session(import_svg(NESTED))
    before = page_box(session, "inner")
    send(session, "select", objects=["inner", "plain"])
    for _ in range(3):
        send(session, "move", dx=1, dy=0, nudge="run-1")
    send(session, "move", dx=0, dy=10, nudge="run-1")
    assert session.state()["undo"] == ["Move selection"]
    assert page_box(session, "inner") == pytest.approx(
        (before[0] + 3, before[1] + 10, before[2] + 3, before[3] + 10)
    )
    # The moves add up into one translate in the rotated, scaled group's frame.
    transform = session.editor.snapshot.document.element("inner").get("transform")
    assert transform == "translate(5.0 -1.5)"
    assert session.editor.snapshot.document.element("plain").get("transform") == (
        "translate(3.0 10.0)"
    )
    # Another run is a step of its own; an edit in between ends a run.
    send(session, "move", dx=1, dy=0, nudge="run-2")
    send(session, "paint", changes={"fill": "green"})
    send(session, "move", dx=1, dy=0, nudge="run-2")
    assert session.state()["undo"] == [
        "Move selection",
        "Move selection",
        "Change paint",
        "Move selection",
    ]
    for _ in range(4):
        send(session, "undo")
    assert page_box(session, "inner") == pytest.approx(before)
    assert session.editor.snapshot.document.element("inner").get("transform") is None
    # Undo between nudges of one run ends it too.
    send(session, "move", dx=1, dy=0, nudge="run-3")
    send(session, "undo")
    send(session, "move", dx=1, dy=0, nudge="run-3")
    send(session, "move", dx=1, dy=0, nudge="run-3")
    assert session.state()["undo"] == ["Move selection"]
    assert page_box(session, "inner") == pytest.approx(
        (before[0] + 2, before[1], before[2] + 2, before[3])
    )


def test_point_nudges_of_one_run_are_one_undo_step():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["a"])
    node = session.editor.snapshot.document.geometry_for("a").subpaths[0].nodes[1]
    for x in (21, 22, 23):
        send(session, "move_nodes", changes={"a": {node.id: [x, 0]}}, nudge="run")
    assert session.state()["undo"] == ["Move points"]
    geometry = session.editor.snapshot.document.geometry_for("a")
    assert geometry.node(node.id).endpoint == (23, 0)
    send(session, "undo")
    geometry = session.editor.snapshot.document.geometry_for("a")
    assert geometry.node(node.id).endpoint == (20, 0)


def test_nudges_respect_position_locks_and_pins():
    session = Session(import_svg(NESTED))
    send(session, "select", objects=["turned"])
    send(session, "locks", object="turned", locks=["transform"])
    send(session, "select", objects=["inner"])
    with pytest.raises(DocumentError, match="locked"):
        send(session, "move", dx=1, dy=0, nudge="run")
    session = Session(import_svg(SVG))
    send(session, "select", objects=["a"])
    node = session.editor.snapshot.document.geometry_for("a").subpaths[0].nodes[1]
    send(session, "pin", points=[["a", node.id]], pinned=True)
    with pytest.raises(DocumentError):
        send(session, "move_nodes", changes={"a": {node.id: [21, 0]}}, nudge="run")


def test_rotate_and_flip_turn_the_selection_about_the_pivot_as_one_edit_each():
    session = Session(import_svg(NESTED))
    before = {oid: page_box(session, oid) for oid in ("inner", "plain")}
    send(session, "select", objects=["inner", "plain"])
    result = send(session, "flip_horizontal", pivot=[100, 0])
    assert result["undo"] == ["Flip horizontal"]
    for oid, (left, top, right, bottom) in before.items():
        assert page_box(session, oid) == pytest.approx(
            (200 - right, top, 200 - left, bottom)
        )
    send(session, "flip_vertical", pivot=[0, 100])
    send(session, "rotate", angle=180, pivot=[100, 100])
    # Both mirrors and a half turn about the same centre cancel out.
    for oid, box in before.items():
        assert page_box(session, oid) == pytest.approx(box)
    assert session.editor.undo_labels[-3:] == (
        "Flip horizontal",
        "Flip vertical",
        "Rotate",
    )
    # Strokes keep their width, the child inside the turned group included.
    document = session.editor.snapshot.document
    assert document.element("inner").get("stroke-width") == "2"
    with pytest.raises(DocumentError, match="whole turn"):
        send(session, "rotate", angle=-360, pivot=[0, 0])
    with pytest.raises(DocumentError, match="pivot"):
        send(session, "rotate", angle=10, pivot=[0])


def test_rotate_and_flip_keep_the_position_lock():
    session = Session(import_svg(NESTED))
    send(session, "select", objects=["turned"])
    send(session, "locks", object="turned", locks=["transform"])
    send(session, "select", objects=["inner"])
    before = session.editor.snapshot.document
    with pytest.raises(DocumentError, match="unlock it to rotate"):
        send(session, "rotate", angle=90, pivot=[0, 0])
    with pytest.raises(DocumentError, match="unlock it to flip"):
        send(session, "flip_vertical", pivot=[0, 0])
    assert session.editor.snapshot.document == before


def test_a_new_group_is_labelled_by_kind_and_number_not_by_its_id():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["a", "b"])
    result = send(session, "group")
    groups = [o for o in result["objects"] if o["tag"] == "g" and o["id"] != "layer"]
    assert len(groups) == 1
    assert groups[0]["label"] == "Group 2"
    # A named or hand-written id still shows as itself.
    assert next(o for o in result["objects"] if o["id"] == "layer")["label"] == "layer"


def png_url(width, height, colour=(200, 40, 40)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def kept_reference(session) -> dict:
    assert session.reference is not None
    return session.reference


def reference_image(session) -> Image.Image:
    data = base64.b64decode(kept_reference(session)["data_url"].split(",", 1)[1])
    return Image.open(io.BytesIO(data)).convert("RGBA")


def pixel(image: Image.Image, xy) -> tuple:
    value = image.getpixel(xy)
    assert isinstance(value, tuple)
    return value


def test_an_empty_drawing_takes_a_reference_s_shape_as_its_artboard():
    session = Session(import_svg('<svg width="100" height="100"/>'))
    reference = {"name": "wide.png", "data_url": png_url(300, 150)}
    send(session, "reference", reference=reference)
    assert session.editor.snapshot.document.artboard() == (0, 0, 300, 150)
    assert reference_image(session).size == (300, 150)


def test_a_reference_of_another_shape_is_padded_not_stretched_over_a_drawing():
    session = Session(import_svg(SVG))
    reference = {"name": "wide.png", "data_url": png_url(300, 150)}
    send(session, "reference", reference=reference)
    # The drawing and its 100 x 100 artboard stay; the image is centred on a
    # transparent square, every pixel of it kept.
    assert session.editor.snapshot.document.artboard()[2:] == (100, 100)
    image = reference_image(session)
    assert image.size == (300, 300)
    assert pixel(image, (150, 10))[3] == 0
    assert pixel(image, (150, 150))[:3] == (200, 40, 40)
    # Changing the opacity sends the kept image back and changes nothing else.
    kept = kept_reference(session)["data_url"]
    send(session, "reference", reference={**kept_reference(session), "opacity": 0.8})
    assert kept_reference(session)["data_url"] == kept
    assert kept_reference(session)["opacity"] == 0.8


def test_a_reference_of_the_artboard_s_shape_is_kept_as_it_is():
    session = Session(import_svg(SVG))
    url = png_url(200, 200)
    send(session, "reference", reference={"name": "square.png", "data_url": url})
    assert kept_reference(session)["data_url"] == url


def test_combine_command_preserves_contours_and_restores_selection_with_undo():
    doc = import_svg("""<svg width="100" height="100">
    <path id="a" fill="red" d="M0 0H20V20H0Z"/>
    <path id="b" fill="blue" d="M10 10H30V30H10Z"/></svg>""")
    session = Session(doc)
    send(session, "select", objects=["b", "a"])
    before = session.editor.snapshot
    for source in [[], "", "missing"]:
        with pytest.raises(DocumentError):
            send(session, "combine_paths", paint_source=source)
        assert session.editor.snapshot == before
    result = send(session, "combine_paths")
    oid = result["selection"]["objects"][0]
    after = session.editor.snapshot.document
    assert after.geometry_for(oid).subpaths == (
        *doc.geometry_for("a").subpaths,
        *doc.geometry_for("b").subpaths,
    )
    assert after.element(oid).get("fill") == "blue"
    assert result["undo"] == ["Combine paths"]
    assert send(session, "undo")["selection"]["objects"] == ["a", "b"]
    assert session.editor.snapshot.document == before.document
    assert send(session, "redo")["selection"]["objects"] == [oid]
