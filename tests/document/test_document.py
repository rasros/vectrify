"""Behavioral checks for the SVG editor foundation, without UI or GPU dependencies."""

import io
import json
from dataclasses import FrozenInstanceError, replace

import cairosvg
import numpy as np
import pytest
from PIL import Image

from vectrify.document import (
    DocumentError,
    EditKind,
    Editor,
    EditRejectedError,
    Rect,
    Selection,
    StaleRevisionError,
    UnsupportedSvgError,
    export_svg,
    import_svg,
    load_project,
    save_project,
)

SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"
viewBox="0 0 64 64">
<g id="layer"><path id="a" fill="red" d="M4 4 L28 4 L28 28 L4 28 Z"/></g>
<path id="b" fill="blue" d="M36 36 C40 30 54 30 60 36 L60 60 L36 60 Z"/>
</svg>"""
SHARED = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">
<defs id="library"><path id="shape" fill-rule="evenodd"
d="M4 4 H28 V28 H4 Z M10 10 H22 V22 H10 Z"/>
<clipPath id="clip"><use id="clip-use" href="#shape"/></clipPath></defs>
<use id="first" href="#shape" fill="red"/>
<use id="second" href="#shape" fill="blue" transform="translate(30 30)"/>
<rect id="clipped" width="32" height="32" fill="green" clip-path="url(#clip)"/>
</svg>"""


def render(svg):
    png = cairosvg.svg2png(bytestring=svg.encode())
    assert png is not None
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def select(*ids, nodes=(), focus=None):
    return Selection(frozenset(ids), frozenset(nodes), focus=focus)


def test_import_export_preserves_rendering_and_shared_dependencies():
    document = import_svg(SHARED)
    assert np.array_equal(render(SHARED), render(export_svg(document)))
    geometry = document.geometry_for("first")
    assert geometry is document.geometry_for("second")
    assert len(geometry.subpaths) == 2
    assert document.geometry_users(geometry.id) == {
        "shape",
        "first",
        "second",
        "clip-use",
        "clipped",
    }
    assert (
        import_svg(export_svg(document)).element("second").get("transform")
        == "translate(30 30)"
    )


def test_geometry_and_object_ids_survive_edits_and_history():
    document = import_svg(SVG)
    editor = Editor(document, selection=select("a"))
    geometry = document.geometry_for("a")
    subpath = geometry.subpaths[0]
    node = subpath.nodes[1]
    with editor.transaction("Refine corner") as tx:
        tx.update_node("a", node.id, (30, 4))
        tx.set_attributes("a", {"fill": "orange"})
        assert editor.snapshot.document == document  # Preview is not committed.
    updated = editor.snapshot.document.geometry_for("a")
    assert updated.id == geometry.id
    assert updated.subpaths[0].id == subpath.id
    assert updated.node(node.id).endpoint == (30, 4)
    assert geometry.node(node.id).endpoint == (28, 4)
    assert editor.snapshot.document.element("b") == document.element("b")
    assert editor.undo_labels == ("Refine corner",)
    assert editor.undo().document == document
    assert editor.redo().document.geometry_for("a") == updated


def test_snapshots_are_immutable():
    document = import_svg(SVG)
    with pytest.raises(FrozenInstanceError):
        document.root.id = "changed"  # pyrefly: ignore[read-only]
    node = document.geometry_for("a").subpaths[0].nodes[0]
    with pytest.raises(FrozenInstanceError):
        node.values = (9, 9)  # pyrefly: ignore[read-only]


def test_empty_selection_and_focus_alone_never_mean_everything():
    for selection in (Selection(), Selection(focus=Rect(0, 0, 32, 32))):
        editor = Editor(import_svg(SVG), selection=selection)
        with (
            pytest.raises(EditRejectedError, match="unselected"),
            editor.transaction("Paint") as tx,
        ):
            tx.set_attributes("a", {"fill": "green"})
        assert editor.snapshot.revision == 0
        assert not editor.undo_labels


def test_group_selection_expands_to_descendants_but_not_other_layers():
    editor = Editor(import_svg(SVG), selection=select("layer"))
    with editor.transaction("Paint child") as tx:
        tx.set_attributes("a", {"fill": "green"})
    with (
        pytest.raises(EditRejectedError, match="unselected"),
        editor.transaction("Paint other layer") as tx,
    ):
        tx.set_attributes("b", {"fill": "green"})


def test_single_property_permission_and_inherited_locks():
    editor = Editor(import_svg(SVG), selection=select("a"))
    editor.set_locks("layer", frozenset({"stroke-width"}))
    with editor.transaction("Colour only", allowed=frozenset({"fill"})) as tx:
        tx.set_attributes("a", {"fill": "green"})
    with (
        pytest.raises(EditRejectedError, match="not permitted"),
        editor.transaction("Colour only", allowed=frozenset({"fill"})) as tx,
    ):
        tx.set_attributes("a", {"stroke": "black"})
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Stroke") as tx,
    ):
        tx.set_attributes("a", {"stroke-width": "3"})
    editor.set_locks("layer", frozenset({EditKind.PAINT}))
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Paint") as tx,
    ):
        tx.set_attributes("a", {"fill": "blue"})


def test_node_selection_and_pinned_endpoint_leave_handles_editable():
    document = import_svg(SVG)
    geometry = document.geometry_for("b")
    node = geometry.subpaths[0].nodes[1]
    editor = Editor(document, selection=select("b", nodes=[node.id]))
    editor.pin_node("b", node.id)
    with editor.transaction("Adjust handle") as tx:
        tx.update_node("b", node.id, (41, 30, 54, 30, 60, 36))
    with (
        pytest.raises(EditRejectedError, match="pinned"),
        editor.transaction("Move endpoint") as tx,
    ):
        tx.update_node("b", node.id, (41, 30, 54, 30, 61, 36))
    with (
        pytest.raises(EditRejectedError, match="selected nodes"),
        editor.transaction("Wrong node") as tx,
    ):
        tx.update_node("b", geometry.subpaths[0].nodes[0].id, (37, 36))
    with (
        pytest.raises(EditRejectedError, match="Whole-object"),
        editor.transaction("Whole transform") as tx,
    ):
        tx.set_attributes("b", {"transform": "translate(1 0)"})


def test_transaction_rolls_back_all_changes_on_error_even_if_caught_inside():
    editor = Editor(import_svg(SVG), selection=select("a"))
    before = editor.snapshot
    tx = editor.transaction("Invalid batch")
    tx.set_attributes("a", {"fill": "green"})
    with pytest.raises(EditRejectedError):
        tx.set_attributes("b", {"fill": "green"})
    with pytest.raises(EditRejectedError, match="failed"):
        tx.commit()
    assert editor.snapshot == before
    assert not editor.undo_labels


def test_abort_and_noop_do_not_create_history():
    editor = Editor(import_svg(SVG), selection=select("a"))
    tx = editor.transaction("Preview")
    tx.set_attributes("a", {"fill": "green"})
    tx.abort()
    with editor.transaction("No-op") as tx:
        tx.set_attributes("a", {"fill": "red"})
    assert editor.snapshot.revision == 0
    assert not editor.undo_labels
    with pytest.raises(EditRejectedError, match="closed"):
        tx.set_attributes("a", {"fill": "black"})


def test_stale_result_is_rejected_even_after_undo_restores_same_content():
    editor = Editor(import_svg(SVG), selection=select("a"))
    pending = editor.transaction("Slow result")
    pending.set_attributes("a", {"fill": "green"})
    with editor.transaction("Manual edit") as tx:
        tx.set_attributes("a", {"fill": "black"})
    editor.undo()
    assert editor.snapshot.revision == 2
    with pytest.raises(StaleRevisionError):
        pending.commit()
    with pytest.raises(StaleRevisionError):
        editor.transaction("Old request", expected_revision=0)
    assert editor.snapshot.document.element("a").get("fill") == "red"


def test_new_edit_after_undo_discards_redo_branch():
    editor = Editor(import_svg(SVG), selection=select("a"))
    with editor.transaction("First") as tx:
        tx.set_attributes("a", {"fill": "green"})
    editor.undo()
    with editor.transaction("Replacement") as tx:
        tx.set_attributes("a", {"fill": "black"})
    assert not editor.redo_labels
    assert editor.redo() == editor.snapshot


def test_scope_is_captured_when_transaction_starts():
    editor = Editor(import_svg(SVG), selection=select("a"))
    tx = editor.transaction("Captured selection")
    editor.select(select("b"))
    tx.set_attributes("a", {"fill": "green"})
    tx.commit()
    assert editor.snapshot.selection == select("b")
    assert editor.snapshot.document.element("a").get("fill") == "green"


def test_shared_geometry_requires_all_visible_users_and_respects_their_locks():
    document = import_svg(SHARED)
    geometry = document.geometry_for("first")
    node = geometry.subpaths[0].nodes[1]
    editor = Editor(document, selection=select("first", "second"))
    with (
        pytest.raises(EditRejectedError, match="unselected"),
        editor.transaction("Move shared edge") as tx,
    ):
        tx.update_node("first", node.id, (30, 4))  # Also affects clipped rect.
    editor.select(select("first", "second", "clipped"))
    editor.set_locks("clipped", frozenset({EditKind.GEOMETRY}))
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Move shared edge") as tx,
    ):
        tx.update_node("first", node.id, (30, 4))
    editor.set_locks("clipped", frozenset())
    with editor.transaction("Move shared edge") as tx:
        tx.update_node("first", node.id, (30, 4))
    assert editor.snapshot.document.geometry_for("second").node(node.id).endpoint == (
        30,
        4,
    )


def test_detach_use_preserves_rendering_then_allows_independent_edit():
    document = import_svg(SHARED)
    editor = Editor(document, selection=select("first"))
    with editor.transaction("Detach") as tx:
        detached_id = tx.detach_geometry("first")
    detached = editor.snapshot.document
    assert np.array_equal(render(export_svg(document)), render(export_svg(detached)))
    assert detached.geometry_for("first").id == detached_id
    assert detached.geometry_for("second").id != detached_id
    node = detached.geometry_for("first").subpaths[0].nodes[1]
    with editor.transaction("Edit independent instance") as tx:
        tx.update_node("first", node.id, (32, 4))
    assert editor.snapshot.document.geometry_for("second") == document.geometry_for(
        "second"
    )
    editor.undo()
    assert editor.undo().document == document


def test_explicit_shared_assets_survive_project_save_and_detach():
    editor = Editor(import_svg(SVG), selection=Selection.all())
    with editor.transaction("Link") as tx:
        tx.share_geometry("b", "a")
    saved, selection = load_project(
        save_project(editor.snapshot.document, editor.snapshot.selection)
    )
    assert saved.geometry_for("a") is saved.geometry_for("b")
    restored = Editor(saved, selection=selection)
    with restored.transaction("Detach") as tx:
        tx.detach_geometry("b")
    assert (
        restored.snapshot.document.geometry_for("a").id
        != restored.snapshot.document.geometry_for("b").id
    )


def test_project_roundtrip_retains_all_identities_constraints_and_selection():
    editor = Editor(import_svg(SHARED), selection=Selection.all())
    node = editor.snapshot.document.geometry_for("first").subpaths[0].nodes[0]
    editor.pin_node("first", node.id)
    editor.set_locks("first", frozenset({"fill", EditKind.TRANSFORM}))
    editor.select(select("first", nodes=[node.id], focus=Rect(0, 0, 10, 10)))
    document, selection = load_project(
        save_project(editor.snapshot.document, editor.snapshot.selection)
    )
    assert document == editor.snapshot.document
    assert selection == editor.snapshot.selection
    assert export_svg(document) == export_svg(editor.snapshot.document)


@pytest.mark.parametrize(
    "body",
    [
        '<path id="same" d="M0 0"/><path id="same" d="M1 1"/>',
        '<use href="#missing"/>',
        '<g id="loop"><use href="#loop"/></g>',
        '<defs><path id="p" d="M0 0" clip-path="url(#c)"/>'
        '<clipPath id="c"><use href="#p"/></clipPath></defs>',
    ],
)
def test_broken_references_and_duplicate_ids_are_rejected(body):
    with pytest.raises(DocumentError):
        import_svg('<svg xmlns="http://www.w3.org/2000/svg">' + body + "</svg>")


@pytest.mark.parametrize(
    "body",
    [
        "<text>Hello</text>",
        '<path d="M0 0 A1 1 0 0 0 3 3"/>',
        '<path d="M0 0" filter="url(#blur)"/>',
        '<g style="opacity:.5"/>',
        '<use href="https://example.com/image.svg#p"/>',
        '<path d="M0 0" fill="url(#gradient)"/>',
    ],
)
def test_unsupported_content_is_reported_instead_of_dropped(body):
    with pytest.raises(DocumentError):
        import_svg('<svg xmlns="http://www.w3.org/2000/svg">' + body + "</svg>")


def test_import_reports_multiple_unsupported_features():
    with pytest.raises(UnsupportedSvgError) as exc:
        import_svg('<svg><text>Hello</text><path d="M0 0 A1 1 0 0 0 3 3"/></svg>')
    assert len(exc.value.issues) >= 2


@pytest.mark.parametrize(
    "data", ["L0 0", "M0", "M0 0 L", "M0 0 C1 2", "M0 0 garbage", "M1e999 0", "Z"]
)
def test_malformed_path_data_is_not_partially_imported(data):
    with pytest.raises(UnsupportedSvgError):
        import_svg(f'<svg><path d="{data}"/></svg>')


def test_relative_and_shorthand_curves_normalize_without_visual_change():
    source = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
        '<path fill="none" stroke="red" '
        'd="m+4e0 4 h20 v4 l-4 4 q-4 6 -8 8 t8 10 c4 0 4 4 8 4 s4 4 8 4"/>'
        "</svg>"
    )
    document = import_svg(source)
    assert {
        node.command for s in document.geometries[0].subpaths for node in s.nodes
    } <= {"M", "L", "C"}
    difference = np.abs(
        render(source).astype(int) - render(export_svg(document)).astype(int)
    )
    assert difference.max() <= 1


def test_invalid_project_versions_and_dangling_geometry_are_rejected():
    data = json.loads(save_project(import_svg(SVG)))
    data["version"] = 999
    with pytest.raises(DocumentError, match="version"):
        load_project(json.dumps(data))
    data["version"] = 1
    data["geometries"] = []
    with pytest.raises(DocumentError, match="geometry"):
        load_project(json.dumps(data))


def test_selection_rejects_unknown_ids_and_nodes_from_other_objects():
    document = import_svg(SVG)
    node = document.geometry_for("b").subpaths[0].nodes[0]
    with pytest.raises(DocumentError, match="Unknown object"):
        Editor(document, selection=select("missing"))
    with pytest.raises(DocumentError, match="Selected nodes"):
        Editor(document, selection=select("a", nodes=[node.id]))
    with pytest.raises(DocumentError, match="explicit objects"):
        Editor(document, selection=replace(select("a"), whole_document=True))


def test_polygon_and_polyline_are_converted_without_changing_rendering():
    source = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
        '<polygon id="p" points="4,4 20,4 20,20" fill="red"/>'
        '<polyline points="32,32 40,40 50,32" stroke="blue" fill="none"/>'
        "</svg>"
    )
    document = import_svg(source)
    assert document.element("p").tag == "path"
    assert np.array_equal(render(source), render(export_svg(document)))


def test_detach_preserves_definition_and_instance_paint_transforms_and_clipping():
    source = SHARED.replace(
        'id="shape" fill-rule="evenodd"',
        'id="shape" fill="orange" transform="scale(.8)" fill-rule="evenodd"',
    ).replace('id="first" href=', 'id="first" x="2" y="3" href=')
    editor = Editor(import_svg(source), selection=select("first"))
    with editor.transaction("Detach") as tx:
        tx.detach_geometry("first")
    assert np.array_equal(render(source), render(export_svg(editor.snapshot.document)))


def test_group_reference_dependency_protects_objects_outside_selection():
    source = (
        '<svg><defs><g id="shape"><path id="p" d="M0 0 L2 2"/></g></defs>'
        '<use id="a" href="#shape"/><use id="b" href="#shape"/></svg>'
    )
    editor = Editor(import_svg(source), selection=select("a"))
    node = editor.snapshot.document.geometry_for("p").subpaths[0].nodes[1]
    with (
        pytest.raises(EditRejectedError, match="unselected"),
        editor.transaction("Definition edit") as tx,
    ):
        tx.update_node("p", node.id, (3, 3))


def test_invalid_property_edit_poisoning_prevents_partial_commit():
    editor = Editor(import_svg(SVG), selection=select("a"))
    tx = editor.transaction("Bad width")
    tx.set_attributes("a", {"fill": "green"})
    with pytest.raises(DocumentError, match="finite"):
        tx.set_attributes("a", {"stroke-width": "nan"})
    with pytest.raises(EditRejectedError, match="failed"):
        tx.commit()
    assert editor.snapshot.document.element("a").get("fill") == "red"


def test_resource_ancestor_lock_cannot_be_bypassed_via_an_instance():
    editor = Editor(import_svg(SHARED), selection=Selection.all())
    editor.set_locks("library", frozenset({EditKind.GEOMETRY}))
    node = editor.snapshot.document.geometry_for("first").subpaths[0].nodes[0]
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Shared edit") as tx,
    ):
        tx.update_node("first", node.id, (5, 4))


def test_pinned_geometry_cannot_be_replaced_by_sharing():
    editor = Editor(import_svg(SVG), selection=Selection.all())
    node = editor.snapshot.document.geometry_for("a").subpaths[0].nodes[0]
    editor.pin_node("a", node.id)
    with (
        pytest.raises(EditRejectedError, match="pinned"),
        editor.transaction("Replace geometry") as tx,
    ):
        tx.share_geometry("a", "b")


def test_project_loader_checks_structural_and_property_constraints():
    data = json.loads(save_project(import_svg(SVG)))
    data["root"]["locks"] = ["unknown-property"]
    with pytest.raises(DocumentError, match="Unknown property lock"):
        load_project(json.dumps(data))
    with pytest.raises(UnsupportedSvgError, match="Invalid XML"):
        import_svg("<svg><path></svg>")


def test_focus_region_validation_and_immutable_selection_inputs():
    with pytest.raises(DocumentError, match="positive"):
        Rect(0, 0, 0, 10)
    with pytest.raises(DocumentError, match="finite"):
        Rect(float("nan"), 0, 10, 10)
    objects = {"a"}
    selection = Selection(object_ids=frozenset(objects))
    objects.add("b")
    assert selection.object_ids == {"a"}


def test_detach_materializes_path_with_instance_and_source_compositing():
    source = """<svg width="80" height="80"><defs fill="purple">
    <clipPath id="outer"><rect width="35" height="32"/></clipPath>
    <clipPath id="inner"><rect x="2" y="2" width="29" height="24"/></clipPath>
    <path id="shape" fill="orange" opacity=".6" transform="scale(.8)"
      clip-path="url(#inner)" d="M0 0H40V40H0Z"/></defs>
    <g fill="blue"><use id="first" href="#shape" x="4" y="5"
      fill="red" stroke="black" transform="translate(8 9)"
      clip-path="url(#outer)"/></g>
    <use id="second" href="#shape" x="40"/></svg>"""
    document = import_svg(source)
    editor = Editor(document, selection=select("first"))
    with editor.transaction("Detach") as tx:
        tx.detach_geometry("first")
    detached = editor.snapshot.document
    assert detached.element("first").tag == "path"
    assert editor.snapshot.selection.object_ids == {"first"}
    assert detached.element("shape") == document.element("shape")
    assert detached.element("second") == document.element("second")
    assert np.array_equal(render(source), render(export_svg(detached)))


def test_detach_rejects_referenced_instance_without_partial_changes():
    document = import_svg(
        SHARED.replace("</svg>", '<use id="third" href="#first" x="10"/></svg>')
    )
    editor = Editor(document, selection=Selection.all())
    with (
        pytest.raises(EditRejectedError, match="references to this instance"),
        editor.transaction("Detach") as tx,
    ):
        tx.detach_geometry("first")
    assert editor.snapshot.document == document
