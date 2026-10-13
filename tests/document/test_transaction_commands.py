"""A public command owns its validation, nested edits and failure state."""

import pytest

from tests.document.test_document import select
from vectrify.document import DocumentError, Editor, EditRejectedError, import_svg


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("scale_objects", (frozenset({"p"}), (0, 0), (0, 1))),
        ("scale_objects", (frozenset(), (0, 0), (1, 1))),
        ("set_node_handles", ("p", "missing", 3)),
        ("set_node_handles", ("p", "missing", 1)),
        ("delete_handle", ("p", "missing", 1)),
        ("set_node_handle_alignment", ("p", "missing", False)),
        ("move_handle", ("p", "missing", 1, (0, 0))),
        ("straighten_node_handles", ("p", "missing", 1)),
        ("break_points", ("p", ())),
        ("delete_segments", ("p", frozenset())),
        ("join_paths", (frozenset(),)),
        ("combine_paths", (frozenset(),)),
        ("cut_out_hole", (frozenset({"p"}),)),
    ],
)
def test_rejected_public_command_cannot_commit_prior_paint(command, args):
    editor = Editor(
        import_svg('<svg><path id="p" d="M0 0 L10 0 L20 10"/></svg>'),
        selection=select("p"),
    )
    before = editor.snapshot
    tx = editor.transaction("Paint then reject")
    tx.set_attributes("p", {"fill": "red"})
    with pytest.raises(DocumentError):
        getattr(tx, command)(*args)
    with pytest.raises(EditRejectedError, match="failed edit"):
        tx.commit()
    with pytest.raises(EditRejectedError, match="failed edit"):
        tx.set_attributes("p", {"fill": "blue"})
    assert editor.snapshot == before
    assert editor.undo_labels == ()


def test_closed_transaction_rejects_a_noop_handle_command():
    editor = Editor(
        import_svg('<svg><path id="p" d="M0 0 L10 0 L20 10"/></svg>'),
        selection=select("p"),
    )
    tx = editor.transaction("No change")
    node = tx.preview.geometry_for("p").subpaths[0].nodes[1]
    tx.commit()
    with pytest.raises(EditRejectedError, match="closed"):
        tx.straighten_node_handles("p", node.id)
    assert editor.undo_labels == ()


def test_composite_resize_preserves_geometry_and_creates_one_history_entry():
    document = import_svg(
        '<svg><path id="p" d="M0 0 L10 0 L20 10" stroke="black" stroke-width="4"/>'
        '<path id="q" d="M0 20 L10 20" stroke="black" stroke-width="2"/></svg>'
    )
    editor = Editor(document, selection=select("p", "q"))
    with editor.transaction("Resize") as tx:
        tx.scale_objects(frozenset({"p", "q"}), (0, 0), (2, 2))
        assert tx.preview_selection == select("p", "q")
        assert tx.node_remapping == ()
        assert tx.object_remapping == ()
    after = editor.snapshot.document
    for oid, width in (("p", "2"), ("q", "1")):
        assert after.geometry_for(oid) is document.geometry_for(oid)
        assert after.element(oid).get("transform") == "scale(2 2)"
        assert after.element(oid).get("stroke-width") == width
    assert editor.undo_labels == ("Resize",)
    assert editor.undo().document == document
    assert editor.redo().document == after


def test_composite_handle_command_normalizes_after_both_mutations(monkeypatch):
    editor = Editor(
        import_svg('<svg><path id="p" d="M0 0 C3 0 7 0 10 0 C13 0 17 0 20 0"/></svg>'),
        selection=select("p"),
    )
    tx = editor.transaction("Move aligned handle")
    geometry = tx.preview.geometry_for("p")
    node = geometry.subpaths[0].nodes[1]
    tx.set_node_handle_alignment("p", node.id, True)
    normalized = []
    original = tx._private_paints

    def normalize():
        normalized.append(tx.preview.geometry_for("p"))
        original()

    monkeypatch.setattr(tx, "_private_paints", normalize)
    tx.move_handle("p", node.id, 2, (7, -3))
    assert len(normalized) == 1
    updated = tx.preview.geometry_for("p")
    assert updated.node(node.id).values[2:4] == (7, -3)
    assert updated.node(node.id).handles_aligned
    assert updated.subpaths[0].nodes[2].values[:2] == pytest.approx(
        (10 + 3 / 2**0.5, 3 / 2**0.5)
    )
    assert [n.id for n in updated.subpaths[0].nodes] == [
        n.id for n in geometry.subpaths[0].nodes
    ]
    tx.commit()
    assert editor.undo_labels == ("Move aligned handle",)


def test_caught_nested_rejection_cannot_clear_command_failure(monkeypatch):
    editor = Editor(
        import_svg('<svg><path id="p" d="M0 0 L10 0 L20 10"/></svg>'),
        selection=select("p"),
    )
    before = editor.snapshot
    tx = editor.transaction("Nested rejection")
    tx.set_attributes("p", {"fill": "red"})
    node = tx.preview.geometry_for("p").subpaths[0].nodes[1]

    def catch_rejection(*_args):
        with pytest.raises(EditRejectedError, match="References and path topology"):
            tx.set_attributes("p", {"d": "M0 0"})

    monkeypatch.setattr(tx, "straighten_node_handles", catch_rejection)
    with pytest.raises(EditRejectedError, match="failed edit"):
        tx.set_node_handle_alignment("p", node.id, True)
    with pytest.raises(EditRejectedError, match="failed edit"):
        tx.commit()
    assert editor.snapshot == before


@pytest.mark.parametrize("refusal", ["tolerance", "scope", "pin"])
def test_snap_preflight_refusal_keeps_prior_edits_and_transaction_usable(refusal):
    document = import_svg(
        '<svg><path id="p" d="M0 0L10 0L10 20L0 20Z"/>'
        '<path id="q" d="M10.2 0L20 0L20 20L10.2 20Z"/></svg>'
    )
    editor = Editor(document, selection=select("p", "q"))
    if refusal == "pin":
        node = document.geometry_for("p").subpaths[0].nodes[1]
        editor.pin_node("p", node.id)
    before = editor.snapshot
    tx = editor.transaction("Paint around refused snap")
    tx.set_attributes("p", {"fill": "red"})
    painted = tx.preview
    tolerance = -1 if refusal == "tolerance" else 0.3
    object_ids = frozenset({"p"}) if refusal == "scope" else None
    with pytest.raises(DocumentError):
        tx.snap_edges(tolerance, object_ids)
    assert tx.preview is painted
    tx.set_attributes("q", {"fill": "blue"})
    tx.commit()
    after = editor.snapshot.document
    assert after.element("p").get("fill") == "red"
    assert after.element("q").get("fill") == "blue"
    assert after.geometries == before.document.geometries
    assert editor.undo().document == before.document
