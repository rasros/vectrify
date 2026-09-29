"""Simplification previews are non-destructive and revision-bound."""

import pytest

from tests.document.test_simplify import circle, document
from vectrify.document import DocumentError, Selection, StaleRevisionError, export_svg
from vectrify.ui.session import Session


def preview(session, **options):
    job = session.operation(
        {
            "command": "start",
            "action": "simplify",
            "method": "curves",
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            "permissions": {"geometry": True, "structure": True},
            "settings": options,
            "bounds": [10, 10, 80, 80],
        }
    )
    return dict(job["result"], id=job["id"], **job["result"]["metrics"])


def test_preview_apply_undo_and_discard():
    session = Session(document(circle()))
    session.editor.select(Selection(object_ids=frozenset({"a"})))
    original = session.editor.snapshot.document
    result = preview(session, tolerance=1)
    assert result["changed"]
    assert result["after"]["nodes"] < result["before"]["nodes"]
    assert result["after"]["coordinates"] < result["before"]["coordinates"]
    assert result["after"]["bytes"] < result["before"]["bytes"]
    assert all(
        value.startswith("data:image/png;base64,")
        for value in result["previews"].values()
    )
    assert session.editor.snapshot.document == original
    assert session.editor.undo_labels == ()
    session.operation({"command": "discard", "job": result["id"]})
    assert session.editor.snapshot.document == original
    result = preview(session, tolerance=1)
    session.operation({"command": "apply", "job": result["id"]})
    assert session.editor.undo_labels == ("Smooth / simplify shapes",)
    assert session.editor.snapshot.document != original
    session.editor.undo()
    assert session.editor.snapshot.document == original


def test_stale_preview_never_overwrites_new_edits_or_opened_document():
    session = Session(document(circle()))
    session.editor.select(Selection(object_ids=frozenset({"a"})))
    result = preview(session)
    with session.editor.transaction("Paint") as tx:
        tx.set_attributes("a", {"fill": "red"})
    with pytest.raises(StaleRevisionError):
        session.operation({"command": "apply", "job": result["id"]})
    assert session.editor.snapshot.document.element("a").get("fill") == "red"


def test_group_scope_does_not_touch_unselected_geometry():
    from vectrify.document import import_svg

    svg = '<svg width="100" height="100"><g id="group">'
    svg += f'<path id="a" d="{circle()}"/></g><path id="b" d="{circle()}"/></svg>'
    session = Session(import_svg(svg))
    session.editor.select(Selection(object_ids=frozenset({"group"})))
    before = session.editor.snapshot.document.geometry_for("b")
    result = preview(session)
    session.operation({"command": "apply", "job": result["id"]})
    assert session.editor.snapshot.document.geometry_for("b") == before


def test_simplify_needs_geometry_and_structure_permission():
    session = Session(document(circle()))
    session.editor.select(Selection(object_ids=frozenset({"a"})))
    with pytest.raises(DocumentError, match="not permitted"):
        session.operation(
            {
                "command": "start",
                "action": "simplify",
                "method": "curves",
                "epoch": session.epoch,
                "revision": 0,
                "settings": {"tolerance": 1},
                "bounds": [10, 10, 80, 80],
            }
        )
    assert session.editor.undo_labels == ()


def test_preview_from_a_replaced_document_cannot_apply():
    session = Session(document(circle()))
    session.editor.select(Selection(object_ids=frozenset({"a"})))
    result = preview(session, tolerance=1)
    session.open(export_svg(session.editor.snapshot.document), "other.svg")
    with pytest.raises(DocumentError, match="expired"):
        session.operation({"command": "apply", "job": result["id"]})
