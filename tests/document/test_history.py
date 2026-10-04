"""History hooks the agent uses: labels by author, one step of several, and
taking back a refused edit's steps."""

import pytest

from vectrify.document import DocumentError, Editor, Selection, import_svg
from vectrify.document.history import HistoryConflictError

SVG = """<svg width="100" height="100">
<path id="a" fill="red" d="M0 0 L20 0 L20 20 Z"/>
<path id="b" fill="blue" d="M30 0 L50 0 L50 20 Z"/>
</svg>"""


def paint(editor: Editor, oid: str, fill: str) -> None:
    with editor.transaction(
        "Change paint", selection=Selection(object_ids=frozenset({oid}))
    ) as tx:
        tx.set_attributes(oid, {"fill": fill})


def test_a_label_prefix_marks_the_edits_made_while_it_is_set():
    editor = Editor(import_svg(SVG))
    paint(editor, "a", "green")
    editor.label_prefix = "Agent: "
    paint(editor, "b", "green")
    editor.label_prefix = ""
    assert editor.undo_labels == ("Change paint", "Agent: Change paint")
    assert [e.revision for e in editor.undo_entries] == [1, 2]
    editor.undo()
    assert editor.redo_entries[0].label == "Agent: Change paint"


def test_squash_makes_several_edits_one_undo_step():
    editor = Editor(import_svg(SVG))
    paint(editor, "a", "black")
    since = len(editor.undo_entries)
    paint(editor, "a", "green")
    paint(editor, "b", "green")
    editor.squash(since, "Agent: Recolour")
    assert editor.undo_labels == ("Change paint", "Agent: Recolour")
    editor.undo()
    document = editor.snapshot.document
    assert document.element("a").get("fill") == "black"
    assert document.element("b").get("fill") == "blue"
    editor.redo()
    assert editor.snapshot.document.element("b").get("fill") == "green"


def test_rollback_takes_edits_back_without_leaving_a_redo():
    editor = Editor(import_svg(SVG))
    original = editor.snapshot.document
    paint(editor, "a", "green")
    paint(editor, "b", "green")
    revision = editor.snapshot.revision
    editor.rollback(0)
    assert editor.snapshot.document == original
    assert editor.undo_labels == editor.redo_labels == ()
    # Taking edits back is a new revision, never an earlier one again.
    assert editor.snapshot.revision == revision + 1


def test_person_undo_and_redo_preserve_agent_edits_on_the_same_object():
    editor = Editor(import_svg(SVG))
    paint(editor, "a", "green")
    person = editor.undo_entries[-1]
    editor.author = "agent"
    with editor.transaction("Move", selection=Selection.all()) as tx:
        tx.set_attributes("a", {"transform": "translate(5 0)"})
    agent = editor.undo_entries[-1]
    editor.author = "person"
    editor.undo(author="person")
    assert editor.snapshot.document.element("a").get("fill") == "red"
    assert editor.snapshot.document.element("a").get("transform") == "translate(5 0)"
    assert editor.undo_entries == (agent,)
    assert editor.redo_entries == (person,)
    editor.redo(author="person")
    assert editor.snapshot.document.element("a").get("fill") == "green"
    assert editor.snapshot.document.element("a").get("transform") == "translate(5 0)"


def test_agent_edits_keep_person_redo_branch():
    editor = Editor(import_svg(SVG))
    paint(editor, "a", "green")
    entry = editor.undo_entries[-1]
    editor.undo(author="person")
    editor.author = "agent"
    paint(editor, "b", "black")
    assert editor.redo_entries == (entry,)
    editor.redo(author="person")
    assert editor.snapshot.document.element("a").get("fill") == "green"
    assert editor.snapshot.document.element("b").get("fill") == "black"


def test_selective_undo_preserves_other_nodes_and_paint():
    editor = Editor(import_svg(SVG))
    first, second = editor.snapshot.document.geometry_for("a").subpaths[0].nodes[:2]
    with editor.transaction("First node", selection=Selection.all()) as tx:
        tx.update_node("a", first.id, (1, 1))
    entry = editor.undo_entries[-1]
    with editor.transaction("Other node", selection=Selection.all()) as tx:
        tx.update_node("a", second.id, (21, 1))
        tx.set_attributes("a", {"fill": "green"})
    editor.undo([entry.id])
    geometry = editor.snapshot.document.geometry_for("a")
    assert geometry.node(first.id) == first
    assert geometry.node(second.id).endpoint == (21, 1)
    assert editor.snapshot.document.element("a").get("fill") == "green"
    editor.redo([entry.id])
    assert editor.snapshot.document.geometry_for("a").node(first.id).endpoint == (1, 1)


def test_structural_undo_preserves_an_independent_deletion():
    editor = Editor(import_svg(SVG))
    original_geometry = editor.snapshot.document.geometry_for("a")
    with editor.transaction("Delete a", selection=Selection.all()) as tx:
        tx.delete_objects(["a"])
    entry = editor.undo_entries[-1]
    with editor.transaction("Delete b", selection=Selection.all()) as tx:
        tx.delete_objects(["b"])
    editor.undo([entry.id])
    assert [e.id for e in editor.snapshot.document.root.children] == ["a"]
    assert editor.snapshot.document.geometry_for("a") == original_geometry


def test_reorder_undo_keeps_later_paint():
    editor = Editor(import_svg(SVG))
    with editor.transaction("Restack", selection=Selection.all()) as tx:
        tx.reorder_object("a", 1)
    entry = editor.undo_entries[-1]
    paint(editor, "b", "green")
    editor.undo([entry.id])
    assert [e.id for e in editor.snapshot.document.root.children] == ["a", "b"]
    assert editor.snapshot.document.element("b").get("fill") == "green"


def test_overlapping_undo_is_refused_and_batch_is_atomic():
    editor = Editor(import_svg(SVG))
    paint(editor, "a", "green")
    first = editor.undo_entries[-1]
    paint(editor, "b", "green")
    second = editor.undo_entries[-1]
    paint(editor, "a", "black")
    latest = editor.undo_entries[-1]
    before, entries = editor.snapshot, editor.undo_entries
    with pytest.raises(HistoryConflictError, match="fill"):
        editor.undo([second.id, first.id])
    assert editor.snapshot == before
    assert editor.undo_entries == entries
    assert not editor.redo_entries
    editor.undo([latest.id, first.id])
    assert editor.snapshot.document.element("a").get("fill") == "red"
    assert editor.snapshot.document.element("b").get("fill") == "green"
    assert editor.snapshot.revision == before.revision + 1
    editor.redo([first.id, latest.id])
    assert editor.snapshot.document.element("a").get("fill") == "black"


@pytest.mark.parametrize("ids", [[], ["missing"], ["duplicate", "duplicate"]])
def test_invalid_history_ids_change_nothing(ids):
    editor = Editor(import_svg(SVG))
    paint(editor, "a", "green")
    before = editor.snapshot
    with pytest.raises(DocumentError):
        editor.undo(ids)
    assert editor.snapshot == before
    assert len(editor.undo_entries) == 1


def test_undoing_creation_refuses_to_delete_later_work():
    editor = Editor(import_svg(SVG))
    with editor.transaction("Delete", selection=Selection.all()) as tx:
        tx.delete_objects(["a"])
    entry = editor.undo_entries[-1]
    editor.undo([entry.id])
    editor.author = "agent"
    paint(editor, "a", "black")
    before = editor.snapshot
    with pytest.raises(HistoryConflictError):
        editor.redo([entry.id])
    assert editor.snapshot == before


def test_redoing_deletion_refuses_to_hide_later_geometry_work():
    editor = Editor(import_svg(SVG))
    with editor.transaction("Delete", selection=Selection.all()) as tx:
        tx.delete_objects(["a"])
    entry = editor.undo_entries[-1]
    editor.undo([entry.id])
    editor.author = "agent"
    node = editor.snapshot.document.geometry_for("a").subpaths[0].nodes[1]
    with editor.transaction("Move node", selection=Selection.all()) as tx:
        tx.update_node("a", node.id, (22, 2))
    before = editor.snapshot
    with pytest.raises(HistoryConflictError, match="geometry"):
        editor.redo([entry.id])
    assert editor.snapshot == before


def test_author_is_metadata_independent_of_label():
    editor = Editor(import_svg(SVG))
    editor.label_prefix = "Agent: "
    paint(editor, "a", "green")
    assert editor.undo_entries[-1].author == "person"
    editor.undo(author="person")
    assert editor.snapshot.document.element("a").get("fill") == "red"
