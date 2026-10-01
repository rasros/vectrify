"""History hooks the agent uses: labels by author, one step of several, and
taking back a refused edit's steps."""

from vectrify.document import Editor, Selection, import_svg

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
