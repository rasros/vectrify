"""Object copy snapshots and atomic, independent, undoable pastes."""

import shutil
import subprocess
from pathlib import Path

import pytest

from vectrify.document import DocumentError, import_svg
from vectrify.document.model import references
from vectrify.ui.session import Session


def send(session, command, **data):
    return session.action(
        {
            "command": command,
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            **data,
        }
    )


def test_copy_is_read_only_and_paste_keeps_the_snapshot_after_source_deletion():
    session = Session(
        import_svg(
            '<svg width="100" height="100"><path id="a" fill="red" '
            'd="M0 0 L20 0 L20 20 Z"/></svg>'
        )
    )
    send(session, "select", objects=["a"])
    before = session.editor.snapshot
    assert send(session, "copy")["clipboard"]
    assert session.editor.snapshot == before
    assert session.state()["undo"] == []
    send(session, "paint", changes={"fill": "blue"})
    send(session, "delete")
    before_paste = session.state()["svg"]
    result = send(session, "paste")
    (oid,) = result["selection"]["objects"]
    document = session.editor.snapshot.document
    copied = document.element(oid)
    assert oid != "a"
    assert copied.tag == "path"
    assert copied.get("fill") == "red"
    assert document.geometry_for(oid).id != before.document.geometry_for("a").id
    assert result["selection"]["nodes"] == []
    assert result["undo"][-1] == "Paste objects"
    assert send(session, "undo")["svg"] == before_paste
    assert send(session, "redo")["svg"] == result["svg"]
    assert session.editor.snapshot.selection.object_ids == frozenset({oid})
    again = send(session, "paste")
    assert again["selection"]["objects"] != [oid]
    send(session, "paint", changes={"fill": "green"})
    assert session.editor.snapshot.document.element(oid).get("fill") == "red"


def test_copy_keeps_group_context_references_names_locks_and_pinned_points():
    session = Session(
        import_svg("""<svg width="100" height="100">
    <defs><linearGradient id="paint"><stop offset="0" stop-color="red"/>
    <stop offset="1" stop-color="blue"/></linearGradient>
    <clipPath id="clip"><rect width="50" height="50"/></clipPath>
    <path id="shape" d="M0 0 L20 0 L20 20 Z"/></defs>
    <g id="parent" transform="translate(10 15)" fill="url(#paint)"
       clip-path="url(#clip)" opacity="0.5">
    <path id="a" d="M0 0 L10 0 L10 10 Z"/>
    <use id="b" href="#shape" x="20"/>
    <rect id="excluded" width="100" height="100"/>
    </g></svg>""")
    )
    send(session, "select", objects=["a"])
    send(session, "rename", object="a", name="Triangle")
    node = session.editor.snapshot.document.geometry_for("a").subpaths[0].nodes[0]
    session.editor.pin_nodes([("a", node.id)], pinned=True)
    send(session, "locks", object="a", locks=["paint"])
    send(session, "select", objects=["a", "b"])
    send(session, "copy")
    original = session.editor.snapshot.document
    result = send(session, "paste")
    document = session.editor.snapshot.document
    (group,) = [document.element(oid) for oid in result["selection"]["objects"]]
    assert group.tag == "g"
    assert group.get("transform") == "translate(10 15)"
    assert group.get("opacity") == "0.5"
    assert len(group.children) == 2
    path, instance = group.children
    assert path.name == "Triangle"
    assert path.locks == frozenset({"paint"})
    assert document.geometry_for(path.id).subpaths[0].nodes[0].pinned
    assert document.geometry_for(path.id).id != original.geometry_for("a").id
    assert document.geometry_for(instance.id).id != original.geometry_for("b").id
    new_ids = {e.id for e in document.elements()} - {e.id for e in original.elements()}
    assert all(ref in new_ids for e in [group, instance] for ref in references(e))
    document.validate()


def test_selected_group_and_descendant_are_copied_once():
    session = Session(
        import_svg('<svg><g id="g"><path id="a" d="M0 0 L10 10"/></g></svg>')
    )
    send(session, "select", objects=["g", "a"])
    send(session, "copy")
    result = send(session, "paste")
    (oid,) = result["selection"]["objects"]
    group = session.editor.snapshot.document.element(oid)
    assert group.tag == "g"
    assert len(group.children) == 1


def test_copy_of_a_child_and_an_instance_keeps_the_complete_referenced_group():
    session = Session(
        import_svg("""<svg>
    <g id="group" fill="red"><path id="a" d="M0 0 L10 10"/>
    <rect id="b" width="10" height="10"/></g>
    <use id="instance" href="#group" x="20"/>
    </svg>""")
    )
    send(session, "select", objects=["a", "instance"])
    send(session, "copy")
    result = send(session, "paste")
    document = session.editor.snapshot.document
    selected = [document.element(oid) for oid in result["selection"]["objects"]]
    (context,) = [e for e in selected if e.tag == "g"]
    (instance,) = [e for e in selected if e.tag == "use"]
    assert len(context.children) == 1
    referenced = document.element(instance.get("href")[1:])
    assert len(referenced.children) == 2
    assert referenced.id != context.id
    assert document.geometry_for(context.children[0].id).id != (
        document.geometry_for("a").id
    )
    document.validate()


def test_paste_refused_by_structure_lock_is_atomic_and_keeps_clipboard():
    session = Session(import_svg('<svg><rect id="a" width="10" height="10"/></svg>'))
    send(session, "select", objects=["a"])
    send(session, "copy")
    send(session, "locks", object="a", locks=["structure"])
    before = session.editor.snapshot
    history = session.editor.undo_entries
    with pytest.raises(DocumentError, match="locked"):
        send(session, "paste")
    assert session.editor.snapshot == before
    assert session.editor.undo_entries == history
    assert session.state()["clipboard"]


def test_clipboard_refusals_leave_drawing_and_history_unchanged():
    session = Session(
        import_svg('<svg><defs><path id="a" d="M0 0 L10 10"/></defs></svg>')
    )
    before = session.editor.snapshot
    with pytest.raises(DocumentError, match="Copy objects first"):
        send(session, "paste")
    with pytest.raises(DocumentError, match="Select an object"):
        send(session, "copy")
    assert session.editor.snapshot == before
    send(session, "select", objects=["a"])
    with pytest.raises(DocumentError, match="not definitions"):
        send(session, "copy")
    assert not session.state()["clipboard"]
    assert not session.state()["undo"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_copy_paste_shortcuts_queue_and_preserve_text_editing():
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("clipboard.mjs"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
