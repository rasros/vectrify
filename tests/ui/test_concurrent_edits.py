"""Manual and MCP requests share a drawing without document-wide refresh errors."""

from threading import Barrier, Thread

import pytest

from tests.ui.test_session import SVG, send
from vectrify.document import DocumentError, Element, Selection, import_svg
from vectrify.document.history import ConcurrentEditError
from vectrify.ui.agent import Agent
from vectrify.ui.session import Session


def manual_payload(session: Session, command: str, **fields) -> dict:
    state = session.state(svg=False)
    return {
        "epoch": state["epoch"],
        "revision": state["revision"],
        "selection": state["selection"],
        "command": command,
        **fields,
    }


def test_simultaneous_manual_and_mcp_edits_both_commit():
    session = Session(import_svg(SVG))
    agent = Agent(session)
    send(session, "select", objects=["a"])
    manual = manual_payload(session, "paint", changes={"fill": "green"})
    seen = [session.epoch, 0]
    barrier = Barrier(3)
    errors = []

    def person():
        try:
            barrier.wait(timeout=5)
            with session.lock:
                session.action(manual)
        except Exception as exc:
            errors.append(exc)

    def mcp():
        try:
            barrier.wait(timeout=5)
            agent.call("properties", {"seen": seen, "ids": ["b"], "fill": "black"})
        except Exception as exc:
            errors.append(exc)

    threads = [Thread(target=person), Thread(target=mcp)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert not errors
    document = session.editor.snapshot.document
    assert document.element("a").get("fill") == "green"
    assert document.element("b").get("fill") == "black"
    assert session.editor.snapshot.revision == 2
    assert session.editor.snapshot.selection.object_ids == {"a"}
    assert sorted(e.author for e in session.editor.undo_entries) == ["agent", "person"]
    send(session, "undo")
    assert session.editor.snapshot.document.element("a").get("fill") == "red"
    assert session.editor.snapshot.document.element("b").get("fill") == "black"


def test_stale_manual_node_drag_merges_with_agent_paint_on_the_same_object():
    session = Session(import_svg(SVG))
    agent = Agent(session)
    send(session, "select", objects=["a"])
    node = session.editor.snapshot.document.geometry_for("a").subpaths[0].nodes[1]
    manual = manual_payload(session, "node", object="a", node=node.id, values=[22, 2])
    painted = agent.call(
        "properties",
        {
            "seen": [session.epoch, 0],
            "ids": ["a"],
            "fill": "green",
        },
    ).data
    result = session.action(manual)
    assert result["revision"] == 2
    assert 'fill="green"' in result["svg"]
    assert session.editor.snapshot.document.geometry_for("a").node(
        node.id
    ).endpoint == (22, 2)
    assert session.editor.undo_entries[-1].before.element("a").get("fill") == "green"
    # The MCP undo still names the paint it made, even without another look.
    agent.call(
        "undo",
        {
            "seen": [painted["epoch"], painted["revision"]],
            "ids": [painted["edit_id"]],
        },
    )
    assert session.editor.snapshot.document.element("a").get("fill") == "red"
    assert session.editor.snapshot.document.geometry_for("a").node(
        node.id
    ).endpoint == (22, 2)


def test_stale_mcp_move_merges_with_manual_paint_and_keeps_selection():
    session = Session(import_svg(SVG))
    agent = Agent(session)
    seen = [session.epoch, 0]
    send(session, "select", objects=["a"])
    send(session, "paint", changes={"fill": "green"})
    send(session, "select", objects=["b"])
    moved = agent.call("transform", {"seen": seen, "ids": ["a"], "dx": 3}).data
    assert moved["revision"] == 2
    assert moved["edit_id"] == session.editor.undo_entries[-1].id
    assert session.editor.snapshot.document.element("a").get("fill") == "green"
    assert (
        session.editor.snapshot.document.element("a").get("transform")
        == "translate(3.0 0.0)"
    )
    assert session.editor.snapshot.selection.object_ids == {"b"}


def test_stale_overlap_is_atomic_and_does_not_ask_for_refresh():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["a"])
    manual = manual_payload(session, "paint", changes={"fill": "green"})
    Agent(session).call(
        "properties",
        {
            "seen": [session.epoch, 0],
            "ids": ["a"],
            "fill": "black",
        },
    )
    before = session.editor.snapshot
    entries = session.editor.undo_entries
    with pytest.raises(ConcurrentEditError, match="fill") as error:
        session.action(manual)
    assert "refresh" not in str(error.value).lower()
    assert session.editor.snapshot == before
    assert session.editor.undo_entries == entries


@pytest.mark.parametrize("constraint", ["lock", "pin", "shared", "parent"])
def test_old_node_edit_respects_new_constraints_and_coordinate_frames(constraint):
    session = Session(import_svg(SVG))
    send(session, "select", objects=["a"])
    node = session.editor.snapshot.document.geometry_for("a").subpaths[0].nodes[1]
    manual = manual_payload(session, "node", object="a", node=node.id, values=[22, 2])
    if constraint == "lock":
        session.editor.set_locks("a", frozenset({"geometry"}))
    elif constraint == "pin":
        session.editor.pin_node("a", node.id)
    elif constraint == "shared":
        with session.editor.transaction("Share", selection=Selection.all()) as tx:
            tx.insert_object(
                tx.preview.root.id,
                Element(
                    "new-consumer",
                    "path",
                    geometry_id=tx.preview.geometry_for("a").id,
                ),
            )
    else:
        with session.editor.transaction("Move parent", selection=Selection.all()) as tx:
            tx.set_attributes("layer", {"transform": "translate(10 0)"})
    before = session.editor.snapshot
    with pytest.raises(DocumentError, match="conflicts"):
        session.action(manual)
    assert session.editor.snapshot == before


def test_stale_manual_undo_restores_its_exact_entry():
    session = Session(import_svg(SVG))
    agent = Agent(session)
    send(session, "select", objects=["a"])
    send(session, "paint", changes={"fill": "green"})
    state = session.state()
    manual = manual_payload(session, "undo", ids=[state["undo_ids"][-1]])
    agent.call(
        "properties", {"seen": [session.epoch, 1], "ids": ["b"], "fill": "black"}
    )
    result = session.action(manual)
    assert result["undo"] == []
    assert session.editor.snapshot.document.element("a").get("fill") == "red"
    assert session.editor.snapshot.document.element("b").get("fill") == "black"
