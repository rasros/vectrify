from __future__ import annotations

import pytest

from tests.mcp.test_agent import fresh
from vectrify.document import DocumentError


def stage(agent, seen):
    return agent.call(
        "edit_batch",
        {
            "seen": seen,
            "edits": [
                {"tool": "properties", "args": {"ids": ["hill"], "fill": "red"}},
                {"tool": "transform", "args": {"ids": ["hill"], "dx": 2}},
            ],
            "close_region": [0, 50, 150, 150],
        },
    ).data


def test_private_batch_merges_unrelated_edit_and_undoes_once():
    agent, seen = fresh()
    editor = agent.session.editor
    before = editor.snapshot.document
    reply = stage(agent, seen)
    assert editor.snapshot.document == before
    assert editor.snapshot.revision == 0
    assert len(agent._batches[reply["id"]].images) == 8
    assert reply["diagnostics"]["effects"]["paint_changed"]
    agent.call("properties", {"seen": seen, "ids": ["sun"], "fill": "blue"})
    applied = agent.call(
        "edit_batch", {"seen": seen, "action": "apply", "id": reply["id"]}
    ).data
    assert editor.snapshot.revision == 2
    assert editor.snapshot.document.element("hill").get("fill") == "red"
    assert editor.snapshot.document.element("sun").get("fill") == "blue"
    agent.call("undo", {"seen": [seen[0], 2], "ids": [applied["edit_id"]]})
    assert editor.snapshot.document.element("hill") == before.element("hill")
    assert editor.snapshot.document.element("sun").get("fill") == "blue"


def test_overlapping_conflict_never_partially_applies():
    agent, seen = fresh()
    batch = stage(agent, seen)
    agent.call("properties", {"seen": seen, "ids": ["hill"], "fill": "blue"})
    editor = agent.session.editor
    before = editor.snapshot
    history = editor.undo_entries
    with pytest.raises(DocumentError, match="conflict"):
        agent.call("edit_batch", {"seen": seen, "action": "apply", "id": batch["id"]})
    assert editor.snapshot == before
    assert editor.undo_entries == history


def test_failed_batch_and_discard_leave_live_state_untouched():
    agent, seen = fresh()
    with pytest.raises(DocumentError):
        agent.call(
            "edit_batch",
            {
                "seen": seen,
                "edits": [
                    {"tool": "properties", "args": {"ids": ["hill"], "fill": "red"}},
                    {
                        "tool": "properties",
                        "args": {"ids": ["missing"], "fill": "blue"},
                    },
                ],
            },
        )
    assert agent.session.editor.snapshot.revision == 0
    batch = stage(agent, seen)
    agent.call("edit_batch", {"seen": seen, "id": batch["id"], "action": "discard"})
    assert agent.session.editor.snapshot.revision == 0
    with pytest.raises(DocumentError, match="expired"):
        agent.call("edit_batch", {"seen": seen, "id": batch["id"], "action": "apply"})


def test_batch_alias_can_edit_new_path():
    agent, seen = fresh()
    reply = agent.call(
        "edit_batch",
        {
            "seen": seen,
            "edits": [
                {"tool": "add_path", "args": {"d": "M0 0 L10 0 L10 10 Z"}, "as": "new"},
                {"tool": "properties", "args": {"ids": ["$new"], "fill": "red"}},
            ],
        },
    ).data
    oid = reply["aliases"]["new"][0]
    agent.call("edit_batch", {"seen": seen, "id": reply["id"], "action": "apply"})
    assert agent.session.editor.snapshot.document.element(oid).get("fill") == "red"
