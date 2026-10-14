"""Align and Distribute: each object moves by its own offset, as one edit."""

import pytest

from vectrify.document import DocumentError, import_svg
from vectrify.document.hit_test import HitIndex
from vectrify.ui.agent import LEFT_OUT
from vectrify.ui.session import ACTION_COMMANDS, Session

# Unstroked shapes, so their painted bounds are their geometric bounds: one
# inside a rotated, scaled group, a group of two, and two on their own.
SVG = """<svg width="200" height="200">
<g id="turned" transform="translate(100 20) rotate(90) scale(2)">
<rect id="inner" width="10" height="5"/>
</g>
<g id="pair"><rect id="left" x="10" y="100" width="10" height="10"/>
<rect id="right" x="30" y="100" width="10" height="20"/></g>
<rect id="wide" x="60" y="150" width="40" height="10"/>
<rect id="key" x="120" y="60" width="20" height="30"/>
</svg>"""


def send(session, command, **data):
    return session.action(
        {
            "command": command,
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            **data,
        }
    )


def box(session, oid):
    found = HitIndex(session.editor.snapshot.document).bounds(frozenset({oid}))
    assert found is not None
    return found


def test_each_object_moves_by_its_own_document_offset_in_one_edit():
    session = Session(import_svg(SVG))
    before = {oid: box(session, oid) for oid in ("inner", "pair", "wide", "key")}
    send(session, "select", objects=["inner", "pair", "wide", "key"])
    # Align left to the key object: each goes as far as its left edge is from
    # the key's, measured on the page; the key has no move and stays.
    goal = before["key"][0]
    moves = {oid: [goal - before[oid][0], 0] for oid in ("inner", "pair", "wide")}
    result = send(session, "align", moves=moves)
    assert result["undo"] == ["Align objects"]
    for oid in ("inner", "pair", "wide"):
        left, top, right, bottom = before[oid]
        assert box(session, oid) == pytest.approx(
            (goal, top, goal + right - left, bottom)
        )
    assert box(session, "key") == before["key"]
    document = session.editor.snapshot.document
    assert document.element("key").get("transform") is None
    # The child of the rotated group takes its move in the group's frame, and
    # a selected group moves its children once, as a whole.
    assert (document.element("inner").get("transform") or "").startswith("translate(")
    assert document.element("left").get("transform") is None
    send(session, "undo")
    for oid, found in before.items():
        assert box(session, oid) == pytest.approx(found)


def test_distribute_moves_without_resizing_under_its_own_label():
    session = Session(import_svg(SVG))
    sizes = {
        oid: (b[2] - b[0], b[3] - b[1])
        for oid in ("pair", "wide", "key")
        for b in [box(session, oid)]
    }
    send(session, "select", objects=["pair", "wide", "key"])
    result = send(session, "distribute", moves={"wide": [-5, 0]})
    assert result["undo"] == ["Distribute objects"]
    for oid, size in sizes.items():
        b = box(session, oid)
        assert (b[2] - b[0], b[3] - b[1]) == pytest.approx(size)


def test_a_locked_position_refuses_and_says_which_object():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["right"])
    send(session, "locks", object="right", locks=["transform"])
    send(session, "select", objects=["key"])
    send(session, "locks", object="key", locks=["transform"])
    send(session, "select", objects=["pair", "wide", "key"])
    before = session.editor.snapshot.document
    # The pair's child is locked: moving the pair would move it.
    with pytest.raises(DocumentError, match="right: position is locked"):
        send(session, "align", moves={"pair": [1, 0], "wide": [2, 0]})
    with pytest.raises(DocumentError, match="key: position is locked"):
        send(session, "align", moves={"key": [0, 3]})
    assert session.editor.snapshot.document == before
    # A locked object that stays, such as a key object, does not refuse.
    send(session, "align", moves={"wide": [0, 4]})
    assert session.editor.undo_labels[-1] == "Align objects"


def test_moves_name_selected_objects_on_their_own():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["pair", "left", "wide"])
    with pytest.raises(DocumentError, match="selected objects"):
        send(session, "align", moves={"key": [1, 0]})
    with pytest.raises(DocumentError, match="moves with its selected group"):
        send(session, "align", moves={"left": [1, 0]})
    with pytest.raises(DocumentError, match="already in place"):
        send(session, "align", moves={})
    with pytest.raises(DocumentError, match="finite"):
        send(session, "align", moves={"wide": [float("inf"), 0]})


def test_align_and_distribute_are_left_out_of_mcp_with_a_reason():
    assert {"align", "distribute"} <= ACTION_COMMANDS.keys()
    assert {"align", "distribute"} <= LEFT_OUT.keys()
