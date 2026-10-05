"""Handle edits preserve anchors, neighbouring controls and undo history."""

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.ui.test_session import send
from vectrify.document import EditRejectedError, import_svg, load_project, save_project
from vectrify.ui.session import Session

CURVE = "M0 0 C3 1 8 4 10 0 C14 2 17 9 20 10"


def session_for(path=CURVE):
    session = Session(import_svg(f'<svg><path id="p" d="{path}"/></svg>'))
    send(session, "select", objects=["p"])
    return session


def nodes(session):
    return session.editor.snapshot.document.geometry_for("p").subpaths[0].nodes


@pytest.mark.parametrize("offset", [0, 2])
def test_delete_one_handle_preserves_every_other_coordinate_and_undo(offset):
    session = session_for()
    before = nodes(session)
    send(session, "select", objects=["p"], nodes=[before[1].id])
    send(session, "delete_handle", object="p", node=before[1].id, offset=offset)
    after = nodes(session)
    expected = list(before[1].values)
    expected[offset : offset + 2] = before[0 if offset == 0 else 1].endpoint
    assert after[1].values == tuple(expected)
    assert after[0] == before[0]
    assert after[2] == before[2]
    assert session.state()["undo"] == ["Delete handle"]
    assert session.editor.snapshot.selection.node_ids == frozenset({before[1].id})
    send(session, "undo")
    assert nodes(session) == before
    send(session, "redo")
    assert nodes(session) == after


def test_delete_keeps_a_neighbours_visible_handle_even_on_the_chord():
    session = session_for("M0 0 C3 0 8 4 10 0")
    node = nodes(session)[1]
    send(session, "delete_handle", object="p", node=node.id, offset=2)
    assert nodes(session)[1].values == (3, 0, 10, 0, 10, 0)
    send(session, "delete_handle", object="p", node=node.id, offset=0)
    assert nodes(session)[1].command == "L"
    assert nodes(session)[1].id == node.id


@pytest.mark.parametrize("side", [None, 0, 2])
@pytest.mark.parametrize("closed", [False, True])
def test_straighten_preserves_lengths_endpoints_and_other_controls(side, closed):
    session = session_for("M0 0 C3 1 8 4 10 0 C14 2 17 9 0 0 Z" if closed else CURVE)
    before = nodes(session)
    target = before[0] if closed else before[1]
    incoming, outgoing = (2, 1) if closed else (1, 2)
    send(session, "straighten_handles", points=[["p", target.id]], side=side)
    after = nodes(session)
    point = target.endpoint
    hin, hout = after[incoming].values[2:4], after[outgoing].values[:2]
    vin = (hin[0] - point[0], hin[1] - point[1])
    vout = (hout[0] - point[0], hout[1] - point[1])
    assert vin[0] * vout[1] - vin[1] * vout[0] == pytest.approx(0, abs=1e-10)
    assert vin[0] * vout[0] + vin[1] * vout[1] < 0
    assert math.dist(point, hin) == pytest.approx(
        math.dist(point, before[incoming].values[2:4])
    )
    assert math.dist(point, hout) == pytest.approx(
        math.dist(point, before[outgoing].values[:2])
    )
    assert [(n.id, n.endpoint) for n in after] == [(n.id, n.endpoint) for n in before]
    assert after[incoming].values[:2] == before[incoming].values[:2]
    assert after[outgoing].values[2:4] == before[outgoing].values[2:4]
    if side == 0:
        assert hout == before[outgoing].values[:2]
    if side == 2:
        assert hin == before[incoming].values[2:4]
    send(session, "undo")
    assert nodes(session) == before
    send(session, "redo")
    assert nodes(session) == after


def test_straighten_does_not_create_retracted_handles_or_nan():
    session = session_for("M0 0 C3 1 10 0 10 0 C14 2 17 9 20 10")
    before = nodes(session)
    send(session, "straighten_handles", points=[["p", before[1].id]])
    assert [n.values for n in nodes(session)] == [n.values for n in before]
    assert nodes(session)[1].handles_aligned
    session = session_for("M0 0 C3 1 14 2 10 0 C14 2 17 9 20 10")
    target = nodes(session)[1]
    send(session, "straighten_handles", points=[["p", target.id]])
    assert all(math.isfinite(v) for n in nodes(session) for v in n.values)
    assert nodes(session)[1].values[2:4] == (6, -2)


@pytest.mark.parametrize("shared", [False, True])
def test_straighten_multiple_paths_is_one_edit_and_undo_restores_all(shared):
    session = Session(
        import_svg(
            f'<svg><path id="p" d="{CURVE}"/>'
            f'<path id="q" d="{CURVE}" transform="translate(30 0)"/></svg>'
        )
    )
    send(session, "select", objects=["p", "q"])
    if shared:
        with session.editor.transaction("Share") as tx:
            tx.share_geometry("q", "p")
    before = session.editor.snapshot.document
    points = [
        [oid, before.geometry_for(oid).subpaths[0].nodes[1].id] for oid in ("p", "q")
    ]
    send(session, "straighten_handles", points=points)
    after = session.editor.snapshot.document
    assert after.geometry_for("p") != before.geometry_for("p")
    assert after.geometry_for("q") != before.geometry_for("q")
    assert session.state()["undo"][-1] == "Toggle straight handles"
    send(session, "undo")
    assert session.editor.snapshot.document == before


@pytest.mark.parametrize("position", [0, -1])
def test_delete_handle_at_closed_seam_keeps_the_opposite_side(position):
    session = session_for("M0 0 C3 1 8 4 10 0 C14 2 -2 -4 0 0 Z")
    before = nodes(session)
    # Incoming belongs to the closing segment; outgoing to the first segment.
    segment, offset = (before[-1], 2) if position == -1 else (before[1], 0)
    send(session, "delete_handle", object="p", node=segment.id, offset=offset)
    after = nodes(session)
    assert [(n.id, n.endpoint) for n in after] == [(n.id, n.endpoint) for n in before]
    if offset == 2:
        assert after[-1].values[2:4] == (0, 0)
        assert after[1] == before[1]
    else:
        assert after[1].values[:2] == (0, 0)
        assert after[-1] == before[-1]


@pytest.mark.parametrize("command", ["delete_handle", "straighten_handles"])
def test_handle_edits_respect_geometry_locks_but_can_edit_pinned_anchors(command):
    session = session_for()
    node = nodes(session)[1]
    send(session, "pin", object="p", node=node.id, pinned=True)
    payload = {"object": "p", "node": node.id, "offset": 2}
    send(session, command, **payload)
    assert nodes(session)[1].endpoint == node.endpoint
    send(session, "locks", object="p", locks=["geometry"])
    before = nodes(session)
    with pytest.raises(EditRejectedError, match="locked"):
        send(session, command, **payload, side=0)
    assert nodes(session) == before


@pytest.mark.parametrize("offset", [0, 2])
def test_alignment_toggle_controls_later_drags_and_undo(offset):
    session = session_for()
    target = nodes(session)[1]
    send(session, "straighten_handles", points=[["p", target.id]], aligned=True)
    before = nodes(session)
    segment = 2 if offset == 0 else 1
    values = list(before[segment].values)
    values[offset : offset + 2] = (10, 8)
    send(
        session,
        "node",
        object="p",
        node=before[segment].id,
        values=values,
        handle_offset=offset,
    )
    after = nodes(session)
    opposite = after[1].values[2:4] if offset == 0 else after[2].values[:2]
    old_opposite = before[1].values[2:4] if offset == 0 else before[2].values[:2]
    assert opposite[0] == pytest.approx(10)
    assert opposite[1] == pytest.approx(-math.dist((10, 0), old_opposite))
    assert after[segment].values[offset : offset + 2] == (10, 8)
    send(session, "undo")
    assert nodes(session) == before
    send(session, "redo")
    assert nodes(session) == after
    send(session, "straighten_handles", points=[["p", target.id]], aligned=False)
    disabled = nodes(session)
    assert not disabled[1].handles_aligned
    assert [n.values for n in disabled] == [n.values for n in after]
    values[offset : offset + 2] = (18, 0)
    send(
        session,
        "node",
        object="p",
        node=disabled[segment].id,
        values=values,
        handle_offset=offset,
    )
    freed = nodes(session)
    opposite = freed[1].values[2:4] if offset == 0 else freed[2].values[:2]
    assert opposite == (after[1].values[2:4] if offset == 0 else after[2].values[:2])


def test_alignment_is_saved_and_old_projects_default_to_independent_handles():
    session = session_for()
    target = nodes(session)[1]
    send(session, "straighten_handles", points=[["p", target.id]], aligned=True)
    source = save_project(session.editor.snapshot.document)
    document, _ = load_project(source)
    assert document.geometry_for("p").node(target.id).handles_aligned
    data = json.loads(source)
    for geometry in data["geometries"]:
        for subpath in geometry["subpaths"]:
            for node in subpath["nodes"]:
                node.pop("handles_aligned")
    document, _ = load_project(json.dumps(data))
    assert not document.geometry_for("p").node(target.id).handles_aligned


def test_closed_seam_toggle_applies_to_both_representations_and_recreated_handles():
    session = session_for("M0 0 C3 1 8 4 10 0 C14 2 -2 -4 0 0 Z")
    first, _, last = nodes(session)
    send(session, "straighten_handles", points=[["p", last.id]], aligned=True)
    assert nodes(session)[0].handles_aligned
    assert nodes(session)[-1].handles_aligned
    send(session, "node_handles", points=[["p", first.id]], count=0)
    send(session, "node_handles", points=[["p", last.id]], count=2)
    assert nodes(session)[0].handles_aligned
    assert nodes(session)[-1].handles_aligned
    send(session, "straighten_handles", points=[["p", first.id]], aligned=False)
    assert not nodes(session)[0].handles_aligned
    assert not nodes(session)[-1].handles_aligned


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_handle_selection_and_commands():
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("handles.mjs"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
