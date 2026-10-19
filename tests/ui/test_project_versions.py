"""Persisted reference changes have a version independent of drawing edits."""

import json

import pytest

from tests.ui.helpers import SVG, png_url, send
from vectrify.document import DocumentError, Selection, import_svg
from vectrify.ui.agent import Agent
from vectrify.ui.server import Backend
from vectrify.ui.session import Session


def test_reference_replacement_metadata_and_removal_change_content_version():
    session = Session(import_svg(SVG))
    agent = Agent(session)
    seen = [session.epoch, 0]
    before = session.state()
    for index, colour in enumerate([(255, 0, 0), (0, 0, 255)], 1):
        reply = agent.call(
            "set_reference",
            {"seen": seen, "name": "same.png", "data_url": png_url(100, 100, colour)},
        )
        state = session.state()
        assert reply.data["changed"] is True
        assert state["revision"] == 0
        assert state["reference_revision"] == index
        assert state["project_version"] != before["project_version"]
        assert state["reference"] == {"name": "same.png", "opacity": 0.5}
        assert (
            json.loads(agent.call("export", {"project": True}).data["content"])[
                "reference"
            ]
            == session.reference
        )
        before = state
    reference = session.reference
    assert reference is not None
    send(session, "reference", reference={**reference, "opacity": 0.8})
    assert session.reference is not None
    send(session, "reference", reference={**session.reference, "name": "renamed.png"})
    assert session.reference_revision == 4
    unchanged = session.project_version
    send(session, "reference", reference=session.reference)
    assert session.project_version == unchanged
    agent.call("set_reference", {"seen": seen})
    assert session.reference_revision == 5
    assert session.state()["reference"] is None
    assert session.editor.snapshot.revision == 0


def test_project_version_excludes_selection_and_view_and_resets_on_open():
    session = Session(import_svg(SVG))
    version = session.project_version
    session.editor.select(Selection(frozenset({"a"})))
    session.set_view(
        {
            "region": [0, 0, 100, 100],
            "pixels": [100, 100],
            "zoom": 1,
            "reference_view": "drawing",
        }
    )
    assert session.project_version == version
    send(session, "paint", changes={"fill": "green"})
    assert session.project_version != version
    send(
        session, "reference", reference={"name": "r.png", "data_url": png_url(100, 100)}
    )
    epoch = session.epoch
    project = session.project()
    session.open(project, "opened.vectrify")
    assert session.epoch != epoch
    assert session.project_version == "0:0"
    assert session.reference is not None


def test_failed_agent_edit_restores_reference_and_its_version():
    session = Session(import_svg(SVG))
    agent = Agent(session)
    before = session.content_version()
    with pytest.raises(DocumentError, match="Unknown editor command"):
        agent._edit(
            [session.epoch, 0],
            [
                {
                    "command": "reference",
                    "reference": {"name": "r.png", "data_url": png_url(100, 100)},
                },
                {"command": "unknown"},
            ],
        )
    assert session.reference is None
    assert session.content_version() == before


def test_export_reports_live_version_and_exact_recovery_refuses_intervening_edit():
    backend = Backend(SVG)
    _, state = backend.handle("/api/session", {}, None)
    sid = state["session"]
    session = backend.sessions[sid]
    send(session, "select", objects=["a"])
    send(session, "paint", changes={"fill": "green"})
    send(
        session, "reference", reference={"name": "r.png", "data_url": png_url(100, 100)}
    )
    # The existing stale-revision export contract still exports live content.
    status, exported = backend.handle(
        "/api/export", {"epoch": state["epoch"], "revision": 0}, sid
    )
    assert status == 200
    assert exported["revision"] == 1
    assert exported["project_version"] == session.project_version
    assert 'fill="green"' in exported["content"]
    assert (
        backend.handle("/api/poll", {}, sid)[1]["project_version"]
        == session.project_version
    )
    status, reference = backend.handle("/api/reference", {}, sid)
    assert status == 200
    assert reference["reference_revision"] == session.reference_revision
    assert reference["reference"] == session.reference
    send(session, "reference", reference=None)
    status, refused = backend.handle("/api/export", {**exported, "project": True}, sid)
    assert status == 409
    assert "project changed" in refused["error"]
    status, project = backend.handle(
        "/api/export", {**session.content_version(), "project": True}, sid
    )
    assert status == 200
    assert json.loads(project["content"])["reference"] is None
    assert project["project_version"] == session.project_version
