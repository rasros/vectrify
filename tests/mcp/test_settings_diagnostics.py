from __future__ import annotations

import pytest

from tests.mcp.test_agent import fresh
from vectrify.document import DocumentError
from vectrify.operations.diagnostics import edit_diagnostics


def test_discovery_and_early_invalid_settings():
    agent, seen = fresh()
    schema = agent.call(
        "settings_schema", {"action": "improve", "method": "nodes"}
    ).data["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["resolution"]["minimum"] == 64
    for settings in (
        {"unknown": True},
        {"tolerance": False},
        {"detail": True, "snap": False},
    ):
        with pytest.raises(DocumentError):
            agent.call("tidy", {"seen": seen, "ids": ["hill"], "settings": settings})
        assert not agent.session.jobs
        assert agent.session.editor.snapshot.revision == 0


def test_diagnostics_identify_geometry_and_paint():
    agent, seen = fresh()
    before = agent.session.editor.snapshot.document
    nodes = before.geometry_for("hill").subpaths[0].nodes
    agent.call(
        "set_points", {"seen": seen, "changes": {"hill": {nodes[1].id: [130, 60]}}}
    )
    after = agent.session.editor.snapshot.document
    diagnostics = edit_diagnostics(before, after)
    assert diagnostics["effects"]["geometry_moved"] == [
        {
            "object": "hill",
            "contour": before.geometry_for("hill").subpaths[0].id,
            "node": nodes[1].id,
        }
    ]
    assert not diagnostics["effects"]["protected_feature_violations"]


def test_job_reports_effective_settings_and_diagnostics():
    agent, seen = fresh()
    state = agent.call("cleanup", {"seen": seen, "ids": ["hill"]}).data
    assert state["effective_settings"] == {}
    assert "effects" in state["result"]["diagnostics"]


def test_transforms_are_geometry_effects_and_nullable_settings_match_schema():
    from vectrify.operations.settings import method_settings, read_settings

    agent, seen = fresh()
    before = agent.session.editor.snapshot.document
    agent.call("transform", {"seen": seen, "ids": ["hill"], "dx": 2})
    diagnostics = edit_diagnostics(before, agent.session.editor.snapshot.document)
    assert diagnostics["effects"]["geometry_moved"]
    assert not diagnostics["effects"]["paint_changed"]
    assert (
        read_settings({"region": None}, method_settings("improve", "nodes"), "nodes")[
            "region"
        ]
        is None
    )
