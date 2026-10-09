from __future__ import annotations

import time

import pytest

from tests.mcp.test_agent import fresh
from vectrify.document import DocumentError
from vectrify.document.svg import parse_path
from vectrify.ui.agent_look import protect_trace


def test_corner_protection_is_distinct_from_pin_and_roundtrips():
    agent, seen = fresh()
    doc = agent.session.editor.snapshot.document
    node = doc.geometry_for("hill").subpaths[0].nodes[1]
    agent.call(
        "protect_features", {"seen": seen, "points": [["hill", node.id]], "kind": "tip"}
    )
    doc = agent.session.editor.snapshot.document
    assert doc.geometry_for("hill").node(node.id).feature == "tip"
    assert not doc.geometry_for("hill").node(node.id).pinned
    with pytest.raises(DocumentError, match="Protected tip"):
        agent.call(
            "set_points",
            {"seen": [seen[0], 1], "changes": {"hill": {node.id: [130, 60]}}},
        )
    assert agent.session.editor.snapshot.document == doc
    project = agent.session.project()
    agent.session.open(project, "protected.vectrify")
    assert (
        agent.session.editor.snapshot.document.geometry_for("hill")
        .node(node.id)
        .feature
        == "tip"
    )


def test_tidy_keeps_tip_and_simplifies_unprotected_nodes():
    from vectrify.document import import_svg
    from vectrify.ui.agent import Agent
    from vectrify.ui.session import Session

    drawing = (
        '<svg width="100" height="100"><path id="grass" fill="green" '
        'd="M0 90 L5 90 L10 90 L15 10 L20 90 L25 90 L30 90 '
        'L40 5 L50 90 L55 90 L60 90 L60 100 L0 100 Z"/></svg>'
    )
    agent = Agent(Session(import_svg(drawing)))
    seen = [agent.session.epoch, 0]
    doc = agent.session.editor.snapshot.document
    tip = doc.geometry_for("grass").subpaths[0].nodes[3]
    agent.call(
        "protect_features", {"seen": seen, "points": [["grass", tip.id]], "kind": "tip"}
    )
    job = agent.call(
        "tidy",
        {
            "seen": [seen[0], 1],
            "ids": ["grass"],
            "settings": {
                "shape": False,
                "snap": False,
                "simplify": True,
                "seconds": 1.0,
            },
        },
    ).data
    deadline = time.monotonic() + 10
    while job["status"] == "running" and time.monotonic() < deadline:
        job = agent.call("job", {"id": job["id"], "wait_seconds": 1}).data
    assert job["status"] == "ready", job
    assert not job["result"]["diagnostics"]["effects"]["protected_feature_violations"]
    assert (
        job["result"]["metrics"]["after"]["nodes"]
        < job["result"]["metrics"]["before"]["nodes"]
    )
    agent.call("job", {"seen": [seen[0], 1], "id": job["id"], "action": "apply"})
    kept = agent.session.editor.snapshot.document.geometry_for("grass").node(tip.id)
    assert kept.endpoint == tip.endpoint
    assert kept.command == "L"


def test_trace_preserves_explicit_tip_and_refuses_unsatisfied_point():
    shapes = [{"d": "M0 10 C1 5 4 1 5 1 C6 1 9 5 10 10 Z"}]
    diagnostics = protect_trace(shapes, [{"x": 5, "y": 0, "kind": "tip"}], 1)
    assert diagnostics[0]["preserved"]
    nodes = parse_path(shapes[0]["d"]).subpaths[0].nodes
    assert nodes[1].endpoint == (5, 0)
    assert nodes[1].command == nodes[2].command == "L"
    with pytest.raises(DocumentError, match="no traced vertex"):
        protect_trace(shapes, [{"x": 100, "y": 100}], 1)


def test_redraw_across_protected_corner_is_refused_without_live_change():
    agent, seen = fresh()
    node = (
        agent.session.editor.snapshot.document.geometry_for("hill").subpaths[0].nodes[1]
    )
    agent.call(
        "protect_features",
        {"seen": seen, "points": [["hill", node.id]], "kind": "corner"},
    )
    before = agent.session.editor.snapshot.document
    with pytest.raises(DocumentError, match="Protected corner"):
        agent.call(
            "redraw_outline",
            {
                "seen": [seen[0], 1],
                "id": "hill",
                "points": [[0, 150], [120, 70], [260, 140]],
            },
        )
    assert agent.session.editor.snapshot.document == before
    assert agent.session.editor.snapshot.revision == 1
