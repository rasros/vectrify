"""The agent's calls on a session, below the MCP layer."""

from __future__ import annotations

import pytest

from tests.mcp.helpers import SVG
from vectrify.document import StaleRevisionError, import_svg
from vectrify.ui import agent as agent_module
from vectrify.ui.agent import Agent, RefusedError
from vectrify.ui.session import Session


def fresh() -> tuple[Agent, list]:
    agent = Agent(Session(import_svg(SVG)))
    hello = agent.call("hello").data
    return agent, [hello["epoch"], hello["revision"]]


def test_renders_of_an_unchanged_revision_and_region_are_cached(monkeypatch):
    agent, seen = fresh()
    calls = []
    real = agent_module.render_document

    def counted(*args):
        calls.append(args[1:])
        return real(*args)

    monkeypatch.setattr(agent_module, "render_document", counted)
    agent.call("render", {"max_side": 100})
    agent.call("render", {"max_side": 100})
    assert len(calls) == 1
    agent.call("render", {"max_side": 100, "region": [0, 0, 100, 100]})
    assert len(calls) == 2
    reply = agent.call("paint", {"seen": seen, "ids": ["sun"], "fill": "red"})
    agent.call("render", {"max_side": 100})
    assert len(calls) == 3
    assert reply.data["revision"] == 1


def test_renders_are_capped_in_size():
    agent, _ = fresh()
    reply = agent.call("render", {"max_side": 100000})
    assert reply.data["pixels"] == [2048, 1024]
    with pytest.raises(agent_module.DocumentError, match="positive width"):
        agent.call("render", {"region": [0, 0, 0, 10]})


def test_describe_pages_and_lists_one_group():
    agent = Agent(
        Session(
            import_svg(
                '<svg width="10" height="10"><g id="g"><rect id="r" width="2" '
                'height="2"/><rect id="s" x="4" width="2" height="2"/></g>'
                '<rect id="t" width="1" height="1"/></svg>'
            )
        )
    )
    first = agent.call("describe", {"page_size": 2}).data
    assert [o["id"] for o in first["objects"]] == ["g", "r"]
    assert (first["pages"], first["total"]) == (2, 4)
    inside = agent.call("describe", {"within": "g"}).data
    assert [o["id"] for o in inside["objects"]] == ["r", "s"]
    assert inside["objects"][1]["bounds"] == [4, 0, 2, 2]


def test_resize_to_a_box_is_one_step():
    agent, seen = fresh()
    reply = agent.call(
        "resize", {"seen": seen, "ids": ["sun"], "box": [0, 0, 20, 20]}
    ).data
    assert reply["step"] == "Agent: Resize"
    bounds = next(
        o["bounds"] for o in agent.call("describe").data["objects"] if o["id"] == "sun"
    )
    assert bounds == pytest.approx([0, 0, 20, 20], abs=1e-6)
    assert agent.session.editor.undo_labels == ("Agent: Resize",)


def test_a_refused_step_takes_back_the_steps_before_it():
    agent, seen = fresh()
    with pytest.raises(RefusedError) as refused:
        agent.call(
            "add_path",
            {"seen": seen, "d": "M0 0 L10 0 L10 10 Z", "parent": "no-such-group"},
        )
    editor = agent.session.editor
    assert editor.undo_labels == editor.redo_labels == ()
    assert [o.id for o in editor.snapshot.document.elements()][1:] == [
        "sky",
        "hill",
        "sun",
    ]
    # Taking it back made a revision, which the refusal reports as seen.
    where = refused.value.where
    agent.call(
        "paint",
        {"seen": [where["epoch"], where["revision"]], "ids": ["sun"], "fill": "red"},
    )


def test_edits_need_the_revision_last_seen():
    agent, seen = fresh()
    with pytest.raises(StaleRevisionError, match="describe"):
        agent.call("paint", {"ids": ["sun"], "fill": "red"})
    agent.call("paint", {"seen": seen, "ids": ["sun"], "fill": "red"})
    with pytest.raises(StaleRevisionError, match="changed since you last looked"):
        agent.call("paint", {"seen": seen, "ids": ["sun"], "fill": "blue"})


def test_get_svg_of_some_objects():
    agent, _ = fresh()
    elements = agent.call("get_svg", {"ids": ["sun"]}).data["elements"]
    assert elements["sun"].startswith("<")
    assert 'id="sun"' in elements["sun"]
    assert "#f1ba77" in elements["sun"]
