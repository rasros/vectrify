"""The agent's calls on a session, below the MCP layer."""

from __future__ import annotations

from collections.abc import Sequence

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
    reply = agent.call("properties", {"seen": seen, "ids": ["sun"], "fill": "red"})
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
        "transform", {"seen": seen, "ids": ["sun"], "box": [0, 0, 20, 20]}
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
        "properties",
        {"seen": [where["epoch"], where["revision"]], "ids": ["sun"], "fill": "red"},
    )


def test_edits_need_the_revision_last_seen():
    agent, seen = fresh()
    with pytest.raises(StaleRevisionError, match="describe"):
        agent.call("properties", {"ids": ["sun"], "fill": "red"})
    agent.call("properties", {"seen": seen, "ids": ["sun"], "fill": "red"})
    with pytest.raises(StaleRevisionError, match="changed since you last looked"):
        agent.call("properties", {"seen": seen, "ids": ["sun"], "fill": "blue"})


def test_get_svg_of_some_objects():
    agent, _ = fresh()
    elements = agent.call("get_svg", {"ids": ["sun"]}).data["elements"]
    assert elements["sun"].startswith("<")
    assert 'id="sun"' in elements["sun"]
    assert "#f1ba77" in elements["sun"]


def nodes(agent: Agent, oid: str) -> list[str]:
    contours = agent.call("points", {"id": oid}).data["contours"]
    return [n["id"] for c in contours for n in c["nodes"]]


def person_selects(agent: Agent, objects: list[str], points: Sequence[str] = ()):
    session = agent.session
    session.action(
        {
            "command": "select",
            "objects": objects,
            "nodes": list(points),
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
        }
    )


def test_agent_edits_leave_the_persons_selection_and_add_one_step_each():
    agent, seen = fresh()
    hill, sun = nodes(agent, "hill"), nodes(agent, "sun")
    person_selects(agent, ["hill"], hill[:2])
    editor = agent.session.editor
    chosen = editor.snapshot.selection
    edits = [
        ("properties", {"ids": ["sun"], "fill": "red"}),
        ("transform", {"ids": ["sun"], "dx": 3, "dy": 0}),
        ("transform", {"ids": ["sun"], "scale": [2, 2]}),
        ("arrange", {"ids": ["sun"], "to": "back"}),
        ("set_points", {"changes": {"sun": {sun[0]: [301, 21]}}}),
        ("point_style", {"points": [["sun", sun[1]]], "handles": 2}),
        ("group", {"ids": ["sun", "sky"]}),
    ]
    reply: dict = {}
    for step, (tool, args) in enumerate(edits, 1):
        revision = editor.snapshot.revision
        reply = agent.call(tool, {"seen": seen, **args}).data
        seen = [reply["epoch"], reply["revision"]]
        assert editor.snapshot.selection == chosen, tool
        # One revision and one undo step each: giving the selection back is
        # neither.
        assert reply["revision"] == revision + 1, tool
        assert len(editor.undo_entries) == step, tool
        assert agent.touched[-1]["change"] == agent.changes
    # What the edit left to work on is the agent's, in its answer: the group.
    (group,) = reply["result"]["objects"]
    assert group in reply["created"]
    # Undoing an agent's step keeps the person's selection too.
    agent.call("undo", {"seen": seen})
    assert editor.snapshot.selection == chosen
    person_selects(agent, [])
    editor.undo()
    assert editor.snapshot.selection == chosen


def test_a_refused_edit_leaves_the_persons_selection():
    agent, seen = fresh()
    person_selects(agent, ["hill"])
    with pytest.raises(RefusedError):
        agent.call(
            "add_path",
            {"seen": seen, "d": "M0 0 L10 0 L10 10 Z", "parent": "no-such-group"},
        )
    assert agent.session.editor.snapshot.selection.object_ids == {"hill"}
    assert not agent.touched


def test_deleting_what_the_person_selected_drops_it_from_their_selection():
    agent, seen = fresh()
    hill, sun = nodes(agent, "hill"), nodes(agent, "sun")
    person_selects(agent, ["hill", "sun"], [hill[0], hill[1], sun[0]])
    reply = agent.call("delete", {"seen": seen, "ids": ["sun"]}).data
    selection = agent.session.editor.snapshot.selection
    assert selection.object_ids == {"hill"}
    assert selection.node_ids == {hill[0], hill[1]}
    assert reply["removed"] == ["sun"]
    seen = [reply["epoch"], reply["revision"]]
    agent.call("delete", {"seen": seen, "points": [["hill", hill[0]]]})
    selection = agent.session.editor.snapshot.selection
    assert selection.object_ids == {"hill"}
    assert selection.node_ids == {hill[1]}


def test_edits_without_targets_are_refused():
    agent, seen = fresh()
    person_selects(agent, ["sun"])
    for tool, args in [
        ("properties", {"fill": "red"}),
        ("properties", {"ids": [], "fill": "red"}),
        ("delete", {}),
        ("tidy", {}),
    ]:
        with pytest.raises(agent_module.DocumentError, match="ids"):
            agent.call(tool, {"seen": seen, **args})
    assert agent.session.editor.undo_labels == ()


def test_touched_names_the_changed_objects():
    agent, seen = fresh()
    reply = agent.call(
        "transform", {"seen": seen, "ids": ["sun", "hill"], "dx": 1, "dy": 0}
    )
    assert agent.touched[-1] == {"change": 1, "ids": ["hill", "sun"]}
    agent.call("undo", {"seen": [reply.data["epoch"], reply.data["revision"]]})
    assert agent.touched[-1] == {"change": 2, "ids": ["hill", "sun"]}
    agent.call("describe")
    assert len(agent.touched) == 2


SHAPES = """<svg xmlns="http://www.w3.org/2000/svg" \
xmlns:xlink="http://www.w3.org/1999/xlink" width="200" height="200" \
viewBox="0 0 200 200">
<defs><path id="shape" d="M0 0 L10 0 L10 10 Z"/></defs>
<path id="a" d="M10 10 L60 10 L60 60 L10 60 Z" fill="red"/>
<path id="b" d="M40 40 L90 40 L90 90 L40 90 Z" fill="blue"/>
<path id="ring" d="M100 100 L190 100 L190 190 L100 190 Z \
M120 120 L120 170 L170 170 L170 120 Z" fill="green"/>
<path id="line" d="M10 150 L50 150 L90 160" fill="none" stroke="black"/>
<path id="line2" d="M93 160 L120 150" fill="none" stroke="black"/>
<g id="grp"><path id="c" d="M150 10 L190 10 L190 50 Z" fill="#888"/></g>
<use id="inst" xlink:href="#shape" x="5" y="180"/>
</svg>
"""


class Calls:
    """An agent on SHAPES that sends the revision it last saw."""

    def __init__(self):
        self.agent = Agent(Session(import_svg(SHAPES)))
        hello = self.agent.call("hello").data
        self.seen = [hello["epoch"], hello["revision"]]

    def __call__(self, tool: str, **args) -> dict:
        data = self.agent.call(tool, {"seen": self.seen, **args}).data
        if "revision" in data:
            self.seen = [data["epoch"], data["revision"]]
        return data

    @property
    def labels(self) -> tuple[str, ...]:
        return self.agent.session.editor.undo_labels

    def element(self, oid: str):
        return self.agent.session.editor.snapshot.document.element(oid)


def test_properties_set_paint_name_and_locks_in_one_step():
    call = Calls()
    reply = call(
        "properties", ids=["a"], fill="#00ff00", name="Square", locks=["stroke"]
    )
    assert reply["step"] == "Agent: Properties"
    assert call.labels == ("Agent: Properties",)
    element = call.element("a")
    assert (element.get("fill"), element.name, set(element.locks)) == (
        "#00ff00",
        "Square",
        {"stroke"},
    )
    # Unlocking comes first, so a lock lifted in the same call does not
    # refuse the change.
    call("properties", ids=["a"], locks=["paint"])
    with pytest.raises(agent_module.DocumentError):
        call("properties", ids=["a"], fill="#0000ff")
    call("properties", ids=["a"], fill="#0000ff", locks=[])
    assert call.element("a").get("fill") == "#0000ff"
    with pytest.raises(agent_module.DocumentError, match="one id"):
        call("properties", ids=["a", "b"], name="Both")


def test_transform_moves_and_scales_in_one_step():
    call = Calls()
    reply = call("transform", ids=["a"], dx=10, dy=0, scale=[2, 2], anchor="top-left")
    assert reply["step"] == "Agent: Transform"
    bounds = next(
        o["bounds"]
        for o in call.agent.call("describe").data["objects"]
        if o["id"] == "a"
    )
    assert bounds == pytest.approx([20, 10, 100, 100], abs=1e-6)
    with pytest.raises(agent_module.DocumentError, match="box alone"):
        call("transform", ids=["a"], dx=1, box=[0, 0, 5, 5])


def test_arrange_restacks_or_moves_into_a_group_in_front():
    call = Calls()
    call("arrange", ids=["a"], to="front")
    call("arrange", ids=["b"], parent="grp")
    assert [c.id for c in call.element("grp").children] == ["c", "b"]
    with pytest.raises(agent_module.DocumentError, match="parent"):
        call("arrange", ids=["a"])


def test_join_does_what_the_editors_join_does():
    call = Calls()
    with pytest.raises(agent_module.DocumentError, match="color_source"):
        call("join", ids=["line", "line2"], color_source="line")
    ends = call("join", ids=["line", "line2"], reach=10)
    assert ends["joined"] == "line ends"
    outlines = call("join", ids=["a", "b"], color_source="a")
    assert outlines["joined"] == "outlines"
    nodes = [
        n["id"]
        for c in call.agent.call("points", {"id": "ring"}).data["contours"]
        for n in c["nodes"]
    ]
    points = call("join", points=[["ring", nodes[0]], ["ring", nodes[5]]])
    assert points["joined"] == "points"
    assert len(call.labels) == 3


def test_points_marks_holes_that_holes_fills_or_makes_shapes():
    call = Calls()
    listed = call.agent.call("points", {"id": "ring", "nodes": False}).data
    assert [c["hole"] for c in listed["contours"]] == [False, True]
    assert listed["contours"][1]["area"] == pytest.approx(2500)
    assert listed["holes_total"] == 1
    hole = listed["contours"][1]["id"]
    made = call("holes", contours=[["ring", hole]], action="shape")
    assert made["created"]
    call("undo")
    call("holes", contours=[["ring", hole]])
    after = call.agent.call("points", {"id": "ring", "nodes": False}).data
    assert after["contours_total"] == 1


def test_points_tools_style_break_and_delete():
    call = Calls()
    line = [
        n["id"]
        for c in call.agent.call("points", {"id": "line"}).data["contours"]
        for n in c["nodes"]
    ]
    styled = call("point_style", points=[["line", line[1]]], handles=2, pinned=True)
    assert styled["step"] == "Agent: Point style"
    # A segment's two ends delete that segment; one point breaks there.
    broke = call("break_points", points=[["line", line[1]], ["line", line[2]]])
    assert broke["broke"] == "segment deleted"
    ring = [
        n["id"]
        for c in call.agent.call("points", {"id": "ring"}).data["contours"]
        for n in c["nodes"]
    ]
    call("delete", points=[["ring", ring[-1]]], contours=True)
    assert call.agent.call("points", {"id": "ring"}).data["contours_total"] == 1
    with pytest.raises(agent_module.DocumentError, match="not both"):
        call("delete", ids=["a"], points=[["ring", ring[0]]])
    with pytest.raises(agent_module.DocumentError, match="contours goes with"):
        call("delete", ids=["a"], contours=True)


def test_convert_to_path_detaches_an_instance():
    call = Calls()
    reply = call("convert", ids=["inst"], to="path")
    assert reply["step"] == "Agent: Detach geometry"
    with pytest.raises(agent_module.DocumentError, match="action is"):
        call("job", id="nope", action="keep")
