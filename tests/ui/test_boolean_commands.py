"""Union, Subtract, Intersect and Exclude from the editor and from agents."""

import pytest

from vectrify.document import DocumentError, import_svg
from vectrify.document.holes import filled_region
from vectrify.document.join import path_style
from vectrify.ui.agent import Agent
from vectrify.ui.session import Session

# A blue square at the back, a red one overlapping it in front, and an
# unselected path between them.
SVG = """<svg width="200" height="200">
<path id="back" fill="blue" d="M0 0H100V100H0Z"/>
<path id="between" fill="green" d="M150 150H190V190H150Z"/>
<path id="front" fill="red" d="M50 50H150V150H50Z"/>
<rect id="box" x="0" y="0" width="10" height="10"/>
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


def area(session, object_id):
    document = session.editor.snapshot.document
    style = path_style(document, document.element(object_id))
    return filled_region(document.geometry_for(object_id), style["fill-rule"]).area


@pytest.mark.parametrize(
    ("command", "label", "expected"),
    [
        ("union", "Union", 17500),
        ("subtract", "Subtract", 7500),
        ("intersect", "Intersect", 2500),
        ("exclude", "Exclude", 15000),
    ],
)
def test_each_command_is_one_undo_step_keeping_the_back_path(command, label, expected):
    session = Session(import_svg(SVG))
    send(session, "select", objects=["front", "back"])
    result = send(session, command)
    assert result["undo"] == [label]
    assert result["selection"]["objects"] == ["back"]
    paths = [o for o in result["objects"] if o["tag"] == "path"]
    assert [o["id"] for o in paths] == ["back", "between"]
    assert paths[0]["attributes"]["fill"] == "blue"
    assert area(session, "back") == pytest.approx(expected)
    undone = send(session, "undo")
    assert [o["id"] for o in undone["objects"] if o["tag"] == "path"] == [
        "back",
        "between",
        "front",
    ]


def test_refusals_change_nothing():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["back", "box"])
    with pytest.raises(DocumentError, match="Convert to path"):
        send(session, "subtract")
    send(session, "select", objects=["back"])
    with pytest.raises(DocumentError, match="two or more"):
        send(session, "union")
    send(session, "select", objects=["back", "between"])
    with pytest.raises(DocumentError, match="Nothing to subtract"):
        send(session, "subtract")
    with pytest.raises(DocumentError, match="Intersect would leave nothing"):
        send(session, "intersect")
    assert session.editor.undo_labels == ()


def test_agents_get_the_same_operand_order_whatever_the_order_of_ids():
    agent = Agent(Session(import_svg(SVG)))
    reply = agent.call(
        "boolean",
        {
            "seen": [agent.session.epoch, 0],
            "ids": ["front", "back"],
            "operation": "subtract",
        },
    )
    assert reply.data["result"]["objects"] == ["back"]
    assert area(agent.session, "back") == pytest.approx(7500)
    with pytest.raises(DocumentError, match="operation is union"):
        agent.call(
            "boolean",
            {
                "seen": [agent.session.epoch, agent.session.editor.snapshot.revision],
                "ids": ["back", "between"],
                "operation": "divide",
            },
        )
