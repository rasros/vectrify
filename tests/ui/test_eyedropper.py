"""The Eyedropper: the session samples the artwork or the reference as MCP's
pick does."""

import base64
import io

import pytest
from PIL import Image

from tests.ui.helpers import send
from vectrify.document import DocumentError, import_svg
from vectrify.ui.agent import Agent
from vectrify.ui.server import Backend
from vectrify.ui.session import Session

SVG = """<svg width="100" height="100">
<rect id="red" x="0" y="0" width="50" height="50" fill="#ff0000"/>
<rect id="glass" x="50" y="0" width="50" height="50" fill="#0000ff" fill-opacity="0.5"/>
<rect id="green" x="0" y="50" width="50" height="50" fill="#00ff00"/>
</svg>"""


def reference_url() -> str:
    """Left half opaque green, right half half-transparent black."""
    image = Image.new("RGBA", (100, 100), (0, 128, 0, 255))
    image.paste((0, 0, 0, 128), (50, 0, 100, 100))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def sample(session, source, x, y, radius=0.0):
    return session.sample(
        {
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            "source": source,
            "at": [x, y],
            "radius": radius,
        }
    )


def channels(colour: str) -> tuple[int, int, int]:
    return tuple(int(colour[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def test_artwork_samples_the_colour_as_seen_on_white_paper():
    session = Session(import_svg(SVG))
    assert sample(session, "artwork", 25, 25)["colour"] == "#ff0000"
    # Translucent paint gives its blend with the paper under it.
    r, g, b = channels(sample(session, "artwork", 75, 25)["colour"])
    assert b == 255
    assert abs(r - 128) <= 1
    assert abs(g - 128) <= 1
    # Empty paper reads as white; an edge gives the average of both sides.
    assert sample(session, "artwork", 75, 75)["colour"] == "#ffffff"
    r, g, b = channels(sample(session, "artwork", 25, 75, radius=60)["colour"])
    assert 0 < r < 255
    assert 0 < b < 255
    result = sample(session, "artwork", 25, 25, radius=0.1)
    assert result["radius"] == 0.5
    assert result["at"] == [25, 25]
    assert result["revision"] == 0
    # Sampling edits nothing.
    assert session.state()["undo"] == []


def test_reference_samples_composited_on_white_and_only_on_the_artboard():
    session = Session(import_svg(SVG))
    with pytest.raises(DocumentError, match="No reference is loaded"):
        sample(session, "reference", 25, 25)
    send(session, "reference", reference={"name": "r.png", "data_url": reference_url()})
    assert sample(session, "reference", 25, 25)["colour"] == "#008000"
    # Half-transparent black over white is mid grey, never black.
    r, g, b = channels(sample(session, "reference", 75, 25)["colour"])
    assert r == g == b
    assert abs(r - 127) <= 1
    with pytest.raises(DocumentError, match="covers only the artboard"):
        sample(session, "reference", 150, 25)
    with pytest.raises(DocumentError, match="artwork or the reference"):
        sample(session, "drawing", 25, 25)
    with pytest.raises(DocumentError, match=r"\[x, y\]"):
        session.sample({"epoch": session.epoch, "revision": 0, "source": "artwork"})


def test_mcp_pick_reports_the_eyedroppers_colours():
    session = Session(import_svg(SVG))
    send(session, "reference", reference={"name": "r.png", "data_url": reference_url()})
    agent = Agent(session)
    seen = [session.epoch, session.editor.snapshot.revision]
    for x, y, radius in [(25, 25, 0), (75, 25, 3), (50, 50, 10)]:
        colour = agent.call("pick", {"seen": seen, "x": x, "y": y, "radius": radius})
        colour = colour.data["colour"]
        assert colour["drawing"] == sample(session, "artwork", x, y, radius)["colour"]
        assert (
            colour["reference"] == sample(session, "reference", x, y, radius)["colour"]
        )


def test_a_sample_applies_as_one_undoable_paint_edit():
    session = Session(import_svg(SVG))
    send(session, "select", objects=["red", "glass"])
    colour = sample(session, "artwork", 25, 75)["colour"]
    state = send(session, "paint", changes={"stroke": colour})
    document = session.editor.snapshot.document
    assert [document.element(i).get("stroke") for i in ("red", "glass")] == [
        "#00ff00",
        "#00ff00",
    ]
    assert state["undo"] == ["Change paint"]
    send(session, "undo")
    document = session.editor.snapshot.document
    assert document.element("red").get("stroke") is None


def test_backend_answers_samples_and_refuses_another_drawings():
    backend = Backend(SVG, "eyedropper.svg")
    status, state = backend.handle("/api/session", {}, None)
    assert status == 200
    request = {
        "epoch": state["epoch"],
        "revision": state["revision"],
        "source": "artwork",
        "at": [25, 25],
        "radius": 2,
    }
    status, result = backend.handle("/api/sample", request, state["session"])
    assert (status, result["colour"], result["radius"]) == (200, "#ff0000", 2)
    status, _ = backend.handle(
        "/api/sample", {**request, "epoch": "another drawing"}, state["session"]
    )
    assert status == 409
    status, result = backend.handle(
        "/api/sample", {**request, "source": "reference"}, state["session"]
    )
    assert status == 400
    assert "No reference is loaded" in result["error"]
