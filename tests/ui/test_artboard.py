"""New drawings and artboard edits: the artwork and the reference stay put,
and the canvas, saves, exports and methods agree on the page."""

import base64
import io
import json

import pytest
from PIL import Image

from vectrify.document import DocumentError, import_svg
from vectrify.ui.agent import Agent
from vectrify.ui.session import Session

SVG = """<svg width="100" height="100" viewBox="0 0 100 100"><g id="layer">
<path id="a" fill="red" d="M10 10 L30 10 L30 30 Z"/>
<rect id="b" x="40" y="20" width="20" height="50" fill="blue"/>
</g></svg>"""


def send(session, command, **data):
    return session.action(
        {
            "command": command,
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            **data,
        }
    )


def halves() -> dict:
    """A 100 by 100 reference, red on the left and blue on the right."""
    image = Image.new("RGB", (100, 100), "red")
    image.paste((0, 0, 255), (50, 0, 100, 100))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    return {"name": "halves.png", "data_url": url, "opacity": 0.5}


def reference_image(source) -> Image.Image:
    """The reference a session or an agent sees, which must be loaded."""
    image = (
        source.reference_image()
        if isinstance(source, Session)
        else source._reference_image()
    )
    assert image is not None
    return image


def pixel(image: Image.Image, x: int, y: int) -> tuple:
    colour = image.getpixel((x, y))
    assert isinstance(colour, tuple)
    return colour


def placed(session: Session) -> dict:
    """The session's reference, which must be loaded."""
    assert session.reference is not None
    return session.reference


def test_artboard_edit_is_one_undoable_step_that_export_and_save_follow():
    session = Session(import_svg(SVG))
    before = session.editor.snapshot.document
    state = send(session, "artboard", box=[-10, 0, 200, 50])
    assert state["bounds"] == [-10, 0, 200, 50]
    assert state["undo"] == ["Change artboard"]
    assert 'viewBox="-10 0 200 50"' in state["svg"]
    assert 'width="200"' in state["svg"]
    assert 'height="50"' in state["svg"]
    # The artwork is untouched.
    after = session.editor.snapshot.document
    assert after.root.children == before.root.children
    restored = Session(import_svg(SVG))
    send(restored, "open", source=session.project(), name="saved.vectrify")
    assert restored.state()["bounds"] == [-10, 0, 200, 50]
    state = send(session, "undo")
    assert state["bounds"] == [0, 0, 100, 100]
    assert session.editor.snapshot.document == before


def test_fit_artboard_to_artwork_takes_the_painted_bounds():
    session = Session(import_svg(SVG))
    state = send(session, "artboard", fit="artwork")
    assert state["bounds"] == pytest.approx([10, 10, 50, 60])
    assert state["undo"] == ["Fit artboard to artwork"]


def test_fit_artboard_refuses_an_empty_drawing_and_bad_boxes():
    session = Session(import_svg('<svg viewBox="0 0 10 10"/>'))
    with pytest.raises(DocumentError, match="Nothing is drawn"):
        send(session, "artboard", fit="artwork")
    for box in ([0, 0, 10], [0, 0, 0, 10], "0 0 10 10"):
        with pytest.raises(DocumentError):
            send(session, "artboard", box=box)
    assert session.editor.snapshot.revision == 0


def test_reference_keeps_its_place_when_the_artboard_changes():
    session = Session(import_svg(SVG))
    send(session, "reference", reference=halves())
    assert placed(session)["box"] == [0, 0, 100, 100]
    revision = session.reference_revision
    # Move the page right by half its width: its left half is the
    # reference's blue half; its right half lies past the image.
    send(session, "artboard", box=[50, 0, 100, 100])
    assert session.reference_revision == revision
    image = reference_image(session)
    assert image.size == (100, 100)
    assert pixel(image, 20, 50) == (0, 0, 255, 255)
    assert pixel(image, 80, 50)[3] == 0
    # Sampling and methods see the same placement; past the image is paper.
    sample = session.sample(
        {**session.content_version(), "source": "reference", "at": [70, 50]}
    )
    assert sample["colour"] == "#0000ff"
    sample = session.sample(
        {**session.content_version(), "source": "reference", "at": [130, 50]}
    )
    assert sample["colour"] == "#ffffff"
    # Undoing the artboard covers it with the whole image again.
    send(session, "undo")
    image = reference_image(session)
    assert pixel(image, 20, 50)[:3] == (255, 0, 0)
    assert pixel(image, 80, 50)[:3] == (0, 0, 255)


def test_grown_artboard_keeps_the_reference_resolution():
    session = Session(import_svg(SVG))
    send(session, "reference", reference=halves())
    send(session, "artboard", box=[0, 0, 400, 100])
    image = reference_image(session)
    assert image.size == (400, 100)
    assert pixel(image, 10, 10)[:3] == (255, 0, 0)
    assert pixel(image, 200, 10)[3] == 0


def test_reference_opacity_change_keeps_its_place():
    session = Session(import_svg(SVG))
    send(session, "reference", reference=halves())
    send(session, "artboard", box=[50, 0, 100, 100])
    # The page sends back what it has; only fitting places the image.
    sent = {**placed(session), "opacity": 0.8, "box": [0, 0, 1, 1]}
    send(session, "reference", reference=sent)
    assert placed(session)["opacity"] == 0.8
    assert placed(session)["box"] == [0, 0, 100, 100]


def test_project_saves_where_the_reference_lies():
    session = Session(import_svg(SVG))
    send(session, "reference", reference=halves())
    send(session, "artboard", box=[50, 0, 100, 100])
    saved = json.loads(session.project())
    assert saved["reference"]["box"] == [0, 0, 100, 100]
    # A project saved before references kept their place lies over its artboard.
    del saved["reference"]["box"]
    restored = Session(import_svg(SVG))
    send(restored, "open", source=json.dumps(saved), name="old.vectrify")
    assert placed(restored)["box"] == [50, 0, 100, 100]
    saved["reference"]["box"] = [0, 0, -1, 1]
    with pytest.raises(DocumentError, match="positive size"):
        send(restored, "open", source=json.dumps(saved), name="bad.vectrify")


def test_agent_sees_the_reference_where_the_editor_shows_it():
    session = Session(import_svg(SVG))
    send(session, "reference", reference=halves())
    agent = Agent(session)
    assert pixel(reference_image(agent), 20, 50) == (255, 0, 0)
    send(session, "artboard", box=[50, 0, 100, 100])
    assert pixel(reference_image(agent), 20, 50) == (0, 0, 255)


def test_new_drawing_replaces_the_drawing_and_its_reference():
    session = Session(import_svg(SVG))
    send(session, "reference", reference=halves())
    send(session, "artboard", box=[0, 0, 300, 100])
    epoch = session.epoch
    # The page opens an empty SVG of the chosen size (projectFiles.newDrawing).
    blank = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" '
        'viewBox="0 0 640 480"/>'
    )
    state = send(session, "open", source=blank, name="Untitled.svg")
    assert state["epoch"] != epoch
    assert state["bounds"] == [0, 0, 640, 480]
    assert state["name"] == "Untitled.svg"
    assert state["reference"] is None
    assert state["undo"] == []
    assert state["objects"][1:] == []
