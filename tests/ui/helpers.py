"""Shared helpers for the UI tests."""

import base64
import io
import time

import pytest
from PIL import Image, ImageDraw

from vectrify.document import import_svg
from vectrify.ui.session import Session


def reference(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return {
        "name": "target.png",
        "opacity": 0.5,
        "data_url": "data:image/png;base64,"
        + base64.b64encode(stream.getvalue()).decode(),
    }


def wait(session, job):
    for _ in range(500):
        state = session.operation({"command": "status", "job": job["id"]})
        if state["status"] != "running":
            return state
        time.sleep(0.01)
    pytest.fail("Operation did not finish")


SQUARE = "M20 20 L80 20 L80 80 L20 80 Z"


# The square with a spike on its top side, as the reference draws it.
SPIKED = [(20, 20), (40, 20), (50, 5), (60, 20), (80, 20), (80, 80), (20, 80)]


# Roughly along the spike, from the top side back to it.
STROKE = [(30, 20.5), (38, 19), (45, 11), (50, 6.5), (55, 11), (62, 19), (70, 20.5)]


# A screen pixel at 400%: two of the reference's per unit, so half of one.
PIXEL = 0.25


def reference_url(polygon):
    image = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(image).polygon([(2 * x, 2 * y) for x, y in polygon], fill="black")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def send(session, command, **data):
    return session.action(
        {
            "command": command,
            "epoch": session.epoch,
            "revision": session.editor.snapshot.revision,
            **data,
        }
    )


def session(d=SQUARE, wrapper="", reference=True):
    path = f'<path id="p" fill="black" d="{d}"/>'
    if wrapper:
        path = f'<g id="g" transform="{wrapper}">{path}</g>'
    result = Session(
        import_svg(f'<svg width="100" height="100" viewBox="0 0 100 100">{path}</svg>')
    )
    if reference:
        send(
            result,
            "reference",
            reference={"name": "r", "data_url": reference_url(SPIKED), "opacity": 0.5},
        )
    send(result, "select", objects=["p"])
    return result


SVG = """<svg width="100" height="100"><g id="layer">
<path id="a" fill="red" d="M0 0 L20 0 L20 20 Z"/>
<rect id="b" x="30" width="20" height="20" fill="blue"/>
</g></svg>"""


def png_url(width, height, colour=(200, 40, 40)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
