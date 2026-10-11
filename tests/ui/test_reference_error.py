"""Reference error pixels, view cycling and asynchronous refreshes."""

import base64
import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.mcp.helpers import reference_png
from vectrify.ui.agent import Agent
from vectrify.ui.server import Backend


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_reference_error_frontend():
    result = subprocess.run(
        ["node", str(Path(__file__).with_name("reference_error.mjs"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_reported_error_view_renders_the_comparison():
    backend = Backend(
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="1200" height="800" viewBox="40 20 1200 800"/>'
    )
    _, initial = backend.handle("/api/session", {}, None)
    session = backend.sessions[initial["session"]]
    session.reference = {
        "name": "reference.png",
        "data_url": "data:image/png;base64,"
        + base64.b64encode(reference_png()).decode(),
        "opacity": 0,
    }
    session.set_view(
        {
            "region": session.state()["bounds"],
            "pixels": [100, 67],
            "zoom": 1,
            "reference_view": "error",
        }
    )
    agent = Agent(session)
    assert session.view is not None
    assert session.view["reference_view"] == "error"
    status, preview = backend.handle(
        "/api/reference-error", initial, initial["session"]
    )
    assert status == 200
    original = base64.b64decode(preview["image"].split(",")[1])
    full = session.reference_error()
    assert session.reference_error() is full
    reply = agent.tool_render(None, region="view", max_side=max(full.size))
    assert "reference error" in reply.data["shows"]
    assert reply.images[0][1] == original
    # Exact matches are black, and differences are visible even at 0% overlay.
    pixels = np.asarray(Image.open(io.BytesIO(reply.images[0][1])))
    assert (pixels == 0).all(axis=2).any()
    assert (pixels > 0).any()
    session.reference["opacity"] = 0.75
    assert session.reference_error() is full, "overlay opacity is irrelevant"

    # A zoomed/panned MCP view crops the shared image, including black outside.
    session.set_view(
        {
            "region": [-80, -60, 600, 400],
            "pixels": [300, 200],
            "zoom": 0.5,
            "reference_view": "error",
        }
    )
    cropped = agent.tool_render(None, region="view")
    crop = Image.open(io.BytesIO(cropped.images[0][1]))
    expected = full.transform(
        (300, 200),
        Image.Transform.EXTENT,
        (-0.1 * full.width, -0.1 * full.height, 0.4 * full.width, 0.4 * full.height),
        Image.Resampling.BILINEAR,
        fillcolor="black",
    )
    assert np.array_equal(np.asarray(crop), np.asarray(expected))

    session.epoch = "replacement"
    assert session.reference_error() is not full
    status, _ = backend.handle("/api/reference-error", initial, initial["session"])
    assert status == 409, "a stale frontend cannot request another drawing's map"
    session.reference = None
    status, result = backend.handle(
        "/api/reference-error",
        {"epoch": session.epoch, "revision": 0},
        initial["session"],
    )
    assert status == 400
    assert "reference" in result["error"]
