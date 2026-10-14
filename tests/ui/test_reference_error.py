"""Reference error pixels, view cycling and asynchronous refreshes."""

import base64
import io
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from PIL import Image

from tests.mcp.helpers import reference_png
from vectrify.ui.agent import Agent
from vectrify.ui.server import Backend
from vectrify.ui.shared import png


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
@pytest.mark.parametrize("script", ["reference_error.mjs", "reference_render.mjs"])
def test_reference_error_frontend(script):
    result = subprocess.run(
        ["node", str(Path(__file__).with_name(script))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_reference_error_in_browser():
    node = shutil.which("node")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not node or not chrome:
        pytest.skip("needs Node.js and Chrome/Chromium")
    result = subprocess.run(
        [node, str(Path(__file__).with_name("reference_render_browser.mjs")), chrome],
        capture_output=True,
        text=True,
        check=False,
        timeout=75,
    )
    if result.returncode == 77:
        pytest.skip("needs Node.js with native WebSocket support")
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
    assert np.array_equal(
        np.asarray(Image.open(io.BytesIO(reply.images[0][1]))),
        np.asarray(Image.open(io.BytesIO(original))),
    )
    # Exact matches are black, and differences are visible even at 0% overlay.
    pixels = np.asarray(Image.open(io.BytesIO(reply.images[0][1])))
    assert (pixels == 0).all(axis=2).any()
    assert (pixels > 0).any()
    reference = session.reference
    assert reference is not None
    session.reference = {**reference, "opacity": 0.75}
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


def test_error_render_allows_edits_and_does_not_cache_an_obsolete_snapshot(monkeypatch):
    from vectrify.svg_render import render_image

    backend = Backend(
        '<svg width="24" height="24"><rect id="shape" '
        'width="24" height="24" fill="white"/></svg>',
        reference={
            "name": "white.png",
            "opacity": 0.5,
            "data_url": "data:image/png;base64,"
            + base64.b64encode(png(Image.new("RGB", (24, 24), "white"))).decode(),
        },
    )
    _, initial = backend.handle("/api/session", {}, None)
    session = backend.sessions[initial["session"]]
    started, release = Event(), Event()

    def slow_render(*args, **kwargs):
        started.set()
        assert release.wait(5), "test did not release the render"
        return render_image(*args, **kwargs)

    monkeypatch.setattr("vectrify.svg_render.render_image", slow_render)
    with ThreadPoolExecutor(max_workers=2) as executor:
        rendering = executor.submit(
            backend.handle, "/api/reference-error", initial, initial["session"]
        )
        try:
            assert started.wait(5)
            selected = executor.submit(
                backend.handle,
                "/api/action",
                {**initial, "command": "select", "objects": ["shape"]},
                initial["session"],
            )
            status, state = selected.result(timeout=2)
            assert status == 200
            edited = executor.submit(
                backend.handle,
                "/api/action",
                {**state, "command": "paint", "changes": {"fill": "black"}},
                initial["session"],
            )
            assert edited.result(timeout=2)[0] == 200
        finally:
            release.set()
        status, result = rendering.result(timeout=5)
    assert status == 200
    old = Image.open(io.BytesIO(base64.b64decode(result["image"].split(",")[1])))
    assert np.asarray(old).max() == 0, "the render uses its original snapshot"
    latest = session.reference_error()
    assert np.asarray(latest).min() == 255, (
        "the obsolete render did not poison the cache"
    )
    assert session.reference_error() is latest


def test_reference_error_invalidates_when_reference_placement_changes():
    backend = Backend(
        '<svg width="24" height="24"/>',
        reference={
            "name": "black.png",
            "opacity": 0.5,
            "data_url": "data:image/png;base64,"
            + base64.b64encode(png(Image.new("RGB", (24, 24), "black"))).decode(),
        },
    )
    _, state = backend.handle("/api/session", {}, None)
    session = backend.sessions[state["session"]]
    before = session.reference_error()
    assert session.reference is not None
    session.reference = {**session.reference, "box": [24, 0, 24, 24]}
    after = session.reference_error()
    assert before is not after
    assert np.asarray(before).min() == 255
    assert np.asarray(after).max() == 0


@pytest.fixture
def browser_comparison():
    backend = Backend(
        '<svg width="24" height="24"/>',
        reference={
            "name": "white.png",
            "opacity": 0.5,
            "data_url": "data:image/png;base64,"
            + base64.b64encode(png(Image.new("RGB", (24, 24), "white"))).decode(),
        },
    )
    _, state = backend.handle("/api/session", {}, None)
    session = backend.sessions[state["session"]]
    return backend, state, session


def test_browser_pixels_are_shared_with_mcp_and_reject_stale_uploads(
    browser_comparison,
):
    backend, state, session = browser_comparison
    image = Image.new("RGB", (24, 24), (42, 0, 0))
    data = {
        **state,
        "image": "data:image/png;base64," + base64.b64encode(png(image)).decode(),
        "view": {
            "region": [0, 0, 24, 24],
            "pixels": [24, 24],
            "zoom": 1,
            "reference_view": "error",
        },
    }
    status, result = backend.handle(
        "/api/reference-error-cache", data, state["session"]
    )
    assert status == 200
    assert result == {"cached": True}
    assert np.array_equal(np.asarray(session.reference_error()), np.asarray(image))
    assert session.view is not None
    assert session.view["reference_view"] == "error"
    rendered = Agent(session).tool_render(None, region="view", max_side=24)
    assert np.array_equal(
        np.asarray(Image.open(io.BytesIO(rendered.images[0][1]))), np.asarray(image)
    )
    for change in (
        {"epoch": "obsolete"},
        {"revision": -1},
        {"reference_revision": -1},
    ):
        status, result = backend.handle(
            "/api/reference-error-cache", {**data, **change}, state["session"]
        )
        assert status == 200
        assert result == {"cached": False}
        assert np.array_equal(np.asarray(session.reference_error()), np.asarray(image))


@pytest.mark.parametrize("size", [(2049, 2049), (24, 12)])
def test_browser_comparison_rejects_wrong_size(browser_comparison, size):
    backend, state, _ = browser_comparison
    image = (
        "data:image/png;base64,"
        + base64.b64encode(png(Image.new("RGB", size))).decode()
    )
    status, result = backend.handle(
        "/api/reference-error-cache", {**state, "image": image}, state["session"]
    )
    assert status == 400
    assert "size" in result["error"]


def test_browser_cache_wins_over_a_render_in_progress(browser_comparison, monkeypatch):
    from vectrify.svg_render import render_image

    backend, state, session = browser_comparison
    started, release = Event(), Event()

    def slow_render(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return render_image(*args, **kwargs)

    monkeypatch.setattr("vectrify.svg_render.render_image", slow_render)
    with ThreadPoolExecutor(max_workers=1) as executor:
        rendering = executor.submit(session.reference_error)
        try:
            assert started.wait(5)
            status, result = backend.handle(
                "/api/reference-error-cache",
                {
                    **state,
                    "image": "data:image/png;base64,"
                    + base64.b64encode(
                        png(Image.new("RGB", (24, 24), (42, 0, 0)))
                    ).decode(),
                },
                state["session"],
            )
            assert status == 200
            assert result == {"cached": True}
        finally:
            release.set()
        assert rendering.result(timeout=5).getpixel((0, 0)) == (42, 0, 0)
    assert session.reference_error().getpixel((0, 0)) == (42, 0, 0)
