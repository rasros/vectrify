"""HTTP integration checks for the local editor boundary and packaged assets."""

import base64
import http.client
import json
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

import vectrify.ui.server
from vectrify.project_file import decode_source
from vectrify.ui.server import Backend, EditorServer


def test_delete_pinned_objects_requests_confirmation_and_preserves_undo():
    backend = Backend('<svg><g id="group"><path id="p" d="M0 0 L10 0"/></g></svg>')
    _, state = backend.handle("/api/session", {}, None)
    session_id = state["session"]
    session = backend.sessions[session_id]
    session.action(
        {
            "command": "select",
            "epoch": session.epoch,
            "revision": 0,
            "objects": ["group"],
        }
    )
    node = session.editor.snapshot.document.geometry_for("p").subpaths[0].nodes[0]
    session.editor.pin_node("p", node.id)
    before = session.editor.snapshot
    payload = {"command": "delete", "epoch": session.epoch, "revision": before.revision}
    status, result = backend.handle("/api/action", payload, session_id)
    assert status == 200
    assert result["confirmation"] == "delete_pinned"
    assert "pinned points" in result["message"]
    assert session.editor.snapshot == before
    status, result = backend.handle(
        "/api/action", {**payload, "confirm_pinned": True}, session_id
    )
    assert status == 200
    assert "confirmation" not in result
    assert result["revision"] == before.revision + 1
    assert not session.editor.snapshot.document.root.children
    assert session.editor.undo().document == before.document


def test_confirmed_delete_refuses_changes_made_while_dialog_is_open():
    backend = Backend('<svg><path id="p" d="M0 0 L10 0"/></svg>')
    _, state = backend.handle("/api/session", {}, None)
    session_id = state["session"]
    session = backend.sessions[session_id]
    session.action(
        {"command": "select", "epoch": session.epoch, "revision": 0, "objects": ["p"]}
    )
    nodes = session.editor.snapshot.document.geometry_for("p").subpaths[0].nodes
    session.editor.pin_node("p", nodes[0].id)
    payload = {
        "command": "delete",
        "epoch": session.epoch,
        "revision": session.editor.snapshot.revision,
    }
    assert backend.handle("/api/action", payload, session_id)[1]["confirmation"]
    session.editor.pin_node("p", nodes[1].id)
    before = session.editor.snapshot
    status, result = backend.handle(
        "/api/action", {**payload, "confirm_pinned": True}, session_id
    )
    assert status == 409
    assert "changed while confirming" in result["error"]
    assert session.editor.snapshot == before


@pytest.fixture
def server():
    sample = (Path(__file__).with_name("sample.svg")).read_text()
    server = EditorServer(("127.0.0.1", 0), sample)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def call(server, path, payload=None, headers=None) -> tuple[int, Any]:
    client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    body = json.dumps(payload).encode() if payload is not None else None
    client.request(
        "POST" if body else "GET",
        path,
        body=body,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    response = client.getresponse()
    data = response.read()
    status = response.status
    content_type = response.getheader("Content-Type") or ""
    client.close()
    return status, json.loads(data) if "json" in content_type else data


def test_static_ui_and_session_edit_export_roundtrip(server):
    status, html = call(server, "/")
    assert status == 200
    assert b"Drawing canvas" in html
    # Every module the editor imports must be servable, not just a listed few.
    static = Path(vectrify.ui.server.__file__).with_name("static")
    for module in static.glob("*.js"):
        assert call(server, f"/{module.name}")[0] == 200, module.name
    assert call(server, "/../server.py")[0] == 404
    assert call(server, "/style.css")[0] == 200
    status, state = call(server, "/api/session", {})
    assert status == 200
    headers = {"X-Vectrify-Session": state["session"]}
    status, selected = call(
        server,
        "/api/action",
        {
            "command": "select",
            "epoch": state["epoch"],
            "revision": 0,
            "objects": ["sun"],
        },
        headers,
    )
    assert status == 200
    assert selected["selection"]["objects"] == ["sun"]
    status, result = call(
        server,
        "/api/action",
        {
            "command": "paint",
            "epoch": state["epoch"],
            "revision": 0,
            "changes": {"fill": "#ff0000"},
        },
        headers,
    )
    assert status == 200
    assert result["revision"] == 1
    assert (
        call(
            server,
            "/api/action",
            {
                "command": "undo",
                "epoch": state["epoch"],
                "revision": 0,
                "ids": [result["undo_ids"][-1]],
            },
            headers,
        )[0]
        == 200
    )
    status, result = call(
        server,
        "/api/export",
        {"epoch": state["epoch"], "revision": 1, "project": True},
        headers,
    )
    assert status == 200
    assert json.loads(result["content"])["vectrify_editor"] == 1
    assert (
        call(server, "/api/session", {"session": state["session"]})[1]["revision"] == 2
    )


def test_external_origins_and_filesystem_paths_are_not_exposed(server):
    assert call(server, "/api/session", {}, {"Origin": "https://example.com"})[0] == 403
    assert call(server, "/", headers={"Host": "example.com"})[0] == 403
    assert call(server, "/../../pyproject.toml")[0] == 404
    assert call(server, "/api/action", {"command": "undo"})[0] == 401


def test_compressed_download_reopens_with_reference_and_recovery_metadata(server):
    _, state = call(server, "/api/session", {})
    headers = {"X-Vectrify-Session": state["session"]}
    session = server.backend.sessions[state["session"]]
    # The actual image bytes must survive download and restore unchanged.
    import io

    from PIL import Image

    image = io.BytesIO()
    Image.new("RGBA", (1200, 800), (20, 30, 40, 128)).save(image, format="PNG")
    session.reference = {
        "name": "original.png",
        "data_url": "data:image/png;base64,"
        + base64.b64encode(image.getvalue()).decode(),
        "opacity": 0.35,
        "box": [0.0, 0.0, 1200.0, 800.0],
    }
    original = session.project()
    status, saved = call(
        server,
        "/api/export",
        {
            "project": True,
            "compressed": True,
            "epoch": state["epoch"],
            "revision": state["revision"],
        },
        headers,
    )
    assert status == 200
    binary = base64.b64decode(saved["content"])
    assert binary.startswith(b"\x1f\x8b")
    assert decode_source(binary) == original
    assert len(binary) < len(original.encode())
    assert (
        saved["summary"]["objects"]
        == len(session.editor.snapshot.document.elements()) - 1
    )
    assert saved["summary"]["nodes"] > 0
    reference = session.reference.copy()
    for source in (original, binary):
        restarted = Backend(source, "saved.vectrify")
        status, fresh = restarted.handle("/api/session", {}, None)
        assert status == 200
        assert restarted.sessions[fresh["session"]].project() == original
    for _ in range(2):  # A second restore models recovery after another restart.
        restored = Backend()
        _, fresh = restored.handle("/api/session", {}, None)
        status, _ = restored.handle(
            "/api/action",
            {
                "command": "open",
                "source": saved["content"],
                "encoding": saved["encoding"],
                "name": "saved.vectrify",
                "epoch": fresh["epoch"],
                "revision": fresh["revision"],
            },
            fresh["session"],
        )
        assert status == 200
        opened = restored.sessions[fresh["session"]]
        assert opened.reference == reference
        assert opened.project() == original


def test_hole_inspection_fill_and_cleanup_follow_session_revision(server):
    _, state = call(server, "/api/session", {})
    headers = {"X-Vectrify-Session": state["session"]}

    def action(command, **fields):
        nonlocal state
        status, state = call(
            server,
            "/api/action",
            dict(
                command=command,
                epoch=state["epoch"],
                revision=state["revision"],
                **fields,
            ),
            headers,
        )
        assert status == 200
        return state

    action(
        "open",
        name="holes.svg",
        source='<svg width="100" height="100"><path id="p" '
        'd="M0 0H100V100H0Z M10 10V30H30V10Z"/>'
        '<circle id="inside" cx="20" cy="20" r="5"/></svg>',
    )
    action("select", objects=["p"])
    request = {"object": "p", "epoch": state["epoch"], "revision": state["revision"]}
    status, result = call(server, "/api/holes", request, headers)
    assert status == 200
    assert len(result["holes"]) == 1
    assert result["holes"][0]["area"] == 400
    holes = [result["holes"][0]["id"]]
    status, candidates = call(
        server, "/api/holes", dict(**request, holes=holes, find_enclosed=True), headers
    )
    assert status == 200
    assert candidates["enclosed"] == ["inside"]
    action("fill_holes", object="p", holes=holes, delete_objects=["inside"])
    assert state["undo"] == ["Fill holes"]
    status, current = call(server, "/api/holes", request, headers)
    assert status == 200
    assert current["holes"] == []
    action("undo")
    assert {obj["id"] for obj in state["objects"]} >= {"p", "inside"}


def test_region_preview_changes_nothing_and_matches_the_extract(server):
    _, state = call(server, "/api/session", {})
    headers = {"X-Vectrify-Session": state["session"]}

    def action(command, **fields):
        nonlocal state
        status, state = call(
            server,
            "/api/action",
            dict(
                command=command,
                epoch=state["epoch"],
                revision=state["revision"],
                **fields,
            ),
            headers,
        )
        assert status == 200
        return state

    action(
        "open",
        name="region.svg",
        source='<svg width="100" height="100"><path id="p" fill="none" '
        'stroke="black" d="M0 50H100"/></svg>',
    )
    action("select", objects=["p"])
    request = {
        "epoch": state["epoch"],
        "revision": state["revision"],
        "region": [20, 20, 40, 60],
        "scope": "selection",
        "cut": True,
    }
    status, result = call(server, "/api/region", request, headers)
    assert status == 200
    assert result["extract"]["targets"] == [{"id": "p", "outcome": "cut"}]
    assert result["crossing"] == ["p"]
    assert result["revision"] == state["revision"]
    _, current = call(
        server, "/api/session", {"session": headers["X-Vectrify-Session"]}
    )
    assert current["undo"] == []
    assert current["revision"] == state["revision"]
    action("extract", region=request["region"], scope="selection", cut=True)
    assert state["undo"] == ["Extract region"]
    status, _ = call(
        server, "/api/region", dict(request, epoch="another drawing"), headers
    )
    assert status == 409


def test_operation_endpoint_previews_applies_and_rejects_unknown(server):
    _, state = call(server, "/api/session", {})
    headers = {"X-Vectrify-Session": state["session"]}
    # The middle point of the top edge is redundant; Clean up removes it.
    source = (
        '<svg width="100" height="100">'
        '<path id="a" d="M10 10 L20 10 L30 10 L30 30 L10 30 Z"/></svg>'
    )
    _, state = call(
        server,
        "/api/action",
        {"command": "open", "source": source, "epoch": state["epoch"], "revision": 0},
        headers,
    )
    _, state = call(
        server,
        "/api/action",
        {
            "command": "select",
            "epoch": state["epoch"],
            "revision": state["revision"],
            "objects": ["a"],
        },
        headers,
    )
    start = {
        "command": "start",
        "epoch": state["epoch"],
        "revision": state["revision"],
        "action": "simplify",
        "method": "cleanup",
        "permissions": {"geometry": True, "structure": True},
    }
    status, job = call(server, "/api/operation", start, headers)
    assert status == 200, job
    assert job["status"] == "ready"
    assert set(job["result"]["previews"]) == {"before", "after"}
    status, _ = call(server, "/api/operation", dict(start, method="nope"), headers)
    assert status == 400
    command = {"command": "apply", "job": job["id"]}
    assert job["result"]["changed"]
    status, applied = call(server, "/api/operation", command, headers)
    assert status == 200
    assert applied["revision"] == state["revision"] + 1
    status, _ = call(server, "/api/operation", command, headers)
    assert status == 400


def test_the_desktop_bridge_answers_like_the_server():
    from vectrify.ui.desktop import Api
    from vectrify.ui.server import Backend

    api = Api(Backend())
    opened = api.request("/api/session", {})
    assert opened["status"] == 200
    session = opened["body"]["session"]
    assert api.request("/api/reference", {}, session) == {
        "status": 200,
        "body": {
            "epoch": opened["body"]["epoch"],
            "revision": 0,
            "reference_revision": 0,
            "project_version": "0:0",
            "reference": None,
        },
    }
    assert api.request("/api/reference", {}, "stale")["status"] == 401
    assert api.request("/api/nope", {}, session)["status"] == 404
