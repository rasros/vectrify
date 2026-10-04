"""The editor in a native window, with the page calling Python directly.

The same page the server delivers, opened by pywebview in the platform's own
web view (WebView2, WebKit, or Qt WebEngine on Linux). There is no port and no
HTTP: the page's requests go straight to ``Backend.handle`` through the
window's JavaScript bridge, agents' calls reach the page as script the window
runs, and files are saved with a native dialog. Agents reach the window on
the one port of the MCP server it hosts while it allows them.
Install with ``vectrify[desktop]``.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vectrify.ui.server import STATIC, Backend


def available() -> bool:
    return importlib.util.find_spec("webview") is not None


def _fix_qt_permissions(window: Any) -> None:
    """Convert pywebview's integer permission policies for PyQt6."""
    qt = window.gui
    if not (getattr(qt, "PYQT6", False) and getattr(qt, "is_webengine", False)):
        return

    # pywebview 6.2.1 passes 1/2 from its feature permission handler, but
    # PyQt6 requires PermissionPolicy enums. Keep the backend's decisions
    # intact and only adapt the argument at the Qt boundary.
    def set_feature_permission(
        page: Any, origin: Any, feature: Any, policy: Any
    ) -> None:
        qt.QWebPage.setFeaturePermission(
            page, origin, feature, qt.QWebPage.PermissionPolicy(policy)
        )

    qt.BrowserView.WebPage.setFeaturePermission = set_feature_permission


class Api:
    """What the page may call: ``window.pywebview.api.<method>``.

    pywebview exposes every public method here, so helpers stay private.
    """

    def __init__(self, backend: Backend):
        self._backend = backend
        self._window: Any = None
        self._pushing: threading.Thread | None = None

    def _push(self, script: Callable[[str], Any], stop: threading.Event) -> None:
        """Tell the page each agent call as it happens, by running *script*
        (the window's ``run_js``) with its pulse: there is no HTTP to push
        over. On a thread of its own, so an agent call never waits on the
        page."""
        backend = self._backend
        channel = backend.agents
        beat = channel.beat
        while not stop.is_set():
            new = channel.wait(beat, 1.0)
            if new == beat:
                continue
            beat = new
            session_id = channel.session_id
            session = backend.sessions.get(session_id or "")
            if session is None or session_id is None:
                continue
            with session.lock:
                pulse = backend.pulse(session, session_id)
            # JSON is a JavaScript expression.
            with contextlib.suppress(Exception):
                script(
                    "window.vectrifyPulse && window.vectrifyPulse("
                    + json.dumps(pulse, allow_nan=False)
                    + ")"
                )

    def _start_pushing(self, script: Callable[[str], Any]) -> threading.Event:
        stop = threading.Event()
        self._pushing = threading.Thread(
            target=self._push, args=(script, stop), daemon=True, name="vectrify-push"
        )
        self._pushing.start()
        return stop

    def request(self, path: str, data: Any, session: str | None = None) -> dict:
        status, body = self._backend.handle(path, data, session)
        return {"status": status, "body": body}

    def save(self, name: str, content: str) -> str | None:
        """Ask where to save *content*; the chosen path, or None if cancelled."""
        webview = importlib.import_module("webview")
        chosen = self._window.create_file_dialog(
            webview.SAVE_DIALOG, save_filename=name
        )
        if not chosen:
            return None
        path = Path(chosen if isinstance(chosen, str) else chosen[0])
        path.write_text(content, encoding="utf-8")
        return str(path)


def run(backend: Backend) -> None:
    # On a Wayland session Qt's native backend reported a scale of 1 on a
    # screen set to 150%, drawing the whole editor at two thirds of its size;
    # through XWayland the page gets the desktop's scale. An explicit
    # QT_QPA_PLATFORM still wins.
    if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    webview = importlib.import_module("webview")
    api = Api(backend)
    # The query tells the page to talk to the bridge rather than fetch().
    api._window = webview.create_window(
        "Vectrify",
        url=(STATIC / "index.html").as_uri() + "?desktop",
        js_api=api,
        width=1440,
        height=900,
        min_size=(960, 640),
    )
    # This synchronous event runs after the backend is selected and before
    # its event loop handles page permission requests.
    api._window.events.before_show += _fix_qt_permissions
    api._start_pushing(api._window.run_js)
    # pywebview tries GTK first on Linux and prints a traceback when its Python
    # bindings are missing, as they are in a virtualenv; the extra installs Qt.
    linux_without_gtk = (
        sys.platform.startswith("linux") and importlib.util.find_spec("gi") is None
    )
    webview.start(gui="qt" if linux_without_gtk else None)
