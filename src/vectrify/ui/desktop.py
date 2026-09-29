"""The editor in a native window, with the page calling Python directly.

The same page the server delivers, opened by pywebview in the platform's own
web view (WebView2, WebKit, or Qt WebEngine on Linux). There is no port and no
HTTP: the page's requests go straight to ``Backend.handle`` through the
window's JavaScript bridge, and files are saved with a native dialog.
Install with ``vectrify[desktop]``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from vectrify.ui.server import STATIC, Backend


def available() -> bool:
    return importlib.util.find_spec("webview") is not None


class Api:
    """What the page may call: ``window.pywebview.api.<method>``.

    pywebview exposes every public method here, so helpers stay private.
    """

    def __init__(self, backend: Backend):
        self._backend = backend
        self._window: Any = None

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
    # pywebview tries GTK first on Linux and prints a traceback when its Python
    # bindings are missing, as they are in a virtualenv; the extra installs Qt.
    linux_without_gtk = (
        sys.platform.startswith("linux") and importlib.util.find_spec("gi") is None
    )
    webview.start(gui="qt" if linux_without_gtk else None)
