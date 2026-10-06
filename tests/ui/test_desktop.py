"""Desktop backend compatibility without requiring a display or Qt install."""

from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vectrify.project_file import encode_project
from vectrify.ui import desktop


class PermissionPolicy(Enum):
    PermissionGrantedByUser = 1
    PermissionDeniedByUser = 2


@pytest.mark.parametrize("policy", [1, 2, *PermissionPolicy])
def test_qt_permission_policies_are_typed_and_decisions_preserved(policy):
    calls = []

    class QtPage:
        def setFeaturePermission(self, origin, feature, permission):  # noqa: N802
            # Match PyQt6's strict enum argument check.
            assert isinstance(permission, PermissionPolicy)
            calls.append((self, origin, feature, permission))

    class WebPage(QtPage):
        pass

    qt = SimpleNamespace(
        PYQT6=True,
        is_webengine=True,
        QWebPage=SimpleNamespace(
            PermissionPolicy=PermissionPolicy,
            setFeaturePermission=QtPage.setFeaturePermission,
        ),
        BrowserView=SimpleNamespace(WebPage=WebPage),
    )
    window = SimpleNamespace(gui=qt)
    # A second window must not wrap the first window's adapter recursively.
    desktop._fix_qt_permissions(window)
    desktop._fix_qt_permissions(window)
    page = WebPage()
    page.setFeaturePermission("origin", "feature", policy)
    assert calls == [(page, "origin", "feature", PermissionPolicy(policy))]


@pytest.mark.parametrize(
    "gui",
    [
        None,
        SimpleNamespace(PYQT6=False),
        SimpleNamespace(PYQT6=True, is_webengine=False),
    ],
)
def test_permission_compatibility_ignores_other_backends(gui):
    desktop._fix_qt_permissions(SimpleNamespace(gui=gui))


def test_desktop_installs_permission_fix_before_start(monkeypatch):
    window = MagicMock()
    before_show = window.events.before_show
    webview = SimpleNamespace(create_window=lambda *_args, **_kwargs: window)

    def start(**_kwargs):
        before_show.__iadd__.assert_called_once_with(desktop._fix_qt_permissions)

    webview.start = start
    monkeypatch.setattr(desktop.importlib, "import_module", lambda _name: webview)
    monkeypatch.setattr(desktop.Api, "_start_pushing", lambda *_args: None)
    desktop.run(desktop.Backend())


@pytest.mark.parametrize("project", [False, True])
def test_native_save_writes_binary_projects_and_plain_svg(
    monkeypatch, tmp_path, project
):
    import base64

    path = tmp_path / ("drawing.vectrify" if project else "drawing.svg")
    api = desktop.Api(desktop.Backend())
    api._window = SimpleNamespace(
        create_file_dialog=lambda *_args, **_kwargs: [str(path)]
    )
    monkeypatch.setattr(
        desktop.importlib, "import_module", lambda _name: SimpleNamespace(SAVE_DIALOG=1)
    )
    source = '{"vectrify_editor":1}' if project else '<svg width="100" height="100"/>'
    expected = encode_project(source) if project else source.encode()
    content = base64.b64encode(expected).decode() if project else source
    assert api.save(path.name, content, "base64" if project else None) == str(path)
    assert path.read_bytes() == expected
