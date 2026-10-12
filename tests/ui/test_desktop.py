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
@pytest.mark.parametrize("modern", [False, True])
def test_native_save_writes_binary_projects_and_plain_svg(
    monkeypatch, tmp_path, project, modern
):
    import base64

    path = tmp_path / ("drawing.vectrify" if project else "drawing.svg")
    api = desktop.Api(desktop.Backend())
    dialog = MagicMock(return_value=[str(path)])
    api._window = SimpleNamespace(create_file_dialog=dialog)
    save = object()
    webview = (
        SimpleNamespace(FileDialog=SimpleNamespace(SAVE=save))
        if modern
        else SimpleNamespace(SAVE_DIALOG=save)
    )
    monkeypatch.setattr(
        desktop.importlib,
        "import_module",
        lambda _name: webview,
    )
    source = '{"vectrify_editor":1}' if project else '<svg width="100" height="100"/>'
    expected = encode_project(source) if project else source.encode()
    content = base64.b64encode(expected).decode() if project else source
    assert api.save(path.name, content, "base64" if project else None) == str(path)
    dialog.assert_called_once_with(save, save_filename=path.name)
    assert path.read_bytes() == expected
    dialog.reset_mock()
    assert api.save_target(path.name) == str(path)
    assert api.save(path.name, content, "base64" if project else None) == str(path)
    dialog.assert_not_called()


def test_native_save_overwrites_initial_file_and_save_as_changes_target(tmp_path):
    original = tmp_path / "original.svg"
    original.write_text("<svg/>")
    api = desktop.Api(desktop.Backend(), original)
    dialog = MagicMock()
    api._window = SimpleNamespace(create_file_dialog=dialog)
    assert api.save_target(original.name) == str(original)
    assert api.save(original.name, "<svg><path/></svg>") == str(original)
    assert original.read_text() == "<svg><path/></svg>"
    dialog.assert_not_called()


def test_native_save_as_cancellation_and_exports_keep_current_file(
    monkeypatch, tmp_path
):
    original = tmp_path / "original.svg"
    copy = tmp_path / "copy.svg"
    export = tmp_path / "export.svg"
    api = desktop.Api(desktop.Backend(), original)
    dialog = MagicMock(side_effect=[None, [str(copy)], [str(export)]])
    api._window = SimpleNamespace(create_file_dialog=dialog)
    monkeypatch.setattr(
        desktop.importlib, "import_module", lambda _name: SimpleNamespace(SAVE_DIALOG=1)
    )
    assert api.save_target(original.name, True) is None
    assert api.save_target(original.name) == str(original)
    assert api.save_target(original.name, True) == str(copy)
    api.save(copy.name, "<svg/>")
    assert api.save_target(original.name) == str(copy)
    assert api.save_target(export.name, True) == str(export)
    api.save(export.name, "<svg/>", remember=False)
    assert export.read_text() == "<svg/>"
    assert api.save_target(original.name) == str(copy)


@pytest.mark.parametrize("modern", [False, True])
def test_native_open_tracks_path_and_recovery_forgets_it(monkeypatch, tmp_path, modern):
    path = tmp_path / "opened.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    api = desktop.Api(desktop.Backend())
    dialog = MagicMock(return_value=[str(path)])
    api._window = SimpleNamespace(create_file_dialog=dialog)
    webview = (
        SimpleNamespace(FileDialog=SimpleNamespace(OPEN=1, SAVE=2))
        if modern
        else SimpleNamespace(OPEN_DIALOG=1, SAVE_DIALOG=2)
    )
    monkeypatch.setattr(desktop.importlib, "import_module", lambda _name: webview)
    session = api.request("/api/session", {})["body"]
    picked = api.open_file()
    assert picked is not None
    opened = api.request(
        "/api/action",
        {
            "command": "open",
            **picked,
            "epoch": session["epoch"],
            "revision": session["revision"],
        },
        session["session"],
    )
    assert opened["status"] == 200
    dialog.reset_mock()
    assert api.save_target(path.name) == str(path)
    dialog.assert_not_called()
    restored = api.request(
        "/api/action",
        {
            "command": "open",
            "name": path.name,
            "source": path.read_text(),
            "epoch": opened["body"]["epoch"],
            "revision": opened["body"]["revision"],
        },
        session["session"],
    )
    assert restored["status"] == 200
    dialog.return_value = None
    assert api.save_target(path.name) is None
    dialog.assert_called_once()
