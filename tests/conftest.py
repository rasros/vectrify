import pytest


@pytest.fixture(autouse=True)
def settings_file(request, tmp_path, monkeypatch):
    """Keep every test away from the user's real settings and API keys.

    Live LLM tests (marked ``llm``) read the real keys: that is their point.
    """
    from vectrify.llm import keys

    if request.node.get_closest_marker("llm"):
        return keys.CONFIG_PATH
    path = tmp_path / "settings.json"
    monkeypatch.setattr(keys, "CONFIG_PATH", path)
    return path
