import importlib.util

import pytest

# The MCP SDK is the optional extra vectrify[mcp] (dev installs it too);
# without it these tests have nothing to drive.
collect_ignore_glob = [] if importlib.util.find_spec("mcp") else ["test_*.py"]


@pytest.fixture(autouse=True)
def state_home(tmp_path, monkeypatch):
    """A discovery file of the test's own, never a running editor's."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path / "state"
