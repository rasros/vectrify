import argparse

import pytest

from vectrify.main import determine_provider_and_model

_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")


@pytest.fixture
def clear_keys(monkeypatch):
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)


def _args(**kwargs):
    defaults = {"provider": "auto", "model": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.mark.usefixtures("clear_keys")
def test_auto_no_key_errors_and_names_env_vars(capsys):
    with pytest.raises(SystemExit) as exc:
        determine_provider_and_model(_args())
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert all(name in err for name in _KEYS)
    assert err.startswith("Error:")


@pytest.mark.usefixtures("clear_keys")
def test_explicit_provider_missing_key_names_var(capsys):
    with pytest.raises(SystemExit) as exc:
        determine_provider_and_model(_args(provider="anthropic"))
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err
    assert err.startswith("Error:")


@pytest.mark.usefixtures("clear_keys")
def test_auto_selects_provider_by_priority(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    provider, model = determine_provider_and_model(_args())
    assert provider == "anthropic"
    assert model == "claude-4-6-sonnet"


@pytest.mark.usefixtures("clear_keys")
def test_explicit_model_is_preserved(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    provider, model = determine_provider_and_model(_args(model="custom-model"))
    assert provider == "openai"
    assert model == "custom-model"
