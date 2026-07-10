import argparse

import pytest

from vectrify.main import (
    _fail,
    determine_provider_and_model,
    format_extension_warning,
)

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


@pytest.mark.parametrize(
    ("output", "fmt", "ext"),
    [
        ("out.svg", "svg", ".svg"),
        ("out.dot", "graphviz", ".dot"),
        ("OUT.SVG", "svg", ".svg"),  # extension check is case-insensitive
    ],
)
def test_extension_match_no_warning(output, fmt, ext):
    assert format_extension_warning(output, fmt, ext) is None


@pytest.mark.parametrize(
    ("output", "fmt", "ext"),
    [
        ("out.svg", "graphviz", ".dot"),
        ("out.dot", "svg", ".svg"),
        ("out", "typst", ".typ"),  # no extension at all
    ],
)
def test_extension_mismatch_warns(output, fmt, ext):
    msg = format_extension_warning(output, fmt, ext)
    assert msg is not None
    assert fmt in msg
    assert ext in msg


def test_fail_without_debug_shows_hint_not_traceback(capsys):
    try:
        raise ValueError("boom")
    except ValueError:
        with pytest.raises(SystemExit) as exc:
            _fail("Error: boom", debug=False)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Error: boom" in err
    assert "--debug" in err
    assert "Traceback" not in err


def test_fail_with_debug_prints_traceback(capsys):
    try:
        raise ValueError("boom")
    except ValueError:
        with pytest.raises(SystemExit) as exc:
            _fail("Error: boom", debug=True)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Error: boom" in err
    assert "Traceback" in err
    assert "ValueError: boom" in err
