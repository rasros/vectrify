"""LLM settings entered in the editor, kept in the user's config dir.

The file is readable by its owner only. Keys leave this module in one direction:
to the provider client. What the editor shows is whether a key is set and its
last four characters. The models, reasoning efforts and the local server's URL
are not secret and are shown as saved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from vectrify.llm.models import (
    DEFAULT_MODELS,
    DEFAULT_REASONING,
    PROVIDERS,
    REASONING,
)

CONFIG_PATH = Path.home() / ".config" / "vectrify" / "settings.json"


def _read() -> dict:
    try:
        data = json.loads(CONFIG_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(data: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Created owner-only, rather than chmod-ed after the key is on disk.
    temp = CONFIG_PATH.with_suffix(".tmp")
    temp.unlink(missing_ok=True)
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as file:
        json.dump(data, file, indent=2)
    temp.replace(CONFIG_PATH)


def load() -> dict[str, str]:
    """The stored key for each provider that has one."""
    keys = _read().get("api_keys")
    if not isinstance(keys, dict):
        return {}
    return {
        name: key
        for name, key in keys.items()
        if name in PROVIDERS and isinstance(key, str) and key
    }


def load_local() -> dict[str, str]:
    """The local server's URL and default model, empty when unset."""
    local = _read().get("local")
    local = local if isinstance(local, dict) else {}
    return {
        field: value if isinstance(value := local.get(field), str) else ""
        for field in ("base_url", "model")
    }


def load_models() -> dict[str, dict[str, str]]:
    """Each hosted provider's model and reasoning effort, empty when unset."""
    saved = _read().get("models")
    saved = saved if isinstance(saved, dict) else {}
    models = {}
    for name in DEFAULT_MODELS:
        entry = saved.get(name)
        entry = entry if isinstance(entry, dict) else {}
        model = entry.get("model")
        reasoning = entry.get("reasoning")
        models[name] = {
            "model": model if isinstance(model, str) else "",
            "reasoning": reasoning if reasoning in REASONING else "",
        }
    return models


def save(
    api_keys: dict[str, str | None] | None = None,
    local: dict[str, str | None] | None = None,
    models: dict[str, dict[str, str | None]] | None = None,
) -> None:
    """Set keys, the local server's URL and model, and each provider's model
    and reasoning effort. An empty value or None removes that entry.
    """
    data = _read()
    keys = load()
    for name, key in (api_keys or {}).items():
        if name not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider: {name}")
        key = (key or "").strip()
        if key:
            keys[name] = key
        else:
            keys.pop(name, None)
    data["api_keys"] = keys
    if local is not None:
        saved = load_local()
        for field, value in local.items():
            if field not in saved:
                raise ValueError(f"Unknown local server setting: {field}")
            saved[field] = (value or "").strip()
        url = urlparse(saved["base_url"])
        if saved["base_url"] and (
            url.scheme not in {"http", "https"} or not url.netloc
        ):
            raise ValueError("The local server URL must start with http:// or https://")
        data["local"] = saved
    if models is not None:
        chosen = load_models()
        for name, entry in models.items():
            if name not in chosen:
                raise ValueError(f"Unknown LLM provider: {name}")
            for field, value in entry.items():
                if field not in chosen[name]:
                    raise ValueError(f"Unknown model setting: {field}")
                value = (value or "").strip()
                if field == "reasoning" and value and value not in REASONING:
                    raise ValueError(f"Reasoning effort must be one of {REASONING}")
                chosen[name][field] = value
        data["models"] = chosen
    _write(data)


def summary() -> dict:
    """What the editor may show: key tails, and the local server as saved."""
    keys = load()
    return {
        "api_keys": {
            name: keys[name][-4:] if name in keys else None for name in PROVIDERS
        },
        "local": load_local(),
        "models": load_models(),
        # What an unset model or effort falls back to, for the editor to show.
        "defaults": {"models": DEFAULT_MODELS, "reasoning": DEFAULT_REASONING},
    }
