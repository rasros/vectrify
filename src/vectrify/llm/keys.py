"""API keys entered in the editor's settings, kept in the user's config dir.

The file is readable by its owner only. Keys leave this module in one direction:
to the provider client. What the editor shows is whether a key is set and its
last four characters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from vectrify.llm.models import PROVIDERS

CONFIG_PATH = Path.home() / ".config" / "vectrify" / "settings.json"


def _read() -> dict:
    try:
        data = json.loads(CONFIG_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


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


def save(changes: dict[str, str | None]) -> None:
    """Set each named provider's key; an empty value or None removes it."""
    keys = load()
    for name, key in changes.items():
        if name not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider: {name}")
        key = (key or "").strip()
        if key:
            keys[name] = key
        else:
            keys.pop(name, None)
    data = _read()
    data["api_keys"] = keys
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Created owner-only, rather than chmod-ed after the key is on disk.
    temp = CONFIG_PATH.with_suffix(".tmp")
    temp.unlink(missing_ok=True)
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as file:
        json.dump(data, file, indent=2)
    temp.replace(CONFIG_PATH)


def summary() -> dict[str, str | None]:
    """Per provider, the key's last four characters, or None when unset."""
    keys = load()
    return {name: keys[name][-4:] if name in keys else None for name in PROVIDERS}
