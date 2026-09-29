"""Single source of truth for supported providers and their default models."""

from __future__ import annotations

from dataclasses import dataclass

# "local" is any server speaking the OpenAI chat API (Ollama, LM Studio,
# llama.cpp, vLLM) at the URL saved in Settings.
PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "gemini", "local")

REASONING: tuple[str, ...] = ("low", "medium", "high")
DEFAULT_REASONING = "medium"

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-5.4",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.1-pro-preview",
}


@dataclass(frozen=True)
class Connection:
    """Where to send a request, and with what."""

    provider: str
    api_key: str
    base_url: str | None = None
    model: str | None = None
    # None for a local server: most reject OpenAI's reasoning_effort.
    reasoning: str | None = None


def _connection(provider: str) -> Connection | None:
    from vectrify.llm import keys

    stored = keys.load()
    if provider == "local":
        local = keys.load_local()
        if not local["base_url"]:
            return None
        return Connection(
            "local",
            stored.get("local", ""),
            local["base_url"],
            local["model"] or None,
        )
    if provider not in stored:
        return None
    chosen = keys.load_models()[provider]
    return Connection(
        provider,
        stored[provider],
        model=chosen["model"] or DEFAULT_MODELS[provider],
        reasoning=chosen["reasoning"] or DEFAULT_REASONING,
    )


def resolve_provider(provider: str = "auto") -> Connection:
    """The provider to use: named, or the first set up in Settings.

    Automatic tries the hosted providers first and the local server last.
    Raises ValueError saying what to add.
    """
    if provider == "auto":
        for name in PROVIDERS:
            connection = _connection(name)
            if connection is not None:
                return connection
        raise ValueError("No LLM is set up; add an API key or local server in Settings")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    connection = _connection(provider)
    if connection is None:
        if provider == "local":
            raise ValueError("Add a local server URL in Settings to use it")
        raise ValueError(f"Add a {provider} API key in Settings to use {provider}")
    return connection
