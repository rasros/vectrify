"""Single source of truth for supported providers and their default models."""

PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "gemini")

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-5.4",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.1-pro-preview",
}


def resolve_provider(provider: str = "auto") -> tuple[str, str]:
    """The provider to use and its API key: named, or the first with a key set.

    Keys come from the editor's settings. Raises ValueError saying what to add.
    """
    from vectrify.llm import keys

    stored = keys.load()
    if provider == "auto":
        for name in PROVIDERS:
            if name in stored:
                return name, stored[name]
        raise ValueError("No LLM API key is set; add one in Settings")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    if provider not in stored:
        raise ValueError(f"Add a {provider} API key in Settings to use {provider}")
    return provider, stored[provider]
