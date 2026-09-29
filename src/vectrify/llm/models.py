"""Single source of truth for supported providers and their default models."""

PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "gemini")

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-5.4",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.1-pro-preview",
}


def api_key_env(provider: str) -> str:
    """Environment variable holding the API key for *provider*."""
    return f"{provider.upper()}_API_KEY"


def resolve_provider(provider: str = "auto") -> tuple[str, str]:
    """The provider to use and its API key: named, or the first with a key set.

    Raises ValueError naming the missing environment variable.
    """
    import os

    if provider == "auto":
        for name in PROVIDERS:
            key = os.getenv(api_key_env(name))
            if key:
                return name, key
        names = ", ".join(api_key_env(name) for name in PROVIDERS)
        raise ValueError(f"No LLM API key is set; export one of {names}")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    key = os.getenv(api_key_env(provider))
    if not key:
        raise ValueError(f"Set {api_key_env(provider)} to use {provider}")
    return provider, key
