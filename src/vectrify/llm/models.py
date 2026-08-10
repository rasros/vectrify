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
