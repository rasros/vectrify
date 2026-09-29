from vectrify.llm.base import LLMConfig, LLMProvider
from vectrify.llm.models import Connection


def get_provider(connection: Connection) -> LLMProvider:
    """A client for *connection*, as models.resolve_provider found it."""
    name, key = connection.provider, connection.api_key
    if name in {"openai", "local"}:
        from vectrify.llm.openai import OpenAIProvider

        # The SDK refuses an empty key, and most local servers ignore it.
        return OpenAIProvider(key or "local", base_url=connection.base_url)
    if name == "anthropic":
        from vectrify.llm.anthropic import AnthropicProvider

        return AnthropicProvider(key)
    if name == "gemini":
        from vectrify.llm.gemini import GeminiProvider

        return GeminiProvider(key)
    raise ValueError(f"Unknown LLM provider: {name}")


__all__ = [
    "LLMConfig",
    "LLMProvider",
    "get_provider",
]
