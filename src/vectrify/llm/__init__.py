from vectrify.llm.base import LLMConfig, LLMProvider


def get_provider(provider_name: str, api_key: str) -> LLMProvider:
    """A client for *provider_name*; models.resolve_provider finds the key."""
    if provider_name == "openai":
        from vectrify.llm.openai import OpenAIProvider

        return OpenAIProvider(api_key)
    if provider_name == "anthropic":
        from vectrify.llm.anthropic import AnthropicProvider

        return AnthropicProvider(api_key)
    if provider_name == "gemini":
        from vectrify.llm.gemini import GeminiProvider

        return GeminiProvider(api_key)
    raise ValueError(f"Unknown LLM provider: {provider_name}")


__all__ = [
    "LLMConfig",
    "LLMProvider",
    "get_provider",
]
