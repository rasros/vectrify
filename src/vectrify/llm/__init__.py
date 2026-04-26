from vectrify.llm.base import LLMConfig, LLMProvider


def get_provider(provider_name: str, api_key: str | None = None) -> LLMProvider:
    provider_name = provider_name.lower()
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
