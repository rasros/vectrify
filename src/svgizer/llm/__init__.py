from svgizer.llm.anthropic import AnthropicProvider
from svgizer.llm.base import LLMConfig, LLMProvider
from svgizer.llm.gemini import GeminiProvider
from svgizer.llm.openai import OpenAIProvider


def get_provider(provider_name: str, api_key: str | None = None) -> LLMProvider:
    provider_name = provider_name.lower()
    if provider_name == "openai":
        return OpenAIProvider(api_key)
    if provider_name == "anthropic":
        return AnthropicProvider(api_key)
    if provider_name == "gemini":
        return GeminiProvider(api_key)
    raise ValueError(f"Unknown LLM provider: {provider_name}")


__all__ = [
    "LLMConfig",
    "LLMProvider",
    "get_provider",
]
