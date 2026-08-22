import dataclasses
import os
from typing import Any, Protocol

from vectrify.llm.models import api_key_env


def reasoning_budget(reasoning: str | None) -> int:
    if reasoning is None:
        return 8192
    return {"low": 1024, "medium": 8192, "high": 24576}.get(reasoning, 8192)


def resolve_api_key(provider: str, api_key: str | None = None) -> str:
    """Return the explicit key, else the provider's env var. Raises if unset."""
    env_var = api_key_env(provider)
    key = api_key or os.getenv(env_var)
    if not key:
        raise ValueError(f"{env_var} must be set.")
    return key


def split_data_url(url: str) -> tuple[str, str]:
    """Split a data URL into (mime_type, base64_payload)."""
    try:
        header, encoded = url.split(",", 1)
        mime_type = header.split(";")[0].split(":")[1]
    except (ValueError, IndexError) as e:
        raise ValueError(f"Malformed image data URL: {url[:50]!r}") from e
    return mime_type, encoded


@dataclasses.dataclass
class LLMConfig:
    model: str
    temperature: float | None = None
    reasoning: str | None = None
    response_schema: dict[str, Any] | None = None
    schema_name: str = "output"


class LLMProvider(Protocol):
    def generate(
        self, content_blocks: list[dict[str, Any]], config: LLMConfig
    ) -> str: ...
