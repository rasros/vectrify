import dataclasses
from typing import Any, Protocol


def reasoning_budget(reasoning: str | None) -> int:
    if reasoning is None:
        return 8192
    return {"low": 1024, "medium": 8192, "high": 24576}.get(reasoning, 8192)


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
    reasoning: str | None = None


class LLMProvider(Protocol):
    def generate(
        self, content_blocks: list[dict[str, Any]], config: LLMConfig
    ) -> str: ...
