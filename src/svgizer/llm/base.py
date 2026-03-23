import dataclasses
from typing import Any, Protocol


@dataclasses.dataclass
class LLMConfig:
    model: str
    temperature: float | None = None
    reasoning: str | None = None
    json_output: bool = False
    response_schema: dict[str, Any] | None = None
    schema_name: str = "output"


class LLMProvider(Protocol):
    def generate(self, content_blocks: list[dict[str, Any]], config: LLMConfig) -> str:
        """Translates internal content blocks to provider-specific SDK calls."""
        ...
