import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class LLMConfig:
    model: str = "gpt-5.4"
    temperature: float | None = None
    reasoning: str | None = None
    json_output: bool = False
    response_schema: dict[str, Any] | None = None
    schema_name: str = "output"


class LLMClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set.")
        self._client = OpenAI(api_key=self.api_key)

    def generate(self, content_blocks: list[dict[str, Any]], config: LLMConfig) -> str:
        kwargs = {
            "model": config.model,
            "input": [{"role": "user", "content": content_blocks}],
        }

        if config.response_schema:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": config.schema_name,
                    "schema": config.response_schema,
                    "strict": True,
                }
            }
        elif config.json_output:
            kwargs["text"] = {"format": {"type": "json_object"}}
        else:
            kwargs["text"] = {"format": {"type": "text"}}

        if config.reasoning:
            kwargs["reasoning"] = config.reasoning
        elif config.temperature is not None:
            kwargs["temperature"] = config.temperature

        response = self._client.responses.create(**kwargs)
        return response.output_text
