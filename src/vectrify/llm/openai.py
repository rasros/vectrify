import os
from typing import Any

from openai import OpenAI

from vectrify.llm.base import LLMConfig, LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set.")
        self._client = OpenAI(api_key=self.api_key)

    def generate(self, content_blocks: list[dict[str, Any]], config: LLMConfig) -> str:
        openai_content = []
        for block in content_blocks:
            if block["type"] == "input_text":
                openai_content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "input_image":
                openai_content.append(
                    {"type": "image_url", "image_url": {"url": block["image_url"]}}
                )
            else:
                openai_content.append(block)

        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": [{"role": "user", "content": openai_content}],
        }

        if config.response_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": config.schema_name,
                    "schema": config.response_schema,
                    "strict": True,
                },
            }
        elif config.json_output:
            kwargs["response_format"] = {"type": "json_object"}

        if config.reasoning:
            kwargs["reasoning_effort"] = config.reasoning
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
