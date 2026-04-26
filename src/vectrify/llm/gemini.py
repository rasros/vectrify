import base64
import os
from typing import Any

from google import genai
from google.genai import types

from vectrify.llm.base import LLMConfig, LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set.")
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, content_blocks: list[dict[str, Any]], config: LLMConfig) -> str:
        prompt_parts = []

        for block in content_blocks:
            if block["type"] == "input_text":
                prompt_parts.append(block["text"])
            elif block["type"] == "input_image":
                try:
                    header, encoded = block["image_url"].split(",", 1)
                    mime_type = header.split(";")[0].split(":")[1]
                except (ValueError, IndexError) as e:
                    raise ValueError(
                        f"Malformed image data URL: {block['image_url'][:50]!r}"
                    ) from e
                prompt_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(encoded), mime_type=mime_type
                    )
                )

        generation_config = types.GenerateContentConfig(temperature=config.temperature)

        if config.reasoning:
            budget_map = {"low": 1024, "medium": 8192, "high": 24576}
            budget = budget_map.get(config.reasoning, 8192)
            generation_config.thinking_config = types.ThinkingConfig(
                thinking_budget=budget
            )

        if config.response_schema or config.json_output:
            generation_config.response_mime_type = "application/json"
            if config.response_schema:
                generation_config.response_schema = config.response_schema

        response = self.client.models.generate_content(
            model=config.model, contents=prompt_parts, config=generation_config
        )
        return response.text or ""
