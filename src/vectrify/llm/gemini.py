import base64
from typing import Any

from google import genai
from google.genai import types

from vectrify.llm.base import LLMConfig, LLMProvider, reasoning_budget, split_data_url


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate(self, content_blocks: list[dict[str, Any]], config: LLMConfig) -> str:
        prompt_parts = []

        for block in content_blocks:
            if block["type"] == "input_text":
                prompt_parts.append(block["text"])
            elif block["type"] == "input_image":
                mime_type, encoded = split_data_url(block["image_url"])
                prompt_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(encoded), mime_type=mime_type
                    )
                )

        generation_config = types.GenerateContentConfig()

        if config.reasoning:
            budget = reasoning_budget(config.reasoning)
            generation_config.thinking_config = types.ThinkingConfig(
                thinking_budget=budget
            )

        response = self.client.models.generate_content(
            model=config.model, contents=prompt_parts, config=generation_config
        )
        return response.text or ""
