import base64
import os
from typing import Any

import google.generativeai as genai

from svgizer.llm.base import LLMConfig, LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set.")
        genai.configure(api_key=self.api_key)

    def generate(self, content_blocks: list[dict[str, Any]], config: LLMConfig) -> str:
        model = genai.GenerativeModel(config.model)
        prompt_parts = []

        for block in content_blocks:
            if block["type"] == "input_text":
                prompt_parts.append(block["text"])
            elif block["type"] == "input_image":
                header, encoded = block["image_url"].split(",", 1)
                mime_type = header.split(";")[0].split(":")[1]
                prompt_parts.append(
                    {"mime_type": mime_type, "data": base64.b64decode(encoded)}
                )

        # Set up generation config
        generation_config = genai.types.GenerationConfig(temperature=config.temperature)

        if config.response_schema or config.json_output:
            generation_config.response_mime_type = "application/json"
            if config.response_schema:
                generation_config.response_schema = config.response_schema

        response = model.generate_content(
            prompt_parts, generation_config=generation_config
        )
        return response.text
