import os
from typing import Any

import anthropic

from .base import LLMConfig, LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set.")
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, content_blocks: list[dict[str, Any]], config: LLMConfig) -> str:
        messages_content = []
        for block in content_blocks:
            if block["type"] == "input_text":
                messages_content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "input_image":
                header, encoded = block["image_url"].split(",", 1)
                mime_type = header.split(";")[0].split(":")[1]
                messages_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": encoded,
                        },
                    }
                )

        kwargs: dict[str, Any] = {
            "model": config.model,
            "max_tokens": 8192,
            "temperature": config.temperature or 0.6,
            "messages": [{"role": "user", "content": messages_content}],
        }

        # Add system prompt dynamically to satisfy the type checker
        if config.response_schema or config.json_output:
            kwargs["system"] = "You must respond with valid JSON."

        message = self._client.messages.create(**kwargs)

        return "".join(block.text for block in message.content if block.type == "text")
