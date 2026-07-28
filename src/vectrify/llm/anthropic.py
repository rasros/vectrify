from typing import Any, cast

import anthropic
from anthropic.types import Message

from vectrify.llm.base import (
    LLMConfig,
    LLMProvider,
    resolve_api_key,
    split_data_url,
)


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = resolve_api_key("anthropic", api_key)
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, content_blocks: list[dict[str, Any]], config: LLMConfig) -> str:
        messages_content = []
        for block in content_blocks:
            if block["type"] == "input_text":
                messages_content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "input_image":
                mime_type, encoded = split_data_url(block["image_url"])
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
            "temperature": config.temperature or 1.0,
            "messages": [{"role": "user", "content": messages_content}],
        }

        if config.reasoning:
            budget_map = {"low": 1024, "medium": 8192, "high": 24576}
            budget = budget_map.get(config.reasoning, 8192)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # max_tokens must exceed the thinking budget, and the API requires
            # temperature 1 when extended thinking is enabled.
            kwargs["max_tokens"] = budget + 8192
            kwargs["temperature"] = 1.0

        # Add system prompt dynamically to satisfy the type checker
        if config.response_schema:
            kwargs["system"] = "You must respond with valid JSON."

        # Same overload-narrowing problem as the OpenAI provider: the arguments
        # are built as a dict, so the checker cannot tell this is the
        # non-streaming form. Streaming is never enabled here.
        message = cast(Message, self._client.messages.create(**kwargs))

        return "".join(block.text for block in message.content if block.type == "text")
