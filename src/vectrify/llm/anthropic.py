from typing import Any, cast

import anthropic
from anthropic.types import Message

from vectrify.llm.base import LLMConfig, LLMProvider, reasoning_budget, split_data_url


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str):
        self._client = anthropic.Anthropic(api_key=api_key)

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
            # The API requires temperature 1 when extended thinking is enabled.
            "temperature": 1.0,
            "messages": [{"role": "user", "content": messages_content}],
        }

        if config.reasoning:
            budget = reasoning_budget(config.reasoning)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # max_tokens must exceed the thinking budget.
            kwargs["max_tokens"] = budget + 8192

        # Same overload-narrowing problem as the OpenAI provider: the arguments
        # are built as a dict, so the checker cannot tell this is the
        # non-streaming form. Streaming is never enabled here.
        message = cast(Message, self._client.messages.create(**kwargs))

        return "".join(block.text for block in message.content if block.type == "text")
