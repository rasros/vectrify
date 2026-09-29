from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletion

from vectrify.llm.base import LLMConfig, LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str | None = None):
        self._client = OpenAI(api_key=api_key, base_url=base_url)

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
        if config.reasoning:
            kwargs["reasoning_effort"] = config.reasoning

        # The SDK overloads on the `stream` literal to decide between a
        # ChatCompletion and a Stream, but building the arguments as a dict
        # hides that from the type checker. Streaming is never enabled here.
        response = cast(ChatCompletion, self._client.chat.completions.create(**kwargs))
        return response.choices[0].message.content or ""
