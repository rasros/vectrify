import pytest

from svgizer.llm import LLMClient, LLMConfig


@pytest.mark.llm
def test_llm_client_basic_generation():
    client = LLMClient()
    config = LLMConfig(model="gpt-5.4-nano", temperature=0.0)

    content_blocks = [
        {
            "type": "input_text",
            "text": "Respond with the exact word 'acknowledged' and nothing else.",
        }
    ]

    response = client.generate(content_blocks, config)

    assert response is not None
    assert "acknowledged" in response.lower()
