import pytest

from svgizer.llm import LLMConfig, get_provider


@pytest.mark.llm
def test_llm_client_basic_generation():
    client = get_provider("openai")
    config = LLMConfig(model="gpt-4o-mini", temperature=0.0)

    content_blocks = [
        {
            "type": "input_text",
            "text": "Respond with the exact word 'acknowledged' and nothing else.",
        }
    ]

    response = client.generate(content_blocks, config)

    assert response is not None
    assert "acknowledged" in response.lower()