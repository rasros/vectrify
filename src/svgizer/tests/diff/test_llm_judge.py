import io
import json

import pytest
from PIL import Image

from svgizer.diff.llm_judge import LLMJudgeScorer
from svgizer.llm import LLMClient, LLMConfig


@pytest.mark.llm
def test_llm_client_structured_output():
    client = LLMClient()

    test_schema = {
        "type": "object",
        "properties": {"magic_number": {"type": "number"}},
        "required": ["magic_number"],
        "additionalProperties": False,
    }

    config = LLMConfig(
        model="gpt-5.4-nano",
        response_schema=test_schema,
        schema_name="magic_number_test",
        temperature=0.0,
    )

    content_blocks = [
        {"type": "input_text", "text": "The magic number is 42. Output it."}
    ]

    response = client.generate(content_blocks, config)
    data = json.loads(response)

    assert "magic_number" in data
    assert data["magic_number"] == 42


@pytest.mark.llm
def test_llm_judge_scorer_identical_images():
    scorer = LLMJudgeScorer()

    img = Image.new("RGB", (64, 64), color="red")
    ref = scorer.prepare_reference(img)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    cand_bytes = buf.getvalue()

    score = scorer.score(ref, cand_bytes)

    assert 0.0 <= score < 0.05


@pytest.mark.llm
def test_llm_judge_scorer_different_images():
    scorer = LLMJudgeScorer()

    img_red = Image.new("RGB", (64, 64), color="red")
    ref = scorer.prepare_reference(img_red)

    img_blue = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    img_blue.save(buf, format="PNG")
    cand_bytes = buf.getvalue()

    score = scorer.score(ref, cand_bytes)

    assert score > 0.5
    assert score <= 1.0
