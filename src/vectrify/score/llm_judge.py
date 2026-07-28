import io
import json
import logging
from dataclasses import dataclass
from typing import Any

from PIL import Image

from vectrify.image_utils import png_bytes_to_data_url
from vectrify.llm import LLMConfig, get_provider
from vectrify.llm.models import JUDGE_MODELS
from vectrify.score.base import Scorer, safe_score
from vectrify.score.utils import clamp01, color_score

log = logging.getLogger(__name__)
TIE_BREAKER_WEIGHT = 0.01


@dataclass
class LLMReference:
    data_url: str
    image: Image.Image


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "similarity": {
            "type": "number",
            "description": "Visual similarity score from 0.0 to 1.0",
        }
    },
    "required": ["similarity"],
    "additionalProperties": False,
}


def _build_judge_prompt(reference_url: str, candidate_url: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "input_text",
            "text": (
                "You are an expert computer vision judge. Compare the reference (1st) "
                "to the candidate (2nd). Rate similarity from 0.0 to 1.0 "
                "(1.0 is identical)."
            ),
        },
        {"type": "input_image", "image_url": reference_url},
        {"type": "input_image", "image_url": candidate_url},
    ]


class LLMJudgeScorer(Scorer):
    def __init__(
        self,
        provider_name: str = "openai",
        api_key: str | None = None,
    ):
        self.provider_name = provider_name
        judge_model = JUDGE_MODELS.get(provider_name, JUDGE_MODELS["openai"])
        self.config = LLMConfig(
            model=judge_model,
            temperature=0.0,
            response_schema=JUDGE_SCHEMA,
            schema_name="similarity_score",
        )
        self.client = get_provider(self.provider_name, api_key)

    def prepare_reference(self, original_rgb: Image.Image) -> LLMReference:
        buf = io.BytesIO()
        original_rgb.save(buf, format="PNG")
        data_url = png_bytes_to_data_url(buf.getvalue())
        return LLMReference(data_url=data_url, image=original_rgb)

    @safe_score
    def score(self, reference: LLMReference, candidate_png: bytes) -> float:
        candidate_data_url = png_bytes_to_data_url(candidate_png)
        content_blocks = _build_judge_prompt(reference.data_url, candidate_data_url)

        response_text = self.client.generate(content_blocks, self.config)

        result = json.loads(response_text)
        similarity = float(result["similarity"])
        llm_score = clamp01(1.0 - similarity)

        tie_breaker = color_score(reference.image, candidate_png)
        final_score = ((1.0 - TIE_BREAKER_WEIGHT) * llm_score) + (
            TIE_BREAKER_WEIGHT * tie_breaker
        )
        return clamp01(final_score)
