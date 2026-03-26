import io
import json
import logging
from dataclasses import dataclass
from typing import Any

from PIL import Image
from PIL.Image import Resampling

from svgizer.image_utils import png_bytes_to_data_url
from svgizer.llm import LLMConfig, get_provider
from svgizer.score.base import Scorer
from svgizer.score.utils import lab_l1

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
        config: LLMConfig | None = None,
        api_key: str | None = None,
    ):
        self.provider_name = provider_name
        self.config = config or LLMConfig(
            model="gpt-5.4-chat",
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

    def score(self, reference: LLMReference, candidate_png: bytes) -> float:
        candidate_data_url = png_bytes_to_data_url(candidate_png)
        content_blocks = _build_judge_prompt(reference.data_url, candidate_data_url)

        try:
            response_text = self.client.generate(content_blocks, self.config)

            result = json.loads(response_text)
            similarity = float(result["similarity"])
            llm_score = float(max(0.0, min(1.0, 1.0 - similarity)))

            cand_img = Image.open(io.BytesIO(candidate_png)).convert("RGB")
            if cand_img.size != reference.image.size:
                cand_img = cand_img.resize(
                    reference.image.size, resample=Resampling.BILINEAR
                )

            color_score = float(max(0.0, min(1.0, lab_l1(reference.image, cand_img))))
            final_score = ((1.0 - TIE_BREAKER_WEIGHT) * llm_score) + (
                TIE_BREAKER_WEIGHT * color_score
            )

            return float(max(0.0, min(1.0, final_score)))

        except Exception as e:
            log.error(f"LLM Judge scoring failed: {e}")
            return 1.0
