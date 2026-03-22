import io
import json
import logging
import os
from dataclasses import dataclass

from openai import OpenAI
from PIL import Image

from .base import DiffScorer
from svgizer.image_utils import png_bytes_to_data_url

log = logging.getLogger(__name__)


@dataclass
class LLMReference:
    data_url: str


class LLMJudgeScorer:
    """Uses a Vision LLM to score the visual difference between two images."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set to use LLMJudgeScorer")

        self.client = OpenAI(api_key=api_key)

    def prepare_reference(self, original_rgb: Image.Image) -> LLMReference:
        """Pre-processes the PIL Image into a base64 Data URL for the LLM."""
        buf = io.BytesIO()
        original_rgb.save(buf, format="PNG")
        data_url = png_bytes_to_data_url(buf.getvalue())
        return LLMReference(data_url=data_url)

    def score(self, reference: LLMReference, candidate_png: bytes) -> float:
        """Asks the LLM to score similarity, returning a difference score (0.0 to 1.0)."""
        candidate_data_url = png_bytes_to_data_url(candidate_png)

        prompt = (
            "You are an expert computer vision judge evaluating image generation quality. "
            "Compare the original reference image (first) to the generated candidate image (second). "
            "Rate how visually and structurally similar the candidate is to the reference on a scale of 0.0 to 1.0. "
            "1.0 means perfectly identical in shape, layout, and color. 0.0 means completely different. "
            "Respond strictly with a JSON object containing a single key 'similarity' mapped to your float value."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": reference.data_url},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": candidate_data_url},
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=50,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            result = json.loads(content)
            similarity = float(result.get("similarity", 0.0))

            # The DiffScorer protocol expects 0.0 for identical and 1.0 for completely different
            distance = 1.0 - similarity
            return float(max(0.0, min(1.0, distance)))

        except Exception as e:
            log.error(f"LLM Judge scoring failed: {e}")
            # Return the worst possible score so the pipeline doesn't crash but rejects the candidate
            return 1.0


_: DiffScorer = LLMJudgeScorer()
