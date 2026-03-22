import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from PIL import Image

from .base import DiffScorer
from svgizer.image_utils import png_bytes_to_data_url

log = logging.getLogger(__name__)

@dataclass
class LLMReference:
    data_url: str

class LLMJudgeScorer(DiffScorer):
    """Uses a Vision LLM to score the visual difference between two images."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set to use LLMJudgeScorer")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def prepare_reference(self, original_rgb: Image.Image) -> LLMReference:
        buf = io.BytesIO()
        original_rgb.save(buf, format="PNG")
        data_url = png_bytes_to_data_url(buf.getvalue())
        return LLMReference(data_url=data_url)

    def score(self, reference: LLMReference, candidate_png: bytes) -> float:
        candidate_data_url = png_bytes_to_data_url(candidate_png)

        prompt = (
            "You are an expert computer vision judge. Compare the reference (1st) "
            "to the candidate (2nd). Rate similarity from 0.0 to 1.0 (1.0 is identical). "
            "Respond strictly with JSON: {'similarity': float}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": reference.data_url}},
                            {"type": "image_url", "image_url": {"url": candidate_data_url}},
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=50,
            )

            result = json.loads(response.choices[0].message.content or "{}")
            similarity = float(result.get("similarity", 0.0))
            return float(max(0.0, min(1.0, 1.0 - similarity)))

        except Exception as e:
            log.error(f"LLM Judge scoring failed: {e}")
            return 1.0