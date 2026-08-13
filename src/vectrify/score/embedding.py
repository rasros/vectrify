"""The round's scorer: cosine distance under a small vision encoder.

The round used to score with hand-built pixel measures because the evaluator is
~300x their cost. A small encoder sits between the two: it ranks candidates the
way the evaluator does at rho 0.84, and where selection actually decides -- one
mutation away from the parent -- it calls 54% of its accepted mutations right
against the pixel blend's 42%. That difference is the search's whole margin,
since roughly one mutation in eight genuinely improves the drawing and a proxy
that is wrong more often than not spends the run undoing them.
"""

import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image

from vectrify.score.base import Scorer, safe_score
from vectrify.score.utils import MAX_SCORE, clamp01, get_device

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "facebook/dinov2-small"


@dataclass
class EmbeddingReference:
    image: Image.Image
    embedding: "torch.Tensor"


class EmbeddingScorer(Scorer):
    def __init__(
        self, model_name: str = DEFAULT_EMBED_MODEL, device: str | None = None
    ):
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device_str: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                f"torch or transformers is not installed: {exc}. "
                "Run 'pip install transformers torch'."
            ) from exc

        device = self._device or get_device()
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).eval().to(device)
        self._torch = torch
        self._device_str = device

    def _embed(self, image: Image.Image) -> "torch.Tensor":
        import torch.nn.functional as functional

        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device_str)
        autocast = self._torch.autocast(
            self._device_str,
            dtype=self._torch.float16,
            enabled=self._device_str == "cuda",
        )
        with self._torch.no_grad(), autocast:
            result = self._model(pixel_values=pixel_values)
        features = getattr(result, "pooler_output", None)
        if features is None:
            features = result.last_hidden_state.mean(dim=1)
        return functional.normalize(features.float(), dim=-1)

    def validate_environment(self) -> None:
        self._load()

    def prepare_reference(self, original_rgb: Image.Image) -> EmbeddingReference:
        self._load()
        return EmbeddingReference(
            image=original_rgb, embedding=self._embed(original_rgb)
        )

    @safe_score
    def score(self, reference: EmbeddingReference, candidate_png: bytes) -> float:
        self._load()
        candidate = Image.open(io.BytesIO(candidate_png)).convert("RGB")
        similarity = float((reference.embedding * self._embed(candidate)).sum().item())
        value = clamp01(1.0 - similarity)
        return value if value == value else MAX_SCORE
