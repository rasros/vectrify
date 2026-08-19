import contextlib
import hashlib
import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from vectrify.image_utils import resize_long_side
from vectrify.score.base import DEFAULT_CONFIG, Scorer
from vectrify.score.utils import MAX_SCORE, clamp01, color_score, get_device

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

# Only used by --scorer vision, which pits a single encoder against the field
# where the default panel votes. SigLIP-so400m held this slot because the
# alternatives were screened by correlating them against SigLIP, which made it
# the winner by construction; on MIEB's NIGHTS task, built on human similarity
# judgements, it places near the bottom of 55 models where dinov2-small places
# fifth.
DEFAULT_VISION_MODEL = "facebook/dinov2-small"


@dataclass
class VisionReference:
    image: Image.Image
    # Embedded once per run: the reference never changes, so only the candidate
    # side costs anything.
    embedding: "torch.Tensor"


class VisionScorer(Scorer):
    def __init__(
        self,
        model_name: str = DEFAULT_VISION_MODEL,
        device: str | None = None,
    ):
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device_str: str | None = None
        # Renders repeat: a front is re-evaluated across epochs and mutation
        # produces byte-identical candidates often enough to matter, and a
        # forward is by far the most expensive thing in a run.
        self._embedding_memo: tuple[bytes, Any] | None = None

    @property
    def comparability(self) -> str:
        return f"vision/{self._model_name.split('/')[-1]}"

    def _embed_cached(self, image: Image.Image) -> "torch.Tensor":
        """Whole-image embedding, memoised on the pixels."""
        key = hashlib.blake2b(image.tobytes(), digest_size=16).digest()
        memo = self._embedding_memo
        if memo is not None and memo[0] == key:
            return memo[1]
        embedding = self._embed(image)
        self._embedding_memo = (key, embedding)
        return embedding

    def _inference(self):
        """Run forwards in half precision on CUDA.

        The scorer's whole output is a cosine distance, and fp16 and fp32
        embeddings of the same image agree to a cosine of 0.999998, so the
        precision is free here while the tower is ~3x faster.
        """
        if self._device_str == "cuda":
            return self._torch.autocast("cuda", dtype=self._torch.float16)
        return contextlib.nullcontext()

    def _load_dependencies(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            # Free on Ampere and later, and it covers the paths autocast does
            # not (CPU offload, older torch without CUDA autocast).
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            device = self._device or get_device()
            processor = AutoImageProcessor.from_pretrained(self._model_name)
            model = AutoModel.from_pretrained(self._model_name)
            model = model.to(device)
            model.eval()

            self._processor = processor
            self._model = model
            self._torch = torch
            self._device_str = device
        except ImportError as e:
            raise ImportError(
                f"transformers or torch not installed or failed to load: {e}. "
                "Run 'pip install transformers torch'."
            ) from e

    def _embed(self, image: Image.Image) -> "torch.Tensor":
        import torch.nn.functional as functional

        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device_str)

        with self._torch.no_grad(), self._inference():
            if hasattr(self._model, "get_image_features"):
                features = self._model.get_image_features(pixel_values=pixel_values)
            else:
                features = self._model(pixel_values=pixel_values)

            # Unwrap dataclass outputs (e.g. BaseModelOutputWithPooling)
            if not isinstance(features, self._torch.Tensor):
                features = features.pooler_output

            return functional.normalize(features.float(), dim=-1)

    def validate_environment(self) -> None:
        self._load_dependencies()

    def prepare_reference(self, original_rgb: Image.Image) -> VisionReference:
        self._load_dependencies()
        return VisionReference(
            image=resize_long_side(original_rgb, DEFAULT_CONFIG.target_long_side),
            embedding=self._embed(original_rgb),
        )

    def score(self, reference: VisionReference, candidate_png: bytes) -> float:
        self._load_dependencies()

        cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")

        # Dot product of L2-normalised vectors = cosine similarity.
        cand_embedding = self._embed_cached(cand)
        cos_sim = float((reference.embedding * cand_embedding).sum().item())
        struct_score = clamp01(1.0 - cos_sim)

        color = color_score(reference.image, candidate_png)

        score = (DEFAULT_CONFIG.w_vision * struct_score) + (
            DEFAULT_CONFIG.w_color * color
        )

        if not np.isfinite(score):
            return MAX_SCORE
        return clamp01(score)
