import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from svgizer.image_utils import resize_long_side
from svgizer.score.base import DEFAULT_CONFIG, Scorer
from svgizer.score.utils import get_device, lab_l1

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = "google/siglip-so400m-patch14-384"


@dataclass
class VisionReference:
    image: Image.Image
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

    def _load_dependencies(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

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

        with self._torch.no_grad():
            if hasattr(self._model, "get_image_features"):
                features = self._model.get_image_features(pixel_values=pixel_values)
            else:
                features = self._model(pixel_values=pixel_values)

            # Unwrap dataclass outputs (e.g. BaseModelOutputWithPooling)
            if not isinstance(features, self._torch.Tensor):
                features = features.pooler_output

            return functional.normalize(features, dim=-1)

    def validate_environment(self) -> None:
        self._load_dependencies()

    def prepare_reference(self, original_rgb: Image.Image) -> VisionReference:
        self._load_dependencies()
        ref_small = resize_long_side(original_rgb, DEFAULT_CONFIG.target_long_side)
        embedding = self._embed(original_rgb)
        return VisionReference(image=ref_small, embedding=embedding)

    def score(self, reference: VisionReference, candidate_png: bytes) -> float:
        self._load_dependencies()

        cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")
        cand_embedding = self._embed(cand)

        # Dot product of L2-normalised vectors = cosine similarity
        cos_sim = float((reference.embedding * cand_embedding).sum().item())
        struct_score = max(0.0, min(1.0, 1.0 - cos_sim))

        cand_small = cand.resize(
            reference.image.size, resample=Image.Resampling.BILINEAR
        )
        color_score = float(max(0.0, min(1.0, lab_l1(reference.image, cand_small))))

        score = (DEFAULT_CONFIG.w_vision * struct_score) + (
            DEFAULT_CONFIG.w_color * color_score
        )

        if not np.isfinite(score):
            return 1.0
        return float(max(0.0, min(1.0, score)))
