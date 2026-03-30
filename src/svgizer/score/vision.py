import io
import logging
import math
from dataclasses import dataclass, field
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
    patch_embeddings: "torch.Tensor | None" = field(default=None)
    grid_hw: "tuple[int, int] | None" = field(default=None)


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

    def _embed_patches(
        self, image: Image.Image
    ) -> "tuple[torch.Tensor, tuple[int, int]] | None":
        """Return (patch_embeddings, grid_hw) or None if patch access is unavailable.

        patch_embeddings: [num_patches, hidden_size], L2-normalised
        grid_hw: (h_patches, w_patches) spatial layout of patches
        """
        import torch.nn.functional as functional

        if not hasattr(self._model, "vision_model"):
            return None

        inputs = self._processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device_str)

        with self._torch.no_grad():
            output = self._model.vision_model(pixel_values=pixel_values)
            # last_hidden_state: [1, num_patches, hidden_size]
            patch_embs = output.last_hidden_state[0]  # [N, D]
            patch_embs = functional.normalize(patch_embs, dim=-1)

        n = patch_embs.shape[0]
        h = w = math.isqrt(n)
        if h * w != n:
            log.warning(
                "Patch count %d is not a perfect square; heatmap unavailable.", n
            )
            return None

        return patch_embs, (h, w)

    def validate_environment(self) -> None:
        self._load_dependencies()

    def prepare_reference(self, original_rgb: Image.Image) -> VisionReference:
        self._load_dependencies()
        ref_small = resize_long_side(original_rgb, DEFAULT_CONFIG.target_long_side)
        embedding = self._embed(original_rgb)
        patch_result = self._embed_patches(original_rgb)
        patch_embeddings, grid_hw = (
            patch_result if patch_result is not None else (None, None)
        )
        return VisionReference(
            image=ref_small,
            embedding=embedding,
            patch_embeddings=patch_embeddings,
            grid_hw=grid_hw,
        )

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

    def diff_heatmap(
        self,
        reference: VisionReference,
        candidate_png: bytes,
        long_side: int,
    ) -> bytes | None:
        """Generate a perceptual diff heatmap using SigLIP patch embeddings.

        Returns a PNG (hot colormap, black=similar → red → yellow → white=different),
        or None when patch embeddings are unavailable for the loaded model.
        """
        if reference.patch_embeddings is None or reference.grid_hw is None:
            return None

        self._load_dependencies()

        cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")
        patch_result = self._embed_patches(cand)
        if patch_result is None:
            return None
        cand_patch_embs, cand_grid_hw = patch_result

        if cand_grid_hw != reference.grid_hw:
            log.warning(
                "Patch grid mismatch: ref=%s cand=%s; skipping heatmap.",
                reference.grid_hw,
                cand_grid_hw,
            )
            return None

        h, w = reference.grid_hw

        # Per-patch cosine distance: 1 - dot(ref_patch, cand_patch)
        # Both are L2-normalised, so dot product = cosine similarity ∈ [-1, 1]
        cos_sim = (reference.patch_embeddings * cand_patch_embs).sum(dim=-1)  # [N]
        distances = (1.0 - cos_sim).clamp(0.0, 2.0) / 2.0  # normalise to [0, 1]

        grid = distances.cpu().float().numpy().reshape(h, w)

        # Boost contrast the same way generate_diff_data_url does (3x scale)
        grid = np.clip(grid * 3.0, 0.0, 1.0)

        # Upsample patch grid to target pixel resolution via PIL
        patch_img = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
        out_w, out_h = reference.image.size
        if long_side > 0:
            scale = long_side / max(out_w, out_h)
            out_w, out_h = max(1, round(out_w * scale)), max(1, round(out_h * scale))
        upsampled = patch_img.resize((out_w, out_h), resample=Image.Resampling.BILINEAR)
        vals = np.asarray(upsampled).astype(np.float32) / 255.0  # [H, W] in [0, 1]

        heatmap_rgb = _apply_hot_colormap(vals)
        heatmap_img = Image.fromarray(heatmap_rgb, mode="RGB")

        buf = io.BytesIO()
        heatmap_img.save(buf, format="PNG")
        return buf.getvalue()


def _apply_hot_colormap(values: np.ndarray) -> np.ndarray:
    """Map a [0, 1] float array to the hot colormap as uint8 RGB.

    Breakpoints: 0.0 → black, 1/3 → red, 2/3 → yellow, 1.0 → white.
    """
    r = np.clip(values * 3.0, 0.0, 1.0)
    g = np.clip(values * 3.0 - 1.0, 0.0, 1.0)
    b = np.clip(values * 3.0 - 2.0, 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
