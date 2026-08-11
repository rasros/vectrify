import contextlib
import hashlib
import io
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from vectrify.image_utils import resize_long_side
from vectrify.score.base import DEFAULT_CONFIG, Scorer
from vectrify.score.regions import crop_tile, tile_boxes, tile_key
from vectrify.score.utils import MAX_SCORE, clamp01, color_score, get_device

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
    # Reference tiles are embedded once per run: they never change, so only the
    # candidate side costs anything per candidate.
    tile_embeddings: "torch.Tensor | None" = field(default=None)
    tile_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    full_size: tuple[int, int] = (0, 0)


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
        # tile key -> distance. Unbounded on purpose: an entry is 16 bytes of
        # key plus a float, and candidates repeat heavily (measured ~69% of
        # tiles already seen, and ~25% of whole renders identical to an earlier
        # one), so evicting would throw away the hit rate that pays for tiling.
        self._tile_cache: dict[bytes, float] = {}
        # Last single-image forward, as (key, pooled, patches). score() wants
        # the pooled vector and region_distance_grid() the patch grid, and at
        # one crop both run on the same pixels -- but the pooled vector is
        # derived from the patches, so the second pass was recomputing what the
        # first already had. Measured 1248 ms per candidate against 614 ms for
        # one forward.
        self._forward_memo: tuple[bytes, Any, Any] | None = None

    def _forward_single(
        self, image: Image.Image
    ) -> "tuple[torch.Tensor, torch.Tensor] | None":
        """One pass returning (pooled, patches), both L2-normalised."""
        import torch.nn.functional as functional

        if not hasattr(self._model, "vision_model"):
            return None

        key = hashlib.blake2b(image.tobytes(), digest_size=16).digest()
        memo = self._forward_memo
        if memo is not None and memo[0] == key:
            return memo[1], memo[2]

        inputs = self._processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device_str)
        with self._torch.no_grad(), self._inference():
            output = self._model.vision_model(pixel_values=pixel_values)
            pooled = functional.normalize(output.pooler_output.float(), dim=-1)
            patches = functional.normalize(output.last_hidden_state[0].float(), dim=-1)

        self._forward_memo = (key, pooled, patches)
        return pooled, patches

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

    def _input_size(self) -> int:
        """The model's expected input edge, read from the loaded processor."""
        size = getattr(self._processor, "size", None) or {}
        if isinstance(size, dict):
            edge = size.get("height") or size.get("width") or size.get("shortest_edge")
            if edge:
                return int(edge)
        return 384

    def _embed_many(self, images: list[Image.Image]) -> "torch.Tensor":
        """Embed a batch of images in one forward pass, L2-normalised."""
        import torch.nn.functional as functional

        if len(images) == 1:
            single = self._forward_single(images[0])
            if single is not None:
                return single[0]

        inputs = self._processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device_str)

        with self._torch.no_grad(), self._inference():
            if hasattr(self._model, "get_image_features"):
                features = self._model.get_image_features(pixel_values=pixel_values)
            else:
                features = self._model(pixel_values=pixel_values)
            if not isinstance(features, self._torch.Tensor):
                features = features.pooler_output
            return functional.normalize(features.float(), dim=-1)

    def _tile_distances(
        self, reference: VisionReference, candidate: Image.Image
    ) -> np.ndarray | None:
        """Cosine distance per tile, computing only the tiles not already known.

        The cache is what makes tiling affordable: a tile's distance depends
        only on its own pixels, and candidates share tiles heavily -- both with
        their parents (mutations are local) and with unrelated candidates
        (blank areas hash identically).
        """
        if reference.tile_embeddings is None or not reference.tile_boxes:
            return None

        if candidate.size != reference.full_size:
            candidate = candidate.resize(
                reference.full_size, resample=Image.Resampling.BILINEAR
            )

        distances: list[float | None] = []
        pending: list[Image.Image] = []
        pending_at: list[int] = []
        keys: list[bytes] = []

        for i, box in enumerate(reference.tile_boxes):
            tile = crop_tile(candidate, box)
            key = tile_key(i, tile)
            keys.append(key)
            hit = self._tile_cache.get(key)
            distances.append(hit)
            if hit is None:
                pending.append(tile)
                pending_at.append(i)

        if pending:
            embs = self._embed_many(pending)
            ref = reference.tile_embeddings[pending_at]
            cos = (ref * embs).sum(dim=-1)
            fresh = ((1.0 - cos).clamp(0.0, 2.0) / 2.0).cpu().float().numpy()
            for slot, value in zip(pending_at, fresh, strict=True):
                distances[slot] = float(value)
                self._tile_cache[keys[slot]] = float(value)

        return np.array(distances, dtype=np.float64)

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

    def _embed_patches(
        self, image: Image.Image
    ) -> "tuple[torch.Tensor, tuple[int, int]] | None":
        """Return (patch_embeddings, grid_hw) or None if patch access is unavailable.

        patch_embeddings: [num_patches, hidden_size], L2-normalised
        grid_hw: (h_patches, w_patches) spatial layout of patches
        """
        single = self._forward_single(image)
        if single is None:
            return None
        patch_embs = single[1]

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
        tile_size = DEFAULT_CONFIG.tile_size or self._input_size()
        boxes = tile_boxes(original_rgb.size, tile_size, DEFAULT_CONFIG.tile_overlap)
        log.info(
            "Scoring on %d crop(s) of %dpx at native resolution.",
            len(boxes),
            tile_size,
        )
        # Coverage stays uniform either way -- the last crop on an axis
        # overhangs and is padded -- but a raster that is not a whole number of
        # crops spends part of every edge pass on padding rather than image.
        real = sum(
            (min(x1, original_rgb.width) - x0) * (min(y1, original_rgb.height) - y0)
            for x0, y0, x1, y1 in boxes
        )
        useful = real / (len(boxes) * tile_size * tile_size)
        if useful < 0.95:
            log.warning(
                "Raster %dx%d is not a whole number of %dpx crops: %.0f%% of each "
                "scoring pass is padding. Set resolution to a multiple of %d.",
                original_rgb.width,
                original_rgb.height,
                tile_size,
                100 * (1 - useful),
                tile_size,
            )
        tile_embeddings = (
            self._embed_many([crop_tile(original_rgb, b) for b in boxes])
            if boxes
            else None
        )

        return VisionReference(
            image=ref_small,
            embedding=embedding,
            patch_embeddings=patch_embeddings,
            grid_hw=grid_hw,
            tile_embeddings=tile_embeddings,
            tile_boxes=boxes,
            full_size=original_rgb.size,
        )

    def score(self, reference: VisionReference, candidate_png: bytes) -> float:
        self._load_dependencies()

        cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")

        tiles = self._tile_distances(reference, cand)
        if tiles is not None and tiles.size:
            # Worst-first over the configured share. A whole-image cosine is
            # just the tiles=1 case of this, so nothing special-cases it.
            k = max(1, round(tiles.size * DEFAULT_CONFIG.score_tile_fraction))
            struct_score = clamp01(float(np.partition(tiles, -k)[-k:].mean()))
        else:
            cand_embedding = self._embed(cand)
            # Dot product of L2-normalised vectors = cosine similarity
            cos_sim = float((reference.embedding * cand_embedding).sum().item())
            struct_score = clamp01(1.0 - cos_sim)

        color = color_score(reference.image, candidate_png)

        score = (DEFAULT_CONFIG.w_vision * struct_score) + (
            DEFAULT_CONFIG.w_color * color
        )

        if not np.isfinite(score):
            return MAX_SCORE
        return clamp01(score)

    def region_distance_grid(
        self, reference: VisionReference, candidate_png: bytes
    ) -> np.ndarray | None:
        """Per-region distances, from the same tiles the score is built on.

        Deliberately not a second tiling. The regions the objective points at
        are exactly the crops the scorer measured, so "the worst region" names
        something the score actually saw. With more than one tile these come
        free from the score's own cached distances; with ``tiles=1`` there is
        no spatial information in them, so it falls back to the finer SigLIP
        patch grid rather than reporting a single number as a "region".
        """
        if len(reference.tile_boxes) > 1:
            cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")
            tiles = self._tile_distances(reference, cand)
            if tiles is not None and tiles.size:
                # Square where the geometry allows it, so the heatmap can
                # upsample the grid back over the canvas it came from.
                side = math.isqrt(tiles.size)
                return tiles.reshape(side, side) if side * side == tiles.size else tiles

        if reference.patch_embeddings is None or reference.grid_hw is None:
            return super().region_distance_grid(reference, candidate_png)

        self._load_dependencies()

        cand = Image.open(io.BytesIO(candidate_png)).convert("RGB")
        patch_result = self._embed_patches(cand)
        if patch_result is None:
            return super().region_distance_grid(reference, candidate_png)
        cand_patch_embs, cand_grid_hw = patch_result

        if cand_grid_hw != reference.grid_hw:
            log.warning(
                "Patch grid mismatch: ref=%s cand=%s; using block distances.",
                reference.grid_hw,
                cand_grid_hw,
            )
            return super().region_distance_grid(reference, candidate_png)

        h, w = reference.grid_hw

        # Per-patch cosine distance normalised to [0, 1]
        cos_sim = (reference.patch_embeddings * cand_patch_embs).sum(dim=-1)
        distances = (1.0 - cos_sim).clamp(0.0, 2.0) / 2.0

        return distances.cpu().float().numpy().reshape(h, w)

    def diff_heatmap(
        self,
        reference: VisionReference,
        candidate_png: bytes,
        long_side: int,
        grid: np.ndarray | None = None,
    ) -> bytes | None:
        """Generate a perceptual diff heatmap using SigLIP patch embeddings.

        Returns a PNG (hot colormap, black=similar → red → yellow → white=different),
        or None when patch distances are unavailable for the loaded model.

        *grid* lets the caller pass distances it already computed for the
        ``worst_region`` metric. Recomputing them here would mean a second
        vision forward pass per candidate purely to draw the same numbers.
        """
        if grid is None:
            grid = self.region_distance_grid(reference, candidate_png)
        if grid is None:
            return None

        # Boost contrast (3x scale)
        grid = np.clip(grid * 3.0, 0.0, 1.0)

        patch_img = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
        out_w, out_h = reference.image.size
        if long_side > 0:
            scale = long_side / max(out_w, out_h)
            out_w, out_h = max(1, round(out_w * scale)), max(1, round(out_h * scale))
        upsampled = patch_img.resize((out_w, out_h), resample=Image.Resampling.BILINEAR)
        vals = np.asarray(upsampled).astype(np.float32) / 255.0

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
