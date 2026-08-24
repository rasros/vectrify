"""The segmentation and tracing stages of SAMVG.

The original SAMVG implementation was not released. This module follows Zhu's
dissertation: automatic SAM masks are filtered on a blank canvas, uncovered
regions are prompted a second time, and every retained mask is traced to a
fixed-count cubic Bezier path.
"""

from __future__ import annotations

import io
import itertools
import json
import logging
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# SAMVG's quality depends directly on the granularity of its automatic masks.
# ViT-H is the paper-quality default; users who need the smaller checkpoint can
# opt down without changing the package through VECTRIFY_SAMVG_MODEL.
SAMVG_MODEL = os.environ.get("VECTRIFY_SAMVG_MODEL", "facebook/sam-vit-huge")
# SAM encodes images at a native 1024px long side.  Keep that encoder-size cap
# as the default even when Vectrify is asked to vectorize a larger original;
# masks are restored to the original canvas before tracing.
SAMVG_MAX_SIDE = int(os.environ.get("VECTRIFY_SAMVG_MAX_SIDE", "1024"))
# This is the decoder prompt batch, not the dissertation's 32x32 sampling
# grid. 64 doubles the old 32 while leaving full-resolution-mask
# headroom on a 16 GB GPU; users with larger cards can raise it by environment.
SAMVG_POINTS_PER_BATCH = int(os.environ.get("VECTRIFY_SAMVG_POINTS_PER_BATCH", "64"))
# SAMVG's own impact filter selects useful masks against the image.  Retaining
# AMG's score gates here discarded the small facial candidates needed by the
# photo seed before that image-aware test could evaluate them.
SAMVG_PRED_IOU_THRESH = 0.0
SAMVG_STABILITY_SCORE_THRESH = 0.0
# The SAMVG seed only needs OCR once and does it after SAM has released its
# automatic-mask pipeline. This is a real VLM pass, not a separate small OCR
# detector: it can decide which visible labels deserve editable text and place
# them in the source coordinate system.
SAMVG_OCR_MODEL = os.environ.get(
    "VECTRIFY_SAMVG_OCR_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"
)
# OCR text is often a few pixels off because its original font is unknown.
# Permit that small mismatch (per affected channel), but never a large visual
# regression just because the VLM claimed confidence.
OCR_TEXT_RMSE_TOLERANCE = 0.02


@dataclass(frozen=True)
class MaskLayer:
    """One painted segmentation mask, in document compositing order."""

    mask: np.ndarray
    colour: tuple[int, int, int]
    impact: float
    overlap_pixels: int = 0


@dataclass(frozen=True)
class TextLayer:
    """A high-confidence OCR word represented as editable SVG text."""

    text: str
    x: float
    y: float
    width: float
    height: float
    colour: tuple[int, int, int]
    angle: float = 0.0


def _text_colour(pixels: np.ndarray) -> tuple[int, int, int]:
    """Estimate ink colour by contrasting a word crop with its border."""
    height, width, _channels = pixels.shape
    if height < 3 or width < 3:
        colour = pixels.reshape(-1, 3).mean(axis=0)
    else:
        border = np.concatenate(
            (pixels[0], pixels[-1], pixels[1:-1, 0], pixels[1:-1, -1])
        )
        background = border.mean(axis=0)
        distance = np.linalg.norm(pixels.astype(np.float32) - background, axis=2)
        ink = pixels[distance >= np.percentile(distance, 80)]
        colour = ink.mean(axis=0) if len(ink) else background
    return cast(tuple[int, int, int], tuple(int(value) for value in np.rint(colour)))


def _ocr_json(response: str) -> list[dict[str, object]]:
    """Decode the strict JSON array requested from the vision-language model."""
    match = re.search(r"\[[\s\S]*\]", response)
    if match is None:
        return []
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def detect_text(image: Image.Image, *, confidence: float = 0.8) -> list[TextLayer]:
    """Read editable text using Qwen2.5-VL's 3B Torch model.

    It returns content and source-pixel bounding boxes in one inference pass.
    We keep only the VLM's high-confidence multi-character labels: a guessed
    font is worse than the normal SAMVG filled-path representation.
    """
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - installation-specific
        raise ImportError(
            "SAMVG OCR requires the samvg extra. Install 'vectrify[samvg]'."
        ) from exc
    source = np.asarray(image.convert("RGB"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    prompt = (
        "Read visible text in this image. Return only a JSON array. Each entry "
        'must be {"text": string, "box": [left, top, right, bottom], '
        '"confidence": number}. Boxes must use this image\'s pixel '
        "coordinates. Include only clearly readable labels of at least two "
        "characters, and do not describe icons, logos, or non-text shapes."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    processor = AutoProcessor.from_pretrained(SAMVG_OCR_MODEL)
    # Transformers currently exposes a descriptor mismatch between this model
    # class and GenerationMixin to Pyrefly; runtime generation is the normal
    # PreTrainedModel API.
    model: Any = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        SAMVG_OCR_MODEL, torch_dtype=dtype
    ).to(device)
    detected: list[TextLayer] = []
    try:
        chat = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[chat], images=[image], padding=True, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=768, do_sample=False)
        generated = output[:, inputs.input_ids.shape[1] :]
        response = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        for entry in _ocr_json(response):
            text = entry.get("text")
            box = entry.get("box")
            score = entry.get("confidence")
            if (
                not isinstance(text, str)
                or not isinstance(box, list)
                or len(box) != 4
                or not isinstance(score, (int, float))
                or float(score) < confidence
                or len(text.strip()) < 2
            ):
                continue
            try:
                x, y, right, bottom = (float(value) for value in box)
            except (TypeError, ValueError):
                continue
            x, y = max(0.0, x), max(0.0, y)
            right = min(float(image.width), right)
            bottom = min(float(image.height), bottom)
            width, height = right - x, bottom - y
            if width < 4 or height < 4:
                continue
            crop = source[
                math.floor(y) : math.ceil(bottom), math.floor(x) : math.ceil(right)
            ]
            if not crop.size:
                continue
            detected.append(
                TextLayer(
                    text=text.strip(),
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    colour=_text_colour(crop),
                )
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log.info("SAMVG OCR: retained %d editable text layer(s).", len(detected))
    return detected


def _text_svg_attributes(layer: TextLayer) -> dict[str, str]:
    """Map OCR geometry to a portable editable SVG text element."""
    colour = f"#{layer.colour[0]:02x}{layer.colour[1]:02x}{layer.colour[2]:02x}"
    attributes = {
        "x": f"{layer.x:.2f}",
        "y": f"{layer.y + layer.height * 0.8:.2f}",
        "font-family": "sans-serif",
        "font-size": f"{layer.height:.2f}",
        "fill": colour,
    }
    if abs(layer.angle) > 1:
        attributes["transform"] = (
            f"rotate({layer.angle:.2f} {layer.x:.2f} {layer.y:.2f})"
        )
    return attributes


def _is_crop_edge_mask(
    mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    tolerance: int = 20,
) -> bool:
    """Match AMG's rejection of masks cut off at an internal crop edge."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return True
    left, top, _right, _bottom = crop_box
    width, height = image_size
    box = np.asarray(
        (left + xs.min(), top + ys.min(), left + xs.max() + 1, top + ys.max() + 1)
    )
    crop = np.asarray(crop_box)
    image = np.asarray((0, 0, width, height))
    at_crop_edge = np.abs(box - crop) <= tolerance
    at_image_edge = np.abs(box - image) <= tolerance
    return bool(np.any(at_crop_edge & ~at_image_edge))


def _run_components(mask: np.ndarray) -> list[list[tuple[int, int, int]]]:
    """Return 4-connected components as row spans in row-major order.

    The old breadth-first walk crossed the Python interpreter once for every
    foreground pixel. SAM masks are usually broad regions, so representing
    each row as contiguous runs reduces that to a small number of intervals
    while retaining scipy.ndimage's 4-connected ordering.
    """
    foreground = np.asarray(mask, dtype=bool)
    _height, width = foreground.shape
    parent = [0]

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def merge(left: int, right: int) -> None:
        left, right = root(left), root(right)
        if left != right:
            parent[right] = left

    rows: list[list[tuple[int, int, int]]] = []
    previous: list[tuple[int, int, int]] = []
    for row in foreground:
        padded = np.empty(width + 2, dtype=bool)
        padded[0] = padded[-1] = False
        padded[1:-1] = row
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        current: list[tuple[int, int, int]] = []
        prior = 0
        for start, end in edges.reshape(-1, 2):
            while prior < len(previous) and previous[prior][1] <= start:
                prior += 1
            index = len(parent)
            parent.append(index)
            candidate = prior
            while candidate < len(previous) and previous[candidate][0] < end:
                merge(index, previous[candidate][2])
                candidate += 1
            current.append((int(start), int(end), index))
        rows.append(current)
        previous = current

    components: list[list[tuple[int, int, int]]] = []
    component_ids: dict[int, int] = {}
    for y, runs in enumerate(rows):
        for start, end, index in runs:
            component = root(index)
            label = component_ids.setdefault(component, len(component_ids))
            if label == len(components):
                components.append([])
            components[label].append((y, start, end))
    return components


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Materialize 4-connected scanline components as an integer label map."""
    foreground = np.asarray(mask, dtype=bool)
    labels = np.zeros(foreground.shape, dtype=np.int32)
    components = _run_components(foreground)
    for index, runs in enumerate(components, start=1):
        for y, start, end in runs:
            labels[y, start:end] = index
    return labels, len(components)


def _edt_1d(values: np.ndarray) -> np.ndarray:
    """Squared lower envelope for the linear-time Euclidean distance transform."""
    size = len(values)
    infinity = np.inf
    sites = np.flatnonzero(np.isfinite(values))
    if not len(sites):
        return np.full(size, infinity, dtype=np.float64)
    vertices = np.empty(len(sites), dtype=np.int32)
    intersections = np.empty(len(sites) + 1, dtype=np.float64)
    count = 0
    vertices[0] = sites[0]
    intersections[0], intersections[1] = -infinity, infinity
    for site in sites[1:]:
        intersection = (
            (values[site] + site * site)
            - (values[vertices[count]] + vertices[count] * vertices[count])
        ) / (2 * (site - vertices[count]))
        while intersection <= intersections[count]:
            count -= 1
            intersection = (
                (values[site] + site * site)
                - (values[vertices[count]] + vertices[count] * vertices[count])
            ) / (2 * (site - vertices[count]))
        count += 1
        vertices[count] = site
        intersections[count], intersections[count + 1] = intersection, infinity
    output = np.empty(size, dtype=np.float64)
    index = 0
    for position in range(size):
        while intersections[index + 1] < position:
            index += 1
        site = vertices[index]
        output[position] = (position - site) ** 2 + values[site]
    return output


def _distance_transform_edt(mask: np.ndarray) -> np.ndarray:
    """Exact CPU Euclidean distance to the nearest false pixel, without SciPy."""
    foreground = np.asarray(mask, dtype=bool)
    height, width = foreground.shape
    squared = np.where(foreground, np.inf, 0.0)
    if not np.isfinite(squared).any():
        yy, xx = np.indices((height, width), dtype=np.float64)
        return np.hypot(yy + 1, xx)
    columns = np.empty_like(squared)
    for column in range(width):
        columns[:, column] = _edt_1d(squared[:, column])
    output = np.empty_like(squared)
    for row in range(height):
        output[row] = _edt_1d(columns[row])
    return np.sqrt(output)


def _binary_dilation(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Apply scipy's default 4-connected binary dilation with Torch kernels."""
    if iterations <= 0:
        return np.asarray(mask, dtype=bool)
    import torch
    import torch.nn.functional as functional

    source = torch.as_tensor(mask, dtype=torch.float32)[None, None]
    cross = source.new_tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]])
    for _ in range(iterations):
        source = (functional.conv2d(source, cross, padding=1) > 0).to(source.dtype)
    return source[0, 0].bool().numpy()


def _mean_shift_centres(points: np.ndarray, bandwidth: float) -> np.ndarray:
    """Deterministic bin-seeded mean shift matching SAMVG's prompt clustering."""
    bins = np.unique(np.rint(points / bandwidth).astype(np.int32), axis=0)
    seeds = bins.astype(np.float64) * bandwidth
    centres: dict[tuple[float, float], int] = {}
    for seed in seeds:
        centre = seed
        members = np.empty(0, dtype=np.int64)
        for _ in range(300):
            delta = points - centre
            members = np.flatnonzero((delta * delta).sum(axis=1) <= bandwidth**2)
            if not len(members):
                break
            updated = points[members].mean(axis=0)
            if np.linalg.norm(updated - centre) < bandwidth * 1e-3:
                centre = updated
                break
            centre = updated
        if len(members):
            centres[tuple(centre)] = len(members)
    # This intentionally follows sklearn's intensity-then-coordinate ordering
    # and radius duplicate suppression, preserving the old prompt priority.
    ordered = sorted(centres.items(), key=lambda item: (item[1], item[0]), reverse=True)
    candidates = np.asarray([centre for centre, _count in ordered], dtype=np.float64)
    unique = np.ones(len(candidates), dtype=bool)
    for index, centre in enumerate(candidates):
        if unique[index]:
            neighbours = np.linalg.norm(candidates - centre, axis=1) <= bandwidth
            unique[neighbours] = False
            unique[index] = True
    return candidates[unique]


def _sam_image(image: Image.Image, max_side: int | None) -> tuple[Image.Image, float]:
    """Bound a SAM pass while retaining masks in the original canvas space."""
    image = image.convert("RGB")
    if max_side is None:
        return image, 1.0
    if max_side < 1:
        raise ValueError("max_side must be positive")
    longest = max(image.size)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    return (
        image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        ),
        scale,
    )


def _restore_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour restore keeps SAM's binary mask semantics."""
    if mask.shape == (size[1], size[0]):
        return np.asarray(mask, dtype=bool)
    return np.asarray(
        Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).resize(
            size, Image.Resampling.NEAREST
        ),
        dtype=bool,
    )


@dataclass
class _SamRuntime:
    """One SAM model lifetime, including a reusable full-image embedding."""

    generator: Any
    processor: Any | None = None
    image_embeddings: Any | None = None
    embedding_size: tuple[int, int] | None = None


def _sam_runtime() -> _SamRuntime:
    """Load SAM once, in half precision when CUDA is available."""
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - installation-specific
        raise ImportError(
            "SAMVG requires the samvg extra. Install 'vectrify[samvg]'."
        ) from exc
    options: dict[str, Any] = {"model": SAMVG_MODEL, "device": 0}
    if torch.cuda.is_available():
        options["dtype"] = torch.float16
    generator = pipeline("mask-generation", **options)
    log.info(
        "SAMVG automatic masks: %s on %s (%s).",
        SAMVG_MODEL,
        generator.device,
        "fp16" if torch.cuda.is_available() else "fp32",
    )
    return _SamRuntime(generator)


def _sam_autocast():
    """Use Tensor Cores for inference while keeping exported masks binary."""
    import torch

    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _automatic_forward(inputs: Any, runtime: _SamRuntime) -> dict[str, Any]:
    """Decode on CUDA, then expand and filter masks on CPU.

    The stock Transformers pipeline expands a prompt batch to the original
    image size on CUDA. At 1024px that transient allocation is larger than the
    decoder itself. Its filtering sequence is unchanged here; only the
    post-decoder device changes.
    """
    generator = runtime.generator
    input_boxes = inputs.pop("input_boxes").detach().cpu().float()
    is_last = inputs.pop("is_last")
    original_sizes = inputs.pop("original_sizes").detach().cpu().tolist()
    reshaped_sizes = inputs.pop("reshaped_input_sizes", None)
    if reshaped_sizes is not None:
        reshaped_sizes = reshaped_sizes.detach().cpu().tolist()
    with _sam_autocast():
        model_outputs = generator.model(**inputs)
    masks = generator.image_processor.post_process_masks(
        model_outputs.pred_masks.detach().cpu(),
        original_sizes,
        mask_threshold=0,
        reshaped_input_sizes=reshaped_sizes,
        binarize=False,
    )
    filtered_masks, scores, boxes = generator.image_processor.filter_masks(
        masks[0],
        model_outputs.iou_scores.detach().cpu().float()[0],
        original_sizes[0],
        input_boxes[0],
        SAMVG_PRED_IOU_THRESH,
        SAMVG_STABILITY_SCORE_THRESH,
        0,
        1,
    )
    return {
        "masks": filtered_masks,
        "is_last": is_last,
        "boxes": boxes,
        "iou_scores": scores,
    }


def _automatic_masks_for(
    source: Image.Image,
    runtime: _SamRuntime,
    *,
    cache_embedding: bool,
    points_per_batch: int = SAMVG_POINTS_PER_BATCH,
) -> list[np.ndarray]:
    """Run one AMG image/crop without recomputing prompt-grid embeddings.

    Transformers' public mask-generation call already encodes an image once
    per 32x32 prompt grid. For the full image we use the same pipeline stages
    directly so the resulting embedding can be reused by coverage/residual
    prompts. Crops intentionally retain their own embeddings.
    """
    generator = runtime.generator
    arguments = {
        "points_per_batch": points_per_batch,
        "points_per_crop": 32,
        "crops_n_layers": 0,
        "pred_iou_thresh": SAMVG_PRED_IOU_THRESH,
        "stability_score_thresh": SAMVG_STABILITY_SCORE_THRESH,
    }
    # Keep a small compatibility path for mocked/older Transformers pipelines.
    if not hasattr(generator, "preprocess"):
        output = generator(source, **arguments)
        return [np.asarray(mask, dtype=bool) for mask in output["masks"]]

    outputs = []
    for inputs in generator.preprocess(
        source,
        points_per_batch=points_per_batch,
        points_per_crop=32,
        crops_n_layers=0,
    ):
        # ChunkPipeline normally performs this transfer between preprocess and
        # _forward. We call those stages directly to retain the embedding.
        inputs = generator._ensure_tensor_on_device(inputs, device=generator.device)
        embedding = inputs.get("image_embeddings")
        if (
            cache_embedding
            and embedding is not None
            and runtime.image_embeddings is None
        ):
            runtime.image_embeddings = embedding
            runtime.embedding_size = source.size
        outputs.append(_automatic_forward(inputs, runtime))
    output = generator.postprocess(outputs)
    return [np.asarray(mask, dtype=bool) for mask in output["masks"]]


def automatic_masks(
    image: Image.Image,
    *,
    max_side: int | None = SAMVG_MAX_SIDE,
    _runtime: _SamRuntime | None = None,
) -> list[np.ndarray]:
    """Retrieve SAM AMG masks with the thesis grid, optionally size-capped."""
    original_size = image.size
    image, _scale = _sam_image(image, max_side)
    runtime = _runtime or _sam_runtime()

    # transformers' built-in crop layer tries to stack unequal crop tensors.
    # Run that first crop layer one crop at a time instead.  Crucially, do not
    # pre-pad a rectangular image: the original AMG formula uses the source's
    # short side for overlap, and black padding changes SAM's visual context.
    width, height = image.size

    def collect(points_per_batch: int) -> list[np.ndarray]:
        collected = _automatic_masks_for(
            image,
            runtime,
            cache_embedding=True,
            points_per_batch=points_per_batch,
        )
        overlap = int((512 / 1500) * min(width, height))
        crop_width = math.ceil((overlap + width) / 2)
        crop_height = math.ceil((overlap + height) / 2)
        for x, y in {
            (0, 0),
            (crop_width - overlap, 0),
            (0, crop_height - overlap),
            (crop_width - overlap, crop_height - overlap),
        }:
            right, bottom = min(x + crop_width, width), min(y + crop_height, height)
            crop_box = (x, y, right, bottom)
            for crop_mask in _automatic_masks_for(
                image.crop(crop_box),
                runtime,
                cache_embedding=False,
                points_per_batch=points_per_batch,
            ):
                if _is_crop_edge_mask(crop_mask, crop_box, image.size):
                    continue
                mask = np.zeros((height, width), dtype=bool)
                mask[y:bottom, x:right] = crop_mask
                collected.append(mask)
        return collected

    collected = collect(SAMVG_POINTS_PER_BATCH)
    return [_restore_mask(mask, original_size) for mask in collected]


def _components(
    mask: np.ndarray, min_pixels: int, *, fill_holes: bool = True
) -> list[np.ndarray]:
    """Return traceable AMG components after its required hole cleanup.

    SAMVG traces each connected component independently.  Filling its mask
    holes before tracing matches AMG's small-region cleanup and prevents a
    noisy mask from becoming hundreds of even-odd SVG contours.
    """
    foreground = np.asarray(mask, dtype=bool)
    if int(foreground.sum()) < min_pixels:
        return []
    height, width = foreground.shape
    components = []
    for runs in _run_components(foreground):
        if sum(end - start for _y, start, end in runs) < min_pixels:
            continue
        component = np.zeros((height, width), dtype=bool)
        for y, start, end in runs:
            component[y, start:end] = True
        if fill_holes:
            # AMG's postprocessing removes *small* enclosed holes, rather
            # than turning meaningful cutouts such as an eye into a solid
            # region.  The same area cutoff as tiny components keeps those
            # two decisions consistent.
            for hole in _run_components(~component):
                area = sum(end - start for _y, start, end in hole)
                if area > min_pixels:
                    continue
                touches_border = any(
                    y in {0, height - 1} or start == 0 or end == width
                    for y, start, end in hole
                )
                if not touches_border:
                    for y, start, end in hole:
                        component[y, start:end] = True
        components.append(np.asarray(component, dtype=bool))
    return components


def _render_layers(
    shape: tuple[int, int], layers: list[MaskLayer]
) -> tuple[np.ndarray, np.ndarray]:
    """Render opaque flat-colour layers and return their alpha coverage."""
    height, width = shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    coverage = np.zeros((height, width), dtype=bool)
    for layer in layers:
        canvas[layer.mask] = layer.colour
        coverage |= layer.mask
    return canvas, coverage


def recolour_visible_layers(
    image: Image.Image, layers: list[MaskLayer]
) -> list[MaskLayer]:
    """Estimate every flat fill from the pixels it remains visible over.

    A layer's initial mask mean includes regions that later opaque layers hide.
    For a portrait this mixes skin into hair and foreground into background.
    Re-estimating in reverse painter order is the least-squares colour for the
    actual visible portion of each fixed mask.
    """
    target = np.asarray(image.convert("RGB"), dtype=np.uint8)
    covered_above = np.zeros(target.shape[:2], dtype=bool)
    revised: list[MaskLayer] = []
    for layer in reversed(layers):
        visible = layer.mask & ~covered_above
        colour = layer.colour
        if visible.any():
            colour = cast(
                tuple[int, int, int],
                tuple(int(value) for value in np.rint(target[visible].mean(axis=0))),
            )
        revised.append(
            MaskLayer(layer.mask, colour, layer.impact, layer.overlap_pixels)
        )
        covered_above |= layer.mask
    return list(reversed(revised))


def _impact_error_map(
    target: np.ndarray, canvas: np.ndarray, coverage: np.ndarray
) -> np.ndarray:
    """SAMVG's blank-canvas error, charging uncovered pixels maximally."""
    error = ((target.astype(np.float32) - canvas.astype(np.float32)) / 255.0) ** 2
    error[~coverage] = 1.0
    return error


def _impact_error(
    target: np.ndarray, canvas: np.ndarray, coverage: np.ndarray
) -> float:
    """Return the scalar blank-canvas reconstruction error."""
    return float(_impact_error_map(target, canvas, coverage).mean())


def filter_by_impact(
    image: Image.Image,
    masks: list[np.ndarray],
    *,
    existing: list[MaskLayer] | None = None,
    initial_canvas: np.ndarray | None = None,
    initial_coverage: np.ndarray | None = None,
    min_pixels: int = 32,
    min_impact: float = 1e-5,
    max_layers: int = 128,
    fill_holes: bool = True,
) -> list[MaskLayer]:
    """Keep masks that lower blank-canvas reconstruction error.

    Masks are sorted largest first; smaller retained masks overwrite their
    parent regions. *existing* makes a prompted second pass use the current
    composite as its starting canvas, as SAMVG does.
    """
    target = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width, _ = target.shape
    accepted = list(existing or [])
    canvas, coverage = _render_layers((height, width), accepted)
    if initial_canvas is not None:
        if initial_canvas.shape != canvas.shape:
            raise ValueError("initial canvas does not match the target size")
        canvas = initial_canvas.astype(np.uint8, copy=True)
    if initial_coverage is not None:
        if initial_coverage.shape != coverage.shape:
            raise ValueError("initial coverage does not match the target size")
        coverage = initial_coverage.astype(bool, copy=True)
    error_map = _impact_error_map(target, canvas, coverage)
    error_total = float(error_map.sum(dtype=np.float64))
    error = error_total / error_map.size
    initial_count = len(accepted)
    candidates = [
        component
        for mask in masks
        if np.asarray(mask).shape == (height, width)
        for component in _components(
            np.asarray(mask, dtype=bool), min_pixels, fill_holes=fill_holes
        )
    ]
    candidates.sort(key=lambda mask: int(mask.sum()), reverse=True)
    for mask in candidates:
        if int(mask.sum()) < min_pixels:
            continue
        colour = cast(
            tuple[int, int, int],
            tuple(int(value) for value in np.rint(target[mask].mean(axis=0))),
        )
        old_error = error_map[mask]
        next_error_values = (
            (target[mask].astype(np.float32) - np.asarray(colour, dtype=np.float32))
            / 255.0
        ) ** 2
        next_error_total = error_total - float(old_error.sum(dtype=np.float64))
        next_error_total += float(next_error_values.sum(dtype=np.float64))
        next_error = next_error_total / error_map.size
        impact = error - next_error
        if impact < min_impact:
            continue
        accepted.append(MaskLayer(mask, colour, impact))
        canvas[mask] = colour
        coverage |= mask
        error_map[mask] = next_error_values
        error_total, error = next_error_total, next_error
        # Each SAMVG stage is allowed its own retained-mask budget.  Applying
        # this to the combined existing+new list silently limited recovery to
        # one path once the automatic stage had filled its budget.
        if len(accepted) - initial_count >= max_layers:
            break
    return accepted


def coverage_prompt_points(
    layers: list[MaskLayer],
    shape: tuple[int, int],
    *,
    radius_fraction: float = 0.06,
    max_points: int = 16,
) -> list[tuple[int, int]]:
    """Find mean-shift centres of large circles untouched by retained masks."""
    _canvas, coverage = _render_layers(shape, layers)
    radius = max(2, round(min(shape) * radius_fraction))
    distance = _distance_transform_edt(~coverage)
    ys, xs = np.nonzero(distance >= radius)
    if len(xs) == 0:
        return []
    stride = max(1, len(xs) // 2_048)
    points = np.column_stack((xs[::stride], ys[::stride]))
    centres = _mean_shift_centres(points, radius)
    ranked = sorted(
        ((float(distance[round(y), round(x)]), round(x), round(y)) for x, y in centres),
        reverse=True,
    )
    return [(x, y) for _distance, x, y in ranked[:max_points]]


def prompted_masks(
    image: Image.Image,
    points: list[tuple[int, int]],
    *,
    max_side: int | None = SAMVG_MAX_SIDE,
    _runtime: _SamRuntime | None = None,
) -> list[np.ndarray]:
    """Prompt SAM at centres and return all three masks per point.

    Predicted IoU is a segmentation-confidence signal, not reconstruction
    impact: a broad candidate can fill an uncovered field while a smaller
    high-IoU candidate captures detail. Filter-by-impact chooses between them.
    """
    if not points:
        return []
    import torch
    from transformers import SamProcessor

    original_size = image.size
    image, scale = _sam_image(image, max_side)
    if scale != 1.0:
        points = [(round(x * scale), round(y * scale)) for x, y in points]
    own_runtime = _runtime is None
    runtime = _runtime or _sam_runtime()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("SAMVG prompted masks: using %s.", device)
    if runtime.processor is None:
        runtime.processor = SamProcessor(runtime.generator.image_processor)
    try:
        input_points = [[[list(point)] for point in points]]
        inputs = runtime.processor(
            images=image, input_points=input_points, return_tensors="pt"
        ).to(device)
        if (
            runtime.embedding_size == image.size
            and runtime.image_embeddings is not None
        ):
            # The full-image automatic pass has already encoded these pixels.
            # Retain only decoder inputs for the coverage/residual prompts.
            inputs.pop("pixel_values")
            inputs["image_embeddings"] = runtime.image_embeddings
        with torch.inference_mode(), _sam_autocast():
            output = runtime.generator.model(**inputs)
        post = runtime.processor.image_processor.post_process_masks(
            output.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]
        return [
            _restore_mask(
                np.asarray(post[prompt, candidate], dtype=bool), original_size
            )
            for prompt in range(post.shape[0])
            for candidate in range(post.shape[1])
        ]
    finally:
        if own_runtime and torch.cuda.is_available():
            torch.cuda.empty_cache()


def retrieve_layers(
    image: Image.Image,
    masks: list[np.ndarray] | None = None,
    *,
    min_pixels: int = 32,
    min_impact: float = 1e-5,
    max_layers: int = 512,
    fill_holes: bool = True,
    max_side: int | None = SAMVG_MAX_SIDE,
    _runtime: _SamRuntime | None = None,
) -> list[MaskLayer]:
    """Run SAMVG's automatic-mask, coverage-prompt, filter sequence."""
    image = image.convert("RGB")
    runtime = _runtime
    if masks is None:
        runtime = runtime or _sam_runtime()
        initial = automatic_masks(image, max_side=max_side, _runtime=runtime)
    else:
        initial = masks
    layers = filter_by_impact(
        image,
        initial,
        min_pixels=min_pixels,
        min_impact=min_impact,
        max_layers=max_layers,
        fill_holes=fill_holes,
    )
    layers = recolour_visible_layers(image, layers)
    points = coverage_prompt_points(layers, (image.height, image.width))
    prompted = prompted_masks(image, points, max_side=max_side, _runtime=runtime)
    recovered = filter_by_impact(
        image,
        prompted,
        existing=layers,
        min_pixels=min_pixels,
        min_impact=min_impact,
        max_layers=max_layers,
        fill_holes=fill_holes,
    )
    recovered = recolour_visible_layers(image, recovered)
    log.info(
        "SAMVG first pass: %d automatic mask(s), %d retained; %d coverage "
        "prompt(s), %d prompted mask(s), %d total retained.",
        len(initial),
        len(layers),
        len(points),
        len(prompted),
        len(recovered),
    )
    return recovered


def _loops(mask: np.ndarray) -> list[list[tuple[float, float]]]:
    """Trace pixel-boundary loops, retaining exterior and hole contours."""
    edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask), strict=True):
        if y == 0 or not mask[y - 1, x]:
            edges[(x, y)].append((x + 1, y))
        if x == width - 1 or not mask[y, x + 1]:
            edges[(x + 1, y)].append((x + 1, y + 1))
        if y == height - 1 or not mask[y + 1, x]:
            edges[(x + 1, y + 1)].append((x, y + 1))
        if x == 0 or not mask[y, x - 1]:
            edges[(x, y + 1)].append((x, y))
    loops: list[list[tuple[float, float]]] = []
    while edges:
        start = next(iter(edges))
        current, loop = start, [cast(tuple[float, float], tuple(map(float, start)))]
        while current in edges:
            following = edges[current].pop()
            if not edges[current]:
                del edges[current]
            current = following
            if current == start:
                break
            loop.append(cast(tuple[float, float], tuple(map(float, current))))
        if current == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def _corners(loop: list[tuple[float, float]], count: int) -> list[int]:
    """Global curvature maxima with the local exclusion SAMVG describes."""
    points = np.asarray(loop, dtype=np.float32)
    size = len(points)
    count = min(count, size)
    step = max(1, size // 12)
    before = points - np.roll(points, step, axis=0)
    after = np.roll(points, -step, axis=0) - points
    denom = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    score = np.divide(
        (before * after).sum(axis=1), denom, out=np.ones(size), where=denom > 0
    )
    blocked = np.zeros(size, dtype=bool)
    chosen: list[int] = []
    exclusion = max(1, size // (count * 2))
    for _ in range(count):
        available = np.where(~blocked)[0]
        if len(available) == 0:
            break
        index = int(available[np.argmin(score[available])])
        chosen.append(index)
        offsets = (np.arange(index - exclusion, index + exclusion + 1) % size).astype(
            int
        )
        blocked[offsets] = True
    return sorted(chosen)


def _fit_cubic(
    points: np.ndarray, *, reparameterize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Fit fixed-endpoint cubic controls, refining the samples' parameters.

    SAMVG starts with uniformly spaced ``t`` values, then applies the
    Newton--Raphson reparameterisation in dissertation equation 3.5 before a
    final least-squares control-point fit.  Pixel contours have highly uneven
    arc-length samples around corners, so this matters even with a fixed number
    of curves.
    """
    start, end = points[0], points[-1]
    t = np.linspace(0.0, 1.0, len(points), dtype=np.float64)

    def solve(parameters: np.ndarray) -> np.ndarray:
        matrix = np.column_stack(
            (
                3 * (1 - parameters) ** 2 * parameters,
                3 * (1 - parameters) * parameters**2,
            )
        )
        base = (1 - parameters)[:, None] ** 3 * start + parameters[:, None] ** 3 * end
        controls, *_ = np.linalg.lstsq(matrix, points - base, rcond=None)
        return controls

    controls = solve(t)
    if reparameterize and len(points) > 2:
        # The endpoints must remain exactly 0 and 1.  Keeping interior values
        # ordered avoids a folded parameterisation on jagged raster contours.
        epsilon = 1e-5
        for _iteration in range(8):
            p0, p1 = controls
            omt = 1 - t
            curve = (
                omt[:, None] ** 3 * start
                + 3 * omt[:, None] ** 2 * t[:, None] * p0
                + 3 * omt[:, None] * t[:, None] ** 2 * p1
                + t[:, None] ** 3 * end
            )
            first = (
                3 * omt[:, None] ** 2 * (p0 - start)
                + 6 * omt[:, None] * t[:, None] * (p1 - p0)
                + 3 * t[:, None] ** 2 * (end - p1)
            )
            second = 6 * omt[:, None] * (p1 - 2 * p0 + start) + 6 * t[:, None] * (
                end - 2 * p1 + p0
            )
            offset = curve - points
            numerator = (offset * first).sum(axis=1)
            denominator = (first * first).sum(axis=1) + (offset * second).sum(axis=1)
            updated = t.copy()
            valid = np.abs(denominator[1:-1]) > 1e-10
            # Raster corners can make an unconstrained Newton step enormous;
            # a short, damped step retains the convergence benefit without
            # collapsing several samples onto one parameter value.
            delta = np.clip(
                numerator[1:-1][valid] / denominator[1:-1][valid], -0.05, 0.05
            )
            interior = updated[1:-1]
            interior[valid] -= delta
            updated[1:-1] = interior
            updated[0], updated[-1] = 0.0, 1.0
            updated[1:-1] = np.clip(updated[1:-1], epsilon, 1 - epsilon)
            updated = np.maximum.accumulate(updated)
            updated[-1] = 1.0
            if np.max(np.abs(updated - t)) < 1e-4:
                break
            t = updated
            controls = solve(t)
    return controls[0], controls[1]


def _cubic_loop(loop: list[tuple[float, float]], segments: int) -> str | None:
    size = len(loop)
    if size < 3:
        return None
    corners = _corners(loop, segments)
    if len(corners) < 3:
        return None
    points = np.asarray(loop, dtype=np.float32)
    parts = [f"M {points[corners[0], 0]:.2f} {points[corners[0], 1]:.2f}"]
    for first, second in zip(corners, [*corners[1:], corners[0]], strict=True):
        indices = (
            np.arange(first, second + 1 if second >= first else second + size + 1)
            % size
        )
        sample = np.vstack((points[indices], points[second]))
        control_a, control_b = _fit_cubic(sample)
        end = points[second]
        parts.append(
            f"C {control_a[0]:.2f} {control_a[1]:.2f} "
            f"{control_b[0]:.2f} {control_b[1]:.2f} {end[0]:.2f} {end[1]:.2f}"
        )
    return " ".join(parts) + " Z"


def mask_path(
    mask: np.ndarray, *, segments: int = 8, overlap_pixels: int = 0
) -> str | None:
    """Fit every mask contour as a fixed-count cubic Bezier SVG path."""
    if overlap_pixels:
        mask = _binary_dilation(mask, overlap_pixels)
    parts = [piece for loop in _loops(mask) if (piece := _cubic_loop(loop, segments))]
    return " ".join(parts) or None


_SKELETON_NEIGHBOURS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _thin_mask(mask: np.ndarray) -> np.ndarray:
    """Zhang--Suen thinning without adding a SciPy/skimage dependency."""
    thin = np.pad(mask.astype(np.uint8), 1).copy()
    changed = True
    while changed:
        changed = False
        for phase in range(2):
            remove: list[tuple[int, int]] = []
            for y, x in zip(*np.nonzero(thin), strict=True):
                if y in {0, thin.shape[0] - 1} or x in {0, thin.shape[1] - 1}:
                    continue
                ring = [
                    thin[y - 1, x],
                    thin[y - 1, x + 1],
                    thin[y, x + 1],
                    thin[y + 1, x + 1],
                    thin[y + 1, x],
                    thin[y + 1, x - 1],
                    thin[y, x - 1],
                    thin[y - 1, x - 1],
                ]
                count = sum(ring)
                transitions = sum(
                    left == 0 and right == 1
                    for left, right in zip(ring, [*ring[1:], ring[0]], strict=True)
                )
                if not (2 <= count <= 6 and transitions == 1):
                    continue
                north, east, south, west = ring[0], ring[2], ring[4], ring[6]
                blocked = (
                    (north and east and south) or (east and south and west)
                    if phase == 0
                    else (north and east and west) or (north and south and west)
                )
                if not blocked:
                    remove.append((y, x))
            if remove:
                changed = True
                for y, x in remove:
                    thin[y, x] = 0
    return thin[1:-1, 1:-1].astype(bool)


def _skeleton_traces(mask: np.ndarray) -> list[np.ndarray]:
    """Split a thinned medial-axis graph into its endpoint/junction traces."""
    points = {tuple(point) for point in np.argwhere(_thin_mask(mask))}
    if len(points) < 2:
        return []

    def adjacent(point: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = point
        output = []
        for dy, dx in _SKELETON_NEIGHBOURS:
            candidate = y + dy, x + dx
            if candidate not in points:
                continue
            # A diagonal across an orthogonal staircase is not another graph
            # edge. Keeping it creates artificial triangles and turns every
            # curved pixel line into a forest of tiny branches.
            if dy and dx and ((y + dy, x) in points or (y, x + dx) in points):
                continue
            output.append(candidate)
        return output

    nodes = {point for point in points if len(adjacent(point)) != 2}
    # Closed loops are better represented by SAMVG's filled path: an open
    # stroke would introduce caps and a stroke-only loop has no stable start.
    if not nodes:
        return []
    traversed: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def edge_key(
        first: tuple[int, int], second: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return (first, second) if first <= second else (second, first)

    traces: list[np.ndarray] = []
    for node in nodes:
        for neighbour in adjacent(node):
            edge = edge_key(node, neighbour)
            if edge in traversed:
                continue
            trace, previous, current = [node], node, neighbour
            traversed.add(edge)
            while current not in nodes:
                trace.append(current)
                choices = [point for point in adjacent(current) if point != previous]
                if len(choices) != 1:
                    trace = []
                    break
                previous, current = current, choices[0]
                traversed.add(edge_key(previous, current))
            if trace:
                trace.append(current)
                if len(trace) >= 2:
                    traces.append(
                        np.asarray([(x, y) for y, x in trace], dtype=np.float64)
                    )
    return traces


def _trace_path_data(trace: np.ndarray, segments: int) -> str:
    """Fit multiple cubic sections to a skeleton rather than one global PCA line."""
    count = max(1, min(segments, math.ceil((len(trace) - 1) / 8)))
    boundaries = np.linspace(0, len(trace) - 1, count + 1, dtype=int)
    output = [f"M {trace[0, 0]:.2f} {trace[0, 1]:.2f}"]
    for first, last in itertools.pairwise(boundaries):
        sample = trace[first : last + 1]
        if len(sample) == 2:
            output.append(f"L {sample[-1, 0]:.2f} {sample[-1, 1]:.2f}")
        else:
            control_a, control_b = _fit_cubic(sample)
            end = sample[-1]
            output.append(
                f"C {control_a[0]:.2f} {control_a[1]:.2f} "
                f"{control_b[0]:.2f} {control_b[1]:.2f} {end[0]:.2f} {end[1]:.2f}"
            )
    return " ".join(output)


def _mask_distance(mask: np.ndarray) -> np.ndarray:
    """Two-pass chamfer distance to the background in mask-pixel units."""
    distance = np.where(mask, np.inf, 0.0).astype(np.float64)
    diagonal = math.sqrt(2.0)
    for y in range(distance.shape[0]):
        for x in range(distance.shape[1]):
            if not mask[y, x]:
                continue
            candidates = []
            if y:
                candidates.append(distance[y - 1, x] + 1)
                if x:
                    candidates.append(distance[y - 1, x - 1] + diagonal)
                if x + 1 < distance.shape[1]:
                    candidates.append(distance[y - 1, x + 1] + diagonal)
            if x:
                candidates.append(distance[y, x - 1] + 1)
            distance[y, x] = min(candidates, default=distance[y, x])
    for y in range(distance.shape[0] - 1, -1, -1):
        for x in range(distance.shape[1] - 1, -1, -1):
            if not mask[y, x]:
                continue
            candidates = [distance[y, x]]
            if y + 1 < distance.shape[0]:
                candidates.append(distance[y + 1, x] + 1)
                if x:
                    candidates.append(distance[y + 1, x - 1] + diagonal)
                if x + 1 < distance.shape[1]:
                    candidates.append(distance[y + 1, x + 1] + diagonal)
            if x + 1 < distance.shape[1]:
                candidates.append(distance[y, x + 1] + 1)
            distance[y, x] = min(candidates)
    return distance


def _trace_sections(trace: np.ndarray, segments: int) -> list[np.ndarray]:
    count = max(1, min(segments, math.ceil((len(trace) - 1) / 8)))
    boundaries = np.linspace(0, len(trace) - 1, count + 1, dtype=int)
    return [trace[first : last + 1] for first, last in itertools.pairwise(boundaries)]


def mask_stroke(
    mask: np.ndarray, *, segments: int = 8, overlap_pixels: int = 0
) -> tuple[str, float] | None:
    """Return a conservative centreline stroke for one thin mask component.

    SAMVG itself uses closed filled shapes.  This optional hybrid extension is
    deliberately strict: a component must be long, narrow, and have no holes
    before it can be represented by a stroke.  Other masks preserve SAMVG's
    original filled-path treatment.
    """
    if overlap_pixels:
        mask = _binary_dilation(mask, overlap_pixels)
    _ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    # A hole is topology that a single centreline cannot preserve.
    if len(_loops(mask)) != 1:
        return None

    traces = _skeleton_traces(mask)
    if len(traces) != 1:
        return None
    trace = traces[0]
    length = float(np.linalg.norm(np.diff(trace, axis=0), axis=1).sum())
    # Arc length, rather than a bounding-box axis, preserves strongly curved
    # thin components whose width and height are similar.
    estimated_width = len(xs) / max(length, 1.0)
    if length < 12 or estimated_width > min(8.0, length * 0.3):
        return None
    distance = _mask_distance(mask)
    widths = [2 * (distance[int(y), int(x)] - 0.5) for x, y in trace]
    data = _trace_path_data(trace, segments)
    return data, max(1.0, float(np.median(widths)))


def mask_strokes(
    mask: np.ndarray, *, segments: int = 8, overlap_pixels: int = 0
) -> list[tuple[str, float]]:
    """Trace a thin component into independently editable constant-width paths.

    A branch becomes one path per medial-axis edge. Each long edge is divided
    into cubic sections and each section gets its local median width, giving an
    SVG approximation of a variable-width centreline without nonstandard SVG
    extensions. Round caps and joins make the adjacent sections continuous.
    """
    if overlap_pixels:
        mask = _binary_dilation(mask, overlap_pixels)
    _ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return []
    if len(_loops(mask)) != 1:
        return []
    traces = _skeleton_traces(mask)
    length = sum(
        float(np.linalg.norm(np.diff(trace, axis=0), axis=1).sum()) for trace in traces
    )
    estimated_width = len(xs) / max(length, 1.0)
    if length < 12 or estimated_width > min(8.0, length * 0.3):
        return []
    distance = _mask_distance(mask)
    output: list[tuple[str, float]] = []
    for trace in traces:
        for section in _trace_sections(trace, segments):
            if len(section) < 2:
                continue
            widths = [2 * (distance[int(y), int(x)] - 0.5) for x, y in section]
            output.append(
                (_trace_path_data(section, 1), max(1.0, float(np.median(widths))))
            )
    return output


def _layer_svg_attributes(
    layer: MaskLayer, segments: int, *, hybrid_strokes: bool = True
) -> list[dict[str, str]]:
    """Trace one SAM mask, using optional strokes only outside the thesis mode."""
    colour = f"#{layer.colour[0]:02x}{layer.colour[1]:02x}{layer.colour[2]:02x}"
    strokes = (
        mask_strokes(layer.mask, segments=segments, overlap_pixels=layer.overlap_pixels)
        if hybrid_strokes
        else []
    )
    if strokes:
        return [
            {
                "d": data,
                "fill": "none",
                "stroke": colour,
                "stroke-width": f"{width:.2f}",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            }
            for data, width in strokes
        ]
    data = mask_path(layer.mask, segments=segments, overlap_pixels=layer.overlap_pixels)
    if data is None:
        return []
    return [{"d": data, "fill": colour, "fill-rule": "evenodd"}]


def generate_svg(
    image: Image.Image,
    masks: list[np.ndarray] | None = None,
    *,
    min_pixels: int = 32,
    min_impact: float = 1e-5,
    max_layers: int = 512,
    segments: int = 16,
    fill_holes: bool = True,
    hybrid_strokes: bool = True,
    ocr: bool = True,
    max_side: int | None = SAMVG_MAX_SIDE,
    rasterize: Callable[[str, int, int], bytes] | None = None,
) -> str:
    """Generate SAMVG's traced, pre-optimisation SVG from a target image."""
    image = image.convert("RGB")
    layers = (
        filter_by_impact(
            image,
            masks,
            min_pixels=min_pixels,
            min_impact=min_impact,
            max_layers=max_layers,
            fill_holes=fill_holes,
        )
        if masks is not None
        else retrieve_layers(
            image,
            min_pixels=min_pixels,
            min_impact=min_impact,
            max_layers=max_layers,
            fill_holes=fill_holes,
            max_side=max_side,
        )
    )
    paths = []
    for layer in layers:
        for attributes in _layer_svg_attributes(
            layer, segments, hybrid_strokes=hybrid_strokes
        ):
            markup = " ".join(f'{key}="{value}"' for key, value in attributes.items())
            paths.append(f"<path {markup} />")
    width, height = image.size
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(paths) + "</svg>"
    )
    text_layers = detect_text(image) if ocr and masks is None else []
    if text_layers and rasterize is not None:
        return _accept_text_layers(svg, image, text_layers, rasterize)
    return _append_text_layers(svg, text_layers)


def residual_prompt_points(
    target: Image.Image,
    rendered: Image.Image,
    *,
    radius_fraction: float = 0.06,
    threshold: float = 0.784,
    max_points: int = 16,
) -> list[tuple[int, int]]:
    """Locate SAMVG's convolved, thresholded residual components."""
    import torch
    import torch.nn.functional as functional

    target_pixels = np.asarray(target.convert("RGB"), dtype=np.float32) / 255.0
    rendered_pixels = np.asarray(rendered.convert("RGB"), dtype=np.float32) / 255.0
    # SAMVG sums RGB-channel difference before applying its 0.784 threshold.
    # Averaging here hides a strongly wrong but uniformly coloured face/body.
    difference = np.abs(target_pixels - rendered_pixels).sum(axis=2)
    height, width = difference.shape
    radius = max(2, round(min(height, width) * radius_fraction))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (xx * xx + yy * yy <= radius * radius).astype(np.float32)
    # Reflected padding preserves the prior symmetric-boundary definition;
    # FFT convolution keeps the full-resolution recovery pass practical.
    padded = np.pad(difference, radius, mode="symmetric")
    smoothed = functional.conv2d(
        torch.from_numpy(padded)[None, None],
        torch.from_numpy((kernel / kernel.sum())[None, None]),
    )[0, 0].numpy()
    labels, count = _label(smoothed >= threshold)
    points: list[tuple[float, int, int]] = []
    for index in range(1, count + 1):
        ys, xs = np.nonzero(labels == index)
        if len(xs):
            points.append(
                (float(smoothed[ys, xs].mean()), round(xs.mean()), round(ys.mean()))
            )
    return [(x, y) for _score, x, y in sorted(points, reverse=True)[:max_points]]


def _append_layers(
    svg: str,
    layers: list[MaskLayer],
    segments: int,
    *,
    hybrid_strokes: bool = True,
) -> str:
    """Add newly prompted paths to an already optimised SVG."""
    root = ET.fromstring(svg)
    for layer in layers:
        for attributes in _layer_svg_attributes(
            layer, segments, hybrid_strokes=hybrid_strokes
        ):
            ET.SubElement(
                root,
                "{http://www.w3.org/2000/svg}path",
                attributes,
            )
    return ET.tostring(root, encoding="unicode")


def _append_text_layers(svg: str, layers: list[TextLayer]) -> str:
    """Append editable OCR text without changing the pre-existing drawing."""
    if not layers:
        return svg
    root = ET.fromstring(svg)
    for layer in layers:
        element = ET.SubElement(
            root, "{http://www.w3.org/2000/svg}text", _text_svg_attributes(layer)
        )
        element.text = layer.text
    return ET.tostring(root, encoding="unicode")


def _render_svg(svg: str, image: Image.Image, rasterize) -> Image.Image:
    return Image.open(io.BytesIO(rasterize(svg, image.width, image.height))).convert(
        "RGB"
    )


def _mse(image: Image.Image, rendered: Image.Image) -> float:
    target = np.asarray(image.convert("RGB"), dtype=np.float32)
    candidate = np.asarray(rendered.convert("RGB"), dtype=np.float32)
    return float(((target - candidate) ** 2).mean())


def _text_error_tolerance(layer: TextLayer, image: Image.Image) -> float:
    """Return the whole-image MSE budget for this one text bounding box."""
    padding = 2
    width = min(image.width, max(1, math.ceil(layer.width) + padding * 2))
    height = min(image.height, max(1, math.ceil(layer.height) + padding * 2))
    affected_fraction = (width * height) / (image.width * image.height)
    return affected_fraction * (255 * OCR_TEXT_RMSE_TOLERANCE) ** 2


def _accept_text_layers(
    svg: str,
    image: Image.Image,
    layers: list[TextLayer],
    rasterize: Callable[[str, int, int], bytes],
) -> str:
    """Retain OCR text that improves, or only negligibly worsens, pixel loss.

    A VLM's asserted confidence is not evidence that a word is present. The
    same rasterisation used to score the seed is the final verifier, including
    font mismatch, positioning, and any existing SAM paths beneath the text.
    """
    accepted = svg
    error = _mse(image, _render_svg(accepted, image, rasterize))
    retained = 0
    for layer in layers:
        candidate = _append_text_layers(accepted, [layer])
        candidate_error = _mse(image, _render_svg(candidate, image, rasterize))
        if candidate_error <= error + _text_error_tolerance(layer, image):
            accepted, error = candidate, candidate_error
            retained += 1
    log.info(
        "SAMVG OCR: retained %d/%d text layer(s) after pixel verification.",
        retained,
        len(layers),
    )
    return accepted


def _accepted_fit(
    svg: str, image: Image.Image, *, rasterize, steps: int
) -> tuple[str, Image.Image]:
    """Keep a differentiable fit only when the actual SVG renderer improves."""
    from vectrify.refine.paths import fit_filled_svg_bounded

    before = _render_svg(svg, image, rasterize)
    fitted = fit_filled_svg_bounded(svg, image, rasterize=rasterize, steps=steps)
    after = _render_svg(fitted, image, rasterize)
    if _mse(image, after) <= _mse(image, before):
        return fitted, after
    log.info("SAMVG fit rejected: the exported SVG MSE increased.")
    return svg, before


def vectorize_svg(
    image: Image.Image,
    *,
    rasterize,
    steps: int = 500,
    min_pixels: int = 32,
    min_impact: float = 1e-5,
    max_layers: int = 512,
    segments: int = 16,
    max_side: int | None = SAMVG_MAX_SIDE,
) -> str:
    """Run SAMVG's two 500-step optimise-and-recover phases.

    ``rasterize`` is the format backend's renderer, used solely to form the
    residual map after the first pass. The actual differentiable fit is the
    built-in filled-path optimiser so SAMVG has no external renderer dependency.
    """
    image = image.convert("RGB")
    runtime = _sam_runtime()
    try:
        layers = retrieve_layers(
            image,
            min_pixels=min_pixels,
            min_impact=min_impact,
            max_layers=max_layers,
            max_side=max_side,
            _runtime=runtime,
        )
        initial = _append_layers(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{image.width}" '
            f'height="{image.height}" '
            f'viewBox="0 0 {image.width} {image.height}"></svg>',
            layers,
            segments,
            hybrid_strokes=False,
        )
        first, first_render = _accepted_fit(
            initial, image, rasterize=rasterize, steps=steps
        )
        points = residual_prompt_points(image, first_render)
        _canvas, coverage = _render_layers((image.height, image.width), layers)
        added = filter_by_impact(
            image,
            prompted_masks(image, points, max_side=max_side, _runtime=runtime),
            existing=layers,
            initial_canvas=np.asarray(first_render, dtype=np.uint8),
            initial_coverage=coverage,
            min_pixels=min_pixels,
            min_impact=min_impact,
            max_layers=max_layers,
        )[len(layers) :]
        log.info(
            "SAMVG residual pass: %d prompt(s), %d accepted added path(s).",
            len(points),
            len(added),
        )
        return _accepted_fit(
            _append_layers(first, added, segments, hybrid_strokes=False),
            image,
            rasterize=rasterize,
            steps=steps,
        )[0]
    finally:
        del runtime
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - installation-specific
            pass
