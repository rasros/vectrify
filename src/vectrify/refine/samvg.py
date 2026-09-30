"""The segmentation and tracing stages of SAMVG.

The original SAMVG implementation was not released. This module follows Zhu's
dissertation for the segmentation: automatic SAM masks are filtered on a blank
canvas and uncovered regions are prompted a second time. The retained masks
are then settled into regions, their edges moved onto the image's own, and
each outline is traced densely with cubic Beziers and simplified.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from PIL import Image

from vectrify.refine.samvg_runtime import device_name, pipeline_options
from vectrify.refine.samvg_types import MaskLayer

log = logging.getLogger(__name__)

# SAMVG's quality depends directly on the granularity of its automatic masks.
# ViT-H is the paper-quality default; ViT-B is faster.
SAMVG_MODEL = "facebook/sam-vit-huge"
# SAM encodes images at a native 1024px long side, the default size cap when
# none is given; masks are restored to the original canvas before tracing.
SAMVG_MAX_SIDE = 1024
# This is the decoder prompt batch, not the dissertation's 32x32 sampling
# grid. 64 leaves full-resolution-mask headroom on a 16 GB GPU.
SAMVG_POINTS_PER_BATCH = 64
# Masks and components smaller than this many pixels are left out, and holes
# up to it filled.
MIN_PIXELS = 32
# The least a mask must lower the blank-canvas error to be kept.
MIN_IMPACT = 3e-6
# Regions narrower than this everywhere, in pixels, are left out: SAM returns
# outlines and hairlines as regions of their own.
MIN_WIDTH = 3
# How far an outline may stray from its region, in pixels: each is traced
# with one curve per DENSITY pixels and then simplified down to it.
TOLERANCE = 0.5
DENSITY = 6
# The share of the impact filter's threshold one merge of neighbouring
# regions may cost; see ``merge_patches``.
MERGE_SHARE = 0.1
# What a pixel pays, in squared RGB (0-1), for each neighbour of another
# region when edges are moved onto the image's; see ``refine_edges``.
EDGE_SMOOTHNESS = 0.01
# How far, in SAM pixels, an edge may move.
EDGE_BAND = 6
# Outlines are smoothed over this many SAM pixels before curves are fitted,
# so the fit does not follow the masks' raster steps.
SMOOTH = 1.0


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
    """Apply scipy's default 4-connected binary dilation with Torch kernels.

    Without Torch, SciPy's own dilation, so the tracing runs on any install.
    """
    if iterations <= 0:
        return np.asarray(mask, dtype=bool)
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError:  # pragma: no cover - installation-specific
        from scipy import ndimage

        return ndimage.binary_dilation(mask, iterations=iterations)

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


def _sam_runtime(*, model: str = SAMVG_MODEL) -> _SamRuntime:
    """Load SAM once, in half precision when CUDA is available."""
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - installation-specific
        raise ImportError(
            "SAMVG requires the samvg extra. Install 'vectrify[samvg]'."
        ) from exc
    options = pipeline_options(torch, model)
    generator = pipeline("mask-generation", **options)
    log.info(
        "SAMVG automatic masks: %s on %s (%s).",
        model,
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
    """Decode and filter one AMG prompt batch in SAM's original order."""
    import torch

    generator = runtime.generator
    input_boxes = inputs.pop("input_boxes").float()
    is_last = inputs.pop("is_last")
    original_sizes = inputs.pop("original_sizes").detach().cpu().tolist()
    reshaped_sizes = inputs.pop("reshaped_input_sizes", None)
    if reshaped_sizes is not None:
        reshaped_sizes = reshaped_sizes.detach().cpu().tolist()
    # `.cpu()` alone preserves the decoder's autograd graph, retaining every
    # prior prompt batch's CUDA activations. AMG is inference-only, so make
    # that lifetime explicit before handing compact candidates to the host.
    with torch.inference_mode(), _sam_autocast():
        model_outputs = generator.model(**inputs)
        # Official AMG interpolates decoder logits to the crop canvas before
        # its confidence, stability, and crop-edge tests.  Filtering at 256px
        # is faster but changes which fine masks survive, so it cannot be used
        # for the dissertation-faithful seed.
        masks_at_size = generator.image_processor.post_process_masks(
            model_outputs.pred_masks,
            original_sizes,
            reshaped_input_sizes=reshaped_sizes,
            mask_threshold=0,
            binarize=False,
        )[0]
        masks, scores, boxes = _filter_automatic_masks(
            masks_at_size,
            model_outputs.iou_scores[0],
            original_sizes[0],
            input_boxes[0],
        )
    return {
        "masks": masks,
        "is_last": is_last,
        "boxes": boxes,
        "scores": scores,
        "original_size": original_sizes[0],
        "reshaped_size": reshaped_sizes[0] if reshaped_sizes is not None else None,
        "crop_box": input_boxes[0],
    }


def _filter_automatic_masks(
    masks: Any,
    iou_scores: Any,
    original_size: list[int],
    cropped_box_image: Any,
) -> tuple[Any, Any, Any]:
    """Apply SAM AMG's full-resolution crop-edge filtering.

    SAMVG's impact filter is the retained-mask decision, so AMG's confidence
    gates are left off: a small, useful candidate reaches that later test
    instead of being discarded by a checkpoint-confidence heuristic.
    """
    from transformers.models.sam.image_processing_sam import (
        _batched_mask_to_box,
        _is_box_near_crop_edge,
        _pad_masks,
    )

    original_height, original_width = original_size
    scores = iou_scores.reshape(-1)
    masks = masks.reshape(-1, *masks.shape[-2:]) > 0
    boxes = _batched_mask_to_box(masks)
    keep = ~_is_box_near_crop_edge(
        boxes, cropped_box_image, [0, 0, original_width, original_height]
    )
    return (
        _pad_masks(masks[keep], cropped_box_image, original_height, original_width),
        scores[keep],
        boxes[keep],
    )


def _nms_indices(boxes: Any, scores: Any) -> Any:
    """Use SAM AMG's box-NMS configuration with dtype-safe scores."""
    import torch
    from torchvision.ops import batched_nms

    return batched_nms(
        boxes=boxes.float(),
        scores=scores.float(),
        idxs=torch.zeros(len(boxes), dtype=torch.long),
        iou_threshold=0.7,
    )


def _finalize_automatic_masks(masks: Any, scores: Any, boxes: Any) -> list[np.ndarray]:
    """Apply AMG's final image-global NMS and transfer binary masks."""
    if not len(masks):
        return []
    keep = _nms_indices(boxes, scores)
    selected_masks = masks[keep]
    return [np.asarray(mask.cpu(), dtype=bool) for mask in selected_masks]


def _automatic_mask_candidates_for(
    source: Image.Image,
    runtime: _SamRuntime,
    *,
    cache_embedding: bool,
) -> tuple[Any, Any, Any]:
    """Return post-filter AMG candidates before its image-global crop NMS.

    Transformers' public mask-generation call already encodes an image once
    per 32x32 prompt grid. For the full image we use the same pipeline stages
    directly so the resulting embedding can be reused by the coverage
    prompts. Crops intentionally retain their own embeddings.
    """
    import torch

    generator = runtime.generator
    arguments = {
        "points_per_batch": SAMVG_POINTS_PER_BATCH,
        "points_per_crop": 32,
        "crops_n_layers": 0,
        "pred_iou_thresh": 0,
        "stability_score_thresh": 0,
    }
    # Keep a small compatibility path for mocked/older Transformers pipelines.
    if not hasattr(generator, "preprocess"):
        output = generator(source, **arguments)
        masks = (
            torch.from_numpy(
                np.stack([np.asarray(mask, dtype=bool) for mask in output["masks"]])
            )
            if output["masks"]
            else torch.empty((0, source.height, source.width), dtype=torch.bool)
        )
        boxes = torch.empty((len(masks), 4), dtype=torch.float32)
        for index, mask in enumerate(masks):
            ys, xs = torch.where(mask)
            boxes[index] = torch.tensor(
                (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), dtype=torch.float32
            )
        scores = torch.ones(len(masks))
        keep = _nms_indices(boxes, scores) if len(masks) else []
        return masks[keep], scores[keep], boxes[keep]

    outputs = []
    for inputs in generator.preprocess(
        source,
        points_per_batch=SAMVG_POINTS_PER_BATCH,
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
        # Keep the GPU bounded to one decoder batch. NMS and score filtering
        # have already retained only the decoder logits that need the
        # full-resolution SAM post-processing.
        outputs[-1]["masks"] = outputs[-1]["masks"].cpu()
        outputs[-1]["scores"] = outputs[-1]["scores"].cpu()
        outputs[-1]["boxes"] = outputs[-1]["boxes"].cpu()
    masks = [output["masks"] for output in outputs if len(output["masks"])]
    if not masks:
        return (
            torch.empty((0, source.height, source.width), dtype=torch.bool),
            torch.empty(0),
            torch.empty((0, 4)),
        )
    masks = torch.cat(masks)
    scores = torch.cat([output["scores"] for output in outputs if len(output["masks"])])
    boxes = torch.cat([output["boxes"] for output in outputs if len(output["masks"])])
    # Meta's AMG first suppresses candidates within each crop by predicted
    # quality.  Its second, cross-crop pass happens in ``automatic_masks``
    # below and deliberately ranks the surviving masks by crop area instead.
    keep = _nms_indices(boxes, scores)
    return masks[keep], scores[keep], boxes[keep]


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
    import torch

    # transformers' built-in crop layer tries to stack unequal crop tensors.
    # Run that first crop layer one crop at a time instead.  Crucially, do not
    # pre-pad a rectangular image: the original AMG formula uses the source's
    # short side for overlap, and black padding changes SAM's visual context.
    width, height = image.size
    masks, scores, boxes = _automatic_mask_candidates_for(
        image, runtime, cache_embedding=True
    )
    all_masks = [masks]
    all_scores = [torch.full_like(scores, 1 / (width * height))]
    all_boxes = [boxes]
    overlap = int((512 / 1500) * min(width, height))
    crop_width = math.ceil((overlap + width) / 2)
    crop_height = math.ceil((overlap + height) / 2)
    for x, y in (
        (0, 0),
        (0, crop_height - overlap),
        (crop_width - overlap, 0),
        (crop_width - overlap, crop_height - overlap),
    ):
        right, bottom = min(x + crop_width, width), min(y + crop_height, height)
        crop_box = (x, y, right, bottom)
        crop_masks, crop_scores, crop_boxes = _automatic_mask_candidates_for(
            image.crop(crop_box), runtime, cache_embedding=False
        )
        for index, crop_mask in enumerate(crop_masks):
            crop_mask_array = np.asarray(crop_mask, dtype=bool)
            if _is_crop_edge_mask(crop_mask_array, crop_box, image.size):
                continue
            mask = np.zeros((height, width), dtype=bool)
            mask[y:bottom, x:right] = crop_mask_array
            all_masks.append(torch.from_numpy(mask)[None])
            crop_area = (right - x) * (bottom - y)
            all_scores.append(
                torch.full_like(crop_scores[index : index + 1], 1 / crop_area)
            )
            box = crop_boxes[index].clone()
            box[[0, 2]] += x
            box[[1, 3]] += y
            all_boxes.append(box[None])
    collected = _finalize_automatic_masks(
        torch.cat(all_masks), torch.cat(all_scores), torch.cat(all_boxes)
    )
    return [_restore_mask(mask, original_size) for mask in collected]


def _components(mask: np.ndarray, min_pixels: int) -> list[np.ndarray]:
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
        min_y = min(y for y, _start, _end in runs)
        max_y = max(y for y, _start, _end in runs)
        min_x = min(start for _y, start, _end in runs)
        max_x = max(end for _y, _start, end in runs)
        local = np.zeros((max_y - min_y + 1, max_x - min_x), dtype=bool)
        for y, start, end in runs:
            local[y - min_y, start - min_x : end - min_x] = True
        component = np.zeros((height, width), dtype=bool)
        for y, start, end in runs:
            component[y, start:end] = True
        # A hole must contain at least one non-component pixel strictly inside
        # this box. Most small SAM fragments are solid or only touch the box
        # boundary, so avoid a connected-components pass when a hole is
        # impossible.
        has_interior_background = (
            local.shape[0] > 2 and local.shape[1] > 2 and not local[1:-1, 1:-1].all()
        )
        if has_interior_background:
            # AMG's postprocessing removes *small* enclosed holes, rather
            # than turning meaningful cutouts such as an eye into a solid
            # region.  The same area cutoff as tiny components keeps those
            # two decisions consistent.
            # The exterior background necessarily reaches a component bounding
            # box edge, while an enclosed hole cannot.  Checking this compact
            # box is equivalent to checking the full mask, without scanning a
            # 1024px canvas once for every small disconnected component.
            local_height, local_width = local.shape
            for hole in _run_components(~local):
                area = sum(end - start for _y, start, end in hole)
                if area > min_pixels:
                    continue
                touches_border = any(
                    y in {0, local_height - 1} or start == 0 or end == local_width
                    for y, start, end in hole
                )
                if not touches_border:
                    for y, start, end in hole:
                        component[y + min_y, start + min_x : end + min_x] = True
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
        revised.append(MaskLayer(layer.mask, colour, layer.impact))
        covered_above |= layer.mask
    return list(reversed(revised))


def _impact_error_map(
    target: np.ndarray, canvas: np.ndarray, coverage: np.ndarray
) -> np.ndarray:
    """SAMVG's blank-canvas error, charging uncovered pixels maximally."""
    error = ((target.astype(np.float32) - canvas.astype(np.float32)) / 255.0) ** 2
    error[~coverage] = 1.0
    return error


def filter_by_impact(
    image: Image.Image,
    masks: list[np.ndarray],
    *,
    existing: list[MaskLayer] | None = None,
    min_pixels: int = MIN_PIXELS,
    min_impact: float = MIN_IMPACT,
    max_layers: int = 128,
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
    error_map = _impact_error_map(target, canvas, coverage)
    error_total = float(error_map.sum(dtype=np.float64))
    error = error_total / error_map.size
    # SAMVG filters an AMG *mask* by its rendered impact, after AMG's component
    # cleanup. Components are independent paths only in the subsequent tracing
    # stage. Scoring every disconnected component here changes the paper's
    # painter-order decision and promotes low-information rectangular fragments.
    candidates = []
    for raw_mask in masks:
        if np.asarray(raw_mask).shape != (height, width):
            continue
        components = _components(np.asarray(raw_mask, dtype=bool), min_pixels)
        if not components:
            continue
        mask = np.logical_or.reduce(components)
        candidates.append((mask, components))
    candidates.sort(key=lambda candidate: int(candidate[0].sum()), reverse=True)
    retained: list[tuple[list[np.ndarray], tuple[int, int, int], float]] = []
    for mask, components in candidates:
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
        retained.append((components, colour, impact))
        canvas[mask] = colour
        coverage |= mask
        error_map[mask] = next_error_values
        error_total, error = next_error_total, next_error
        # Each SAMVG stage is allowed its own retained-mask budget.  Applying
        # this to the combined existing+new list silently limited recovery to
        # one path once the automatic stage had filled its budget.
        if len(retained) >= max_layers:
            break
    for components, colour, impact in retained:
        accepted.extend(
            MaskLayer(component, colour, impact) for component in components
        )
    return accepted


def coverage_prompt_points(
    layers: list[MaskLayer],
    shape: tuple[int, int],
    *,
    radius_fraction: float = 0.06,
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
    return [(x, y) for _distance, x, y in ranked]


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
    device = device_name(torch)
    log.info("SAMVG prompted masks: using %s.", device)
    if runtime.processor is None:
        runtime.processor = SamProcessor(runtime.generator.image_processor)
    try:
        output_masks = []
        for start in range(0, len(points), SAMVG_POINTS_PER_BATCH):
            batch = points[start : start + SAMVG_POINTS_PER_BATCH]
            input_points = [[[list(point)] for point in batch]]
            inputs = runtime.processor(
                images=image, input_points=input_points, return_tensors="pt"
            ).to(device)
            if (
                runtime.embedding_size == image.size
                and runtime.image_embeddings is not None
            ):
                # The full-image automatic pass has already encoded these pixels.
                # Retain only decoder inputs for the coverage prompts.
                inputs.pop("pixel_values")
                inputs["image_embeddings"] = runtime.image_embeddings
            with torch.inference_mode(), _sam_autocast():
                output = runtime.generator.model(**inputs)
            post = runtime.processor.image_processor.post_process_masks(
                output.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                inputs["reshaped_input_sizes"].detach().cpu(),
            )[0]
            output_masks.extend(
                _restore_mask(
                    np.asarray(post[prompt, candidate], dtype=bool), original_size
                )
                for prompt in range(post.shape[0])
                for candidate in range(post.shape[1])
            )
        return output_masks
    finally:
        if own_runtime:
            _release(runtime)


def retrieve_layers(
    image: Image.Image,
    *,
    min_pixels: int = MIN_PIXELS,
    min_impact: float = MIN_IMPACT,
    max_layers: int = 512,
    max_side: int | None = SAMVG_MAX_SIDE,
    model: str = SAMVG_MODEL,
    _runtime: _SamRuntime | None = None,
) -> list[MaskLayer]:
    """Run SAMVG's automatic-mask, coverage-prompt, filter sequence.

    A model loaded here is released, and its GPU memory returned, before
    this returns: the tracing that follows needs no GPU.
    """
    image = image.convert("RGB")
    runtime = _runtime or _sam_runtime(model=model)
    try:
        initial = automatic_masks(image, max_side=max_side, _runtime=runtime)
        layers = filter_by_impact(
            image,
            initial,
            min_pixels=min_pixels,
            min_impact=min_impact,
            max_layers=max_layers,
        )
        points = coverage_prompt_points(layers, (image.height, image.width))
        prompted = prompted_masks(image, points, max_side=max_side, _runtime=runtime)
    finally:
        if _runtime is None:
            _release(runtime)
    recovered = filter_by_impact(
        image,
        prompted,
        existing=layers,
        min_pixels=min_pixels,
        min_impact=min_impact,
        max_layers=max_layers,
    )
    log.info(
        "SAMVG first pass: %d automatic mask(s), %d retained; %d coverage "
        "prompt(s), %d prompted mask(s), %d total retained.",
        len(initial),
        len(layers),
        len(points),
        len(prompted),
        len(recovered),
    )
    # Mask selection intentionally scores the initially painted colours: that
    # is the paper's greedy impact procedure.  Once painter order is fixed,
    # however, a lower layer should be coloured from only the pixels it still
    # exposes.  This is the least-squares fill for the emitted seed and does
    # not alter its accepted masks, ordering, or coverage prompts.
    return recolour_visible_layers(image, recovered)


def _release(runtime: _SamRuntime) -> None:
    """Drop *runtime*'s model and embedding and hand their memory back."""
    import gc

    runtime.generator = runtime.processor = runtime.image_embeddings = None
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - installation-specific
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _loops(mask: np.ndarray) -> list[list[tuple[float, float]]]:
    """Trace pixel-boundary loops, retaining exterior and hole contours."""
    edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    height, width = mask.shape
    # Only a pixel with a neighbour outside has an edge; walking just those,
    # in the same order, leaves a large region's interior out of the loop.
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    inner = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2]
    inner &= padded[1:-1, 2:]
    for y, x in zip(*np.nonzero(mask & ~inner), strict=True):
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


def _curvature_scores(loop: list[tuple[float, float]]) -> np.ndarray:
    """Return SAMVG's scale-aware cosine curvature score for a contour."""
    points = np.asarray(loop, dtype=np.float32)
    size = len(points)
    step = max(1, size // 12)
    before = points - np.roll(points, step, axis=0)
    after = np.roll(points, -step, axis=0) - points
    denom = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    return np.divide(
        (before * after).sum(axis=1), denom, out=np.ones(size), where=denom > 0
    )


def _corners(loop: list[tuple[float, float]], count: int) -> list[int]:
    """Global curvature maxima with the local exclusion SAMVG describes."""
    size = len(loop)
    count = min(count, size)
    score = _curvature_scores(loop)
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


def _smoothed(loop: list[tuple[float, float]], sigma: float):
    """*loop* smoothed along its length by a Gaussian of *sigma* pixels.

    A pixel loop walks every step of a raster staircase, and with enough
    curves the fit follows each one: a mask SAM made at a lower resolution
    than the image comes out as steps a SAM pixel wide. Smoothing over about
    that width takes the steps out and rounds a real corner by as much.
    """
    if sigma <= 0 or len(loop) < 3:
        return loop
    points = np.asarray(loop, dtype=np.float64)
    reach = min(int(np.ceil(3 * sigma)), (len(points) - 1) // 2)
    offsets = np.arange(-reach, reach + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    smooth = np.zeros_like(points)
    for offset, weight in zip(offsets, weights, strict=True):
        smooth += weight * np.roll(points, -offset, axis=0)
    return [(float(x), float(y)) for x, y in smooth]


def _cubic_loop(
    loop: list[tuple[float, float]], segments: int, *, smooth: float = 0.0
) -> str | None:
    size = len(loop)
    if size < 3:
        return None
    loop = _smoothed(loop, smooth)
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
        # ``indices`` already includes the endpoint.  Repeating it adds an
        # artificial least-squares weight at every selected corner and bends
        # each fitted cubic toward its end point rather than the contour data.
        sample = points[indices]
        control_a, control_b = _fit_cubic(sample)
        end = points[second]
        parts.append(
            f"C {control_a[0]:.2f} {control_a[1]:.2f} "
            f"{control_b[0]:.2f} {control_b[1]:.2f} {end[0]:.2f} {end[1]:.2f}"
        )
    return " ".join(parts) + " Z"


def mask_path(
    mask: np.ndarray, *, smooth: float = 0.0, density: int = DENSITY
) -> str | None:
    """Fit every mask contour with cubic Beziers, one per *density* pixels of
    its length, each contour first smoothed over *smooth* pixels."""
    parts = [
        piece
        for loop in _loops(mask)
        if (piece := _cubic_loop(loop, max(4, len(loop) // density), smooth=smooth))
    ]
    return " ".join(parts) or None


def thinner_than(mask: np.ndarray, width: int) -> bool:
    """Whether *mask* is narrower than *width* pixels everywhere.

    Eroding by half the width empties a region that nowhere reaches that
    thickness: a traced outline or hairline, where a region of any real size
    keeps a core.
    """
    radius = (width - 1) // 2
    if radius <= 0:
        return False
    return not (~_binary_dilation(~np.asarray(mask, dtype=bool), radius)).any()


def arrange_layers(
    layers: list[MaskLayer], *, min_width: int = MIN_WIDTH
) -> list[MaskLayer]:
    """Settle which layers are traced.

    Too-thin layers go first, since removing one can uncover what is beneath.
    A layer the ones above it hide completely paints nothing and is dropped.
    """
    if min_width:
        layers = [layer for layer in layers if not thinner_than(layer.mask, min_width)]
    if not layers:
        return layers
    above = np.zeros(layers[0].mask.shape, dtype=bool)
    kept: list[MaskLayer] = []
    for layer in reversed(layers):
        if (layer.mask & ~above).any():
            kept.append(layer)
        above |= layer.mask
    return kept[::-1]


def refine_edges(
    layers: list[MaskLayer],
    image: Image.Image,
    *,
    band: int,
    smoothness: float = EDGE_SMOOTHNESS,
) -> list[MaskLayer]:
    """Move region edges onto the image's own edges, at its full resolution.

    SAM decodes a mask on a coarse grid, a few image pixels a cell, so a thin
    spike or notch narrower than that is smoothed away. Here every pixel
    within *band* of an edge between regions may change to a neighbouring
    region whose colour it matches better, paying *smoothness* (squared RGB,
    0-1) for each neighbour it then differs from; repeating that grows a
    region back into the spikes it lost, one pixel a round. A pixel only
    changes to a region it touches, and only which region shows where
    changes: a pixel that changed belongs to its new region alone.
    """
    if not layers or band <= 0:
        return layers
    target = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    owner = np.zeros(layers[0].mask.shape, dtype=np.int32)
    for index, layer in enumerate(layers, start=1):
        owner[layer.mask] = index
    start = owner.copy()
    count = len(layers) + 1
    sums = np.zeros((count, 3))
    for channel in range(3):
        sums[:, channel] = np.bincount(
            owner.ravel(), weights=target[..., channel].ravel(), minlength=count
        )
    sizes = np.bincount(owner.ravel(), minlength=count)
    colours = sums / np.maximum(sizes, 1)[:, None]
    edge = np.zeros(owner.shape, dtype=bool)
    edge[1:] |= owner[1:] != owner[:-1]
    edge[:-1] |= owner[1:] != owner[:-1]
    edge[:, 1:] |= owner[:, 1:] != owner[:, :-1]
    edge[:, :-1] |= owner[:, 1:] != owner[:, :-1]
    near = _binary_dilation(edge, band) & (owner > 0)
    shifts = ((0, 1), (0, -1), (1, 1), (1, -1))
    for _ in range(band):
        neighbours = [np.roll(owner, shift, axis=axis) for axis, shift in shifts]
        best = owner.copy()
        best_cost = np.full(owner.shape, np.inf)
        for candidate in [owner, *neighbours]:
            data = np.sum((target - colours[candidate]) ** 2, axis=2)
            differing = sum((n != candidate).astype(np.float64) for n in neighbours)
            cost = data + smoothness * differing
            take = near & (candidate > 0) & (cost < best_cost)
            best[take] = candidate[take]
            best_cost[take] = cost[take]
        if np.array_equal(best, owner):
            break
        owner = best
    changed = owner != start
    if not changed.any():
        return layers
    result = []
    for index, layer in enumerate(layers, start=1):
        # A pixel that changed region belongs to that region alone.
        mask = layer.mask & ~changed | (changed & (owner == index))
        result.append(replace(layer, mask=mask))
    return result


def merge_patches(
    layers: list[MaskLayer], image: Image.Image, *, min_impact: float
) -> list[MaskLayer]:
    """Merge neighbouring regions whose difference is not worth a shape.

    SAM returns many small masks along the edges between larger regions, and
    regions of one colour cut in pieces, each traced as a shape of its own.
    Two regions painted in their own mean colours differ from one painted in
    the mean of both by n1 n2 / (n1 + n2) |c1 - c2|^2, summed over what each
    shows. Where that is below *min_impact*, the least the impact filter
    keeps a layer for, the smaller region joins its neighbour: a tiny patch
    merges even when its colour is off, a large region only when the colours
    all but match. Smallest first, into the neighbour it costs least to join.
    """
    if not layers:
        return layers
    target = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    height, width = layers[0].mask.shape
    # The impact filter's error is a mean over every pixel and channel; each
    # merge may cost a share of what keeps a layer, since hundreds add up.
    budget = MERGE_SHARE * min_impact * height * width * 3
    # Which layer shows at each pixel: the last one to paint it.
    owner = np.zeros((height, width), dtype=np.int32)
    for index, layer in enumerate(layers, start=1):
        owner[layer.mask] = index
    masks = {index: layer.mask.copy() for index, layer in enumerate(layers, 1)}
    areas = np.bincount(owner.ravel(), minlength=len(layers) + 1)
    colours = np.zeros((len(layers) + 1, 3))
    for index in masks:
        if areas[index]:
            colours[index] = target[owner == index].mean(axis=0)
    for patch in sorted(masks, key=lambda index: areas[index]):
        if not areas[patch]:
            continue
        ys, xs = np.nonzero(owner == patch)
        top, left = max(ys.min() - 1, 0), max(xs.min() - 1, 0)
        window = owner[top : ys.max() + 2, left : xs.max() + 2]
        shown = window == patch
        ring = _binary_dilation(shown, 1) & ~shown
        best, cheapest = 0, budget
        for neighbour in np.unique(window[ring]):
            if neighbour in (0, patch) or not areas[neighbour]:
                continue
            n1, n2 = areas[patch], areas[neighbour]
            cost = (
                n1
                * n2
                / (n1 + n2)
                * float(np.sum((colours[patch] - colours[neighbour]) ** 2))
            )
            if cost < cheapest:
                best, cheapest = int(neighbour), cost
        if not best:
            continue
        n1, n2 = areas[patch], areas[best]
        colours[best] = (colours[patch] * n1 + colours[best] * n2) / (n1 + n2)
        areas[best] += n1
        areas[patch] = 0
        masks[best] |= owner == patch
        owner[owner == patch] = best
        masks.pop(patch)
    return [
        replace(
            layer,
            mask=masks[index],
            colour=cast(
                tuple[int, int, int],
                tuple(int(v) for v in np.rint(colours[index] * 255)),
            ),
        )
        if areas[index]
        else replace(layer, mask=masks[index])
        for index, layer in enumerate(layers, start=1)
        if index in masks
    ]


def backdrop_colour(
    image: Image.Image, layers: list[MaskLayer]
) -> tuple[int, int, int]:
    """The colour of what no layer claims, for a rectangle beneath them all.

    SAM leaves drawn outlines to neither neighbour, so the unclaimed pixels are
    mostly outline, and filling beneath in their colour makes the gaps read as
    the outlines they were.
    """
    pixels = np.asarray(image.convert("RGB"))
    unclaimed = ~np.any([layer.mask for layer in layers], axis=0) if layers else None
    chosen = pixels[unclaimed] if unclaimed is not None and unclaimed.any() else None
    if chosen is None:
        chosen = pixels.reshape(-1, 3)
    red, green, blue = (int(v) for v in np.median(chosen, axis=0))
    return red, green, blue


def _layer_svg_attributes(
    layer: MaskLayer, *, smooth: float = 0.0, tolerance: float = TOLERANCE
) -> dict[str, str] | None:
    """Trace one SAM mask as a filled path, simplified to *tolerance* pixels."""
    data = mask_path(layer.mask, smooth=smooth)
    if data is None:
        return None
    colour = f"#{layer.colour[0]:02x}{layer.colour[1]:02x}{layer.colour[2]:02x}"
    return {
        "d": _simplified_data(data, tolerance),
        "fill": colour,
        "fill-rule": "evenodd",
    }


def _simplified_data(data: str, tolerance: float) -> str:
    """Path *data*, in pixels, with the points it does not need removed."""
    from vectrify.document.svg import parse_path
    from vectrify.refine.frozen import Frozen
    from vectrify.refine.simplify import simplified_geometry

    geometry = simplified_geometry(parse_path(data), Frozen(frozenset()), tolerance)
    # Hundredths of a pixel, as the tracer writes them.
    return re.sub(
        r"-?\d+\.\d+(?:e-?\d+)?",
        lambda number: f"{float(number.group()):.2f}",
        geometry.path_data(),
    )


def generate_svg(
    image: Image.Image,
    masks: list[np.ndarray] | None = None,
    *,
    max_layers: int = 512,
    max_side: int | None = SAMVG_MAX_SIDE,
    model: str = SAMVG_MODEL,
    min_pixels: int = MIN_PIXELS,
    min_impact: float = MIN_IMPACT,
    min_width: int = MIN_WIDTH,
    tolerance: float = TOLERANCE,
) -> str:
    """Trace *image*'s SAM segments, or the given *masks*, into an SVG.

    Regions too thin to be one, and those hidden by the ones above, are left
    out, the edges moved onto the image's own and small patches merged into
    a neighbour. A rectangle beneath them all fills what none claims.
    """
    image = image.convert("RGB")
    if masks is None:
        layers = retrieve_layers(
            image,
            min_pixels=min_pixels,
            min_impact=min_impact,
            max_layers=max_layers,
            max_side=max_side,
            model=model,
        )
    else:
        layers = filter_by_impact(
            image,
            masks,
            min_pixels=min_pixels,
            min_impact=min_impact,
            max_layers=max_layers,
        )
        # ``retrieve_layers`` has already done this for SAM's own masks.
        layers = recolour_visible_layers(image, layers)
    layers = arrange_layers(layers, min_width=min_width)
    width, height = image.size
    # SAM's masks have steps of one SAM pixel, which is more than one of the
    # image's when SAM worked at a smaller size.
    scale = max(1.0, max(width, height) / max_side) if max_side else 1.0
    layers = refine_edges(layers, image, band=round(EDGE_BAND * scale))
    layers = merge_patches(layers, image, min_impact=min_impact)
    red, green, blue = backdrop_colour(image, layers)
    paths = [
        f'<rect width="{width}" height="{height}" '
        f'fill="#{red:02x}{green:02x}{blue:02x}" />'
    ]
    for layer in layers:
        attributes = _layer_svg_attributes(
            layer, smooth=SMOOTH * scale, tolerance=tolerance
        )
        if attributes is not None:
            markup = " ".join(f'{key}="{value}"' for key, value in attributes.items())
            paths.append(f"<path {markup} />")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(paths) + "</svg>"
    )
