"""Retrace a shape: give a path the outline of its object in the reference.

SAM is prompted with the path itself: its bounding box, points deep inside
it, points just outside it that look different, and its coverage as a mask.
Of the masks that come back, the one that best agrees with the path while
keeping to one colour in the reference wins. Without a GPU, or when asked,
the object is instead grown from the path's interior by colour similarity.
Either region is then traced the way SAMVG traces its segments: edges moved
onto the reference's own, outlines smoothed, fitted densely with cubics and
simplified to a pixel tolerance. Holes in the region stay holes.

Encoding the reference is the slow part of a SAM prompt, so the model and
the reference's embedding are kept in ``SAM_CACHE`` between retraces and
dropped after a few idle minutes, when the reference changes, or before
another GPU job needs the memory.
"""

from __future__ import annotations

import functools
import hashlib
import io
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from threading import Lock, Timer
from typing import Any, cast

import cairosvg
import numpy as np
from PIL import Image
from scipy import ndimage

from vectrify.document import Document, Geometry
from vectrify.document.svg import parse_path
from vectrify.operations.generate import Region
from vectrify.refine.samvg import (
    DENSITY,
    EDGE_BAND,
    EDGE_SMOOTHNESS,
    MIN_PIXELS,
    SAMVG_MODEL,
    SMOOTH,
    TOLERANCE,
    MaskLayer,
    _binary_dilation,
    _components,
    _release,
    _sam_autocast,
    _sam_runtime,
    _simplified_data,
    mask_path,
    refine_edges,
)
from vectrify.refine.snap import _Frame, _frame

log = logging.getLogger(__name__)

# How long, in seconds, an unused model and embedding stay loaded.
IDLE_SECONDS = 180.0
# SAM's input size: it encodes an image at this long side.
SAM_SIDE = 1024
# How many points go inside the path, and at most outside it.
POSITIVES = 3
NEGATIVES = 4
# A point outside the path is a negative prompt only when its colour is at
# least this far (RGB, 0-1) from the path's inside.
NEGATIVE_CONTRAST = 0.15
# How far outside the path, as a share of its size, the negatives sit.
NEGATIVE_REACH = 0.08
# The path's coverage as a mask prompt: SAM takes logits, so inside and
# outside become this much either side of its threshold.
MASK_LOGIT = 6.0
# How much a mask's outline lying on the reference's edges counts in its
# choice, against its IoU with the path, and how near (pixels, a square's
# side) an edge must be to count.
EDGE_WEIGHT = 1.0
EDGE_REACH = 5
# The colour fallback grows over pixels this close (RGB distance, 0-1) to
# the path's inside, within this share of the path's size around it.
COLOUR_TOLERANCE = 0.1
COLOUR_MARGIN = 0.25
# The reference is blurred by this many pixels first, so texture and noise
# do not stop the growth.
COLOUR_BLUR = 1.0
# How far, in pixels, edges may move onto the reference's in colour mode;
# its regions already follow the reference's pixels.
COLOUR_BAND = 2


@dataclass(frozen=True)
class Prompt:
    """One SAM prompt, in the reference's pixels."""

    box: tuple[float, float, float, float] | None = None
    points: tuple[tuple[float, float], ...] = ()
    labels: tuple[int, ...] = ()
    # The path's coverage over the whole reference.
    mask: np.ndarray | None = field(default=None, compare=False)


@dataclass
class Encoding:
    """A reference's SAM embedding and the sizes that locate it."""

    embedding: Any
    # (height, width) of the reference and of SAM's resized input.
    original: tuple[int, int]
    reshaped: tuple[int, int]


class TransformersSam:
    """SAM through transformers: load, encode an image once, decode prompts."""

    def load(self, model: str) -> Any:
        from transformers import SamProcessor

        runtime = _sam_runtime(model=model)
        runtime.processor = SamProcessor(runtime.generator.image_processor)
        return runtime

    def encode(self, runtime: Any, image: Image.Image) -> Encoding:
        import torch

        model = runtime.generator.model
        inputs = runtime.processor(images=image, return_tensors="pt")
        pixels = inputs["pixel_values"].to(model.device, model.dtype)
        with torch.inference_mode(), _sam_autocast():
            embedding = model.get_image_embeddings(pixels)
        runtime.image_embeddings, runtime.embedding_size = embedding, image.size
        original = inputs["original_sizes"][0].tolist()
        reshaped = inputs["reshaped_input_sizes"][0].tolist()
        return Encoding(embedding, tuple(original), tuple(reshaped))

    def decode(
        self, runtime: Any, encoding: Encoding, prompt: Prompt
    ) -> list[tuple[np.ndarray, float]]:
        import torch

        model = runtime.generator.model
        device = model.device
        (height, width), (inner_h, inner_w) = encoding.original, encoding.reshaped
        scale = np.array([inner_w / width, inner_h / height])
        inputs: dict[str, Any] = {"image_embeddings": encoding.embedding}
        if prompt.points:
            points = np.asarray(prompt.points, dtype=np.float32) * scale
            inputs["input_points"] = torch.tensor(points, device=device)[None, None]
            inputs["input_labels"] = torch.tensor(
                prompt.labels, dtype=torch.int64, device=device
            )[None, None]
        if prompt.box is not None:
            box = np.asarray(prompt.box, dtype=np.float32) * np.tile(scale, 2)
            inputs["input_boxes"] = torch.tensor(box, device=device)[None, None]
        if prompt.mask is not None:
            inputs["input_masks"] = torch.tensor(
                _mask_logits(prompt.mask, (inner_h, inner_w)), device=device
            )[None, None]
        with torch.inference_mode(), _sam_autocast():
            output = model(**inputs, multimask_output=True)
            masks = runtime.processor.image_processor.post_process_masks(
                output.pred_masks.float(),
                [list(encoding.original)],
                [list(encoding.reshaped)],
            )[0][0]
        scores = output.iou_scores[0, 0].float().cpu().tolist()
        return [
            (np.asarray(mask.cpu(), dtype=bool), float(score))
            for mask, score in zip(masks, scores, strict=True)
        ]

    def release(self, runtime: Any) -> None:
        _release(runtime)


def _mask_logits(mask: np.ndarray, reshaped: tuple[int, int]) -> np.ndarray:
    """*mask* as the low-resolution logits SAM takes as a mask prompt.

    SAM's mask input covers its padded square input at a quarter of its size.
    """
    height, width = reshaped
    resized = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    square = np.zeros((SAM_SIDE, SAM_SIDE), dtype=np.float32)
    square[:height, :width] = np.asarray(resized, dtype=np.float32) / 255
    low = Image.fromarray(square).resize(
        (SAM_SIDE // 4, SAM_SIDE // 4), Image.Resampling.BOX
    )
    return ((np.asarray(low, dtype=np.float32) * 2 - 1) * MASK_LOGIT).astype(np.float32)


def _digest(image: Image.Image) -> str:
    return hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()


class SamCache:
    """One SAM model and one reference's embedding, kept between retraces.

    A retrace of the same reference with the same model reuses both, so only
    SAM's prompt decoder runs. Everything is dropped, and the GPU memory
    handed back, after *idle* seconds without use or on ``release``.
    """

    def __init__(
        self,
        backend: Any = None,
        *,
        idle: float | None = IDLE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.backend = backend or TransformersSam()
        self.idle = idle
        self.clock = clock
        self._lock = Lock()
        self._runtime: Any = None
        self._model: str | None = None
        self._key: tuple | None = None
        self._encoding: Encoding | None = None
        self._used = 0.0
        self._timer: Timer | None = None

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    @property
    def encoded(self) -> bool:
        return self._encoding is not None

    def masks(
        self,
        image: Image.Image,
        prompts: list[Prompt],
        *,
        model: str = SAMVG_MODEL,
        progress: Callable[[str], None] | None = None,
    ) -> list[list[tuple[np.ndarray, float]]]:
        """SAM's three masks and predicted IoUs for each prompt on *image*."""
        report = progress or (lambda _message: None)
        with self._lock:
            if self._runtime is not None and self._model != model:
                self._drop()
            if self._runtime is None:
                report("Loading SAM…")
                self._runtime, self._model = self.backend.load(model), model
            key = (model, image.size, image.mode, _digest(image))
            if self._key != key or self._encoding is None:
                report("Encoding the reference with SAM…")
                self._encoding = None
                self._encoding = self.backend.encode(self._runtime, image)
                self._key = key
            report("Prompting SAM…")
            results = [
                self.backend.decode(self._runtime, self._encoding, prompt)
                for prompt in prompts
            ]
            self._used = self.clock()
            self._schedule()
            return results

    def release(self) -> None:
        """Drop the model and embedding now."""
        with self._lock:
            self._drop()

    def expire(self) -> bool:
        """Release if unused for the idle time; whether it did."""
        with self._lock:
            if self._runtime is None or self.idle is None:
                return False
            if self.clock() - self._used < self.idle:
                return False
            self._drop()
            return True

    def _schedule(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
        if self.idle is None:
            return
        self._timer = Timer(self.idle, self.expire)
        self._timer.daemon = True
        self._timer.start()

    def _drop(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        runtime, self._runtime = self._runtime, None
        self._encoding = self._key = self._model = None
        _EDGES.clear()
        if runtime is not None:
            self.backend.release(runtime)
            log.info("Released the retrace SAM model and its embedding.")


# The one cache the editor's retraces share: a single model in memory.
SAM_CACHE = SamCache()


@functools.cache
def sam_problem() -> str | None:
    """Why SAM cannot retrace here, or None when it can."""
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError:
        return "Retracing with SAM needs the samvg extra"
    if not torch.cuda.is_available():
        return "Retracing with SAM needs an NVIDIA GPU"
    return None


@dataclass(frozen=True)
class Target:
    """One path to retrace, located in the reference's pixels."""

    oid: str
    geometry: Geometry
    fill_rule: str
    frame: _Frame
    # Its coverage over the whole reference, and its pixel bounds.
    coverage: np.ndarray
    bounds: tuple[int, int, int, int]


def reference_region(document: Document, reference: Image.Image) -> Region:
    """The whole reference over the artboard it is stretched across."""
    return Region(*document.artboard(), reference)


def target(
    document: Document, oid: str, reference: Image.Image, fill_rule: str
) -> Target | None:
    """*oid* in the reference's pixels, or None when it covers none of them."""
    frame = _frame(document, oid, reference_region(document, reference), reference.size)
    if frame is None:
        return None
    geometry = document.geometry_for(oid)
    coverage = _coverage(geometry, fill_rule, frame, reference.size)
    ys, xs = np.nonzero(coverage)
    if not len(xs):
        return None
    bounds = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return Target(oid, geometry, fill_rule, frame, coverage, bounds)


def _coverage(
    geometry: Geometry, fill_rule: str, frame: _Frame, size: tuple[int, int]
) -> np.ndarray:
    """Which of the reference's pixels *geometry* covers at least half of."""
    (a, c), (b, d) = frame.matrix.tolist()
    e, f = frame.offset.tolist()
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size[0]}" '
        f'height="{size[1]}"><path transform="matrix({a!r} {b!r} {c!r} {d!r} '
        f'{e!r} {f!r})" d="{geometry.path_data()}" fill="#000" '
        f'fill-rule="{fill_rule}"/></svg>'
    )
    png = cairosvg.svg2png(bytestring=svg.encode())
    assert png is not None
    with Image.open(io.BytesIO(png)) as image:
        alpha = np.asarray(image.convert("RGBA"))[..., 3]
    return alpha >= 128


def _spread(candidates: np.ndarray, count: int) -> list[tuple[int, int]]:
    """Up to *count* of *candidates* (x, y rows), each far from the others."""
    if not len(candidates):
        return []
    chosen = [candidates[0]]
    distance = np.linalg.norm(candidates - candidates[0], axis=1)
    while len(chosen) < count and distance.max() > 0:
        index = int(distance.argmax())
        chosen.append(candidates[index])
        distance = np.minimum(
            distance, np.linalg.norm(candidates - candidates[index], axis=1)
        )
    return [(int(x), int(y)) for x, y in chosen]


def _depth(mask: np.ndarray) -> np.ndarray:
    """How far each pixel of *mask* is from the nearest pixel outside it."""
    return np.asarray(ndimage.distance_transform_edt(mask), dtype=np.float64)


def _inside_colour(pixels: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """The colour of the path's inside: the median of its deepest pixels."""
    depth = _depth(coverage)
    deep = depth >= max(1.0, 0.5 * float(depth.max()))
    return np.median(pixels[deep], axis=0)


def prompts(item: Target, reference: np.ndarray) -> list[Prompt]:
    """The SAM prompts tried for *item*; *reference* is RGB in 0-1.

    With and without the box and the mask, since SAM reads each differently
    and the choice between the masks sorts it out.
    """
    left, top, right, bottom = item.bounds
    height, width = item.coverage.shape
    size = max(right - left, bottom - top)
    pad = max(4, round(size * 0.5))
    x0, y0 = max(0, left - pad), max(0, top - pad)
    x1, y1 = min(width, right + pad), min(height, bottom + pad)
    coverage = item.coverage[y0:y1, x0:x1]
    pixels = reference[y0:y1, x0:x1]
    colour = _inside_colour(pixels, coverage)
    contrast = np.linalg.norm(pixels - colour, axis=2)
    depth = _depth(coverage)
    # Deep inside, where the colour is the inside's: the path may overlap
    # its neighbours, and a positive there would ask for them too.
    ys, xs = np.nonzero((depth >= 0.5 * depth.max()) & (contrast < NEGATIVE_CONTRAST))
    if not len(xs):
        ys, xs = np.nonzero(depth >= 0.5 * depth.max())
    order = np.argsort(-depth[ys, xs], kind="stable")
    positives = _spread(np.column_stack((xs, ys))[order], POSITIVES)
    reach = max(2.0, NEGATIVE_REACH * size)
    outside = _depth(~coverage)
    ring = (outside >= reach) & (outside < 2 * reach) & (contrast > NEGATIVE_CONTRAST)
    ys, xs = np.nonzero(ring)
    negatives = _spread(np.column_stack((xs, ys)), NEGATIVES)
    points = tuple((x + x0 + 0.5, y + y0 + 0.5) for x, y in positives + negatives)
    labels = (1,) * len(positives) + (0,) * len(negatives)
    box = (float(left), float(top), float(right), float(bottom))
    return [
        Prompt(box, points, labels, item.coverage),
        Prompt(box, points, labels),
        Prompt(None, points, labels),
    ]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 0.0


def edge_map(reference: np.ndarray) -> np.ndarray:
    """How strongly the reference changes colour at or near each pixel.

    The colour gradient, taken within a couple of pixels, relative to the
    reference's strong edges and capped at one.
    """
    blur = ndimage.gaussian_filter(reference, (1, 1, 0))
    gradient = np.sqrt(
        sum(
            ndimage.sobel(blur[..., channel], axis=axis) ** 2
            for channel in range(3)
            for axis in (0, 1)
        )
    )
    near = ndimage.grey_dilation(gradient, size=EDGE_REACH)
    strong = float(np.percentile(gradient, 99)) or 1.0
    return np.minimum(near / strong, 1.0)


_EDGES: dict[str, np.ndarray] = {}


def _edges(image: Image.Image, reference: np.ndarray) -> np.ndarray:
    """``edge_map`` of *image*, kept for the last reference retraced."""
    key = _digest(image)
    if key not in _EDGES:
        _EDGES.clear()
        _EDGES[key] = edge_map(reference)
    return _EDGES[key]


def choose_mask(
    candidates: list[np.ndarray],
    coverage: np.ndarray,
    reference: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """The candidate that best matches the path and the reference.

    Its score is its IoU with the path's coverage, less the RMS distance
    (RGB, 0-1) of its pixels from their mean, plus how strong the
    reference's edges are along its outline (see ``edge_map``). A mask that
    spills over a neighbour of another colour pays for it even where the
    path spills too, and the edge term rules out SAM repeating the path's
    own rough outline back where the reference has no edge.
    """
    best, best_score = None, -np.inf
    for mask in candidates:
        if np.count_nonzero(mask) < MIN_PIXELS:
            continue
        pixels = reference[mask]
        rms = float(np.sqrt(np.mean(np.sum((pixels - pixels.mean(0)) ** 2, 1))))
        ys, xs = np.nonzero(mask)
        box = np.s_[
            max(0, ys.min() - 1) : ys.max() + 2, max(0, xs.min() - 1) : xs.max() + 2
        ]
        inside = mask[box]
        outline = inside & ~ndimage.binary_erosion(inside)
        edge = float(edges[box][outline].mean())
        score = _iou(mask, coverage) - rms + EDGE_WEIGHT * edge
        if score > best_score:
            best, best_score = mask, score
    return best, float(best_score)


def colour_mask(
    item: Target, reference: np.ndarray, tolerance: float = COLOUR_TOLERANCE
) -> np.ndarray:
    """The region grown from *item*'s inside over pixels of its colour.

    Only pixels within a margin around the path are reached, and only
    regions a good part of the path's inside belongs to are kept.
    """
    left, top, right, bottom = item.bounds
    height, width = item.coverage.shape
    pad = max(4, round(COLOUR_MARGIN * max(right - left, bottom - top)))
    x0, y0 = max(0, left - pad), max(0, top - pad)
    x1, y1 = min(width, right + pad), min(height, bottom + pad)
    window = reference[y0:y1, x0:x1]
    if COLOUR_BLUR:
        window = ndimage.gaussian_filter(window, (COLOUR_BLUR, COLOUR_BLUR, 0))
    coverage = item.coverage[y0:y1, x0:x1]
    colour = _inside_colour(window, coverage)
    similar = np.linalg.norm(window - colour, axis=2) <= tolerance
    labels, count = ndimage.label(similar)
    mask = np.zeros(item.coverage.shape, dtype=bool)
    if not count:
        return mask
    overlap = np.bincount(labels[coverage & similar], minlength=count + 1)
    overlap[0] = 0
    if not overlap.any():
        return mask
    keep = np.flatnonzero(overlap >= 0.1 * overlap.max())
    mask[y0:y1, x0:x1] = np.isin(labels, keep)
    return mask


def _window(
    mask: np.ndarray, bounds: tuple[int, int, int, int], margin: int
) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    left, top, right, bottom = bounds
    if len(xs):
        left, top = min(left, int(xs.min())), min(top, int(ys.min()))
        right, bottom = max(right, int(xs.max()) + 1), max(bottom, int(ys.max()) + 1)
    height, width = mask.shape
    return (
        max(0, left - margin),
        max(0, top - margin),
        min(width, right + margin),
        min(height, bottom + margin),
    )


def _kept(mask: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """*mask*'s pieces that overlap the path, with pinholes filled."""
    pieces = [c for c in _components(mask, MIN_PIXELS) if (c & coverage).any()]
    return np.logical_or.reduce(pieces) if pieces else np.zeros_like(mask)


def trace(
    item: Target,
    mask: np.ndarray,
    reference: np.ndarray,
    image: Image.Image,
    *,
    band: int,
    smooth: float,
) -> Geometry | None:
    """*mask*'s outline in *item*'s own coordinates, holes and all.

    The edges are moved onto the reference's first, against a ring of what
    surrounds the region, then the outline is traced as SAMVG traces its
    regions and simplified to its pixel tolerance.
    """
    coverage = item.coverage
    mask = _kept(mask, coverage)
    if not mask.any():
        return None
    x0, y0, x1, y1 = _window(mask, item.bounds, 3 * band + 2)
    region = mask[y0:y1, x0:x1]
    crop = image.crop((x0, y0, x1, y1))
    if band > 0:
        ring = _binary_dilation(region, 3 * band) & ~region
        window = reference[y0:y1, x0:x1]
        layers = [MaskLayer(m, _mean_colour(window, m), 0.0) for m in (ring, region)]
        region = refine_edges(layers, crop, band=band, smoothness=EDGE_SMOOTHNESS)[
            1
        ].mask
        region = _kept(region, coverage[y0:y1, x0:x1])
    data = mask_path(region, smooth=smooth, density=DENSITY)
    if data is None:
        return None
    pixels = parse_path(_simplified_data(data, TOLERANCE))
    offset = np.array([x0, y0], dtype=np.float64)
    return _local(pixels, item.frame, offset)


def _mean_colour(pixels: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    mean = pixels[mask].mean(0) if mask.any() else np.zeros(3)
    return cast(tuple[int, int, int], tuple(int(v) for v in np.rint(mean * 255)))


def _local(geometry: Geometry, frame: _Frame, offset: np.ndarray) -> Geometry:
    """*geometry*, in a window's pixels at *offset*, in the path's own frame."""

    def mapped(values: tuple[float, ...]) -> tuple[float, ...]:
        points = np.asarray(values, dtype=np.float64).reshape(-1, 2) + offset
        return tuple(round(v, 4) for v in frame.local(points))

    return replace(
        geometry,
        subpaths=tuple(
            replace(
                s, nodes=tuple(replace(n, values=mapped(n.values)) for n in s.nodes)
            )
            for s in geometry.subpaths
        ),
    )


def sam_trace_options(reference: Image.Image) -> tuple[int, float]:
    """(edge band, smoothing) in reference pixels for SAM's masks.

    SAM's masks step once per SAM pixel, more than one reference pixel when
    the reference is larger than SAM's input.
    """
    scale = max(1.0, max(reference.size) / SAM_SIDE)
    return round(EDGE_BAND * scale), SMOOTH * scale


@dataclass
class Retraced:
    """What retracing one path gave."""

    geometry: Geometry | None
    iou: float = 0.0
    reason: str | None = None


def retrace_colour(
    item: Target,
    reference: np.ndarray,
    image: Image.Image,
    tolerance: float = COLOUR_TOLERANCE,
) -> Retraced:
    mask = colour_mask(item, reference, tolerance)
    if np.count_nonzero(mask) < MIN_PIXELS:
        return Retraced(None, reason="no region of its colour found")
    geometry = trace(item, mask, reference, image, band=COLOUR_BAND, smooth=SMOOTH)
    if geometry is None:
        return Retraced(None, reason="the region was too small to trace")
    return Retraced(geometry, _iou(mask, item.coverage))


def retrace_sam(
    items: list[Target],
    reference: np.ndarray,
    image: Image.Image,
    *,
    model: str = SAMVG_MODEL,
    cache: SamCache | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[Retraced]:
    """Each of *items* retraced from SAM's best mask for it."""
    cache = cache or SAM_CACHE
    asked = [prompts(item, reference) for item in items]
    flat = [prompt for group in asked for prompt in group]
    answers = iter(cache.masks(image, flat, model=model, progress=progress))
    band, smooth = sam_trace_options(image)
    edges = _edges(image, reference)
    results = []
    for item, group in zip(items, asked, strict=True):
        candidates = [mask for _ in group for mask, _score in next(answers)]
        mask, _score = choose_mask(candidates, item.coverage, reference, edges)
        if mask is None:
            results.append(Retraced(None, reason="SAM found no object there"))
            continue
        geometry = trace(item, mask, reference, image, band=band, smooth=smooth)
        if geometry is None:
            results.append(Retraced(None, reason="the object was too small to trace"))
            continue
        results.append(Retraced(geometry, _iou(mask, item.coverage)))
    return results
