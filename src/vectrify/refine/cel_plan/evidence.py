"""Multi-scale ink and surface evidence without a learned runtime dependency."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from threading import RLock

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label, median_filter

from vectrify.refine import cel
from vectrify.refine.cel_plan import opacity as opacity_models
from vectrify.refine.cel_plan.model import Evidence, Options, Work
from vectrify.refine.colour_regions import fit_palette, nearest_indices

EVIDENCE_VERSION = 3
MAX_ANALYSIS_SIDE = 1536
CACHE_BYTES = 96 * 1024 * 1024
_CACHE: OrderedDict[str, Evidence] = OrderedDict()
_LOCK = RLock()


def _size(value: Evidence) -> int:
    return sum(v.nbytes for v in vars(value).values() if isinstance(v, np.ndarray))


def _cache_put(key: str, value: Evidence):
    # Published arrays are read-only; candidates allocate their own label maps.
    for array in vars(value).values():
        if isinstance(array, np.ndarray):
            array.flags.writeable = False
    with _LOCK:
        if _size(value) > CACHE_BYTES:
            return
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while sum(_size(v) for v in _CACHE.values()) > CACHE_BYTES:
            _CACHE.popitem(last=False)


def collect(
    image: Image.Image, alpha: np.ndarray | None, options: Options, work: Work
) -> Evidence:
    started = time.monotonic()
    raw = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255
    if alpha is not None:
        opacity = np.asarray(alpha, dtype=np.float32)
        if opacity.shape != raw.shape[:2] or not np.isfinite(opacity).all():
            raise ValueError("Reference alpha must match the image and be finite")
        if np.any((opacity < 0) | (opacity > 1)):
            raise ValueError("Reference alpha must lie between zero and one")
        if not image.has_transparency_data:
            raw[..., :3] = np.clip(
                (raw[..., :3] - 1 + opacity[..., None])
                / np.maximum(opacity[..., None], 1e-6),
                0,
                1,
            )
        raw[..., 3] = opacity
    partial = opacity_models.needed(raw[..., 3])
    digest = hashlib.sha256(raw.tobytes())
    digest.update(
        str(
            (
                image.size,
                options.palette_size,
                options.line_width if partial else 0,
                EVIDENCE_VERSION,
            )
        ).encode()
    )
    key = digest.hexdigest()
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            work.timings["evidence"] = time.monotonic() - started
            return cached
    source = raw[..., :3] * 255
    opaque = raw[..., 3] > (opacity_models.VISIBLE if partial else 0.5 - 1e-8)
    background = None
    foreground = opaque.copy()
    if (raw[..., 3] >= 1 - 1 / 255).all():
        drawing = cel.silhouette(source)
        if drawing is not None:
            foreground = drawing
            border = np.concatenate(
                (source[0], source[-1], source[:, 0], source[:, -1])
            )
            color = np.median(border, axis=0) / 255
            background = (float(color[0]), float(color[1]), float(color[2]))
    ys, xs = np.nonzero(foreground)
    if len(xs):
        margin = 8
        x0, y0 = max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin)
        x1 = min(image.width, int(xs.max()) + margin + 1)
        y1 = min(image.height, int(ys.max()) + margin + 1)
    else:
        x0, y0, x1, y1 = 0, 0, image.width, image.height
    crop = raw[y0:y1, x0:x1]
    # Translucent topology needs native samples when a long, narrow crop still
    # fits the same bounded analysis pixel allowance. Resizing its byte-alpha
    # fringe can close holes and delete low-opacity connected marks.
    scale = min(
        1.0,
        MAX_ANALYSIS_SIDE / np.sqrt(crop.shape[0] * crop.shape[1])
        if partial
        else MAX_ANALYSIS_SIDE / max(crop.shape[:2]),
    )
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    if scale < 1:
        crop = (
            np.asarray(
                Image.fromarray((crop * 255).round().astype(np.uint8)).resize(
                    size, Image.Resampling.LANCZOS
                ),
                dtype=np.float32,
            )
            / 255
        )
    shown = crop[..., 3] > opacity_models.VISIBLE if partial else crop[..., 3] >= 0.5
    target = np.where(shown[..., None], crop[..., :3] * 255, 255)
    foreground = np.asarray(
        Image.fromarray(foreground[y0:y1, x0:x1]).resize(size, Image.Resampling.NEAREST)
    )
    grainy = cel.noise_level(target, ~shown) > cel.NOISE
    found = median_filter(target, size=(3, 3, 1)) if grainy else target
    line, darkness = cel.detect_lines(found, 3, shading=not grainy)
    line = cel.without_shapes(line) & shown
    drawn = line.copy()
    if grainy and not work.interrupted:
        kept, second = cel.detect_lines(cel.denoised(target), 3, shading=False)
        drawn |= cel.without_shapes(kept) | cel.ridge_lines(target, second)
        darkness = np.maximum(darkness, second)
    drawn &= shown
    # Color evidence is smoothed within the opaque support, never against the
    # white used only to make detection at the transparency border well-defined.
    support = shown & ~line
    weight = gaussian_filter(support.astype(np.float32), 2)
    smooth = gaussian_filter(target * support[..., None], (2, 2, 0))
    smooth /= np.maximum(weight[..., None], 1e-6)
    coarse = gaussian_filter(smooth, (4, 4, 0))
    residual = np.linalg.norm(target - smooth, axis=-1)
    texture = np.clip(gaussian_filter(residual, 3) / 24, 0, 1)
    if partial and options.line_width > 0:
        target, drawn = opacity_models.ink_width(
            target,
            smooth,
            drawn,
            shown,
            options.line_width,
            (size[0] / (x1 - x0), size[1] / (y1 - y0)),
        )
    filled = cel.trapped_ball_fill(shown & ~line)
    free = shown & ~line & (filled > 0)
    if partial:
        labels = opacity_models.labels(
            target,
            crop[..., 3],
            shown,
            options.palette_size,
            work,
            ink=drawn,
            smooth=smooth,
        )
    elif free.any() and not work.interrupted:
        palette = np.zeros(shown.shape, dtype=np.int32)
        count = min(options.palette_size, int(free.sum()))
        palette[free] = fit_palette(smooth[free][:, None, :], count, 16, gpu=False)[
            :, 0
        ]
        pieces = np.zeros(shown.shape, dtype=np.int32)
        offset = 0
        for index in range(count):
            found, n = label(free & (palette == index))
            pieces[found > 0] = found[found > 0] + offset
            offset += n
        _, labels = np.unique(
            pieces * (int(filled.max()) + 1) + filled, return_inverse=True
        )
        labels = labels.reshape(shown.shape).astype(np.int32)
        labels[~free] = 0
        sizes = np.bincount(labels.ravel())
        valid = (sizes[labels] >= 12) & free
        if valid.any():
            labels[~valid] = labels[nearest_indices(~valid)][~valid]
        else:
            labels = filled
    else:
        labels = filled
    if not partial:
        if labels.any():
            labels = labels[nearest_indices(labels == 0)]
        labels = np.where(shown, labels + 1, 0)
        _, labels = np.unique(labels, return_inverse=True)
    result = Evidence(
        raw,
        target,
        smooth,
        coarse,
        ~shown,
        foreground,
        line,
        drawn,
        darkness,
        texture,
        labels.reshape(shown.shape).astype(np.int32),
        image.size,
        (x0, y0),
        (size[0] / (x1 - x0), size[1] / (y1 - y0)),
        background,
        grainy,
        crop[..., 3].copy() if partial else None,
        options.line_width if partial else 0,
    )
    _cache_put(key, result)
    work.timings["evidence"] = time.monotonic() - started
    return result
