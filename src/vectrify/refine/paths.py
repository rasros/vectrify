"""Fit a group of stroked cubic paths to a target by gradient descent.

The differentiable rasterizer supports stroked cubics only. Pixels are sampled
at their centres, and paths in a group are fitted jointly so overlapping
strokes can move together.
"""

from __future__ import annotations

import io
import itertools
import logging
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from vectrify.formats.svg.ownership import drawable_elements

log = logging.getLogger(__name__)

# Absolute commands only: normalize_svg has already run, and a relative command
# here means the drawing skipped that pass rather than that it needs handling.
_TOKEN = re.compile(r"([MLCZmlcz])|(-?(?:\d+\.\d+|\.\d+|\d+))")
_SUPPORTED = frozenset("MLCZ")


class UnsupportedPathError(ValueError):
    """Raised for path data this rasterizer cannot represent exactly."""


def parse_cubics(d: str) -> list[list[tuple[float, float]]]:
    """Path data as a list of cubic segments, four control points each.

    Lines become cubics with their controls spaced along them, so the fit has
    one uniform representation to move and a straight edge can bend if the
    target curves.
    """
    groups: list[tuple[str, list[float]]] = []
    numbers: list[float] = []
    for token in _TOKEN.finditer(d):
        if token.group(1):
            command = token.group(1).upper()
            if command not in _SUPPORTED:
                raise UnsupportedPathError(
                    f"unsupported path command {token.group(1)!r}"
                )
            numbers = []
            groups.append((command, numbers))
        elif groups:
            numbers.append(float(token.group(2)))

    segments: list[list[tuple[float, float]]] = []
    current: tuple[float, float] | None = None
    start: tuple[float, float] | None = None
    for command, args in groups:
        if command == "M":
            for index in range(0, len(args) - 1, 2):
                point = (args[index], args[index + 1])
                if index == 0 or current is None:
                    current = start = point
                else:
                    segments.append(_as_cubic(current, point))
                    current = point
        elif command == "L":
            for index in range(0, len(args) - 1, 2):
                if current is None:
                    raise UnsupportedPathError("a lineto before any moveto")
                point = (args[index], args[index + 1])
                segments.append(_as_cubic(current, point))
                current = point
        elif command == "C":
            for index in range(0, len(args) - 5, 6):
                if current is None:
                    raise UnsupportedPathError("a curve before any moveto")
                point = (args[index + 4], args[index + 5])
                segments.append(
                    [
                        current,
                        (args[index], args[index + 1]),
                        (args[index + 2], args[index + 3]),
                        point,
                    ]
                )
                current = point
        elif current is not None and start is not None and current != start:
            segments.append(_as_cubic(current, start))
            current = start
    if not segments:
        raise UnsupportedPathError("path has no drawable segment")
    return segments


def parse_filled_cubics(d: str) -> list[list[list[tuple[float, float]]]]:
    """Parse an SVG fill into its independently closed cubic contours.

    SVG fills implicitly close an open subpath, and a path may contain several
    ``M … Z`` contours.  The stroke parser above intentionally flattens a
    single chain; doing that to a fill joins separate contours and makes an
    even-odd hole impossible to rasterise correctly.
    """
    groups: list[tuple[str, list[float]]] = []
    numbers: list[float] = []
    for token in _TOKEN.finditer(d):
        if token.group(1):
            command = token.group(1).upper()
            if command not in _SUPPORTED:
                raise UnsupportedPathError(
                    f"unsupported path command {token.group(1)!r}"
                )
            numbers = []
            groups.append((command, numbers))
        elif groups:
            numbers.append(float(token.group(2)))

    contours: list[list[list[tuple[float, float]]]] = []
    segments: list[list[tuple[float, float]]] = []
    current: tuple[float, float] | None = None
    start: tuple[float, float] | None = None

    def finish() -> None:
        nonlocal segments, current, start
        if current is not None and start is not None and current != start:
            segments.append(_as_cubic(current, start))
        if segments:
            contours.append(segments)
        segments, current, start = [], None, None

    for command, args in groups:
        if command == "M":
            if current is not None:
                finish()
            if len(args) < 2:
                continue
            current = start = (args[0], args[1])
            for index in range(2, len(args) - 1, 2):
                point = (args[index], args[index + 1])
                segments.append(_as_cubic(current, point))
                current = point
        elif command == "L":
            if current is None:
                raise UnsupportedPathError("a lineto before any moveto")
            for index in range(0, len(args) - 1, 2):
                point = (args[index], args[index + 1])
                segments.append(_as_cubic(current, point))
                current = point
        elif command == "C":
            if current is None:
                raise UnsupportedPathError("a curve before any moveto")
            for index in range(0, len(args) - 5, 6):
                point = (args[index + 4], args[index + 5])
                segments.append(
                    [
                        current,
                        (args[index], args[index + 1]),
                        (args[index + 2], args[index + 3]),
                        point,
                    ]
                )
                current = point
        elif command == "Z":
            finish()
    if current is not None:
        finish()
    if not contours:
        raise UnsupportedPathError("path has no drawable contour")
    return contours


def to_knots(segments) -> list[tuple[float, float]]:
    """Flatten segments into one 3n+1 chain so joins share parameters."""
    for before, after in itertools.pairwise(segments):
        if before[3] != after[0]:
            raise UnsupportedPathError("path is not one connected chain")
    points = [segments[0][0]]
    for segment in segments:
        points.extend(segment[1:])
    return points


def weld(chains, tolerance: float = 0.01):
    """Weld coincident points so shared junctions use one parameter."""
    points: list[tuple[float, float]] = []
    index: list[list[int]] = []
    for chain in chains:
        row: list[int] = []
        for point in chain:
            for position, existing in enumerate(points):
                if abs(existing[0] - point[0]) <= tolerance and (
                    abs(existing[1] - point[1]) <= tolerance
                ):
                    row.append(position)
                    break
            else:
                points.append(point)
                row.append(len(points) - 1)
        index.append(row)
    return points, index


def knots_to_path_d(points) -> str:
    """A 3n+1 chain back to path data."""
    parts = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    for index in range(1, len(points) - 2, 3):
        trio = points[index : index + 3]
        parts.append("C " + " ".join(f"{x:.1f} {y:.1f}" for x, y in trio))
    return " ".join(parts)


def _as_cubic(a, b):
    return [
        a,
        (a[0] + (b[0] - a[0]) / 3, a[1] + (b[1] - a[1]) / 3),
        (a[0] + 2 * (b[0] - a[0]) / 3, a[1] + 2 * (b[1] - a[1]) / 3),
        b,
    ]


def to_path_d(segments) -> str:
    """Cubic segments back to path data, one C per segment."""
    head = segments[0][0]
    parts = [f"M {head[0]:.1f} {head[1]:.1f}"]
    for segment in segments:
        parts.append("C " + " ".join(f"{x:.1f} {y:.1f}" for x, y in segment[1:]))
    return " ".join(parts)


def coverage(
    control: Any,
    width: float | Any,
    box: tuple[int, int, int, int],
    samples: int | None = None,
    softness: float = 0.25,
    chunk: int = 16384,
) -> Any:
    """Soft stroke coverage in [0, 1] over *box*, differentiable in *control*.

    A hard inside/outside test has zero gradient almost everywhere and none at
    all at the edge, so coverage falls off through a sigmoid instead: a pixel
    just outside the stroke still knows which way the stroke is.

    *box* is (left, top, right, bottom) in the drawing's own units, keeping cost
    proportional to the part being fitted rather than the full canvas.
    """
    import torch

    if control.is_cuda and control.shape[0] <= _FUSED_CUBICS:
        from vectrify.refine.cuda_renderer import stroke_coverage

        padded = _pad_fused_cubics(control[None])
        stroke_width = (
            width.reshape(1)
            if isinstance(width, torch.Tensor)
            else torch.full(
                (1,), float(width), dtype=control.dtype, device=control.device
            )
        )
        native = stroke_coverage(padded, stroke_width, box, subpixels=2)
        if native is not None:
            return native[0]

    if samples is None:
        samples = _samples_for(control)
    left, top, right, bottom = box
    height, width_px = bottom - top, right - left
    steps = torch.linspace(0, 1, samples, device=control.device, dtype=control.dtype)
    basis = torch.stack(
        [
            (1 - steps) ** 3,
            3 * steps * (1 - steps) ** 2,
            3 * steps**2 * (1 - steps),
            steps**3,
        ],
        dim=-1,
    )
    points = torch.einsum("sk,nkc->nsc", basis, control)
    head = points[:, :-1].reshape(-1, 2)
    tail = points[:, 1:].reshape(-1, 2)

    ys, xs = torch.meshgrid(
        torch.arange(height, device=control.device, dtype=control.dtype) + top + 0.5,
        torch.arange(width_px, device=control.device, dtype=control.dtype) + left + 0.5,
        indexing="ij",
    )
    pixels = torch.stack([xs, ys], dim=-1).reshape(-1, 2)

    span = tail - head
    length = (span * span).sum(-1).clamp_min(1e-9)
    nearest = []
    for start in range(0, pixels.shape[0], chunk):
        block = pixels[start : start + chunk]
        offset = block[:, None, :] - head[None]
        along = ((offset * span[None]).sum(-1) / length[None]).clamp(0, 1)
        foot = head[None] + along[..., None] * span[None]
        nearest.append((block[:, None, :] - foot).norm(dim=-1).min(dim=1).values)
    distance = torch.cat(nearest).reshape(height, width_px)
    return torch.sigmoid((width / 2 - distance) / softness)


# Sample density scales with curve length because coverage is measured to
# sampled chords.
_UNITS_PER_SAMPLE = 15.0
_MIN_SAMPLES, _MAX_SAMPLES = 8, 48
_FUSED_CUBICS = 16


def _samples_for(control: Any) -> int:
    """Samples per cubic, from the longest control polygon in the chain."""

    spans = control[:, 1:] - control[:, :-1]
    longest = float(spans.detach().norm(dim=-1).sum(dim=-1).max())
    wanted = math.ceil(longest / _UNITS_PER_SAMPLE)
    return max(_MIN_SAMPLES, min(_MAX_SAMPLES, wanted))


def _focus_mask(covers: list[Any], reach: int) -> Any:
    """Return a fixed dilated band around the group's initial coverage."""
    import torch
    import torch.nn.functional as functional

    union = 1 - torch.prod(torch.stack([1 - c for c in covers]), dim=0)
    band = functional.max_pool2d(
        union[None, None], kernel_size=2 * reach + 1, stride=1, padding=reach
    )
    return (band[0, 0] > 0.01).float()


def _bounds(segments_list, margin: float, size: int) -> tuple[int, int, int, int]:
    xs = [p[0] for segs in segments_list for seg in segs for p in seg]
    ys = [p[1] for segs in segments_list for seg in segs for p in seg]
    left = max(0, int(min(xs) - margin))
    top = max(0, int(min(ys) - margin))
    right = min(size, int(max(xs) + margin) + 1)
    bottom = min(size, int(max(ys) + margin) + 1)
    if right - left < 2 or bottom - top < 2:
        raise UnsupportedPathError("group occupies no area")
    return left, top, right, bottom


def fit_group(
    paths: list[str],
    widths: float | list[float],
    target: Image.Image,
    backdrop: Image.Image,
    colours: list[tuple[float, float, float]] | None = None,
    size: int = 700,
    steps: int = 200,
    samples: int | None = None,
    pinned: set[int] | None = None,
    learning_rate: float = 0.4,
    margin: float = 24.0,
    redundancy: float = 0.15,
    smooth: float = 0.0,
    anchor: float = 0.001,
) -> tuple[list[str], list[float], list[tuple[float, float, float]], float, float]:
    """Fit every path in *paths* together, returning new path data and losses.

    *backdrop* is the drawing rendered with this group removed; *target* is the
    picture being matched. Both are RGB and the same size as the canvas.

    *pinned* names welded vertices that must not move: a point this set shares
    with a path outside it. Without them a partial fit tears the drawing at
    exactly the junctions welding exists to hold -- the fitted side walks away
    while the neighbour it meets stays put.

    Each path is composited in SVG document order over the fixed backdrop. This
    retains different stroke colours and makes width, colour, and cubic controls
    jointly differentiable parameters of the same local move.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parsed = [parse_cubics(d) for d in paths]
    box = _bounds(parsed, margin, size)
    chains = [to_knots(segs) for segs in parsed]
    left, top, right, bottom = box

    def crop(image: Image.Image) -> Any:

        array = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32)
        return torch.tensor(array[top:bottom, left:right] / 255.0, device=device)

    goal = crop(target)
    under = crop(backdrop)

    # The optimized parameters are the chains; the control tensor a fit needs
    # is a view onto them, so each join is one number shared by the two curves
    # that meet there rather than two that drift apart.
    each = (
        [float(widths)] * len(paths)
        if isinstance(widths, int | float)
        else list(widths)
    )
    colour_values = colours or [(0.0, 0.0, 0.0)] * len(paths)
    if len(colour_values) != len(paths):
        raise ValueError("each stroked path needs one RGB colour")
    welded, index = weld(chains)
    vertices = torch.tensor(welded, device=device, dtype=torch.float32)
    vertices.requires_grad_(True)
    stroke_widths = torch.tensor(each, device=device, dtype=torch.float32)
    stroke_widths.requires_grad_(True)
    stroke_colours = torch.tensor(colour_values, device=device, dtype=torch.float32)
    stroke_colours.requires_grad_(True)
    original = vertices.detach().clone()
    rows = [torch.tensor(r, device=device, dtype=torch.long) for r in index]
    optimizer = torch.optim.Adam(
        [
            {"params": [vertices], "lr": learning_rate},
            {"params": [stroke_widths], "lr": learning_rate * 0.1},
            {"params": [stroke_colours], "lr": learning_rate * 0.05},
        ]
    )

    def chain_of(row: Any) -> Any:
        return vertices[row]

    def controls_of(chain: Any) -> Any:
        return chain.unfold(0, 4, 3).permute(0, 2, 1)

    with torch.no_grad():
        mask = _focus_mask(
            [
                coverage(controls_of(chain_of(r)), w, box, samples=samples)
                for r, w in zip(rows, stroke_widths, strict=True)
            ],
            int(margin),
        )
    weight = mask / mask.sum().clamp_min(1.0)

    first = last = 0.0
    for step in range(steps):
        covers = [
            coverage(controls_of(chain_of(r)), w, box, samples=samples)
            for r, w in zip(rows, stroke_widths, strict=True)
        ]
        stacked = torch.stack(covers)
        drawn = under
        for alpha, colour in zip(stacked, stroke_colours, strict=True):
            drawn = drawn * (1 - alpha[..., None]) + (
                colour.clamp(0, 1) * alpha[..., None]
            )
        loss = ((drawn - goal).abs() * weight[..., None]).sum()
        if redundancy:
            loss = (
                loss + redundancy * ((stacked.sum(0) - 1).clamp_min(0) * weight).sum()
            )
        if smooth:
            # Penalize non-collinear handles without forbidding real corners.
            loss = loss + smooth * sum(
                ((k[:-2:3] + k[2::3] - 2 * k[1:-1:3]) ** 2).mean()
                for k in (chain_of(r) for r in rows)
                if k.shape[0] >= 4
            )
        if anchor:
            loss = loss + anchor * ((vertices - original) ** 2).mean()
        if step == 0:
            first = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            stroke_widths.clamp_(min=0.1)
            stroke_colours.clamp_(0, 1)
        if pinned:
            with torch.no_grad():
                held = torch.tensor(sorted(pinned), device=device, dtype=torch.long)
                vertices[held] = original[held]
        last = float(loss.detach())

    fitted = [knots_to_path_d(chain_of(r).detach().cpu().tolist()) for r in rows]
    return (
        fitted,
        stroke_widths.detach().cpu().tolist(),
        [tuple(colour) for colour in stroke_colours.detach().cpu().tolist()],
        first,
        last,
    )


def _fill_winding(
    control: Any,
    box: tuple[int, int, int, int],
    samples: int = 32,
    x_offset: float = 0.5,
    y_offset: float = 0.5,
) -> Any:
    """Return a differentiable winding-angle field for one closed contour."""
    import torch

    left, top, right, bottom = box
    height, width = bottom - top, right - left
    steps = torch.linspace(0, 1, samples, device=control.device, dtype=control.dtype)
    basis = torch.stack(
        [
            (1 - steps) ** 3,
            3 * steps * (1 - steps) ** 2,
            3 * steps**2 * (1 - steps),
            steps**3,
        ],
        dim=-1,
    )
    curve = torch.einsum("sk,nkc->nsc", basis, control).reshape(-1, 2)
    curve = torch.cat((curve, curve[:1]))
    ys, xs = torch.meshgrid(
        torch.arange(height, device=control.device, dtype=control.dtype)
        + top
        + y_offset,
        torch.arange(width, device=control.device, dtype=control.dtype)
        + left
        + x_offset,
        indexing="ij",
    )
    pixels = torch.stack((xs, ys), dim=-1).reshape(-1, 2)
    start = curve[:-1][None] - pixels[:, None]
    end = curve[1:][None] - pixels[:, None]
    cross = start[..., 0] * end[..., 1] - start[..., 1] * end[..., 0]
    dot = (start * end).sum(dim=-1)
    return torch.atan2(cross, dot).sum(dim=-1).reshape(height, width)


def _fill_coverage(
    control: Any,
    box: tuple[int, int, int, int],
    samples: int = 32,
    softness: float = 0.25,
    subpixels: int = 4,
) -> Any:
    """Differentiable soft fill coverage for one closed cubic contour.

    The path's winding angle is smooth with respect to its sampled curve
    points. A sigmoid around pi turns it into antialiased inside coverage while
    retaining gradients for every point coordinate. This is the filled-path
    counterpart to :func:`coverage`, used by the SAMVG optimiser.
    """
    import torch

    coverages = []
    for y in range(subpixels):
        for x in range(subpixels):
            winding = _fill_winding(
                control,
                box,
                samples=samples,
                x_offset=(x + 0.5) / subpixels,
                y_offset=(y + 0.5) / subpixels,
            ).abs()
            coverages.append(torch.sigmoid((winding - math.pi) / softness))
    return torch.stack(coverages).mean(dim=0)


def _fill_winding_chunk(start: Any, end: Any, pixels: Any) -> Any:
    """Sum the winding angles of batched sampled contours at ``pixels``.

    Keeping this primitive separate gives ``torch.compile`` one regular,
    side-effect-free GPU expression to fuse.  In eager mode it is deliberately
    the same arithmetic previously in :func:`_fill_coverages`.
    """
    import torch

    offset_start = start[:, None] - pixels[None, :, None]
    offset_end = end[:, None] - pixels[None, :, None]
    cross = (
        offset_start[..., 0] * offset_end[..., 1]
        - offset_start[..., 1] * offset_end[..., 0]
    )
    dot = (offset_start * offset_end).sum(dim=-1)
    return torch.atan2(cross, dot).sum(dim=-1)


def _dynamic_fill_winding_chunk(start: Any, end: Any, pixels: Any) -> Any:
    """The tiled counterpart with only its pixel dimension left symbolic."""
    return _fill_winding_chunk(start, end, pixels)


@lru_cache(maxsize=1)
def _compiled_fill_winding_chunk() -> Any:
    """Return the CUDA-fused winding primitive when this torch supports it.

    This is intentionally lazy: installing Vectrify must not require a CUDA
    compiler, and the normal CPU renderer remains useful for tests and small
    jobs.  Inductor generates a kernel for this exact operation rather than
    adding an external renderer dependency.
    """
    import torch

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return _fill_winding_chunk
    try:
        return compile_fn(
            _fill_winding_chunk,
            fullgraph=True,
            dynamic=False,
            # This primitive is invoked repeatedly while its earlier outputs
            # still participate in one optimisation graph.  CUDA graph replay
            # cannot safely reuse those outputs and adds substantial overhead.
            options={"triton.cudagraphs": False},
        )
    except (RuntimeError, TypeError):
        log.warning("CUDA winding fusion is unavailable; using eager torch.")
        return _fill_winding_chunk


@lru_cache(maxsize=1)
def _compiled_tiled_fill_winding_chunk() -> Any:
    """Fuse arbitrary-size clipped tiles after path shapes were normalized."""
    import torch

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return _fill_winding_chunk
    try:
        return compile_fn(
            _dynamic_fill_winding_chunk,
            fullgraph=True,
            dynamic=True,
            options={"triton.cudagraphs": False},
        )
    except (RuntimeError, TypeError):
        log.warning("CUDA tiled winding fusion is unavailable; using eager torch.")
        return _fill_winding_chunk


def _pad_fused_cubics(controls: Any) -> Any:
    """Pad a short closed contour with zero-length cubics for CUDA fusion."""
    import torch

    count = controls.shape[1]
    if count >= _FUSED_CUBICS:
        return controls
    # All four points are the contour origin, so every added sampled segment
    # has zero angle at every pixel and cannot change its winding number.
    point = controls[:, :1, :1].expand(-1, _FUSED_CUBICS - count, 4, -1)
    return torch.cat((controls, point), dim=1)


def _fill_batched_windings(
    controls: Any,
    box: tuple[int, int, int, int],
    *,
    samples: int,
    x_offset: float,
    y_offset: float,
    batch_size: int = 4,
    pixel_chunk: int = 4_096,
    winding_chunk: Any | None = None,
) -> Any:
    """Return one winding field per equal-sized contour on CUDA.

    Unlike :func:`_fill_coverages`, this stops before applying a fill rule.
    A multi-contour SVG path needs its contour windings summed before that
    nonlinearity, so this is the reusable GPU building block for holes.
    """
    import torch

    # The native primitive is fixed-width, but winding is additive over cubic
    # ranges.  Chunk a long contour into padded 16-cubic ranges and sum its
    # exact native winding fields before applying the SVG fill rule.  This is
    # the same representation as the fixed SAMVG path, not a tessellation or
    # geometry approximation, and keeps SAMVG+var off Torch's huge broadcast
    # fallback.
    if samples in {8, 16, 32}:
        from vectrify.refine.cuda_renderer import winding as cuda_winding

        chunks = math.ceil(controls.shape[1] / _FUSED_CUBICS)
        padded = controls
        if chunks > 1:
            count = chunks * _FUSED_CUBICS - controls.shape[1]
            point = controls[:, :1, :1].expand(-1, count, 4, -1)
            padded = torch.cat((controls, point), dim=1)
        native = cuda_winding(
            padded.reshape(-1, _FUSED_CUBICS, 4, 2),
            box,
            samples=samples,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        if native is not None:
            return native.reshape(len(controls), chunks, *native.shape[1:]).sum(dim=1)

    left, top, right, bottom = box
    height, width = bottom - top, right - left
    steps = torch.linspace(0, 1, samples, device=controls.device, dtype=controls.dtype)
    basis = torch.stack(
        [
            (1 - steps) ** 3,
            3 * steps * (1 - steps) ** 2,
            3 * steps**2 * (1 - steps),
            steps**3,
        ],
        dim=-1,
    )
    ys, xs = torch.meshgrid(
        torch.arange(height, device=controls.device, dtype=controls.dtype)
        + top
        + y_offset,
        torch.arange(width, device=controls.device, dtype=controls.dtype)
        + left
        + x_offset,
        indexing="ij",
    )
    pixels = torch.stack((xs, ys), dim=-1).reshape(-1, 2)
    if winding_chunk is None:
        winding_chunk = (
            _compiled_fill_winding_chunk()
            if controls.shape[1] <= _FUSED_CUBICS
            else _fill_winding_chunk
        )
    output = []
    for control in controls.split(batch_size):
        count = len(control)
        if count < batch_size:
            # Keep the compiled kernel's leading dimension static.  The
            # padding is sliced away before it reaches the caller, so it has
            # no effect on pixels or gradients of the real contours.
            control = torch.cat(
                (control, control[:1].expand(batch_size - count, -1, -1, -1))
            )
        if control.shape[1] <= _FUSED_CUBICS:
            control = _pad_fused_cubics(control)
        curve = torch.einsum("sk,nqkc->nqsc", basis, control).flatten(1, 2)
        curve = torch.cat((curve, curve[:, :1]), dim=1)
        winding = []
        for pixel_start in range(0, len(pixels), pixel_chunk):
            winding.append(
                winding_chunk(
                    curve[:, :-1],
                    curve[:, 1:],
                    pixels[pixel_start : pixel_start + pixel_chunk],
                )
            )
        output.append(torch.cat(winding, dim=1)[:count])
    return torch.cat(output).reshape(-1, height, width)


def _fill_path_coverage(
    contours: list[Any],
    box: tuple[int, int, int, int],
    *,
    fill_rule: str = "nonzero",
    samples: int = 32,
    softness: float = 0.25,
    subpixels: int = 4,
    fuse: bool = True,
    dynamic_fuse: bool = False,
) -> Any:
    """Rasterise every contour according to SVG's fill-rule semantics."""
    import torch

    if contours and contours[0].is_cuda:
        # SAM's detailed masks can have many contours.  Sum each contour's
        # winding before applying the SVG fill rule, exactly as the eager
        # implementation below does, but keep the pixel/segment loops in the
        # fused CUDA primitive.
        fused = [contour for contour in contours if contour.shape[0] <= _FUSED_CUBICS]
        unfused: dict[tuple[int, ...], list[Any]] = defaultdict(list)
        for contour in contours:
            if contour.shape[0] > _FUSED_CUBICS:
                unfused[tuple(contour.shape)].append(contour)
        # SAMVG's tracer normally emits at most 16 cubics per contour.  Pad
        # those contours once and render a whole path in one large GPU batch,
        # rather than launching a tiny batch for each contour-shape group.
        fused_controls = (
            torch.cat([_pad_fused_cubics(contour[None]) for contour in fused])
            if fused
            else None
        )
        if fuse:
            winding_chunk = _compiled_fill_winding_chunk()
        elif dynamic_fuse:
            winding_chunk = _compiled_tiled_fill_winding_chunk()
        else:
            winding_chunk = _fill_winding_chunk
        contour_batch_size = 64 if (fuse or dynamic_fuse) else 4
        coverages = []
        for y in range(subpixels):
            for x in range(subpixels):
                winding = torch.zeros(
                    (box[3] - box[1], box[2] - box[0]),
                    dtype=contours[0].dtype,
                    device=contours[0].device,
                )
                if fused_controls is not None:
                    winding = winding + _fill_batched_windings(
                        fused_controls,
                        box,
                        samples=samples,
                        x_offset=(x + 0.5) / subpixels,
                        y_offset=(y + 0.5) / subpixels,
                        batch_size=contour_batch_size,
                        winding_chunk=winding_chunk,
                    ).sum(dim=0)
                for group in unfused.values():
                    winding = winding + _fill_batched_windings(
                        torch.stack(group),
                        box,
                        samples=samples,
                        x_offset=(x + 0.5) / subpixels,
                        y_offset=(y + 0.5) / subpixels,
                        winding_chunk=winding_chunk,
                    ).sum(dim=0)
                if fill_rule == "evenodd":
                    coverages.append(0.5 * (1 - torch.cos(winding / 2)))
                else:
                    coverages.append(
                        torch.sigmoid((winding.abs() - math.pi) / softness)
                    )
        return torch.stack(coverages).mean(dim=0)

    def contour_winding(contour: Any, x_offset: float, y_offset: float) -> Any:
        # A noisy SAM mask can contain dozens of enclosed contours.  Keeping
        # every pixel-by-segment intermediate alive until its layer loss is
        # backpropagated exhausts VRAM even on a small working canvas.
        # Checkpointing recomputes the same differentiable winding field during
        # backward, preserving renderer semantics and gradients exactly.
        if contour.requires_grad:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(
                lambda value: _fill_winding(
                    value,
                    box,
                    samples=samples,
                    x_offset=x_offset,
                    y_offset=y_offset,
                ),
                contour,
                use_reentrant=False,
            )
        return _fill_winding(
            contour,
            box,
            samples=samples,
            x_offset=x_offset,
            y_offset=y_offset,
        )

    coverages = []
    for y in range(subpixels):
        for x in range(subpixels):
            winding = torch.stack(
                [
                    contour_winding(
                        contour,
                        (x + 0.5) / subpixels,
                        (y + 0.5) / subpixels,
                    )
                    for contour in contours
                ]
            ).sum(dim=0)
            if fill_rule == "evenodd":
                # Winding changes by 2π for every crossing.  This periodic
                # expression is zero for an even count and one for an odd one.
                coverages.append(0.5 * (1 - torch.cos(winding / 2)))
            else:
                coverages.append(torch.sigmoid((winding.abs() - math.pi) / softness))
    return torch.stack(coverages).mean(dim=0)


def _large_path_tile_candidates(
    contours: list[Any],
    width: int,
    height: int,
    *,
    tile_size: int = 16,
    margin: float = 2.0,
) -> list[tuple[int, int, int, int, tuple[int, ...]]]:
    """Build conservative ray-crossing candidates for a large filled path.

    A horizontal ray from a tile pixel can only cross a contour whose control
    hull overlaps the tile vertically and reaches to the pixel's right.  The
    latter becomes ``max_x >= tile_left`` for every pixel in a tile.  Cubic
    curves lie inside their control hulls, making this a conservative spatial
    index: it may retain an unnecessary contour but never drops a crossing.
    ``margin`` also admits nearby contours to the boundary-gradient pass.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    bounds = []
    for contour in contours:
        points = contour.detach().reshape(-1, 2)
        bounds.append(
            (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )
        )
    tiles = []
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            right = min(width, left + tile_size)
            bottom = min(height, top + tile_size)
            candidates = tuple(
                index
                for index, (_min_x, min_y, max_x, max_y) in enumerate(bounds)
                if max_y >= top - margin
                and min_y <= bottom + margin
                and max_x >= left - margin
            )
            if candidates:
                tiles.append((left, top, right - left, bottom - top, candidates))
    return tiles


def _large_path_tile_boundary_candidates(
    contours: list[Any],
    tiles: list[tuple[int, int, int, int, tuple[int, ...]]],
    *,
    margin: float = 2.0,
) -> list[tuple[int, ...]]:
    """Return nearby-contour subsets for the boundary-gradient pass.

    Winding rays must retain any contour extending to a tile's right.  The
    closest-boundary surrogate is local, so it only needs contours whose
    conservative control hull overlaps the tile plus its antialias band.
    """
    bounds = []
    for contour in contours:
        points = contour.detach().reshape(-1, 2)
        bounds.append(
            (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )
        )
    return [
        tuple(
            index
            for index in ray_candidates
            if bounds[index][2] >= left - margin
            and bounds[index][0] <= left + tile_width + margin
            and bounds[index][3] >= top - margin
            and bounds[index][1] <= top + tile_height + margin
        )
        for left, top, tile_width, tile_height, ray_candidates in tiles
    ]


def _tiled_large_path_coverage(
    contours: list[Any],
    box: tuple[int, int, int, int],
    tiles: list[tuple[int, int, int, int, tuple[int, ...]]],
    *,
    fill_rule: str,
    subpixels: int,
    packed_contours: Any | None = None,
    candidate_indices: list[Any] | None = None,
    boundary_candidate_indices: list[Any] | None = None,
    topology_workspaces: dict[tuple[int, int], Any] | None = None,
) -> Any | None:
    """Analytically rasterise a large path from conservative contour tiles.

    ``packed_contours`` is normally a contiguous fixed-16 slice of the fit
    parameter.  Reusing it and the device-resident ``candidate_indices``
    avoids rebuilding the same per-tile Python concatenations each Adam step.
    """
    import torch

    left, top, right, bottom = box
    height, width = bottom - top, right - left
    output = None
    from vectrify.refine.cuda_renderer import multi_coverage

    # Tile dimensions have only edge variants.  Rendering one CUDA batch per
    # dimension replaces the old one-launch-per-tile graph without mixing
    # candidate sets: each tile remains an independent SVG compound path.
    tile_groups: dict[tuple[int, int], list[tuple[int, Any]]] = defaultdict(list)
    for tile_number, tile in enumerate(tiles):
        tile_groups[(tile[2], tile[3])].append((tile_number, tile))
    for (tile_width, tile_height), group in tile_groups.items():
        packed_tiles = []
        offsets = [0]
        boundary_offsets = [0]
        boundary_indices = []
        for tile_number, tile in group:
            tile_left, tile_top, _tile_width, _tile_height, candidates = tile
            offset = contours[0].new_tensor((left + tile_left, top + tile_top))
            if packed_contours is None:
                packed = torch.cat(
                    [
                        _pad_fused_cubics((contours[candidate] - offset)[None])
                        for candidate in candidates
                    ]
                )
            else:
                indices = (
                    candidate_indices[tile_number]
                    if candidate_indices is not None
                    else torch.tensor(
                        candidates, dtype=torch.long, device=packed_contours.device
                    )
                )
                packed = packed_contours.index_select(0, indices) - offset
            packed_tiles.append(packed)
            offsets.append(offsets[-1] + len(candidates))
            if boundary_candidate_indices is None:
                local_boundary = torch.arange(
                    len(candidates), dtype=torch.long, device=packed.device
                )
            else:
                local_boundary = boundary_candidate_indices[tile_number]
            boundary_indices.append(local_boundary)
            boundary_offsets.append(boundary_offsets[-1] + local_boundary.numel())
        topology_workspace = None
        if topology_workspaces is not None:
            shape = (len(group), tile_height, tile_width)
            topology_workspace = topology_workspaces.get((tile_width, tile_height))
            if topology_workspace is None or tuple(topology_workspace.shape) != shape:
                topology_workspace = torch.empty(
                    shape, dtype=torch.uint16, device=packed_tiles[0].device
                )
                topology_workspaces[(tile_width, tile_height)] = topology_workspace
        coverage = multi_coverage(
            torch.cat(packed_tiles),
            offsets,
            (0, 0, tile_width, tile_height),
            subpixels=subpixels,
            fill_rule=fill_rule,
            boundary_indices=torch.cat(boundary_indices),
            boundary_offsets=boundary_offsets,
            topology_workspace=topology_workspace,
        )
        if coverage is None:
            return None
        for alpha, (_tile_number, tile) in zip(coverage, group, strict=True):
            tile_left, tile_top, _tile_width, _tile_height, _candidates = tile
            restored = torch.nn.functional.pad(
                alpha,
                (
                    tile_left,
                    width - tile_left - tile_width,
                    tile_top,
                    height - tile_top - tile_height,
                ),
            )
            output = restored if output is None else output + restored
    if output is None:
        return torch.zeros(
            (height, width), dtype=contours[0].dtype, device=contours[0].device
        )
    return output


def _fill_coverages(
    controls: Any,
    box: tuple[int, int, int, int],
    samples: int = 32,
    softness: float = 0.25,
    batch_size: int = 4,
    fill_rule: str = "nonzero",
    pixel_chunk: int = 1_024,
    subpixels: int = 4,
    fuse: bool = True,
    dynamic_fuse: bool = False,
) -> Any:
    """Rasterise equal-sized closed cubic paths together on the GPU.

    The SAMVG tracer deliberately emits a fixed number of cubics per contour.
    Keeping that regularity here removes Python's per-path kernel-launch loop;
    this is substantially faster on CUDA for the 500-step optimisation pass.
    """
    import torch

    left, top, right, bottom = box
    height, width = bottom - top, right - left
    # Simple closed contours are the common SAMVG case.  Use the native
    # cubic-intersection renderer here; the sampled winding implementation
    # below remains the portable oracle and handles arbitrary layouts.
    if controls.is_cuda and controls.shape[1] <= _FUSED_CUBICS:
        from vectrify.refine.cuda_renderer import coverage as cuda_coverage

        native = cuda_coverage(
            _pad_fused_cubics(controls),
            box,
            subpixels=subpixels,
            fill_rule=fill_rule,
        )
        if native is not None:
            return native
    steps = torch.linspace(0, 1, samples, device=controls.device, dtype=controls.dtype)
    basis = torch.stack(
        [
            (1 - steps) ** 3,
            3 * steps * (1 - steps) ** 2,
            3 * steps**2 * (1 - steps),
            steps**3,
        ],
        dim=-1,
    )
    output = []
    can_fuse = controls.is_cuda and controls.shape[1] <= _FUSED_CUBICS
    if fuse and can_fuse:
        winding_chunk = _compiled_fill_winding_chunk()
    elif dynamic_fuse and can_fuse:
        winding_chunk = _compiled_tiled_fill_winding_chunk()
    else:
        winding_chunk = _fill_winding_chunk
    # The fused CUDA kernel consumes far less temporary memory than eager
    # broadcasting, so one complete small SAMVG working raster is faster than
    # many 1,024-pixel launches.  Retain the conservative caller-selected
    # chunking on CPU.
    active_pixel_chunk = (
        max(pixel_chunk, 4_096)
        if controls.is_cuda and (fuse or dynamic_fuse)
        else pixel_chunk
    )
    for control in controls.split(batch_size):
        count = len(control)
        if controls.is_cuda and count < batch_size:
            control = torch.cat(
                (control, control[:1].expand(batch_size - count, -1, -1, -1))
            )
        if controls.is_cuda and control.shape[1] <= _FUSED_CUBICS:
            control = _pad_fused_cubics(control)
        curve = torch.einsum("sk,nqkc->nqsc", basis, control).flatten(1, 2)
        curve = torch.cat((curve, curve[:, :1]), dim=1)
        start = curve[:, :-1]
        end = curve[:, 1:]
        coverage_sum = None
        for y in range(subpixels):
            for x in range(subpixels):
                ys, xs = torch.meshgrid(
                    torch.arange(height, device=controls.device, dtype=controls.dtype)
                    + top
                    + (y + 0.5) / subpixels,
                    torch.arange(width, device=controls.device, dtype=controls.dtype)
                    + left
                    + (x + 0.5) / subpixels,
                    indexing="ij",
                )
                pixels = torch.stack((xs, ys), dim=-1).reshape(-1, 2)
                coverages = []
                for pixel_start in range(0, len(pixels), active_pixel_chunk):
                    pixel_block = pixels[pixel_start : pixel_start + active_pixel_chunk]
                    winding = winding_chunk(start, end, pixel_block)
                    if fill_rule == "evenodd":
                        coverages.append(0.5 * (1 - torch.cos(winding / 2)))
                    else:
                        coverages.append(
                            torch.sigmoid((winding.abs() - math.pi) / softness)
                        )
                coverage = torch.cat(coverages, dim=1)
                coverage_sum = (
                    coverage if coverage_sum is None else coverage_sum + coverage
                )
        assert coverage_sum is not None
        output.append((coverage_sum / (subpixels * subpixels))[:count])
    return torch.cat(output).reshape(-1, height, width)


def _fill_distance_surrogate_coverages(
    controls: Any,
    box: tuple[int, int, int, int],
    *,
    samples: int = 32,
    softness: float = 0.25,
    subpixels: int = 2,
    pixel_chunk: int = 1_024,
) -> Any:
    """Exact sampled-fill forward values with closest-cubic surrogate gradients.

    This is the same forward/backward split used by differentiable vector
    renderers: topology is a discrete winding decision, while a local signed
    distance provides useful boundary gradients.  The hard forward samples
    retain the existing renderer's fill semantics; only the expensive
    pixel-by-512-segment autograd graph is replaced for simple contours.
    """
    import torch

    left, top, right, bottom = box
    height, width = bottom - top, right - left
    result = []
    seeds = torch.linspace(0, 1, 5, dtype=controls.dtype, device=controls.device)
    for y in range(subpixels):
        for x in range(subpixels):
            x_offset, y_offset = (x + 0.5) / subpixels, (y + 0.5) / subpixels
            with torch.no_grad():
                winding = _fill_batched_windings(
                    controls.detach(),
                    box,
                    samples=samples,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    batch_size=4,
                    pixel_chunk=pixel_chunk,
                    winding_chunk=_compiled_tiled_fill_winding_chunk(),
                )
                inside = winding.abs() >= math.pi
            ys, xs = torch.meshgrid(
                torch.arange(height, device=controls.device, dtype=controls.dtype)
                + top
                + y_offset,
                torch.arange(width, device=controls.device, dtype=controls.dtype)
                + left
                + x_offset,
                indexing="ij",
            )
            pixels = torch.stack((xs, ys), dim=-1).reshape(-1, 2)
            surrogate_parts = []
            for start in range(0, len(pixels), pixel_chunk):
                pixel = pixels[start : start + pixel_chunk]
                parameter = seeds[None, None, None].expand(
                    len(controls), len(pixel), controls.shape[1], -1
                )
                cubic = controls[:, None, :, None]
                for _ in range(5):
                    inverse = 1 - parameter
                    point = (
                        inverse[..., None] ** 3 * cubic[..., 0, :]
                        + 3
                        * inverse[..., None] ** 2
                        * parameter[..., None]
                        * cubic[..., 1, :]
                        + 3
                        * inverse[..., None]
                        * parameter[..., None] ** 2
                        * cubic[..., 2, :]
                        + parameter[..., None] ** 3 * cubic[..., 3, :]
                    )
                    tangent = (
                        3
                        * inverse[..., None] ** 2
                        * (cubic[..., 1, :] - cubic[..., 0, :])
                        + 6
                        * inverse[..., None]
                        * parameter[..., None]
                        * (cubic[..., 2, :] - cubic[..., 1, :])
                        + 3
                        * parameter[..., None] ** 2
                        * (cubic[..., 3, :] - cubic[..., 2, :])
                    )
                    acceleration = 6 * inverse[..., None] * (
                        cubic[..., 2, :] - 2 * cubic[..., 1, :] + cubic[..., 0, :]
                    ) + 6 * parameter[..., None] * (
                        cubic[..., 3, :] - 2 * cubic[..., 2, :] + cubic[..., 1, :]
                    )
                    delta = point - pixel[None, :, None, None, :]
                    parameter = (
                        parameter
                        - (delta * tangent).sum(dim=-1)
                        / (
                            (tangent * tangent).sum(dim=-1)
                            + (delta * acceleration).sum(dim=-1)
                        ).clamp_min(1e-8)
                    ).clamp(0, 1)
                inverse = 1 - parameter
                point = (
                    inverse[..., None] ** 3 * cubic[..., 0, :]
                    + 3
                    * inverse[..., None] ** 2
                    * parameter[..., None]
                    * cubic[..., 1, :]
                    + 3
                    * inverse[..., None]
                    * parameter[..., None] ** 2
                    * cubic[..., 2, :]
                    + parameter[..., None] ** 3 * cubic[..., 3, :]
                )
                distance = (
                    (point - pixel[None, :, None, None, :])
                    .norm(dim=-1)
                    .amin(dim=(-1, -2))
                )
                sign = torch.where(
                    inside.reshape(len(controls), -1)[:, start : start + len(pixel)],
                    -1.0,
                    1.0,
                )
                surrogate_parts.append(torch.sigmoid(-sign * distance / softness))
            surrogate = torch.cat(surrogate_parts, dim=1).reshape(-1, height, width)
            hard = inside.to(dtype=controls.dtype)
            result.append(hard + surrogate - surrogate.detach())
    return torch.stack(result).mean(dim=0)


def _xing_penalties(control: Any) -> Any:
    """Return SAMVG's normalized Xing penalty for every cubic in ``control``."""
    import torch

    start_handle = control[:, 1] - control[:, 0]
    end_handle = control[:, 3] - control[:, 2]
    cross = (
        start_handle[:, 0] * end_handle[:, 1] - start_handle[:, 1] * end_handle[:, 0]
    )
    sine = cross / (start_handle.norm(dim=-1) * end_handle.norm(dim=-1) + 1e-12)
    return torch.where(cross < 0, torch.relu(-sine), torch.relu(sine))


def _xing_loss(control: Any) -> Any:
    """Return SAMVG's normalized per-cubic Xing regularizer (Eq. 3-6--3-8)."""
    return _xing_penalties(control).mean()


_HEX_FILL = re.compile(r"^#([0-9a-fA-F]{6})$")


def _fill_rgb(value: str | None) -> tuple[float, float, float] | None:
    match = _HEX_FILL.match((value or "").strip())
    if not match:
        return None
    digits = match.group(1)
    return (
        int(digits[0:2], 16) / 255,
        int(digits[2:4], 16) / 255,
        int(digits[4:6], 16) / 255,
    )


def _composite_opaque_fills(
    alphas: Any, colours: Any, backdrop: Any | None = None
) -> Any:
    """Composite opaque SVG fills in document order without a layer loop.

    Each layer contributes its premultiplied colour through the product of the
    transparencies above it.  This is algebraically identical to repeatedly
    applying ``canvas * (1 - alpha) + colour * alpha`` over a black canvas,
    but lets Torch execute the 223-layer cat seed in a few large operations.
    """
    import torch

    transparency = 1 - alphas
    above_inclusive = torch.cumprod(transparency.flip(0), dim=0).flip(0)
    above = torch.cat((above_inclusive[1:], torch.ones_like(alphas[:1])), dim=0)
    painted = (
        colours.clamp(0, 1)[:, None, None, :] * alphas[..., None] * above[..., None]
    ).sum(dim=0)
    if backdrop is None:
        return painted
    return painted + backdrop * above_inclusive[0][..., None]


@lru_cache(maxsize=1)
def _compiled_opaque_fill_composite() -> Any:
    """Return the CUDA-fused painter's-order compositing kernel when possible."""
    import torch

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return _composite_opaque_fills
    try:
        return compile_fn(
            _composite_opaque_fills,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
    except (RuntimeError, TypeError):
        log.warning("CUDA fill compositing fusion is unavailable; using eager torch.")
        return _composite_opaque_fills


def fit_filled_svg(
    svg: str,
    target: Image.Image,
    *,
    steps: int = 500,
    point_learning_rate: float = 1.0,
    color_learning_rate: float = 0.01,
    xing_weight: float = 0.02,
    optimisation_long_side: int | None = None,
    subpixels: int = 2,
    monolithic: bool | None = None,
    curve_samples: int | None = None,
    backdrop: Image.Image | None = None,
    learn_alpha: bool = False,
    sparse_replay: bool = False,
) -> str:
    """Optimise filled cubic SVG paths against an RGB target.

    SAMVG optimises opaque path coordinates and fill colours for 500 Adam
    iterations in each of its two passes.  ``learn_alpha`` enables the
    dissertation's SAMVG+alpha variation: each selected path receives a
    learnable fill opacity.  The standard SAMVG configuration deliberately
    keeps it disabled and treats fills as opaque. This implementation reuses
    Vectrify's torch renderer instead of requiring DiffVG, while retaining the
    dissertation's full-resolution Adam defaults: point LR 1, colour LR .01,
    and MSE plus .02 Xing loss.  It uses DiffVG's standard 2x2 optimisation
    sampling; the standalone renderer retains its stricter 4x4 default for
    Cairo-fidelity checks.  Small clipped tiles use fewer cubic samples because
    their screen-space deviation is bounded by the tile size; pass
    ``curve_samples`` to override that adaptive choice.  ``optimisation_long_side``
    is available only as an explicit caller-selected preview mode. CUDA uses
    one monolithic compositor graph for 64px-or-smaller working canvases by
    default; larger canvases retain the memory-bounded replay.
    ``sparse_replay`` retains the same painter-order MSE derivative while
    saving layer state only within each path's raster tile; it makes a full
    1024px SAMVG phase practical without a monolithic alpha stack.
    """
    import xml.etree.ElementTree as ET

    import torch

    root = ET.fromstring(svg)

    def opacity(element) -> float:
        """Read the directly applied SVG fill opacity, clamped for Adam."""
        try:
            fill_opacity = float(element.get("fill-opacity", "1"))
            element_opacity = float(element.get("opacity", "1"))
        except ValueError:
            return 1.0
        return min(1.0, max(0.0, fill_opacity * element_opacity))

    entries = []
    for element in root.iter():
        if element.tag.split("}")[-1] != "path" or not element.get("d"):
            continue
        colour = _fill_rgb(element.get("fill"))
        if colour is None:
            continue
        try:
            contours = parse_filled_cubics(element.get("d", ""))
        except UnsupportedPathError:
            continue
        fill_rule = element.get("fill-rule", "nonzero").strip().lower()
        if fill_rule not in {"evenodd", "nonzero"}:
            continue
        entries.append((element, contours, colour, fill_rule, opacity(element)))
    if not entries:
        raise UnsupportedPathError("no opaque filled cubic paths to optimise")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    width, height = target.size
    scale = (
        1.0
        if optimisation_long_side is None
        else min(1.0, optimisation_long_side / max(width, height))
    )
    work_width, work_height = round(width * scale), round(height * scale)
    if monolithic is None:
        # At SAMVG's 64px seed-fitting resolution the complete opaque-layer
        # graph is small, avoids renderer replay for each bounded layer batch,
        # and has exactly the same painter-order MSE derivative.  Preserve the
        # bounded path at larger resolutions, where its saved alpha/canvas
        # representation is intentionally memory conservative.
        monolithic = device == "cuda" and work_width * work_height <= 64 * 64
    # The target is resized to the integer working raster.  Map coordinates
    # with those exact axis scales too: applying the single nominal scale to
    # both axes subtly shifts every horizontal edge when rounding makes the
    # working raster's aspect ratio differ from the source image.
    coordinate_scale = torch.tensor(
        [work_width / width, work_height / height],
        dtype=torch.float32,
        device=device,
    )
    initial_controls = [
        [
            (
                torch.tensor(contour, dtype=torch.float32, device=device)
                * coordinate_scale
            )
            for contour in contours
        ]
        for _element, contours, _colour, _fill_rule, _opacity in entries
    ]
    # A detailed SAMVG seed has hundreds of contours.  Keeping each one as a
    # separate Adam parameter turns one optimiser update into hundreds of tiny
    # CUDA kernels.  Store fixed-width contour slots in one parameter and use
    # narrow views below, retaining every original contour length in the SVG
    # and Xing terms.
    flat_controls = [control for path in initial_controls for control in path]
    contour_sizes = [len(control) for control in flat_controls]
    control_storage = torch.nn.Parameter(
        torch.cat([_pad_fused_cubics(control[None]) for control in flat_controls])
    )
    controls = []
    path_storage_spans = []
    storage_offset = 0
    for path in initial_controls:
        path_start = storage_offset
        views = []
        for _control in path:
            size = contour_sizes[storage_offset]
            views.append(control_storage[storage_offset, :size])
            storage_offset += 1
        controls.append(views)
        path_storage_spans.append((path_start, storage_offset))
    # Match the fixed-width geometry storage above: one colour parameter
    # avoids launching Adam's tiny update kernels once per SVG layer.
    color_storage = torch.nn.Parameter(
        torch.tensor(
            [colour for _element, _contours, colour, _fill_rule, _opacity in entries],
            dtype=torch.float32,
            device=device,
        )
    )
    goal = torch.tensor(
        np.asarray(
            target.convert("RGB").resize((work_width, work_height)), dtype=np.float32
        )
        / 255.0,
        device=device,
    )
    under = (
        None
        if backdrop is None
        else torch.tensor(
            np.asarray(
                backdrop.convert("RGB").resize((work_width, work_height)),
                dtype=np.float32,
            )
            / 255.0,
            device=device,
        )
    )
    alpha_values = (
        torch.nn.Parameter(
            torch.tensor(
                [entry[4] for entry in entries],
                dtype=torch.float32,
                device=device,
            )
        )
        if learn_alpha
        else None
    )
    point_optimizer = torch.optim.Adam(
        [control_storage], lr=point_learning_rate, fused=device == "cuda"
    )
    colour_optimizer = torch.optim.Adam(
        [color_storage, *([] if alpha_values is None else [alpha_values])],
        lr=color_learning_rate,
        fused=device == "cuda",
    )
    # The dissertation averages Xing within each contour then sums contours.
    # Keep that weighting while evaluating the 413 cat contours in one CUDA
    # expression rather than launching one tiny graph for each.
    xing_contour_weights = torch.cat(
        [
            torch.full(
                (len(control),),
                1 / len(control),
                dtype=goal.dtype,
                device=device,
            )
            for path in controls
            for control in path
        ]
    )

    def close_contours() -> None:
        """Restore the shared joins of every traced closed Bezier contour.

        SAMVG traces closed fixed-segment loops.  The packed parameter storage
        keeps their cubic endpoints as separate Adam values for efficient
        rasterisation, so project them back to a continuous closed contour
        after each update.  Otherwise a subpixel gap becomes an extra implicit
        SVG closing cubic on export, violating the fixed-segment invariant.
        """
        with torch.no_grad():
            for path in controls:
                for contour in path:
                    contour[1:, 0].copy_(contour[:-1, 3])
                    contour[-1, 3].copy_(contour[0, 0])

    def tile_for(path: list[Any]) -> tuple[int, int, int, int]:
        """A fixed, antialiased raster tile covering a path's control hull."""
        points = torch.cat([control.detach().reshape(-1, 2) for control in path])
        # Cubic Beziers lie in their control hull.  Two pixels retain the
        # entire soft edge while avoiding the full-canvas work DiffVG culls.
        left = max(0, math.floor(float(points[:, 0].min())) - 2)
        top = max(0, math.floor(float(points[:, 1].min())) - 2)
        right = min(work_width, math.ceil(float(points[:, 0].max())) + 2)
        bottom = min(work_height, math.ceil(float(points[:, 1].max())) + 2)

        # Bucket dimensions keep unrelated paths in the same CUDA batch.  A
        # 32px bucket roughly halves the distinct sizes of the 1024px cat
        # seed versus 8px buckets while adding only a small protected fringe
        # to the right/bottom of each tile.  The origin—and therefore every
        # coverage sample belonging to the path—remains unchanged.  Shift a
        # bucket at the canvas edge rather than clipping its antialias margin.
        tile_bucket = 32
        tile_width = min(
            work_width, tile_bucket * math.ceil(max(right - left, 1) / tile_bucket)
        )
        tile_height = min(
            work_height, tile_bucket * math.ceil(max(bottom - top, 1) / tile_bucket)
        )
        left = min(left, work_width - tile_width)
        top = min(top, work_height - tile_height)
        return left, top, tile_width, tile_height

    def samples_for(tile_width: int, tile_height: int) -> int:
        """Choose winding tessellation from the cubic's visible pixel extent."""
        if curve_samples is not None:
            return curve_samples
        longest_side = max(tile_width, tile_height)
        if longest_side <= 32:
            return 8
        if longest_side <= 64:
            return 16
        return 32

    def restore_tile(alpha: Any, left: int, top: int) -> Any:
        return torch.nn.functional.pad(
            alpha,
            (
                left,
                work_width - left - alpha.shape[1],
                top,
                work_height - top - alpha.shape[0],
            ),
        )

    def cropped_simple_groups() -> dict[
        tuple[tuple[int, ...], str, int, int], list[tuple[int, int, int]]
    ]:
        groups: dict[
            tuple[tuple[int, ...], str, int, int], list[tuple[int, int, int]]
        ] = defaultdict(list)
        for index, path in enumerate(controls):
            if len(path) != 1:
                continue
            left, top, tile_width, tile_height = tile_for(path)
            groups[
                (tuple(path[0].shape), entries[index][3], tile_width, tile_height)
            ].append((index, left, top))
        return groups

    def rasterise_simple_tiles(
        fill_rule: str,
        tile_width: int,
        tile_height: int,
        items: list[tuple[int, int, int]],
    ) -> list[tuple[int, Any, int, int]]:
        # SAMVG+var can emit a contour longer than the fixed-width coverage
        # primitive.  Route those through the chunked native winding path;
        # packing them into the old batched coverage call would force eager
        # Torch broadcasting over every cubic and pixel.
        if controls[items[0][0]][0].shape[0] > _FUSED_CUBICS:
            output = []
            for index, left, top in items:
                offset = controls[index][0].new_tensor((left, top))
                alpha = _fill_path_coverage(
                    [controls[index][0] - offset],
                    (0, 0, tile_width, tile_height),
                    fill_rule=fill_rule,
                    samples=samples_for(tile_width, tile_height),
                    subpixels=subpixels,
                    fuse=False,
                )
                output.append((index, alpha, left, top))
            return output
        translated = torch.stack(
            [
                controls[index][0] - controls[index][0].new_tensor((left, top))
                for index, left, top in items
            ]
        )
        rasterised = _fill_coverages(
            translated,
            (0, 0, tile_width, tile_height),
            fill_rule=fill_rule,
            samples=samples_for(tile_width, tile_height),
            subpixels=subpixels,
            fuse=False,
            dynamic_fuse=len(items) >= 4,
        )
        return [
            (index, alpha, left, top)
            for (index, left, top), alpha in zip(items, rasterised, strict=True)
        ]

    def rasterise_simple(
        fill_rule: str,
        tile_width: int,
        tile_height: int,
        items: list[tuple[int, int, int]],
    ) -> list[tuple[int, Any]]:
        return [
            (index, restore_tile(alpha, left, top))
            for index, alpha, left, top in rasterise_simple_tiles(
                fill_rule, tile_width, tile_height, items
            )
        ]

    def rasterise_multi(index: int, path: list[Any]) -> Any:
        # Large paths use fixed conservative candidate tiles.  Every tile
        # sees all contours that can cross one of its horizontal rays, while
        # avoiding the old all-contours-at-every-pixel winding fallback.
        if len(path) >= 16:
            # The index has a two-pixel conservative guard band.  Rebuild it
            # after local optimisation consumes half that allowance, so a
            # stale index cannot exclude a valid ray crossing.
            reference = large_multi_index_references[index]
            movement = max(
                float((control.detach() - saved).abs().amax())
                for control, saved in zip(path, reference, strict=True)
            )
            if movement > 1.0:
                initial_large_multi_tiles[index] = _large_path_tile_candidates(
                    path, work_width, work_height
                )
                large_multi_tile_indices[index] = [
                    torch.tensor(candidates, dtype=torch.long, device=device)
                    for _left, _top, _width, _height, candidates in (
                        initial_large_multi_tiles[index]
                    )
                ]
                large_multi_boundary_indices[index] = [
                    torch.tensor(
                        [ray_candidates.index(candidate) for candidate in candidates],
                        dtype=torch.long,
                        device=device,
                    )
                    for (
                        _left,
                        _top,
                        _width,
                        _height,
                        ray_candidates,
                    ), candidates in zip(
                        initial_large_multi_tiles[index],
                        _large_path_tile_boundary_candidates(
                            path, initial_large_multi_tiles[index]
                        ),
                        strict=True,
                    )
                ]
                large_multi_index_references[index] = tuple(
                    control.detach().clone() for control in path
                )
                large_multi_topology_workspaces[index].clear()
            tiled_alpha = _tiled_large_path_coverage(
                path,
                (0, 0, work_width, work_height),
                initial_large_multi_tiles[index],
                fill_rule=entries[index][3],
                subpixels=subpixels,
                packed_contours=control_storage[slice(*path_storage_spans[index])],
                candidate_indices=large_multi_tile_indices[index],
                boundary_candidate_indices=large_multi_boundary_indices[index],
                topology_workspaces=large_multi_topology_workspaces[index],
            )
            if tiled_alpha is not None:
                return tiled_alpha
            return _fill_path_coverage(
                path,
                (0, 0, work_width, work_height),
                fill_rule=entries[index][3],
                samples=samples_for(work_width, work_height),
                subpixels=subpixels,
            )
        left, top, tile_width, tile_height = initial_multi_tiles[index]
        offset = path[0].new_tensor((left, top))
        from vectrify.refine.cuda_renderer import multi_coverage

        packed = torch.cat(
            [_pad_fused_cubics((control - offset)[None]) for control in path]
        )
        analytic = multi_coverage(
            packed,
            [0, len(path)],
            (0, 0, tile_width, tile_height),
            subpixels=subpixels,
            fill_rule=entries[index][3],
        )
        if analytic is not None:
            return restore_tile(analytic[0], left, top)
        alpha = _fill_path_coverage(
            [control - offset for control in path],
            (0, 0, tile_width, tile_height),
            fill_rule=entries[index][3],
            samples=samples_for(tile_width, tile_height),
            subpixels=subpixels,
            fuse=False,
        )
        return restore_tile(alpha, left, top)

    def rasterise_multi_group(
        fill_rule: str,
        tile_width: int,
        tile_height: int,
        items: list[tuple[int, int, int]],
    ) -> list[tuple[int, Any]]:
        """Rasterise equal-sized multi-contour paths in one contour batch."""
        translated = []
        spans = []
        for index, left, top in items:
            offset = controls[index][0].new_tensor((left, top))
            start = len(translated)
            translated.extend(control - offset for control in controls[index])
            spans.append((start, len(translated)))
        # Native winding uses one CUDA block per contour.  Combining contours
        # from otherwise independent paths lets its blocks occupy the GPU at
        # once, while summing each recorded span before the fill nonlinearity
        # preserves SVG path semantics (including holes).
        packed = torch.cat([_pad_fused_cubics(control[None]) for control in translated])
        from vectrify.refine.cuda_renderer import multi_coverage

        analytic = multi_coverage(
            packed,
            [start for start, _end in spans] + [spans[-1][1]],
            (0, 0, tile_width, tile_height),
            subpixels=subpixels,
            fill_rule=fill_rule,
        )
        if analytic is not None:
            return [
                (index, restore_tile(alpha, left, top))
                for (index, left, top), alpha in zip(items, analytic, strict=True)
            ]
        from vectrify.refine.cuda_renderer import windings as cuda_windings

        native_winding = cuda_windings(
            packed,
            (0, 0, tile_width, tile_height),
            samples=samples_for(tile_width, tile_height),
            subpixels=subpixels,
        )
        if native_winding is not None:
            path_winding = torch.stack(
                [native_winding[start:end].sum(dim=0) for start, end in spans]
            )
            if fill_rule == "evenodd":
                coverage = 0.5 * (1 - torch.cos(path_winding / 2))
            else:
                coverage = torch.sigmoid((path_winding.abs() - math.pi) / 0.25)
            return [
                (index, restore_tile(alpha, left, top))
                for (index, left, top), alpha in zip(
                    items, coverage.mean(dim=1), strict=True
                )
            ]
        coverage_sum = None
        for y in range(subpixels):
            for x in range(subpixels):
                winding = _fill_batched_windings(
                    packed,
                    (0, 0, tile_width, tile_height),
                    samples=samples_for(tile_width, tile_height),
                    x_offset=(x + 0.5) / subpixels,
                    y_offset=(y + 0.5) / subpixels,
                    batch_size=64,
                )
                path_winding = torch.stack(
                    [winding[start:end].sum(dim=0) for start, end in spans]
                )
                if fill_rule == "evenodd":
                    coverage = 0.5 * (1 - torch.cos(path_winding / 2))
                else:
                    coverage = torch.sigmoid((path_winding.abs() - math.pi) / 0.25)
                coverage_sum = (
                    coverage if coverage_sum is None else coverage_sum + coverage
                )
        assert coverage_sum is not None
        return [
            (index, restore_tile(alpha, left, top))
            for (index, left, top), alpha in zip(
                items,
                coverage_sum / (subpixels * subpixels),
                strict=True,
            )
        ]

    # Tile layout is part of the seed rasterisation setup, not optimisation
    # state.  Re-reading each CUDA control tensor's extrema every Adam step
    # introduces hundreds of device synchronisations on a detailed SAMVG
    # seed.  The two-pixel antialias margin already makes these fixed tiles
    # conservative for the local coordinate updates used by the fit.
    initial_simple_groups = cropped_simple_groups()
    initial_multi_tiles = {
        index: tile_for(path)
        for index, path in enumerate(controls)
        if len(path) != 1 and len(path) < 16
    }
    initial_large_multi_tiles = {
        index: _large_path_tile_candidates(path, work_width, work_height)
        for index, path in enumerate(controls)
        if len(path) >= 16
    }
    large_multi_tile_indices = {
        index: [
            torch.tensor(candidates, dtype=torch.long, device=device)
            for _left, _top, _width, _height, candidates in tiles
        ]
        for index, tiles in initial_large_multi_tiles.items()
    }
    large_multi_boundary_indices = {
        index: [
            torch.tensor(
                [ray_candidates.index(candidate) for candidate in candidates],
                dtype=torch.long,
                device=device,
            )
            for (_left, _top, _width, _height, ray_candidates), candidates in zip(
                tiles,
                _large_path_tile_boundary_candidates(controls[index], tiles),
                strict=True,
            )
        ]
        for index, tiles in initial_large_multi_tiles.items()
    }
    large_multi_topology_workspaces: dict[int, dict[tuple[int, int], Any]] = {
        index: {} for index in initial_large_multi_tiles
    }
    large_multi_index_references = {
        index: tuple(control.detach().clone() for control in path)
        for index, path in enumerate(controls)
        if len(path) >= 16
    }
    initial_multi_groups: dict[tuple[str, int, int], list[tuple[int, int, int]]] = (
        defaultdict(list)
    )
    for index, (left, top, tile_width, tile_height) in initial_multi_tiles.items():
        initial_multi_groups[(entries[index][3], tile_width, tile_height)].append(
            (index, left, top)
        )

    log.info(
        "Filled-path optimisation: %d path(s), %dx%d working raster on %s.",
        len(entries),
        work_width,
        work_height,
        device,
    )
    for _step in range(steps):
        point_optimizer.zero_grad()
        colour_optimizer.zero_grad()
        simple_groups = initial_simple_groups
        all_controls = torch.cat([control for path in controls for control in path])

        if monolithic:
            alphas: list[Any | None] = [None] * len(entries)
            for (
                _shape,
                fill_rule,
                tile_width,
                tile_height,
            ), items in simple_groups.items():
                for index, alpha in rasterise_simple(
                    fill_rule, tile_width, tile_height, items
                ):
                    alphas[index] = alpha
            for (
                fill_rule,
                tile_width,
                tile_height,
            ), items in initial_multi_groups.items():
                for index, alpha in rasterise_multi_group(
                    fill_rule, tile_width, tile_height, items
                ):
                    alphas[index] = alpha
            for index, path in enumerate(controls):
                if alphas[index] is None:
                    alphas[index] = rasterise_multi(index, path)
            alpha_stack = torch.stack([alpha for alpha in alphas if alpha is not None])
            if alpha_values is not None:
                alpha_stack = alpha_stack * alpha_values.clamp(0, 1)[:, None, None]
            composite = (
                _compiled_opaque_fill_composite()
                if goal.is_cuda
                else _composite_opaque_fills
            )
            rendered = (
                composite(alpha_stack, color_storage)
                if under is None
                else _composite_opaque_fills(alpha_stack, color_storage, under)
            )
            loss = ((rendered - goal) ** 2).mean()
            loss = (
                loss
                + xing_weight
                * (_xing_penalties(all_controls) * xing_contour_weights).sum()
            )
            loss.backward()
            point_optimizer.step()
            colour_optimizer.step()
            close_contours()
            continue

        if sparse_replay:
            # Dense replay previously saved a full alpha, pre-layer canvas and
            # downstream-transparency map for every SVG path.  Painter-order
            # compositing is local to a path's coverage tile, so retain only
            # those slices while keeping the current canvas/transparency as
            # full images. This is algebraically the same replay derivative.
            with torch.no_grad():
                coverages: list[tuple[Any, int, int] | None] = [None] * len(entries)
                for (
                    _shape,
                    fill_rule,
                    tile_width,
                    tile_height,
                ), items in simple_groups.items():
                    for index, alpha, left, top in rasterise_simple_tiles(
                        fill_rule, tile_width, tile_height, items
                    ):
                        coverages[index] = (alpha, left, top)
                for index, path in enumerate(controls):
                    if coverages[index] is None:
                        coverages[index] = (rasterise_multi(index, path), 0, 0)

                opacity_values = (
                    alpha_values.detach().clamp(0, 1)
                    if alpha_values is not None
                    else None
                )
                stored_alphas: list[Any] = []
                before_tiles: list[Any] = []
                rendered = torch.zeros_like(goal) if under is None else under.clone()
                for index, item in enumerate(coverages):
                    assert item is not None
                    alpha, left, top = item
                    if opacity_values is not None:
                        alpha = alpha * opacity_values[index]
                    bottom, right = top + alpha.shape[0], left + alpha.shape[1]
                    canvas = rendered[top:bottom, left:right]
                    before_tiles.append(canvas.clone())
                    rendered[top:bottom, left:right] = (
                        canvas * (1 - alpha[..., None])
                        + color_storage[index].detach().clamp(0, 1) * alpha[..., None]
                    )
                    stored_alphas.append(alpha)

                suffix_tiles: list[Any] = [None] * len(entries)
                transparency = torch.ones(
                    (work_height, work_width), dtype=goal.dtype, device=device
                )
                for index in range(len(entries) - 1, -1, -1):
                    item = coverages[index]
                    assert item is not None
                    alpha, left, top = item
                    bottom, right = top + alpha.shape[0], left + alpha.shape[1]
                    suffix = transparency[top:bottom, left:right]
                    suffix_tiles[index] = suffix.clone()
                    suffix.mul_(1 - stored_alphas[index])
                image_gradient = 2 * (rendered - goal) / rendered.numel()

            def sparse_layer_loss(
                index: int,
                alpha: Any,
                left: int,
                top: int,
                *,
                saved_coverages: list[tuple[Any, int, int] | None] = coverages,
                saved_alphas: list[Any] = stored_alphas,
                saved_suffixes: list[Any] = suffix_tiles,
                saved_canvases: list[Any] = before_tiles,
                gradient: Any = image_gradient,
            ) -> Any:
                item = saved_coverages[index]
                assert item is not None
                coverage, _stored_left, _stored_top = item
                stored_alpha = saved_alphas[index]
                suffix = saved_suffixes[index]
                canvas = saved_canvases[index]
                bottom, right = top + alpha.shape[0], left + alpha.shape[1]
                gradient = gradient[top:bottom, left:right]
                colour = color_storage[index]
                colour_delta = colour.detach().clamp(0, 1) - canvas
                alpha_gradient = (gradient * suffix[..., None] * colour_delta).sum(
                    dim=-1
                )
                opacity = (
                    alpha_values[index].clamp(0, 1)
                    if alpha_values is not None
                    else None
                )
                colour_gradient = (
                    gradient * suffix[..., None] * stored_alpha[..., None]
                ).sum(dim=(0, 1))
                geometry_loss = (alpha * alpha_gradient.detach()).sum()
                if opacity is not None:
                    geometry_loss = geometry_loss * opacity
                    geometry_loss = (
                        geometry_loss
                        + opacity * (coverage * alpha_gradient.detach()).sum()
                    )
                return (
                    geometry_loss
                    + (colour.clamp(0, 1) * colour_gradient.detach()).sum()
                )

            for (
                _shape,
                fill_rule,
                tile_width,
                tile_height,
            ), items in simple_groups.items():
                # Sparse replay keeps only a tile-local graph, so it can
                # batch more equal-size paths than the legacy dense replay.
                # This reduces native coverage launches without increasing the
                # full-canvas memory footprint.
                for offset in range(0, len(items), 16):
                    loss = torch.zeros((), device=device)
                    for index, alpha, left, top in rasterise_simple_tiles(
                        fill_rule, tile_width, tile_height, items[offset : offset + 16]
                    ):
                        loss = loss + sparse_layer_loss(index, alpha, left, top)
                    loss.backward()
            simple_indices = {
                index
                for group in simple_groups.values()
                for index, _left, _top in group
            }
            for index, path in enumerate(controls):
                if index not in simple_indices:
                    alpha = rasterise_multi(index, path)
                    sparse_layer_loss(index, alpha, 0, 0).backward()
            (
                xing_weight
                * (_xing_penalties(all_controls) * xing_contour_weights).sum()
            ).backward()
            point_optimizer.step()
            colour_optimizer.step()
            close_contours()
            continue

        # First composite the exact same soft fills without recording an
        # autograd graph.  The saved canvases and suffix transparencies are
        # enough to derive the MSE gradient of each layer independently.
        with torch.no_grad():
            initial_alphas: list[Any | None] = [None] * len(entries)
            for (
                _shape,
                fill_rule,
                tile_width,
                tile_height,
            ), items in simple_groups.items():
                for index, alpha in rasterise_simple(
                    fill_rule, tile_width, tile_height, items
                ):
                    initial_alphas[index] = alpha
            for (
                fill_rule,
                tile_width,
                tile_height,
            ), items in initial_multi_groups.items():
                for index, alpha in rasterise_multi_group(
                    fill_rule, tile_width, tile_height, items
                ):
                    initial_alphas[index] = alpha
            for index, path in enumerate(controls):
                if initial_alphas[index] is None:
                    initial_alphas[index] = rasterise_multi(index, path)

            before: list[Any] = []
            rendered = torch.zeros_like(goal) if under is None else under
            opacity_values = (
                alpha_values.detach().clamp(0, 1) if alpha_values is not None else None
            )
            initial_coverages: list[Any | None] = initial_alphas.copy()
            for index, alpha in enumerate(initial_alphas):
                assert alpha is not None
                if opacity_values is not None:
                    alpha = alpha * opacity_values[index]
                    initial_alphas[index] = alpha
                colour = color_storage[index]
                before.append(rendered)
                rendered = (
                    rendered * (1 - alpha[..., None])
                    + colour.detach().clamp(0, 1) * alpha[..., None]
                )
            downstream: list[Any | None] = [None] * len(entries)
            transparency = torch.ones(
                (work_height, work_width), dtype=goal.dtype, device=device
            )
            for index in range(len(entries) - 1, -1, -1):
                downstream[index] = transparency
                alpha = initial_alphas[index]
                assert alpha is not None
                transparency = transparency * (1 - alpha)
            image_gradient = 2 * (rendered - goal) / rendered.numel()

        def layer_loss(
            index: int,
            alpha: Any,
            *,
            alphas: list[Any | None] = initial_alphas,
            suffixes: list[Any | None] = downstream,
            canvases: list[Any] = before,
            coverages: list[Any | None] = initial_coverages,
            gradient: Any = image_gradient,
        ) -> Any:
            stored_alpha = alphas[index]
            suffix = suffixes[index]
            assert stored_alpha is not None
            assert suffix is not None
            colour = color_storage[index]
            colour_delta = colour.detach().clamp(0, 1) - canvases[index]
            alpha_gradient = (gradient * suffix[..., None] * colour_delta).sum(dim=-1)
            opacity = (
                alpha_values[index].clamp(0, 1) if alpha_values is not None else None
            )
            colour_gradient = (
                gradient * suffix[..., None] * stored_alpha[..., None]
            ).sum(dim=(0, 1))
            geometry_loss = (alpha * alpha_gradient.detach()).sum()
            if opacity is not None:
                geometry_loss = geometry_loss * opacity
                coverage = coverages[index]
                assert coverage is not None
                opacity_gradient = (coverage * alpha_gradient.detach()).sum()
                geometry_loss = geometry_loss + opacity * opacity_gradient
            return geometry_loss + (colour.clamp(0, 1) * colour_gradient.detach()).sum()

        # Backpropagate a bounded batch at a time.  The compositing derivative
        # above accounts for all later opaque layers, so this has the same MSE
        # gradient as one monolithic render without its peak-memory cost.
        for (
            _shape,
            fill_rule,
            tile_width,
            tile_height,
        ), items in simple_groups.items():
            for offset in range(0, len(items), 4):
                batch = items[offset : offset + 4]
                loss = torch.zeros((), device=device)
                for index, alpha in rasterise_simple(
                    fill_rule, tile_width, tile_height, batch
                ):
                    loss = loss + layer_loss(index, alpha)
                loss.backward()
        simple_indices = {
            index for group in simple_groups.values() for index, _left, _top in group
        }
        multi_indices = {
            index
            for group in initial_multi_groups.values()
            for index, _left, _top in group
        }
        for (fill_rule, tile_width, tile_height), items in initial_multi_groups.items():
            for offset in range(0, len(items), 64):
                batch = items[offset : offset + 64]
                loss = torch.zeros((), device=device)
                for index, alpha in rasterise_multi_group(
                    fill_rule, tile_width, tile_height, batch
                ):
                    loss = loss + layer_loss(index, alpha)
                loss.backward()
        for index, path in enumerate(controls):
            if index not in simple_indices and index not in multi_indices:
                layer_loss(
                    index,
                    rasterise_multi(index, path),
                ).backward()
        (
            xing_weight * (_xing_penalties(all_controls) * xing_contour_weights).sum()
        ).backward()
        point_optimizer.step()
        colour_optimizer.step()
        close_contours()

    coordinate_scale_cpu = coordinate_scale.cpu()
    for index, ((element, _contours, _colour, _fill_rule, _opacity), path) in enumerate(
        zip(entries, controls, strict=True)
    ):
        colour = color_storage[index]
        data = " ".join(
            to_path_d((control.detach().cpu() / coordinate_scale_cpu).tolist()) + " Z"
            for control in path
        )
        element.set("d", data)
        red, green, blue = (
            round(float(v) * 255) for v in colour.detach().clamp(0, 1).cpu()
        )
        element.set("fill", f"#{red:02x}{green:02x}{blue:02x}")
        if alpha_values is not None:
            element.set(
                "fill-opacity",
                f"{float(alpha_values[index].detach().clamp(0, 1).cpu()):.8g}",
            )
    return ET.tostring(root, encoding="unicode")


PATH_FIT = "Mutation: path fit"


def fittable_opaque_fills(svg: str) -> bool:
    """Whether *svg* contains a fill the analytic CUDA fitter can optimise."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return False
    for element in root.iter():
        if (
            element.tag.split("}")[-1] != "path"
            or _fill_rgb(element.get("fill")) is None
        ):
            continue
        try:
            parse_filled_cubics(element.get("d", ""))
        except UnsupportedPathError:
            continue
        if element.get("fill-rule", "nonzero").strip().lower() in {
            "evenodd",
            "nonzero",
        }:
            return True
    return False


_FillBounds = tuple[float, float, float, float]
_FittableFill = tuple[int, Any, _FillBounds]


def _fittable_fill_elements(root) -> list[_FittableFill]:
    """Return document-indexed opaque fills with conservative control bounds."""
    entries = []
    for document_index, element in enumerate(root.iter()):
        if element.tag.split("}")[-1] != "path" or not element.get("d"):
            continue
        if _fill_rgb(element.get("fill")) is None:
            continue
        try:
            contours = parse_filled_cubics(element.get("d", ""))
        except UnsupportedPathError:
            continue
        if element.get("fill-rule", "nonzero").strip().lower() not in {
            "evenodd",
            "nonzero",
        }:
            continue
        points = [point for contour in contours for cubic in contour for point in cubic]
        entries.append(
            (
                document_index,
                element,
                (
                    min(point[0] for point in points),
                    min(point[1] for point in points),
                    max(point[0] for point in points),
                    max(point[1] for point in points),
                ),
            )
        )
    return entries


def _select_fill_group(
    entries: list[_FittableFill],
    *,
    weights: Mapping[int, float] | None,
    maximum_paths: int,
) -> set[int]:
    """Choose one bounded spatial fill group, biased toward attributed error."""
    if maximum_paths < 1:
        raise ValueError("maximum_paths must be positive")
    scores = [max(0.0, (weights or {}).get(index, 0.0)) for index, _el, _box in entries]
    focal = (
        random.choices(entries, weights=scores, k=1)[0]
        if sum(scores) > 0
        else random.choice(entries)
    )
    focal_index, _element, (left, top, right, bottom) = focal
    centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
    extent = max(right - left, bottom - top, 8.0)

    def distance(entry: _FittableFill) -> tuple[int, float, int]:
        (
            index,
            _candidate,
            (
                candidate_left,
                candidate_top,
                candidate_right,
                candidate_bottom,
            ),
        ) = entry
        candidate_x = (candidate_left + candidate_right) / 2
        candidate_y = (candidate_top + candidate_bottom) / 2
        overlap = not (
            candidate_right < left - extent
            or candidate_left > right + extent
            or candidate_bottom < top - extent
            or candidate_top > bottom + extent
        )
        return (
            0 if overlap else 1,
            (candidate_x - centre_x) ** 2 + (candidate_y - centre_y) ** 2,
            index,
        )

    selected = sorted(entries, key=distance)[:maximum_paths]
    return {index for index, _element, _box in selected} | {focal_index}


def fill_groups(svg: str, *, maximum_paths: int = 16) -> list[set[int]]:
    """Partition opaque fills into bounded spatial groups for coordinate descent."""
    import xml.etree.ElementTree as ET

    entries = _fittable_fill_elements(ET.fromstring(svg))
    remaining = {index for index, _element, _box in entries}
    groups = []
    while remaining:
        focal = next(entry for entry in entries if entry[0] in remaining)
        focal_index, _element, (left, top, right, bottom) = focal
        centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
        extent = max(right - left, bottom - top, 8.0)

        def key(
            entry: _FittableFill,
            bounds: _FillBounds = (left, top, right, bottom),
            radius: float = extent,
            centre: tuple[float, float] = (centre_x, centre_y),
        ) -> tuple[int, float, int]:
            (
                index,
                _candidate,
                (
                    candidate_left,
                    candidate_top,
                    candidate_right,
                    candidate_bottom,
                ),
            ) = entry
            focal_left, focal_top, focal_right, focal_bottom = bounds
            focal_x, focal_y = centre
            candidate_x = (candidate_left + candidate_right) / 2
            candidate_y = (candidate_top + candidate_bottom) / 2
            overlap = not (
                candidate_right < focal_left - radius
                or candidate_left > focal_right + radius
                or candidate_bottom < focal_top - radius
                or candidate_top > focal_bottom + radius
            )
            return (
                0 if overlap else 1,
                (candidate_x - focal_x) ** 2 + (candidate_y - focal_y) ** 2,
                index,
            )

        group = {
            index
            for index, _element, _bounds in sorted(
                (entry for entry in entries if entry[0] in remaining), key=key
            )[:maximum_paths]
        }
        group.add(focal_index)
        groups.append(group)
        remaining -= group
    return groups


def fit_opaque_fills_locally(
    svg: str,
    reference_png: bytes,
    *,
    steps: int = 8,
    rasterize=None,
    weights: Mapping[int, float] | None = None,
    maximum_paths: int = 16,
    selected_indices: set[int] | None = None,
    optimisation_long_side: int | None = 64,
    gpu_gate: Any = None,
    learn_alpha: bool = False,
) -> str:
    """Fit one spatially bounded fill group as a local-search move.

    Unlike the legacy stroke fitter this operates on complete filled shapes,
    including compound paths and holes.  It deliberately keeps the 64px
    optimisation raster used by SAMVG; this is a local move, not its 500-step
    seed-fitting phase.
    """
    from PIL import Image

    target = Image.open(io.BytesIO(reference_png)).convert("RGB")
    if rasterize is None:
        raise UnsupportedPathError("bounded fill fitting needs an SVG rasterizer")
    import xml.etree.ElementTree as ET

    original = ET.fromstring(svg)
    entries = _fittable_fill_elements(original)
    if not entries:
        raise UnsupportedPathError("no opaque filled cubic paths to fit")
    selected_indices = selected_indices or _select_fill_group(
        entries, weights=weights, maximum_paths=maximum_paths
    )
    backdrop_root = ET.fromstring(svg)
    working_root = ET.fromstring(svg)
    for index, element in enumerate(backdrop_root.iter()):
        if index in selected_indices:
            element.set("d", "")
    for index, element in enumerate(working_root.iter()):
        if index not in selected_indices and element.tag.split("}")[-1] == "path":
            element.set("d", "")
    backdrop = Image.open(
        io.BytesIO(
            rasterize(
                ET.tostring(backdrop_root, encoding="unicode"),
                target.width,
                target.height,
            )
        )
    ).convert("RGB")
    with gpu_slot(gpu_gate):
        fitted = fit_filled_svg(
            ET.tostring(working_root, encoding="unicode"),
            target,
            steps=steps,
            optimisation_long_side=optimisation_long_side,
            backdrop=backdrop,
            learn_alpha=learn_alpha,
        )
    fitted_root = ET.fromstring(fitted)
    fitted_by_index = dict(enumerate(fitted_root.iter()))
    for index, element in enumerate(original.iter()):
        if index not in selected_indices:
            continue
        updated = fitted_by_index[index]
        element.set("d", updated.get("d", ""))
        element.set("fill", updated.get("fill", element.get("fill", "")))
        if learn_alpha:
            element.set("fill-opacity", updated.get("fill-opacity", "1"))
    return ET.tostring(original, encoding="unicode")


def fit_filled_svg_bounded(
    svg: str,
    target: Image.Image,
    *,
    rasterize,
    steps: int = 500,
    maximum_paths: int = 16,
    gpu_gate: Any = None,
    measurements: list[dict[str, int | float]] | None = None,
    learn_alpha: bool = False,
    global_replay: bool = True,
) -> str:
    """Run one full SAMVG fill phase as bounded spatial coordinate descent.

    ``steps`` is the per-group phase budget.  ``global_replay`` uses the
    sparse painter-order replay to give every path the dissertation's one
    simultaneous Adam update per iteration without materialising a full alpha
    stack.  The older coordinate-descent path remains available for local
    experiments. Coordinate descent needs to give
    every group the same fitting opportunity that it would have had in the
    original global graph; splitting that budget between groups loses detail.
    It consequently trades wall time for a strictly bounded differentiable
    graph.  When requested, ``measurements`` receives one timing and CUDA-peak
    record for each local group mutation.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if global_replay:
        import xml.etree.ElementTree as ET

        started = perf_counter()
        peak_before = 0
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                peak_before = int(torch.cuda.max_memory_allocated())
        except ImportError:
            torch = None  # type: ignore[assignment]
        with gpu_slot(gpu_gate):
            fitted = fit_filled_svg(
                svg,
                target,
                steps=steps,
                learn_alpha=learn_alpha,
                sparse_replay=True,
            )
        if measurements is not None:
            peak = peak_before
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
            measurements.append(
                {
                    "group": 0,
                    "paths": len(_fittable_fill_elements(ET.fromstring(svg))),
                    "seconds": perf_counter() - started,
                    "peak_cuda_bytes": peak,
                }
            )
        return fitted
    groups = fill_groups(svg, maximum_paths=maximum_paths)
    if not groups:
        raise UnsupportedPathError("no opaque filled cubic paths to optimise")
    encoded = io.BytesIO()
    target.convert("RGB").save(encoded, format="PNG")
    fitted = svg
    for index, group in enumerate(groups):
        peak_before = 0
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                peak_before = int(torch.cuda.max_memory_allocated())
        except ImportError:
            torch = None  # type: ignore[assignment]
        started = perf_counter()
        fitted = fit_opaque_fills_locally(
            fitted,
            encoded.getvalue(),
            steps=steps,
            rasterize=rasterize,
            maximum_paths=maximum_paths,
            selected_indices=group,
            optimisation_long_side=None,
            gpu_gate=gpu_gate,
            learn_alpha=learn_alpha,
        )
        if measurements is not None:
            peak = peak_before
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
            measurements.append(
                {
                    "group": index,
                    "paths": len(group),
                    "seconds": perf_counter() - started,
                    "peak_cuda_bytes": peak,
                }
            )
    return fitted


def fittable_strokes(svg: str) -> bool:
    """Whether the unified cubic-stroke fitter can select a stroke group."""
    import xml.etree.ElementTree as ET

    try:
        return bool(fittable_clusters(ET.fromstring(svg)))
    except ET.ParseError:
        return False


def fit_svg_primitives_locally(
    svg: str,
    reference_png: bytes,
    *,
    rasterize,
    weights: Mapping[int, float] | None = None,
    steps: int = 8,
    gpu_gate: Any = None,
) -> str:
    """Fit one selected fill or stroke primitive group over fixed SVG context.

    Each fitter rasterizes the non-active document as a fixed backdrop.  This
    prevents a fill from being rewarded for covering a line or editable text,
    while the subsequent stroke move sees the newly fitted fills unchanged.
    """
    fills = fittable_opaque_fills(svg)
    strokes = fittable_strokes(svg)
    if fills and (not strokes or random.random() < 0.5):
        return fit_opaque_fills_locally(
            svg,
            reference_png,
            steps=steps,
            rasterize=rasterize,
            weights=weights,
            gpu_gate=gpu_gate,
        )
    if strokes:
        return fit_random_group(
            svg,
            reference_png,
            rasterize=rasterize,
            steps=steps,
            weights=weights,
            gpu_gate=gpu_gate,
        )
    raise UnsupportedPathError("no supported filled or stroked cubics to fit")


def _stroke_width(element, ancestors) -> float | None:
    """Return the inherited stroke width, or ``None`` for an unpainted path."""
    for node in (element, *ancestors):
        raw = node.get("stroke-width")
        if raw:
            try:
                return abs(float(str(raw).rstrip("px")))
            except ValueError:
                return None
        if (node.get("stroke") or "").strip() in ("none",):
            return None
    return None


def _stroke_rgb(
    element: Any, ancestors: list[Any]
) -> tuple[float, float, float] | None:
    """Return an inherited opaque hex stroke colour, if the path paints one."""
    for node in (element, *ancestors):
        raw = node.get("stroke")
        if raw is None:
            continue
        if raw.strip().lower() == "none":
            return None
        return _fill_rgb(raw)
    return None


def _parents(root) -> dict[int, Any]:
    """id(child) -> parent, so a path can be read in the context it inherits."""
    table: dict[int, Any] = {}
    for parent in root.iter():
        for child in parent:
            table[id(child)] = parent
    return table


def _ancestry(element, table, root) -> list[Any]:
    chain, node = [], table.get(id(element))
    while node is not None:
        chain.append(node)
        node = table.get(id(node))
    if root not in chain:
        chain.append(root)
    return chain


def fittable_groups(root) -> list[tuple[Any, list[Any], list[float]]]:
    """Groups whose every path this rasterizer can represent exactly.

    A group rather than a path because paths that meet -- the two halves of a
    bill -- have to move together, and a group is the drawing's own statement
    about which those are.
    """
    out = []
    table = _parents(root)
    for group in root.iter():
        if group.tag.split("}")[-1] != "g":
            continue
        paths = [
            child
            for child in group
            if child.tag.split("}")[-1] == "path" and child.get("d")
        ]
        if not paths:
            continue
        keep, widths = [], []
        for path in paths:
            width = _stroke_width(path, _ancestry(path, table, root))
            if width is None or width <= 0:
                continue
            try:
                to_knots(parse_cubics(path.get("d", "")))
            except UnsupportedPathError:
                continue
            keep.append(path)
            widths.append(width)
        if keep:
            out.append((group, keep, widths))
    return out


def fit_random_group(
    svg: str,
    reference_png: bytes,
    *,
    rasterize,
    steps: int = 8,
    samples: int | None = None,
    weights: Mapping[int, float] | None = None,
    reach: tuple[float, float] = (0.8, 3.5),
    gpu_gate: Any = None,
) -> str:
    """Fit one cluster of touching paths to the reference, returning the edit.

    Raises UnsupportedPathError when the drawing offers nothing this can fit,
    which the caller reports as an operator that found nothing to change.

    *weights* is the error attribution the other operators already use, keyed by
    element index in document order; a cluster is drawn in proportion to the
    error its own paths answer for, so the fit is spent where the drawing is
    wrong rather than on whichever cluster came first.

    *reach* is the range the contact threshold is drawn from, and drawing it
    rather than fixing it is deliberate: the threshold decides what counts as
    one part, and there is no single right answer. Near the bottom of the range
    a wing's feathers are their own unit and can be shaped without disturbing
    the sweep they hang off; near the top they move with it. An operator that
    fixed the threshold would make that choice once for the whole run, where the
    search can afford to make it differently each time and keep what worked.

    The default uses a small number of full-resolution steps so this remains an
    occasional, expensive operator rather than dominating the search.
    """
    import xml.etree.ElementTree as ET

    from PIL import Image

    root = ET.fromstring(svg)
    clusters = fittable_clusters(root, random.uniform(*reach))
    if not clusters:
        raise UnsupportedPathError("no touching stroked cubics to fit")

    order = {id(el): index for index, el in enumerate(drawable_elements(root))}
    scores = [
        sum((weights or {}).get(order.get(id(p), -1), 0.0) for p in paths)
        for paths, _widths in clusters
    ]
    if sum(scores) > 0:
        paths, widths = random.choices(clusters, weights=scores, k=1)[0]
    else:
        paths, widths = random.choice(clusters)

    # Sometimes shape part of a cluster rather than all of it. What counts as
    # one part is a judgement the drawing does not settle -- a wing's feathers
    # can be shaped on their own or carried with the sweep they hang off -- and
    # an operator that decided once would make that choice for the whole run.
    # Deciding per draw lets selection keep whichever worked.
    #
    # Whatever is left out still has to be met: a point the chosen paths share
    # with an excluded one is pinned, so a partial fit cannot tear a junction
    # that welding exists to hold.
    excluded: list[Any] = []
    if len(paths) > 1 and random.random() < 0.5:
        keep = random.randint(1, len(paths) - 1)
        chosen = random.sample(range(len(paths)), keep)
        excluded = [p for i, p in enumerate(paths) if i not in chosen]
        widths = [w for i, w in enumerate(widths) if i in chosen]
        paths = [p for i, p in enumerate(paths) if i in chosen]

    size = int(_canvas_side(root))
    target = Image.open(io.BytesIO(reference_png)).convert("RGB").resize((size, size))

    # The backdrop is the drawing without these paths, so the fit sees the rest
    # of the picture as a constant and cannot be rewarded for redrawing it.
    # Blanking `d` in place and restoring it afterwards avoids having to match
    # elements across a copied tree.
    original = [path.get("d", "") for path in paths]
    for path in paths:
        path.set("d", "")
    backdrop = Image.open(
        io.BytesIO(rasterize(ET.tostring(root, encoding="unicode"), size, size))
    ).convert("RGB")
    for path, data in zip(paths, original, strict=True):
        path.set("d", data)

    held = _shared_vertices([parse_cubics(d) for d in original], excluded)
    parents = _parents(root)
    colours = [
        _stroke_rgb(path, _ancestry(path, parents, root)) or (0.0, 0.0, 0.0)
        for path in paths
    ]
    with gpu_slot(gpu_gate):
        fitted, fitted_widths, fitted_colours, _first, _last = fit_group(
            original,
            widths,
            target,
            backdrop,
            colours,
            size=size,
            steps=steps,
            samples=samples,
            pinned=held,
        )
    for path, data, width, colour in zip(
        paths, fitted, fitted_widths, fitted_colours, strict=True
    ):
        path.set("d", data)
        path.set("stroke-width", f"{width:.2f}")
        path.set(
            "stroke",
            "#" + "".join(f"{round(channel * 255):02x}" for channel in colour),
        )
    return ET.tostring(root, encoding="unicode")


def _shared_vertices(parsed, excluded, tolerance: float = 0.01) -> set[int]:
    """Welded vertices the fitted paths share with a path left out of the fit."""
    if not excluded:
        return set()
    outside = []
    for path in excluded:
        try:
            outside.extend(to_knots(parse_cubics(path.get("d", ""))))
        except UnsupportedPathError:
            continue
    welded, _index = weld([to_knots(segs) for segs in parsed])
    return {
        position
        for position, (x, y) in enumerate(welded)
        if any(
            abs(x - qx) <= tolerance and abs(y - qy) <= tolerance for qx, qy in outside
        )
    }


def _canvas_side(root) -> float:
    box = root.get("viewBox", "").replace(",", " ").split()
    if len(box) == 4:
        try:
            return max(abs(float(box[2])), abs(float(box[3]))) or 700.0
        except ValueError:
            pass
    return 700.0


# Enough device memory to be worth starting: a context plus the crop's tensors.
_FIT_HEADROOM = 512 * 1024 * 1024


@contextmanager
def gpu_slot(gpu_gate: Any) -> Iterator[None]:
    """Hold the run-wide GPU slot for a path fit, when one is configured."""
    if gpu_gate is None:
        yield
        return
    gpu_gate.acquire()
    try:
        yield
    finally:
        gpu_gate.release()


def fit_available() -> bool:
    """Whether CUDA has enough capacity for a path fit."""
    try:
        import torch
    except ImportError:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        # Each fitting worker owns a CUDA context, so reserve headroom first.
        free, _total = torch.cuda.mem_get_info()
        return free > _FIT_HEADROOM
    except Exception:  # pragma: no cover - driver trouble is not our business
        return False


def _touching(
    a: list[tuple[float, float]], b: list[tuple[float, float]], reach: float
) -> bool:
    """Whether two chains come within *reach* of each other anywhere."""
    return any(
        (px - qx) ** 2 + (py - qy) ** 2 <= reach * reach for px, py in a for qx, qy in b
    )


def fittable_clusters(root, reach_multiple: float = 2.5):
    """Group touching stroked paths, with contact reach scaled by width."""
    table = _parents(root)
    entries = []
    for element in root.iter():
        if element.tag.split("}")[-1] != "path" or not element.get("d"):
            continue
        width = _stroke_width(element, _ancestry(element, table, root))
        if width is None or width <= 0:
            continue
        try:
            chain = to_knots(parse_cubics(element.get("d", "")))
        except UnsupportedPathError:
            continue
        entries.append((element, width, chain))

    parent = list(range(len(entries)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            reach = reach_multiple * max(entries[i][1], entries[j][1])
            if _touching(entries[i][2], entries[j][2], reach):
                parent[find(i)] = find(j)

    clusters: dict[int, tuple[list[Any], list[float]]] = {}
    for index, (element, width, _chain) in enumerate(entries):
        paths, widths = clusters.setdefault(find(index), ([], []))
        paths.append(element)
        widths.append(width)
    return list(clusters.values())
