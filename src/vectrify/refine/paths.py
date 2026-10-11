"""Fit a group of stroked cubic paths to a target by gradient descent.

The differentiable rasterizer supports stroked cubics only. Pixels are sampled
at their centres, and paths in a group are fitted jointly so overlapping
strokes can move together.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Absolute commands only: normalize_svg has already run, and a relative command
# here means the drawing skipped that pass rather than that it needs handling.
_TOKEN = re.compile(r"([MLCZmlcz])|(-?(?:\d+\.\d+|\.\d+|\d+))")
_SUPPORTED = frozenset("MLCZ")


class UnsupportedPathError(ValueError):
    """Raised for path data this rasterizer cannot represent exactly."""


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


def _as_cubic(a, b):
    return [
        a,
        (a[0] + (b[0] - a[0]) / 3, a[1] + (b[1] - a[1]) / 3),
        (a[0] + 2 * (b[0] - a[0]) / 3, a[1] + 2 * (b[1] - a[1]) / 3),
        b,
    ]


def to_path_d(segments, *, precision: int = 1) -> str:
    """Cubic segments back to path data, one C per segment."""
    head = segments[0][0]
    parts = [f"M {head[0]:.{precision}f} {head[1]:.{precision}f}"]
    for segment in segments:
        parts.append(
            "C "
            + " ".join(f"{x:.{precision}f} {y:.{precision}f}" for x, y in segment[1:])
        )
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


def _torch_compile_enabled() -> bool:
    """Whether this process permits Torch Dynamo to compile renderer kernels."""
    # PyTorch raises from an already-created ``torch.compile`` wrapper when
    # this environment switch is set.  Respect it before creating the cached
    # wrapper so the advertised eager renderer is actually usable for
    # debugging, constrained deployments, and compiler-cache recovery.
    return os.environ.get("TORCH_COMPILE_DISABLE", "0") not in {"1", "true", "True"}


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
    if compile_fn is None or not _torch_compile_enabled():
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
    if compile_fn is None or not _torch_compile_enabled():
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


def _fused_chunks(control: Any) -> Any:
    """One contour's cubics as padded 16-cubic ranges, shape (k, 16, 4, 2).

    The analytic kernel sums signed ray crossings per cubic, with no closing
    edge of its own, so a contour's winding is the sum over any split of its
    cubics; a zero-length padding cubic crosses nothing. Chunking this way
    gives long contours the same exact, differentiable coverage as short ones,
    where the sampled-winding fallback has no gradient to speak of.
    """
    import torch

    count = control.shape[0]
    chunks = max(1, math.ceil(count / _FUSED_CUBICS))
    missing = chunks * _FUSED_CUBICS - count
    if missing:
        control = torch.cat((control, control[:1, :1].expand(missing, 4, -1)))
    return control.reshape(chunks, _FUSED_CUBICS, 4, 2)


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
    # the same representation as a fixed-length contour, not a tessellation
    # or geometry approximation, and keeps long contours off Torch's huge
    # broadcast fallback.
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
        # A detailed path can have many contours.  Sum each contour's
        # winding before applying the SVG fill rule, exactly as the eager
        # implementation below does, but keep the pixel/segment loops in the
        # fused CUDA primitive.
        fused = [contour for contour in contours if contour.shape[0] <= _FUSED_CUBICS]
        unfused: dict[tuple[int, ...], list[Any]] = defaultdict(list)
        for contour in contours:
            if contour.shape[0] > _FUSED_CUBICS:
                unfused[tuple(contour.shape)].append(contour)
        # Most contours have at most 16 cubics.  Pad
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
        # A noisy path can contain dozens of enclosed contours.  Keeping
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
                packed = _pad_fused_cubics(
                    packed_contours.index_select(0, indices) - offset
                )
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
        # Scatter the batched tiles into one canvas. Padding every small tile
        # to a full image creates a full-canvas addition and backward node per
        # tile, making the assembly cost grow with canvas area times tile count.
        origins = torch.tensor(
            [(tile[0], tile[1]) for _tile_number, tile in group],
            dtype=torch.long,
            device=coverage.device,
        )
        rows = (
            origins[:, 1, None, None]
            + torch.arange(tile_height, device=coverage.device)[None, :, None]
        )
        columns = (
            origins[:, 0, None, None]
            + torch.arange(tile_width, device=coverage.device)[None, None, :]
        )
        indices = (rows * width + columns).reshape(-1)
        restored = (
            coverage.new_zeros(height * width)
            .scatter_add(0, indices, coverage.reshape(-1))
            .reshape(height, width)
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

    Paths with a fixed number of cubics per contour are batched together.
    Keeping that regularity here removes Python's per-path kernel-launch loop;
    this is substantially faster on CUDA for the 500-step optimisation pass.
    """
    import torch

    left, top, right, bottom = box
    height, width = bottom - top, right - left
    # Simple closed contours are the common case.  Use the native
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
    # broadcasting, so one complete small working raster is faster than
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


def _xing_penalties(control: Any) -> Any:
    """Return the normalized Xing (handle crossing) penalty for every cubic
    in ``control``."""
    import torch

    start_handle = control[:, 1] - control[:, 0]
    middle_edge = control[:, 2] - control[:, 1]
    end_handle = control[:, 3] - control[:, 2]
    orientation = (
        start_handle[:, 0] * middle_edge[:, 1] - start_handle[:, 1] * middle_edge[:, 0]
    )
    cross = (
        start_handle[:, 0] * end_handle[:, 1] - start_handle[:, 1] * end_handle[:, 0]
    )
    sine = cross / (start_handle.norm(dim=-1) * end_handle.norm(dim=-1) + 1e-12)
    # AB x BC chooses the turn direction; AB x CD measures the violation.
    # Using AB x CD for both reduces this to abs(sine), penalizing valid bends.
    return torch.where(orientation > 0, torch.relu(-sine), torch.relu(sine))


def _xing_loss(control: Any) -> Any:
    """Return the normalized per-cubic Xing regularizer."""
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


def _composite_in_context(alphas: Any, colours: Any, context: tuple[Any, ...]) -> Any:
    """Apply a common frozen SVG response to a selected painter-order run."""
    import torch

    base, delta, transmission = context
    if len(alphas) == 1:
        rgb = colours[0].clamp(0, 1)
        paint = torch.cat((rgb, rgb.new_ones(1))) if base.shape[-1] == 4 else rgb
        return base + alphas[0, ..., None] * (delta + transmission * paint)
    opacity = 1 - (1 - alphas).prod(dim=0)
    paint = _composite_opaque_fills(alphas, colours)
    if base.shape[-1] == 4:
        paint = torch.cat((paint, opacity[..., None]), dim=-1)
    return base + opacity[..., None] * delta + transmission * paint


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
    max_point_displacement: float | None = None,
    fit_context: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    project_controls: Any = None,
    control_transform: Any = None,
    control_loss: Any = None,
    observe: Any = None,
    coverage_transform: Any = None,
    loss_transform: Any = None,
    device: str | None = None,
) -> str:
    """Optimise filled cubic SVG paths against an RGB target.

    Path coordinates and fill colours are optimised with Adam.
    ``learn_alpha`` gives each selected path a learnable fill opacity;
    by default it is off and fills are opaque. This implementation reuses
    Vectrify's torch renderer instead of requiring DiffVG, with full-resolution
    Adam defaults: point LR 1, colour LR .01,
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
    1024px fit practical without a monolithic alpha stack.
    ``max_point_displacement`` optionally bounds every control's Euclidean
    displacement from its seed in working-raster pixels. This limits contour
    drift without restricting fill colours; ``None`` retains the unbounded fit.
    ``fit_context`` supplies the editor's frozen affine compositing response
    for one selected path or a contiguous run of sibling paths
    (base, black-minus-base, white-minus-black), in
    white-backed RGB or white-backed RGB plus alpha for transparent references.
    In the latter case mean colour error and opacity error have equal weight.
    ``project_controls`` enforces editor coordinate constraints after each Adam
    update; ``observe`` reports/retains candidates and returns False to stop.
    ``control_transform`` is the differentiable counterpart of that projection
    in monolithic fits. Its Jacobian combines the gradients of tied endpoints,
    implicit straight controls and reversed stroke copies before Adam updates
    their independent coordinates. Projection runs before restoring duplicate
    endpoints, so those copies do not overwrite the independent update.
    ``loss_transform`` optionally prepares the rendered image and target for
    monolithic fitting, for example a coarse image comparison before refinement.
    These optional hooks leave the automatic path-fit mutation unchanged.
    ``control_loss`` adds a differentiable geometry penalty in monolithic fits.
    ``device`` overrides the default of CUDA whenever Torch sees a GPU.
    Without CUDA or the native extension, coverage comes from the portable
    polyline renderer in :mod:`vectrify.refine.soft_coverage`, whose gradient
    moves the geometry where the sampled winding numbers barely do.
    """
    from vectrify.refine.filled_fit_optimize import FilledFitOptimizer
    from vectrify.refine.filled_fit_raster import FilledFitRaster
    from vectrify.refine.filled_fit_scene import (
        FilledFitHooks,
        FilledFitOptions,
        prepare_scene,
    )

    options = FilledFitOptions(
        xing_weight=xing_weight,
        subpixels=subpixels,
        curve_samples=curve_samples,
        sparse_replay=sparse_replay,
        max_point_displacement=max_point_displacement,
    )
    hooks = FilledFitHooks(
        project_controls=project_controls,
        control_transform=control_transform,
        control_loss=control_loss,
        observe=observe,
        coverage_transform=coverage_transform,
        loss_transform=loss_transform,
    )
    scene = prepare_scene(
        svg,
        target,
        options=options,
        hooks=hooks,
        optimisation_long_side=optimisation_long_side,
        monolithic=monolithic,
        backdrop=backdrop,
        learn_alpha=learn_alpha,
        fit_context=fit_context,
        device=device,
    )
    raster = FilledFitRaster(scene, options)
    optimizer = FilledFitOptimizer(
        scene,
        raster,
        options,
        hooks,
        point_learning_rate=point_learning_rate,
        color_learning_rate=color_learning_rate,
    )
    log.info(
        "Filled-path optimisation: %d path(s), %dx%d working raster on %s.",
        len(scene.entries),
        scene.work_width,
        scene.work_height,
        scene.device,
    )
    optimizer.run(steps)
    return scene.serialize(max_point_displacement)
