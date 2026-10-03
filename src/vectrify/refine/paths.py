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
    observe: Any = None,
    coverage_transform: Any = None,
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
    for one selected path (base, black-minus-base, white-minus-black).
    ``project_controls`` enforces editor coordinate constraints after each Adam
    update; ``observe`` reports/retains candidates and returns False to stop.
    These optional hooks leave the automatic path-fit mutation unchanged.
    ``device`` overrides the default of CUDA whenever Torch sees a GPU.
    Without CUDA or the native extension, coverage comes from the portable
    polyline renderer in :mod:`vectrify.refine.soft_coverage`, whose gradient
    moves the geometry where the sampled winding numbers barely do.
    """
    import xml.etree.ElementTree as ET

    import torch

    if max_point_displacement is not None and (
        not math.isfinite(max_point_displacement) or max_point_displacement < 0
    ):
        raise ValueError("max_point_displacement must be finite and nonnegative")

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

    from vectrify.refine.cuda_renderer import available as native_available
    from vectrify.refine.soft_coverage import soft_coverage

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    soft = device != "cuda" or not native_available()
    width, height = target.size
    scale = (
        1.0
        if optimisation_long_side is None
        else min(1.0, optimisation_long_side / max(width, height))
    )
    work_width, work_height = round(width * scale), round(height * scale)
    if monolithic is None:
        # At a 64px fitting resolution the complete opaque-layer
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
    # A detailed drawing has hundreds of contours.  Keeping each one as a
    # separate Adam parameter turns one optimiser update into hundreds of tiny
    # CUDA kernels.  Store equal-width contour slots in one parameter and use
    # narrow views below, retaining every original contour length in the SVG
    # and Xing terms.  The native coverage primitive itself uses 16-cubic
    # chunks, but a drawing can have longer contours; storage must
    # therefore use the document maximum rather than that renderer chunk size.
    flat_controls = [control for path in initial_controls for control in path]
    contour_sizes = [len(control) for control in flat_controls]
    storage_width = max(contour_sizes)

    def pad_storage_control(control: Any) -> Any:
        if len(control) == storage_width:
            return control
        return torch.cat(
            (
                control,
                control[:1].expand(storage_width - len(control), -1, -1),
            )
        )

    control_storage = torch.nn.Parameter(
        torch.stack([pad_storage_control(control) for control in flat_controls])
    )
    seed_controls = (
        None if max_point_displacement is None else control_storage.detach().clone()
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

    # An editor-selected path needs the original painter-order response,
    # including clipping and isolated group opacity, rather than a backdrop
    # with the selected path implicitly painted on top of everything.
    context_tensors = None
    if fit_context is not None:
        if len(entries) != 1 or not monolithic or scale != 1:
            raise ValueError(
                "Selected-path context requires one unscaled monolithic path"
            )
        if any(array.shape != (height, width, 3) for array in fit_context):
            raise ValueError("Selected-path context must match the target raster")
        context_tensors = tuple(
            torch.tensor(array, dtype=torch.float32, device=device)
            for array in fit_context
        )

    def close_contours() -> None:
        """Restore the shared joins of every traced closed Bezier contour.

        Traced paths are closed fixed-segment loops.  The packed parameter storage
        keeps their cubic endpoints as separate Adam values for efficient
        rasterisation, so project them back to a continuous closed contour
        after each update.  Otherwise a subpixel gap becomes an extra implicit
        SVG closing cubic on export, violating the fixed-segment invariant.
        """
        with torch.no_grad():
            if seed_controls is not None:
                assert max_point_displacement is not None
                delta = control_storage - seed_controls
                factor = (
                    max_point_displacement / delta.norm(dim=-1).clamp_min(1e-12)
                ).clamp_max(1)
                control_storage.copy_(seed_controls + delta * factor[..., None])
            for path in controls:
                for contour in path:
                    contour[1:, 0].copy_(contour[:-1, 3])
                    contour[-1, 3].copy_(contour[0, 0])
            if project_controls is not None:
                project_controls(controls)

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
        if soft:
            return [
                (
                    index,
                    soft_coverage(
                        controls[index],
                        (left, top, left + tile_width, top + tile_height),
                        fill_rule=fill_rule,
                    ),
                    left,
                    top,
                )
                for index, left, top in items
            ]
        # A contour can be longer than the fixed-width coverage
        # primitive.  Route those through the chunked native winding path;
        # packing them into the old batched coverage call would force eager
        # Torch broadcasting over every cubic and pixel.
        if controls[items[0][0]][0].shape[0] > _FUSED_CUBICS:
            from vectrify.refine.cuda_renderer import multi_coverage

            chunked = [
                _fused_chunks(
                    controls[index][0] - controls[index][0].new_tensor((left, top))
                )
                for index, left, top in items
            ]
            offsets = [0]
            for chunks in chunked:
                offsets.append(offsets[-1] + len(chunks))
            analytic = multi_coverage(
                torch.cat(chunked),
                offsets,
                (0, 0, tile_width, tile_height),
                subpixels=subpixels,
                fill_rule=fill_rule,
            )
            if analytic is not None:
                return [
                    (index, alpha, left, top)
                    for (index, left, top), alpha in zip(items, analytic, strict=True)
                ]
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
            # Sparse replay keeps this graph alive through the layer's
            # backward pass.  Torch's dynamic compiler can specialise one
            # large tile batch into an unbounded graph here; eager chunking
            # has the same coverage/gradient while retaining the documented
            # tile-local memory bound.
            dynamic_fuse=False,
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

    def sparse_backward_batch_size(tile_width: int, tile_height: int) -> int:
        """Bound the live tile-local autograd graph while filling the GPU.

        Sparse replay never needs a full-canvas alpha stack, but it does keep
        the coverage graph for one backward batch alive.  A fixed 16-path
        batch underutilises CUDA for common 32--64px tiles, but a
        recovery pass can create a much larger equal-tile group than the
        initial seed.  Retain the proven 16-path graph cap and apply the
        tile-area budget beneath it.  This bounds peak memory for every
        document without changing the rendered image or its derivative.
        """
        return max(1, min(16, (1 << 20) // max(1, tile_width * tile_height)))

    def soft_multi(index: int, path: list[Any]) -> Any:
        # The tile follows the current controls, so movement never clips it.
        left, top, tile_width, tile_height = tile_for(path)
        alpha = soft_coverage(
            path,
            (left, top, left + tile_width, top + tile_height),
            fill_rule=entries[index][3],
        )
        return restore_tile(alpha, left, top)

    def rasterise_multi(index: int, path: list[Any]) -> Any:
        if soft:
            return soft_multi(index, path)
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

        packed = torch.cat([_fused_chunks(control - offset) for control in path])
        analytic = multi_coverage(
            packed,
            [0, len(packed)],
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
        if soft:
            return [
                (index, soft_multi(index, controls[index]))
                for index, _left, _top in items
            ]
        chunks = []
        spans = []
        for index, left, top in items:
            offset = controls[index][0].new_tensor((left, top))
            start = sum(len(c) for c in chunks)
            chunks.extend(
                _fused_chunks(control - offset) for control in controls[index]
            )
            spans.append((start, sum(len(c) for c in chunks)))
        # Native winding uses one CUDA block per contour chunk.  Combining
        # contours from otherwise independent paths lets its blocks occupy the
        # GPU at once, while summing each recorded span before the fill
        # nonlinearity preserves SVG path semantics (including holes); long
        # contours are split into 16-cubic chunks, whose windings add up.
        packed = torch.cat(chunks)
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
    # introduces hundreds of device synchronisations on a detailed
    # drawing.  The two-pixel antialias margin already makes these fixed tiles
    # conservative for the local coordinate updates used by the fit.
    initial_simple_groups = cropped_simple_groups()
    # Simple paths use cropped tiles, so unlike the full-canvas compositor
    # their bounds are optimisation state.  A coordinate update can move
    # a boundary outside its initial two-pixel antialias fringe; continuing to
    # rasterise the old crop silently clips that fill and creates the holes and
    # spikes visible in long fits.  Keep one packed reference so the movement
    # check is a single device reduction; rebuild the inexpensive Python tile
    # grouping only after a meaningful move.
    simple_tile_reference = control_storage.detach().clone()

    def refresh_simple_tiles() -> None:
        nonlocal initial_simple_groups, simple_tile_reference

        movement = (control_storage.detach() - simple_tile_reference).abs().amax()
        if float(movement) <= 1.0:
            return
        initial_simple_groups = cropped_simple_groups()
        simple_tile_reference = control_storage.detach().clone()

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
    completed_steps = 0
    for _step in range(steps):
        if observe is not None and not observe(_step, controls, color_storage):
            break
        completed_steps = _step + 1
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
            if coverage_transform is not None:
                alpha_stack = coverage_transform(controls, alpha_stack)
            if alpha_values is not None:
                alpha_stack = alpha_stack * alpha_values.clamp(0, 1)[:, None, None]
            if context_tensors is not None:
                # Selected-path fitting supplies its own compositing response.
                # Avoid compiling an unused full-document composite for every
                # crop size (which can exhaust Torch's recompilation cache).
                base, delta, transmission = context_tensors
                rendered = base + alpha_stack[0, ..., None] * (
                    delta + transmission * color_storage[0].clamp(0, 1)
                )
            else:
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
            loss = loss + xing_weight * _xing_loss(all_controls)
            loss.backward()
            point_optimizer.step()
            colour_optimizer.step()
            close_contours()
            refresh_simple_tiles()
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
                batch_size = sparse_backward_batch_size(tile_width, tile_height)
                for offset in range(0, len(items), batch_size):
                    loss = torch.zeros((), device=device)
                    for index, alpha, left, top in rasterise_simple_tiles(
                        fill_rule,
                        tile_width,
                        tile_height,
                        items[offset : offset + batch_size],
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
            (xing_weight * _xing_loss(all_controls)).backward()
            point_optimizer.step()
            colour_optimizer.step()
            close_contours()
            refresh_simple_tiles()
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
        (xing_weight * _xing_loss(all_controls)).backward()
        point_optimizer.step()
        colour_optimizer.step()
        close_contours()
        refresh_simple_tiles()

    if observe is not None:
        observe(completed_steps, controls, color_storage)
    coordinate_scale_cpu = coordinate_scale.cpu()
    for index, ((element, _contours, _colour, _fill_rule, _opacity), path) in enumerate(
        zip(entries, controls, strict=True)
    ):
        colour = color_storage[index]
        data = " ".join(
            to_path_d(
                (control.detach().cpu() / coordinate_scale_cpu).tolist(),
                precision=3 if max_point_displacement is not None else 1,
            )
            + " Z"
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
