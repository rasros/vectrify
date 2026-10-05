"""Differentiable cubic fill and round-stroke coverage in plain PyTorch.

The native CUDA kernel is exact; the sampled-winding fallback has almost no
gradient because winding is piecewise constant. This renderer works on any
device: each cubic becomes a short polyline, the fill rule decides inside or
outside from scanline ray crossings (no gradient needed), and the antialiased
edge is ``clamp(0.5 + sign * distance, 0, 1)`` with the distance to the
nearest polyline segment, through which the gradient flows.

Only pixels within half a pixel of the outline have fractional coverage, so
the distance is evaluated for (pixel, segment) pairs near each segment rather
than for every pixel against every segment. Memory grows with the outline's
length, not with the tile's area times the segment count.
"""

from __future__ import annotations

import math
from typing import Any

# Target polyline chord length in pixels, and the per-cubic sample bounds.
# A 2px chord on a 20px radius arc deviates from it by about 0.03px.
CHORD = 2.0
MIN_SAMPLES = 2
MAX_SAMPLES = 64


def _polyline(control: Any, chord: float) -> Any:
    """Sample one contour's cubics (n, 4, 2) into its closed polyline's vertices."""
    import torch

    with torch.no_grad():
        hull = (control[:, 1:] - control[:, :-1]).norm(dim=-1).sum(dim=-1).max()
        samples = int(
            min(MAX_SAMPLES, max(MIN_SAMPLES, math.ceil(float(hull) / chord)))
        )
    t = torch.arange(samples, dtype=control.dtype, device=control.device) / samples
    basis = torch.stack(
        (
            (1 - t) ** 3,
            3 * t * (1 - t) ** 2,
            3 * t**2 * (1 - t),
            t**3,
        ),
        dim=-1,
    )
    # Each cubic contributes its start and interior samples; the next cubic's
    # start (or, for the last, the contour's first point) closes the segment.
    return torch.einsum("sk,nkc->nsc", basis, control).reshape(-1, 2)


def soft_coverage(
    contours: list[Any],
    box: tuple[int, int, int, int],
    *,
    fill_rule: str = "nonzero",
    chord: float = CHORD,
    subpixels: int = 4,
) -> Any:
    """Coverage (height, width) of one path's closed contours inside *box*.

    *contours* are (n, 4, 2) cubic control tensors in pixel coordinates; the
    fill rule combines all of them, so holes work. Pixel ``(i, j)`` of the
    result covers ``[left + i, left + i + 1] x [top + j, top + j + 1]``.
    Integrating subpixel coverage distinguishes both sides of a thin feature;
    a single centre-distance sample otherwise paints it at least half opaque.
    """
    import torch

    if type(subpixels) is not int or subpixels < 1:
        raise ValueError("Subpixel count must be a positive integer")
    if subpixels == 1:
        return _sample_coverage(contours, box, fill_rule=fill_rule, chord=chord)
    samples = _sample_coverage(
        [control * subpixels for control in contours],
        tuple(value * subpixels for value in box),
        fill_rule=fill_rule,
        chord=chord * subpixels,
    )
    return torch.nn.functional.avg_pool2d(samples[None, None], subpixels, subpixels)[
        0, 0
    ]


def _sample_coverage(contours, box, *, fill_rule, chord):
    import torch

    left, top, right, bottom = box
    width, height = right - left, bottom - top
    reference = contours[0]
    origin = reference.new_tensor((left, top))
    starts, ends = [], []
    for control in contours:
        points = _polyline(control, chord) - origin
        starts.append(points)
        ends.append(torch.roll(points, -1, dims=0))
    a = torch.cat(starts)
    b = torch.cat(ends)

    with torch.no_grad():
        # Winding at each pixel centre from crossings of a ray to +x.
        ad, bd = a.detach(), b.detach()
        centre_y = torch.arange(height, dtype=a.dtype, device=a.device)[:, None] + 0.5
        above_a = ad[None, :, 1] <= centre_y
        above_b = bd[None, :, 1] <= centre_y
        crossing = above_a != above_b
        dy = bd[:, 1] - ad[:, 1]
        safe = torch.where(dy == 0, torch.ones_like(dy), dy)
        x = ad[:, 0] + (centre_y - ad[:, 1]) / safe * (bd[:, 0] - ad[:, 0])
        # Pixel i counts a crossing when its centre i + 0.5 lies left of x.
        bucket = torch.ceil(x - 0.5).clamp(0, width).long()
        direction = torch.where(dy > 0, 1, -1).to(torch.int32) * crossing
        counts = torch.zeros(
            (height, width + 1), dtype=torch.int32, device=a.device
        ).scatter_add_(1, bucket, direction.expand(height, -1).contiguous())
        winding = counts.flip(1).cumsum(1).flip(1)[:, 1:]
        inside = winding % 2 != 0 if fill_rule == "evenodd" else winding != 0
        sign = inside.to(a.dtype) * 2 - 1

        # Pixels whose centre can lie within half a pixel of each segment.
        low = torch.minimum(ad, bd)
        high = torch.maximum(ad, bd)
        x0 = torch.floor(low[:, 0] - 1).clamp(0, width - 1).long()
        x1 = torch.ceil(high[:, 0]).clamp(-1, width - 1).long()
        y0 = torch.floor(low[:, 1] - 1).clamp(0, height - 1).long()
        y1 = torch.ceil(high[:, 1]).clamp(-1, height - 1).long()
        spans = (x1 - x0 + 1).clamp_min(0)
        counts_per = spans * (y1 - y0 + 1).clamp_min(0)
        segment = torch.repeat_interleave(
            torch.arange(len(a), device=a.device), counts_per
        )
        first = torch.cumsum(counts_per, 0) - counts_per
        local = torch.arange(len(segment), device=a.device) - first[segment]
        px = x0[segment] + local % spans[segment]
        py = y0[segment] + local // spans[segment]
        pixel = py * width + px

    centre = torch.stack((px, py), dim=-1).to(a.dtype) + 0.5
    sa, sb = a[segment], b[segment]
    edge = sb - sa
    t = (((centre - sa) * edge).sum(-1) / (edge * edge).sum(-1).clamp_min(1e-12)).clamp(
        0, 1
    )
    offset = centre - (sa + t[:, None] * edge)
    squared = (offset * offset).sum(-1)
    distance = (squared + 1e-12).sqrt()
    boundary = squared <= 1e-10
    if bool(boundary.any()):
        # The norm has zero derivative at the edge itself. A diagonal can
        # hit every antialiased sample centre exactly, leaving no direction
        # for filling a seam. Use the filled side's oriented normal there,
        # retaining the original coverage values and all other derivatives.
        with torch.no_grad():
            indices = torch.unique(segment[boundary])
            vector = bd - ad
            normal = torch.stack((-vector[:, 1], vector[:, 0]), -1)
            normal /= vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            middle = (ad + bd) / 2
            epsilon = torch.maximum(
                middle.abs().amax(-1) * torch.finfo(a.dtype).eps * 8,
                middle.new_tensor(1e-3),
            )
            side = torch.zeros(len(a), dtype=a.dtype, device=a.device)
            for batch in indices.split(128):
                probe = torch.stack(
                    (
                        middle[batch] + normal[batch] * epsilon[batch, None],
                        middle[batch] - normal[batch] * epsilon[batch, None],
                    ),
                    1,
                ).reshape(-1, 2)
                y = probe[:, 1, None]
                crossed = (ad[None, :, 1] <= y) != (bd[None, :, 1] <= y)
                dy = vector[:, 1]
                safe = torch.where(dy == 0, torch.ones_like(dy), dy)
                x = ad[:, 0] + (y - ad[:, 1]) / safe * vector[:, 0]
                winding = (
                    (crossed & (probe[:, :1] < x)) * torch.where(dy > 0, 1, -1)
                ).sum(-1)
                filled = winding % 2 != 0 if fill_rule == "evenodd" else winding != 0
                filled = filled.reshape(-1, 2).to(a.dtype)
                side[batch] = filled[:, 0] - filled[:, 1]
            inward = normal * side[:, None]
        oriented = (offset * inward[segment]).sum(-1) * sign.flatten()[pixel]
        distance = torch.where(
            boundary, distance.detach() + (oriented - oriented.detach()), distance
        )
    nearest = torch.full(
        (height * width,), 1e4, dtype=a.dtype, device=a.device
    ).scatter_reduce(0, pixel, distance, "amin", include_self=True)
    return (0.5 + sign * nearest.reshape(height, width)).clamp(0, 1)


def soft_stroke_coverage(
    contours: list[Any], box, width: float, *, subpixels: int = 4
) -> Any:
    """Differentiable round-cap, round-join strokes, including open contours.

    Only pixels near the centreline are evaluated, as for soft fill coverage.
    Unlike a fill, a stroke never implicitly closes an open contour. Integrating
    subpixels keeps thin diagonal strokes from changing width with orientation.
    """
    import torch

    if type(subpixels) is not int or subpixels < 1:
        raise ValueError("Subpixel count must be a positive integer")
    if subpixels == 1:
        return _sample_stroke_coverage(contours, box, width, CHORD)
    samples = _sample_stroke_coverage(
        [control * subpixels for control in contours],
        tuple(value * subpixels for value in box),
        width * subpixels,
        CHORD * subpixels,
    )
    return torch.nn.functional.avg_pool2d(samples[None, None], subpixels, subpixels)[
        0, 0
    ]


def _sample_stroke_coverage(contours, box, width, chord):
    import torch

    left, top, right, bottom = box
    w, h = right - left, bottom - top
    reference = contours[0]
    starts, ends = [], []
    for control in contours:
        points = torch.cat((_polyline(control, chord), control[-1:, 3]))
        points = points - points.new_tensor((left, top))
        starts.append(points[:-1])
        ends.append(points[1:])
    a, b = torch.cat(starts), torch.cat(ends)
    radius = width / 2
    with torch.no_grad():
        low = torch.minimum(a, b) - radius - 1
        high = torch.maximum(a, b) + radius + 1
        x0 = low[:, 0].floor().clamp(0, w - 1).long()
        x1 = high[:, 0].ceil().clamp(-1, w - 1).long()
        y0 = low[:, 1].floor().clamp(0, h - 1).long()
        y1 = high[:, 1].ceil().clamp(-1, h - 1).long()
        spans = (x1 - x0 + 1).clamp_min(0)
        counts = spans * (y1 - y0 + 1).clamp_min(0)
        segment = torch.repeat_interleave(torch.arange(len(a), device=a.device), counts)
        first = counts.cumsum(0) - counts
        index = torch.arange(len(segment), device=a.device) - first[segment]
        px = x0[segment] + index % spans[segment]
        py = y0[segment] + index // spans[segment]
        pixel = py * w + px
    center = torch.stack((px, py), dim=-1).to(a.dtype) + 0.5
    start, end = a[segment], b[segment]
    edge = end - start
    t = ((center - start) * edge).sum(-1) / (edge * edge).sum(-1).clamp_min(1e-12)
    delta = center - (start + t.clamp(0, 1)[:, None] * edge)
    distance = (delta.square().sum(-1) + 1e-12).sqrt()
    nearest = reference.new_full((w * h,), 1e4).scatter_reduce(
        0, pixel, distance, "amin", include_self=True
    )
    return (radius + 0.5 - nearest.reshape(h, w)).clamp(0, min(1, width))
