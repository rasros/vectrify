"""Differentiable miter stroke outlines, rasterized by the native CUDA filler.

Curve offsets are sampled in pixel space; joins use exact endpoint tangents.
Each strip and join has the same winding so the nonzero fill is their union.
The generated outlines are temporary fitting data, never written to the SVG.
"""

from __future__ import annotations

import math
from typing import Any


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _normal(v):
    import torch

    return torch.stack((-v[..., 1], v[..., 0]), dim=-1)


def endpoint_directions(controls):
    """SVG tangent fallback for repeated endpoint control points."""
    import torch

    start = controls[..., 3, :] - controls[..., 0, :]
    end = start
    for i in (2, 1):
        candidate = controls[..., i, :] - controls[..., 0, :]
        start = torch.where(
            candidate.norm(dim=-1, keepdim=True) > 1e-8, candidate, start
        )
        candidate = controls[..., 3, :] - controls[..., 3 - i, :]
        end = torch.where(candidate.norm(dim=-1, keepdim=True) > 1e-8, candidate, end)
    return _unit(start), _unit(end)


def incoming_directions(contour):
    """Include the closing seam and skip zero-length segments at a join."""
    import torch

    _, end = endpoint_directions(contour)
    valid = torch.nonzero(end.detach().norm(dim=-1) > 1e-8).flatten()
    if not len(valid):
        return end
    positions = torch.arange(len(contour), device=contour.device)
    previous = torch.searchsorted(valid, positions) - 1
    return end[valid[previous]]


def _edges(vertices):
    import torch

    following = vertices.roll(-1, dims=-2)
    return torch.stack(
        (
            vertices,
            (2 * vertices + following) / 3,
            (vertices + 2 * following) / 3,
            following,
        ),
        dim=-2,
    )


def miter_stroke_coverage(
    controls: Any,
    incoming: Any,
    widths: Any,
    box: tuple[int, int, int, int],
    *,
    miter_limit: float = 4,
    samples: int | None = None,
    subpixels: int = 2,
) -> Any | None:
    """Stroke batches of cubic chunks, with the incoming tangent of each cubic.

    Chunks need not be complete contours. Supplying tangents from the full
    contour preserves joins at chunk boundaries and at the closing seam.
    """
    import torch

    from vectrify.refine.cuda_renderer import multi_coverage

    if samples is None:
        # Cubic chord error <= max|B''| / (8*n*n). Keep it below .05 px,
        # with extra samples for curved offsets. Straight edges need one strip.
        curvature = float(controls.detach().diff(n=2, dim=-2).norm(dim=-1).max())
        samples = max(1, math.ceil(math.sqrt(15 * curvature)))
        if samples > 1:
            samples = max(8, samples)
    batch = controls.shape[0]
    radius = widths[:, None, None] / 2
    t = torch.linspace(0, 1, samples + 1, device=controls.device, dtype=controls.dtype)
    u = 1 - t
    basis = torch.stack((u**3, 3 * u * u * t, 3 * u * t * t, t**3), dim=-1)
    derivative = torch.stack(
        (-3 * u * u, 3 * u * u - 6 * u * t, 6 * u * t - 3 * t * t, 3 * t * t), dim=-1
    )
    points = torch.einsum("sk,bnkd->bnsd", basis, controls)
    tangent = torch.einsum("sk,bnkd->bnsd", derivative, controls)
    start, end = endpoint_directions(controls)
    tangent = torch.cat(
        (start[..., None, :], tangent[..., 1:-1, :], end[..., None, :]), dim=-2
    )
    normals = _normal(_unit(tangent)) * radius[..., None]
    strips = torch.cat((points + normals, (points - normals).flip(-2)), dim=-2)
    strip_edges = _edges(strips).reshape(batch, -1, 4, 2)

    # The outer offset rays meet at the miter. Over-limit corners become
    # bevels, not truncated spikes or round joins (SVG stroke-miterlimit).
    turn = incoming[..., 0] * start[..., 1] - incoming[..., 1] * start[..., 0]
    side = -turn.sign()[..., None]
    n1, n2 = _normal(incoming), _normal(start)
    denominator = (1 + (incoming * start).sum(-1, keepdim=True)).clamp_min(1e-12)
    offset = (n1 + n2) / denominator
    allowed = (offset.norm(dim=-1, keepdim=True) <= miter_limit) & (denominator > 1e-7)
    offset = torch.where(allowed, offset, n1)
    vertex = controls[..., 0, :]
    joins = torch.stack(
        (
            vertex,
            vertex + n1 * side * radius,
            vertex + offset * side * radius,
            vertex + n2 * side * radius,
        ),
        dim=-2,
    )
    # Strips are clockwise. Reverse counterclockwise outer wedges.
    joins = torch.where((turn > 0)[..., None, None], joins.flip(-2), joins)
    edges = torch.cat((strip_edges, _edges(joins).reshape(batch, -1, 4, 2)), dim=1)
    padding = (-edges.shape[1]) % 16
    if padding:
        edges = torch.cat((edges, edges[:, -1:, 3:4].expand(-1, padding, 4, -1)), dim=1)
    packets = edges.shape[1] // 16
    return multi_coverage(
        edges.reshape(-1, 16, 4, 2),
        [i * packets for i in range(batch + 1)],
        box,
        subpixels=subpixels,
        fill_rule="nonzero",
    )
