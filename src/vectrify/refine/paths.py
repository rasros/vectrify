"""Fit a group's stroked paths to the target by gradient descent.

Every other operator is a pure markup transform: it edits numbers without ever
seeing the picture it is trying to match, and finds a better curve only by
proposing random nudges until one measures better. That works for placement and
colour and does not work for shape -- four consecutive runs left the beak with
the same notch, because reaching the right curve means moving several control
points together and each one alone measures worse.

This is the same idea as the differentiable-rasterizer line of work (diffvg,
Li et al. 2020, and CLIPasso, Vinker et al. 2022, which descends on encoder
features rather than pixels): render the paths in a way that has a derivative,
compare against the target, and let the gradient move the control points. The
rasterizer here covers only stroked cubics -- no fills, no clipping, no z-order
-- because that is what the drawings this search produces are made of, and the
general case is what makes diffvg a build rather than a file.

Two details are load-bearing, both learned by getting them wrong:

Pixels are sampled at their centres. Sampling at integer coordinates puts every
stroke half a pixel off its rendered position, which on a 1.75px stroke is most
of its width: measured against the real renderer, intersection-over-union rose
from 0.573 to 0.888 when the offset was added, and a fit without it spends its
budget compensating for an error that exists only in the proxy.

A group is fitted jointly, not path by path. With one path optimized at a time
and its neighbours baked into the backdrop, a misplaced neighbour's ink reads as
target already covered, and the path under optimization is pushed away from the
region its neighbour should have covered. Overlapping strokes -- a wing drawn as
three -- are exactly the case that needs this.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image

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
    width: float,
    box: tuple[int, int, int, int],
    samples: int = 24,
    softness: float = 0.25,
    chunk: int = 16384,
) -> Any:
    """Soft stroke coverage in [0, 1] over *box*, differentiable in *control*.

    A hard inside/outside test has zero gradient almost everywhere and none at
    all at the edge, so coverage falls off through a sigmoid instead: a pixel
    just outside the stroke still knows which way the stroke is.

    *box* is (left, top, right, bottom) in the drawing's own units, which keeps
    the cost proportional to the part being fitted rather than to the canvas --
    a beak is a fraction of a percent of a 700x700 page, and computing the other
    99% both wastes the time and dilutes the loss.
    """
    import torch

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
    width: float,
    target: Image.Image,
    backdrop: Image.Image,
    size: int = 700,
    steps: int = 200,
    learning_rate: float = 0.4,
    margin: float = 24.0,
    redundancy: float = 0.15,
    anchor: float = 0.001,
) -> tuple[list[str], float, float]:
    """Fit every path in *paths* together, returning new path data and losses.

    *backdrop* is the drawing rendered with this group removed; *target* is the
    picture being matched. Both are greyscale and the same size as the canvas.

    The paths composite as a soft union -- one minus the product of their
    complements -- which is what "any of these strokes covers this pixel" means
    and what the real renderer shows. A redundancy term charges for pixels more
    than one stroke covers, because the union alone is indifferent between three
    strokes doing a third of the work each and one doing all of it while the
    other two collapse onto it.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parsed = [parse_cubics(d) for d in paths]
    box = _bounds(parsed, margin, size)
    left, top, right, bottom = box

    def crop(image: Image.Image) -> Any:
        import numpy as np

        array = np.asarray(image.convert("L").resize((size, size)), dtype=np.float32)
        return torch.tensor(array[top:bottom, left:right] / 255.0, device=device)

    goal = crop(target)
    under = crop(backdrop)

    controls = [
        torch.tensor(segs, device=device, dtype=torch.float32) for segs in parsed
    ]
    original = [c.clone() for c in controls]
    for control in controls:
        control.requires_grad_(True)
    optimizer = torch.optim.Adam(controls, lr=learning_rate)

    first = last = 0.0
    for step in range(steps):
        covers = [coverage(c, width, box) for c in controls]
        stacked = torch.stack(covers)
        union = 1 - torch.prod(1 - stacked, dim=0)
        drawn = under * (1 - union)
        loss = (drawn - goal).abs().mean()
        if redundancy:
            loss = loss + redundancy * (stacked.sum(0) - 1).clamp_min(0).mean()
        if anchor:
            loss = loss + anchor * sum(
                ((c - o) ** 2).mean() for c, o in zip(controls, original, strict=True)
            )
        if step == 0:
            first = float(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last = float(loss)

    fitted = [to_path_d(c.detach().cpu().tolist()) for c in controls]
    return fitted, first, last
