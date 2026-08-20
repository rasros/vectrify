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

import copy
import io
import itertools
import logging
import random
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
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


def to_knots(segments) -> list[tuple[float, float]]:
    """Segments as one chain of 3n+1 points, sharing each join.

    Stored per segment, a join is two numbers that happen to be equal, and an
    optimizer moves them apart: measured on a real beak, joins drifted a mean
    of 2.08px and up to 4.00px on a stroke 3.5px wide. `to_path_d` then writes
    `M` once and a `C` per segment, so each curve silently starts wherever the
    last one ended and the drift is discarded -- the fit optimizes one curve and
    emits a different one.

    Shared here instead, so a join cannot come apart and a gradient arriving
    from either side moves both curves together, which is what continuity is.
    """
    for before, after in itertools.pairwise(segments):
        if before[3] != after[0]:
            raise UnsupportedPathError("path is not one connected chain")
    points = [segments[0][0]]
    for segment in segments:
        points.extend(segment[1:])
    return points


def weld(chains, tolerance: float = 0.01):
    """One point per distinct location, and an index per chain into it.

    Coincident points are how a drawing states that two curves meet, and a fit
    that gives each its own parameter lets the meeting come apart. Two cases
    appear in real output and neither is inside a single chain, so sharing
    knots along a chain does not reach them: the beak tip, where the upper and
    lower bill are separate paths ending at the same coordinate, and a closed
    path, whose last point is its first. Measured on one beak, fitting split the
    tip by 2.42px and opened the closure by 1.06px, on a stroke 3.5px wide.

    Welding them makes the junction a single parameter, so a gradient from
    either curve moves both and the meeting is preserved by construction rather
    than by a penalty that has to be tuned.
    """
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
    width: float,
    box: tuple[int, int, int, int],
    samples: int = 8,
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


def _focus_mask(covers: list[Any], reach: int) -> Any:
    """Where in the crop this group's strokes can actually be judged.

    A bounding box is the wrong unit once a path is long: the body outline is
    one stroke across half the canvas, so its box is most of the page and its
    own ink is a fraction of a percent of it. Averaging over that box buries
    the signal exactly as averaging over the whole canvas did -- measured, the
    body's loss moved 9.3% where a compact group's moved 44.6%.

    So the loss is averaged over a band around the strokes instead: dilate
    their starting coverage by roughly how far a control point should travel,
    and judge inside that. Fixed at the start rather than recomputed as they
    move, because a mask that follows the strokes would let them escape their
    own errors by walking away from them.
    """
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
    size: int = 700,
    steps: int = 200,
    samples: int = 8,
    learning_rate: float = 0.4,
    margin: float = 24.0,
    redundancy: float = 0.15,
    smooth: float = 0.0,
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
    chains = [to_knots(segs) for segs in parsed]
    left, top, right, bottom = box

    def crop(image: Image.Image) -> Any:
        import numpy as np

        array = np.asarray(image.convert("L").resize((size, size)), dtype=np.float32)
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
    welded, index = weld(chains)
    vertices = torch.tensor(welded, device=device, dtype=torch.float32)
    vertices.requires_grad_(True)
    original = vertices.detach().clone()
    rows = [torch.tensor(r, device=device, dtype=torch.long) for r in index]
    optimizer = torch.optim.Adam([vertices], lr=learning_rate)

    def chain_of(row: Any) -> Any:
        return vertices[row]

    def controls_of(chain: Any) -> Any:
        return chain.unfold(0, 4, 3).permute(0, 2, 1)

    with torch.no_grad():
        mask = _focus_mask(
            [
                coverage(controls_of(chain_of(r)), w, box, samples=samples)
                for r, w in zip(rows, each, strict=True)
            ],
            int(margin),
        )
    weight = mask / mask.sum().clamp_min(1.0)

    first = last = 0.0
    for step in range(steps):
        covers = [
            coverage(controls_of(chain_of(r)), w, box, samples=samples)
            for r, w in zip(rows, each, strict=True)
        ]
        stacked = torch.stack(covers)
        union = 1 - torch.prod(1 - stacked, dim=0)
        drawn = under * (1 - union)
        loss = ((drawn - goal).abs() * weight).sum()
        if redundancy:
            loss = (
                loss + redundancy * ((stacked.sum(0) - 1).clamp_min(0) * weight).sum()
            )
        if smooth:
            # Two curves meeting at a knot continue smoothly when their handles
            # are collinear with it: P2 + P1' = 2*P3. The distance from that is
            # the corner, and nothing else in the loss objects to one -- ink
            # lands in much the same place whether a join turns or flows, so a
            # fit leaves the kink it started with (measured: a mean bend of 52
            # degrees before, 50 after, with the sharpest corner getting worse).
            #
            # A penalty rather than a constraint, because the target has real
            # corners too -- a beak tip is one -- and a chain forced smooth
            # everywhere cannot draw them.
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
        last = float(loss.detach())

    fitted = [knots_to_path_d(chain_of(r).detach().cpu().tolist()) for r in rows]
    return fitted, first, last


PATH_FIT = "Mutation: path fit"


def _stroke_width(element, ancestors) -> float | None:
    """The stroke width in force for *element*, walking SVG inheritance.

    A width can be declared on the element, on any group above it, or on the
    root, and real output puts it in all three places -- one model wrote it on
    every `<path>` and none on their groups, which an element-and-root lookup
    misses entirely. That silently made every group unfittable, so the operator
    never once fired in a full run.
    """
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
    steps: int = 25,
    samples: int = 8,
    weights: Mapping[int, float] | None = None,
) -> str:
    """Fit one group's paths to the reference, returning the edited document.

    Raises UnsupportedPathError when the drawing offers nothing this can fit,
    which the caller reports as an operator that found nothing to change.

    *weights* is the error attribution the other operators already use, keyed by
    element index in document order; a group is drawn in proportion to the error
    its own elements answer for, so the fit is spent where the drawing is wrong
    rather than on whichever group came first.
    """
    import xml.etree.ElementTree as ET

    from PIL import Image

    root = ET.fromstring(svg)
    candidates = fittable_groups(root)
    if not candidates:
        raise UnsupportedPathError("no group of stroked cubics to fit")

    order = {id(el): index for index, el in enumerate(drawable_elements(root))}
    scores = []
    for _group, paths, _widths in candidates:
        error = sum((weights or {}).get(order.get(id(p), -1), 0.0) for p in paths)
        scores.append(error)
    total = sum(scores)
    if total > 0:
        group, paths, widths = random.choices(candidates, weights=scores, k=1)[0]
    else:
        group, paths, widths = random.choice(candidates)

    size = int(_canvas_side(root))
    target = Image.open(io.BytesIO(reference_png)).convert("L").resize((size, size))

    # The backdrop is the drawing without this group, so the fit sees the rest
    # of the picture as a constant and cannot be rewarded for redrawing it.
    stripped = copy.deepcopy(root)
    for parent in stripped.iter():
        for child in list(parent):
            if child.get("id") == group.get("id") and child.tag == group.tag:
                parent.remove(child)
    backdrop = Image.open(
        io.BytesIO(rasterize(ET.tostring(stripped, encoding="unicode"), size, size))
    ).convert("L")

    fitted, _first, _last = fit_group(
        [p.get("d", "") for p in paths],
        widths,
        target,
        backdrop,
        size=size,
        steps=steps,
        samples=samples,
    )
    for path, data in zip(paths, fitted, strict=True):
        path.set("d", data)
    return ET.tostring(root, encoding="unicode")


def _canvas_side(root) -> float:
    box = root.get("viewBox", "").replace(",", " ").split()
    if len(box) == 4:
        try:
            return max(abs(float(box[2])), abs(float(box[3]))) or 700.0
        except ValueError:
            pass
    return 700.0


def fit_available() -> bool:
    """Whether fitting is cheap enough to hand a worker.

    Gated on CUDA rather than on torch alone. Measured on one beak group at 25
    steps: 0.5s on a GPU, 9.3s on one CPU thread, against about 1ms for an
    ordinary mutation. At GPU speed the fit is a rare expensive operator the
    policy can weigh against the others; at CPU speed it holds a worker for
    seconds while its siblings complete thousands of tasks, which is not a
    trade worth offering.
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - driver trouble is not our business
        return False
