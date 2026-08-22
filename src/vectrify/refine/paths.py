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
from collections.abc import Mapping
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
    width: float,
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
    size: int = 700,
    steps: int = 200,
    samples: int | None = None,
    pinned: set[int] | None = None,
    learning_rate: float = 0.4,
    margin: float = 24.0,
    redundancy: float = 0.15,
    smooth: float = 0.0,
    anchor: float = 0.001,
) -> tuple[list[str], float, float]:
    """Fit every path in *paths* together, returning new path data and losses.

    *backdrop* is the drawing rendered with this group removed; *target* is the
    picture being matched. Both are greyscale and the same size as the canvas.

    *pinned* names welded vertices that must not move: a point this set shares
    with a path outside it. Without them a partial fit tears the drawing at
    exactly the junctions welding exists to hold -- the fitted side walks away
    while the neighbour it meets stays put.

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
        if pinned:
            with torch.no_grad():
                held = torch.tensor(sorted(pinned), device=device, dtype=torch.long)
                vertices[held] = original[held]
        last = float(loss.detach())

    fitted = [knots_to_path_d(chain_of(r).detach().cpu().tolist()) for r in rows]
    return fitted, first, last


PATH_FIT = "Mutation: path fit"


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
    target = Image.open(io.BytesIO(reference_png)).convert("L").resize((size, size))

    # The backdrop is the drawing without these paths, so the fit sees the rest
    # of the picture as a constant and cannot be rewarded for redrawing it.
    # Blanking `d` in place and restoring it afterwards avoids having to match
    # elements across a copied tree.
    original = [path.get("d", "") for path in paths]
    for path in paths:
        path.set("d", "")
    backdrop = Image.open(
        io.BytesIO(rasterize(ET.tostring(root, encoding="unicode"), size, size))
    ).convert("L")
    for path, data in zip(paths, original, strict=True):
        path.set("d", data)

    held = _shared_vertices([parse_cubics(d) for d in original], excluded)
    fitted, _first, _last = fit_group(
        original,
        widths,
        target,
        backdrop,
        size=size,
        steps=steps,
        samples=samples,
        pinned=held,
    )
    for path, data in zip(paths, fitted, strict=True):
        path.set("d", data)
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
