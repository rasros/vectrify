"""Source chains become editable strokes before surrounding paint is decoded.

Unsupported runs retain filled evidence. Existing skeleton junctions are
anchors; no joining or gap completion uses a human redraw. Different supported
widths and source paints receive distinct bounded models.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import pathops
from scipy.ndimage import distance_transform_edt, find_objects, gaussian_filter, label

from vectrify.document import Geometry
from vectrify.document.join import curve_path, path_geometry
from vectrify.refine import cel
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.ink import measure
from vectrify.refine.cel_plan.local import Box

MAX_COMPONENTS = 128
MAX_SOURCE_COMPONENTS = 8192
MAX_RUNS = 128
MAX_DISCOVERY_RUNS = 4096
MAX_POINTS = 16_384
MAX_RUN_POINTS = 2048
MAX_WIDTH = 16
MAX_RUN_VARIATION = 3
MAX_PIXELS = 1536**2
MAX_MODELS = 8
MAX_MASK_BYTES = 16 * 1024**2
PAINT_SPREAD = 24


@dataclass(frozen=True)
class InkModel:
    geometry: Geometry
    footprint: Geometry
    paint: np.ndarray
    selected: np.ndarray
    details: dict


def footprint(geometry, width):
    shape = curve_path(geometry)
    shape.stroke(width, pathops.LineCap.ROUND_CAP, pathops.LineJoin.ROUND_JOIN, 4)
    shape.convertConicsToQuads(0.05)
    shape.simplify()
    return path_geometry(shape)


def connected_runs(runs, work):
    """Join only a degree-two source junction; never join distinct endpoints."""
    ends = {}
    for i, run in enumerate(runs):
        for side in (0, -1):
            ends.setdefault(tuple(run[side]), []).append((i, side))
    seen, result = set(), []
    for seed, run in enumerate(runs):
        if work.interrupted:
            return []
        if seed in seen:
            continue
        seen.add(seed)
        pieces = deque([run])
        for side in (-1, 0):
            point = run[side]
            while len(ends[tuple(point)]) == 2:
                if work.interrupted:
                    return []
                other = [r for r in ends[tuple(point)] if r[0] not in seen]
                if not other:
                    break
                index, endpoint = other[0]
                seen.add(index)
                part = runs[index]
                if endpoint == side:
                    part = part[::-1]
                if side == -1:
                    pieces.append(part[1:])
                else:
                    pieces.appendleft(part[:-1])
                point = part[side]
        result.append(np.vstack(pieces))
    return result


def models(mask, evidence, options, work, *, carrier=None, prune_spurs=False):
    """Return complete bounded models; interruption discards partial discovery."""
    if work.interrupted or mask.size > MAX_PIXELS or not mask.any():
        return ()
    components, count = label(mask, np.ones((3, 3)))
    if count > MAX_SOURCE_COMPONENTS:
        return ()
    groups = []
    claimed = np.zeros(mask.shape, np.uint8)
    rejected = np.zeros(mask.shape, bool)
    light = gaussian_filter(cel.lightness(evidence.target), 0.5)
    areas = np.bincount(components.ravel(), minlength=count + 1)
    order = sorted(range(1, count + 1), key=lambda i: (-areas[i], i))[:MAX_COMPONENTS]
    rejected |= mask & ~np.isin(components, order)
    boxes = find_objects(components)
    scanned = points_count = 0
    scale = float(np.sqrt(np.prod(evidence.scale)))
    model_limit = min(MAX_MODELS, MAX_MASK_BYTES // mask.nbytes)
    for index in order:
        if work.interrupted:
            return ()
        slices = boxes[index - 1]
        assert slices is not None
        box = Box(slices[1].start, slices[0].start, slices[1].stop, slices[0].stop)
        box = box.expand(2, mask.shape)
        own = components[box.slices] == index
        depth = np.asarray(distance_transform_edt(own))
        skeleton = cel.thin(own)
        runs = cel.line_runs(skeleton, spur=0, depth=depth)
        if not runs or len(runs) > MAX_DISCOVERY_RUNS:
            rejected[box.slices] |= own
            continue
        if prune_spurs and skeleton.any():
            typical = float(np.median(2 * depth[skeleton]))
            # The existing depth proof retains a short branch which genuinely
            # extends beyond its incident ink. Only thinning whiskers go; the
            # newly degree-two junctions can then form complete source chains.
            runs = connected_runs(
                cel.line_runs(skeleton, spur=2 * typical + 2, depth=depth), work
            )
        # Long source chains compete before small skeleton spurs. Unexamined
        # chains stay filled; the bounded scan never discards their evidence.
        runs.sort(key=lambda r: -len(r))
        for run in runs:
            if work.interrupted:
                return ()
            x = np.clip(np.floor(run[:, 0]).astype(int), 0, own.shape[1] - 1)
            y = np.clip(np.floor(run[:, 1]).astype(int), 0, own.shape[0] - 1)
            xx, yy = x + box.x, y + box.y
            thickness = 2 * depth[y, x]
            middle = thickness[2:-2] if len(thickness) > 8 else thickness
            typical = max(0.8, float(np.median(middle)))
            if (
                scanned >= MAX_RUNS
                or len(run) > MAX_RUN_POINTS
                or points_count + len(run) > MAX_POINTS
                or typical > MAX_WIDTH
                or np.percentile(middle, 90)
                > MAX_RUN_VARIATION * max(np.percentile(middle, 10), 0.5)
            ):
                rejected[yy, xx] = True
                continue
            scanned += 1
            points_count += len(run)
            points = run + np.array((box.x, box.y))
            proof = measure(points, evidence.target, typical, light=light)
            if proof is None:
                rejected[yy, xx] = True
                continue
            native = proof.points / evidence.scale + evidence.offset
            model = fitted(native, options.tolerance or 0.75)
            width = proof.width / scale
            if carrier is not None:
                # Keep boundary contacts filled while decoding internal runs
                # of the same paint. The allowance bounds a later shared width.
                upper = options.line_width or 1.6 * width
                shape = footprint(Geometry("run", (model.contour,)), upper)
                outside = pathops.op(
                    curve_path(shape), carrier, pathops.PathOp.DIFFERENCE
                )
                if abs(outside.area) > 1e-8:
                    rejected[yy, xx] = True
                    continue
            group_index = None
            for i, group in enumerate(groups):
                widths = [*group["widths"], width]
                paints = np.array([*group["paints"], proof.paint])
                if (options.line_width or max(widths) <= 1.6 * min(widths)) and (
                    np.ptp(paints, axis=0) <= PAINT_SPREAD
                ).all():
                    group_index = i
                    break
            if group_index is None:
                if len(groups) >= model_limit:
                    rejected[yy, xx] = True
                    continue
                group_index = len(groups)
                groups.append(
                    {"widths": [], "paints": [], "contours": [], "proofs": []}
                )
            group = groups[group_index]
            group["widths"].append(width)
            group["paints"].append(proof.paint)
            group["contours"].append(model.contour)
            group["proofs"].append(proof)
            collision = (claimed[yy, xx] != 0) & (claimed[yy, xx] != group_index + 1)
            rejected[yy[collision], xx[collision]] = True
            claimed[yy, xx] = group_index + 1
    if work.interrupted or not groups:
        return ()
    # Source samples belong to their nearest existing skeleton run. Ambiguous
    # junctions and unsupported chains retain their independently owned fill.
    nearest = distance_transform_edt(
        ~((claimed > 0) | rejected), return_distances=False, return_indices=True
    )
    assert nearest is not None
    ownership = np.where(rejected, 0, claimed)[tuple(nearest)]
    # A nearest run in another physical component cannot own this source mark.
    # Tiny islands can thin to a single point with no discoverable line run.
    ownership[components[tuple(nearest)] != components] = 0
    result = []
    for i, group in enumerate(groups):
        if work.interrupted:
            return ()
        selected = mask & (ownership == i + 1)
        if not selected.any():
            continue
        selected.flags.writeable = False
        width = options.line_width or float(np.median(group["widths"]))
        # Standalone compound geometry also needs distinct chain/node IDs.
        contours = tuple(
            replace(
                sub,
                id=f"ink-{i}-{j}",
                nodes=tuple(
                    replace(n, id=f"ink-{i}-{j}-{k}") for k, n in enumerate(sub.nodes)
                ),
            )
            for j, sub in enumerate(group["contours"])
        )
        geometry = Geometry(f"source-ink-{i}", contours)
        shape = footprint(geometry, width)
        result.append(
            InkModel(
                geometry,
                shape,
                np.median(group["paints"], axis=0),
                selected,
                {
                    "model": "source-stroke",
                    "runs": len(contours),
                    "width": width,
                    "support": min(p.support for p in group["proofs"]),
                    "peak_gap": max(p.peak_gap for p in group["proofs"]),
                    "source_runs_scanned": scanned,
                    "source_points_scanned": points_count,
                },
            )
        )
    return () if work.interrupted else tuple(result)


def decoded(mask, evidence, options, work, *, carrier=None):
    """A single-paint consumer may use only a single compatible source model."""
    found = models(mask, evidence, options, work, carrier=carrier)
    return found[0] if len(found) == 1 else None
