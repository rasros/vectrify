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
from vectrify.document.lines import open_path
from vectrify.refine import cel
from vectrify.refine.cel_plan.geometry import fitted
from vectrify.refine.cel_plan.ink import measure
from vectrify.refine.cel_plan.local import MAX_CROP_PIXELS, MAX_TILES, Box
from vectrify.refine.cel_plan.score import render
from vectrify.refine.crossings import crossings

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
MAX_CONTACT_PIECES = 8


@dataclass(frozen=True)
class InkModel:
    geometry: Geometry
    footprint: Geometry
    paint: np.ndarray
    selected: np.ndarray
    details: dict


def footprint(geometry, width, *, cap="round"):
    # Fill boolean paths close all contours; stroke coverage must not include
    # an invisible endpoint-to-endpoint chord in an actual open source chain.
    shape = open_path(geometry)
    shape.stroke(
        width,
        {"round": pathops.LineCap.ROUND_CAP, "butt": pathops.LineCap.BUTT_CAP}[cap],
        pathops.LineJoin.ROUND_JOIN,
        4,
    )
    shape.convertConicsToQuads(0.05)
    shape.simplify()
    return path_geometry(shape)


def covered_selection(selected, geometry, stroke_width, evidence, work, *, cap="round"):
    """Own only source pixels hit by the actual replacement stroke body.

    Nearest-run assignment determines style, not replacement coverage. A dark
    connected component may contain broad material or unmodeled ink beyond a
    supported chain. Actual stroke rasterization on the source-atom grid
    retains those pixels independently. Nonzero antialias coverage admits a
    boundary pixel without adding a distance/dilation tolerance or repairing a
    source gap. Bounded tiles preserve the full-grid sampling phase.
    """
    if work.interrupted:
        return None
    rows = np.flatnonzero(selected.any(axis=1))
    columns = np.flatnonzero(selected.any(axis=0))
    if not len(rows) or not len(columns):
        return selected.copy()
    box = Box(int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1)
    tiles = tuple(box.chunks(MAX_CROP_PIXELS))
    if len(tiles) > MAX_TILES:
        return None
    data = geometry.path_data()
    sx, sy = evidence.scale
    ox, oy = evidence.offset
    output = selected.copy()
    for tile in tiles:
        if work.interrupted:
            return None
        if not output[tile.slices].any():
            continue
        width, height = tile.right - tile.x, tile.bottom - tile.y
        # The stroke is in native coordinates. The viewport is the exact
        # source-atom pixel rectangle, including anisotropic analysis scale and
        # any source crop offset, rather than a separately normalized crop.
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="{tile.x / sx + ox} {tile.y / sy + oy} '
            f'{width / sx} {height / sy}" preserveAspectRatio="none">'
            f'<path d="{data}" fill="none" stroke="#000000" '
            f'stroke-width="{stroke_width}" stroke-linecap="{cap}" '
            'stroke-linejoin="round"/></svg>'
        )
        pixels = render(svg, (width, height))
        if work.interrupted:
            return None
        output[tile.slices] &= pixels[..., 3] > 0
    return None if work.interrupted else output


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


def owned_model(model, owned, evidence, work):
    """Retain complete discovered chains, never crop a run to owner eligibility.

    Compatible paint groups can contain independent physical chains. A held
    owner touching one chain must not suppress all the others, or turn the
    held chain into budget-dependent fragments. Raster queries use the actual
    complete path and the same source-grid coverage contract as discovery.
    """
    if work.interrupted:
        return None
    if not np.any(model.selected & ~owned):
        return model
    contours = []
    selected = np.zeros(model.selected.shape, bool)
    for sub in model.geometry.subpaths:
        if work.interrupted:
            return None
        geometry = Geometry(model.geometry.id, (sub,))
        covered = covered_selection(
            model.selected,
            geometry,
            model.details["width"],
            evidence,
            work,
            cap=model.details["linecap"],
        )
        if covered is None:
            return None
        if not covered.any() or np.any(covered & ~owned):
            continue
        contours.append(sub)
        selected |= covered
    if work.interrupted or not contours:
        return None
    geometry = replace(model.geometry, subpaths=tuple(contours))
    shape = footprint(geometry, model.details["width"], cap=model.details["linecap"])
    if work.interrupted:
        return None
    selected.flags.writeable = False
    return replace(
        model,
        geometry=geometry,
        footprint=shape,
        selected=selected,
        details={
            **model.details,
            "source_style_runs": model.details["runs"],
            "runs": len(contours),
            "owner_excluded_runs": len(model.geometry.subpaths) - len(contours),
        },
    )


def carrier_width(geometry, width, carrier, work, *, fixed=False, cap="round"):
    """A proved width ceiling for one source run, before compatible grouping.

    Every retained width has an exact footprint/carrier difference proof. A
    bounded search may recover safe headroom below the grouping factor; its
    unproved upper endpoint is never used. Cancellation discards discovery.
    """
    if carrier is None:
        return float("inf")

    def fits(value):
        if work.interrupted:
            return False
        shape = footprint(geometry, value, cap=cap)
        if work.interrupted:
            return False
        outside = pathops.op(curve_path(shape), carrier, pathops.PathOp.DIFFERENCE)
        return abs(outside.area) <= 1e-8

    if not fits(width):
        return None
    if fixed:
        return width
    low, high = width, 1.6 * width
    if fits(high):
        return high
    for _ in range(6):
        middle = (low + high) / 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return None if work.interrupted else low


def carried(
    run, proof, evidence, options, carrier, work, contacts, budget, light, visible
):
    """Fit source chains inside the carrier with proved complete footprints.

    No endpoint is extended or joined. The optional contact hypothesis tries
    an original-ended butt-cap body before shortening at existing samples and
    retaining filled contact ends. Every choice needs exact footprint proofs.
    """
    tolerance = options.tolerance or 0.75
    scale = float(np.sqrt(np.prod(evidence.scale)))

    def fit(part, supported, cap="round"):
        if work.interrupted:
            return None
        native = supported.points / evidence.scale + evidence.offset
        model = fitted(native, tolerance)
        if work.interrupted:
            return None
        geometry = Geometry("run", (model.contour,))
        if crossings(geometry):
            model = fitted(native, min(tolerance, 0.25))
            if work.interrupted:
                return None
            geometry = Geometry("run", (model.contour,))
        if crossings(geometry):
            # Profile centering can fold a noisy run even though its original
            # source skeleton is simple. Retain that anchored source chain as
            # a precise competitor; an unstable chain stays filled.
            native = part / evidence.scale + evidence.offset
            model = fitted(native, min(tolerance, 0.25))
            if work.interrupted:
                return None
            geometry = Geometry("run", (model.contour,))
        if crossings(geometry) or work.interrupted:
            return None
        width = supported.width / scale
        ceiling = carrier_width(
            geometry,
            options.line_width or width,
            carrier,
            work,
            fixed=bool(options.line_width),
            cap=cap,
        )
        return None if ceiling is None else (part, supported, model, ceiling, cap)

    complete = fit(run, proof)
    if complete is not None:
        return [complete]
    if not contacts or carrier is None or work.interrupted:
        return []
    # Some original source endpoints touch the carrier. A round cap spills
    # even when the complete open stroke body fits. Prove a butt-cap model
    # before shortening that source chain; its endpoints and junctions stay
    # exact, and its width is bounded using the same actual cap geometry.
    complete = fit(run, proof, "butt")
    if complete is not None:
        return [complete]
    if work.interrupted:
        return []
    upper = options.line_width or 1.6 * proof.width / scale
    edge = pathops.Path(carrier)
    edge.stroke(
        upper + 2 * tolerance,
        pathops.LineCap.ROUND_CAP,
        pathops.LineJoin.ROUND_JOIN,
        4,
    )
    edge.convertConicsToQuads(0.05)
    interior = pathops.op(carrier, edge, pathops.PathOp.DIFFERENCE)
    native = proof.points / evidence.scale + evidence.offset
    inside = np.array([interior.contains(tuple(p)) for p in native], bool)
    starts = np.flatnonzero(inside & ~np.r_[False, inside[:-1]])
    ends = np.flatnonzero(inside & ~np.r_[inside[1:], False]) + 1
    spans = sorted(zip(starts, ends, strict=True), key=lambda p: -(p[1] - p[0]))
    result = []
    for first, last in spans[:MAX_CONTACT_PIECES]:
        if work.interrupted:
            return []
        part = run[first:last]
        if len(part) < 4:
            continue
        if budget["runs"] >= MAX_RUNS or budget["points"] + len(part) > MAX_POINTS:
            break
        budget["runs"] += 1
        budget["points"] += len(part)
        supported = measure(
            part, evidence.target, proof.width, light=light, visible=visible
        )
        if supported is not None:
            candidate = fit(part, supported)
            if candidate is not None:
                result.append(candidate)
    return result


def models(
    mask,
    evidence,
    options,
    work,
    *,
    carrier=None,
    prune_spurs=False,
    boundary_contacts=False,
):
    """Return complete bounded models; interruption discards partial discovery."""
    if work.interrupted or mask.size > MAX_PIXELS or not mask.any():
        return ()
    components, count = label(mask, np.ones((3, 3)))
    if count > MAX_SOURCE_COMPONENTS:
        return ()
    groups = []
    claimed = np.zeros(mask.shape, np.uint8)
    rejected = np.zeros(mask.shape, bool)
    visible = ~evidence.empty
    weight = gaussian_filter(visible.astype(np.float32), 0.5)
    light = gaussian_filter(cel.lightness(evidence.target) * visible, 0.5)
    light /= np.maximum(weight, 1e-12)
    del weight
    areas = np.bincount(components.ravel(), minlength=count + 1)
    order = sorted(range(1, count + 1), key=lambda i: (-areas[i], i))[:MAX_COMPONENTS]
    rejected |= mask & ~np.isin(components, order)
    boxes = find_objects(components)
    scanned = points_count = 0
    contact_budget = {"runs": 0, "points": 0}
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
            proof = measure(
                points, evidence.target, typical, light=light, visible=visible
            )
            if proof is None:
                rejected[yy, xx] = True
                continue
            offered = carried(
                points,
                proof,
                evidence,
                options,
                carrier,
                work,
                boundary_contacts,
                contact_budget,
                light,
                visible,
            )
            accepted = set()
            for part, supported, model, ceiling, cap in offered:
                accepted.update(tuple(p) for p in part)
                px = np.clip(np.floor(part[:, 0]).astype(int), 0, mask.shape[1] - 1)
                py = np.clip(np.floor(part[:, 1]).astype(int), 0, mask.shape[0] - 1)
                width = supported.width / scale
                group_index = None
                for i, group in enumerate(groups):
                    widths = [*group["widths"], width]
                    paints = np.array([*group["paints"], supported.paint])
                    common_width = options.line_width or float(np.median(widths))
                    if (
                        cap == group["cap"]
                        and common_width <= min(ceiling, *group["ceilings"])
                        and (options.line_width or max(widths) <= 1.6 * min(widths))
                        and (np.ptp(paints, axis=0) <= PAINT_SPREAD).all()
                    ):
                        group_index = i
                        break
                if group_index is None:
                    if len(groups) >= model_limit:
                        rejected[py, px] = True
                        continue
                    group_index = len(groups)
                    groups.append(
                        {
                            "widths": [],
                            "ceilings": [],
                            "paints": [],
                            "contours": [],
                            "proofs": [],
                            "cap": cap,
                        }
                    )
                group = groups[group_index]
                group["widths"].append(width)
                group["ceilings"].append(ceiling)
                group["paints"].append(supported.paint)
                group["contours"].append(model.contour)
                group["proofs"].append(supported)
                collision = (claimed[py, px] != 0) & (
                    claimed[py, px] != group_index + 1
                )
                rejected[py[collision], px[collision]] = True
                claimed[py, px] = group_index + 1
            unoffered = np.array([tuple(p) not in accepted for p in points], bool)
            rejected[yy[unoffered], xx[unoffered]] = True
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
        claimed_pixels = int(selected.sum())
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
        shape = footprint(geometry, width, cap=group["cap"])
        if carrier is not None:
            outside = pathops.op(curve_path(shape), carrier, pathops.PathOp.DIFFERENCE)
            if abs(outside.area) > 1e-8:
                continue
        selected = covered_selection(
            selected, geometry, width, evidence, work, cap=group["cap"]
        )
        if selected is None:
            if work.interrupted:
                return ()
            continue
        if not selected.any():
            continue
        selected.flags.writeable = False
        result.append(
            InkModel(
                geometry,
                shape,
                np.median(group["paints"], axis=0),
                selected,
                {
                    "model": "source-stroke",
                    "ownership": "rendered-stroke-coverage",
                    "claimed_pixels": claimed_pixels,
                    "retained_ink_pixels": claimed_pixels - int(selected.sum()),
                    "runs": len(contours),
                    "width": width,
                    "linecap": group["cap"],
                    "carrier_width_ceiling": min(group["ceilings"])
                    if carrier is not None
                    else None,
                    "support": min(p.support for p in group["proofs"]),
                    "peak_gap": max(p.peak_gap for p in group["proofs"]),
                    "source_runs_scanned": scanned,
                    "source_points_scanned": points_count,
                    "boundary_contacts": boundary_contacts,
                    "contact_runs_scanned": contact_budget["runs"],
                    "contact_points_scanned": contact_budget["points"],
                },
            )
        )
    return () if work.interrupted else tuple(result)


def decoded(mask, evidence, options, work, *, carrier=None):
    """A single-paint consumer may use only a single compatible source model."""
    found = models(mask, evidence, options, work, carrier=carrier)
    return found[0] if len(found) == 1 else None
