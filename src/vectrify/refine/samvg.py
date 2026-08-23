"""The segmentation and tracing stages of SAMVG.

The original SAMVG implementation was not released. This module follows Zhu's
dissertation: automatic SAM masks are filtered on a blank canvas, uncovered
regions are prompted a second time, and every retained mask is traced to a
fixed-count cubic Bezier path.
"""

from __future__ import annotations

import io
import logging
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# SAMVG's quality depends directly on the granularity of its automatic masks.
# ViT-H is the paper-quality default; users who need the smaller checkpoint can
# opt down without changing the package through VECTRIFY_SAMVG_MODEL.
SAMVG_MODEL = os.environ.get("VECTRIFY_SAMVG_MODEL", "facebook/sam-vit-huge")


@dataclass(frozen=True)
class MaskLayer:
    """One painted segmentation mask, in document compositing order."""

    mask: np.ndarray
    colour: tuple[int, int, int]
    impact: float
    overlap_pixels: int = 0


def _is_crop_edge_mask(
    mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    tolerance: int = 20,
) -> bool:
    """Match AMG's rejection of masks cut off at an internal crop edge."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return True
    left, top, _right, _bottom = crop_box
    width, height = image_size
    box = np.asarray(
        (left + xs.min(), top + ys.min(), left + xs.max() + 1, top + ys.max() + 1)
    )
    crop = np.asarray(crop_box)
    image = np.asarray((0, 0, width, height))
    at_crop_edge = np.abs(box - crop) <= tolerance
    at_image_edge = np.abs(box - image) <= tolerance
    return bool(np.any(at_crop_edge & ~at_image_edge))


def automatic_masks(image: Image.Image) -> list[np.ndarray]:
    """Retrieve SAM AMG masks with the thesis's 32-point grid and crops."""
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - installation-specific
        raise ImportError(
            "SAMVG requires the vision extra. Install 'vectrify[vision]'."
        ) from exc
    image = image.convert("RGB")
    generator = pipeline("mask-generation", model=SAMVG_MODEL, device=0)
    log.info("SAMVG automatic masks: %s on %s.", SAMVG_MODEL, generator.device)

    def masks_for(source: Image.Image) -> list[np.ndarray]:
        return [
            np.asarray(mask, dtype=bool)
            for mask in generator(
                source,
                points_per_batch=32,
                points_per_crop=32,
                crops_n_layers=0,
            )["masks"]
        ]

    # transformers' built-in crop layer tries to stack unequal crop tensors.
    # Run that first crop layer one crop at a time instead.  Crucially, do not
    # pre-pad a rectangular image: the original AMG formula uses the source's
    # short side for overlap, and black padding changes SAM's visual context.
    width, height = image.size
    collected = masks_for(image)
    overlap = int((512 / 1500) * min(width, height))
    crop_width = math.ceil((overlap + width) / 2)
    crop_height = math.ceil((overlap + height) / 2)
    for x, y in {
        (0, 0),
        (crop_width - overlap, 0),
        (0, crop_height - overlap),
        (crop_width - overlap, crop_height - overlap),
    }:
        right, bottom = min(x + crop_width, width), min(y + crop_height, height)
        crop_box = (x, y, right, bottom)
        for crop_mask in masks_for(image.crop(crop_box)):
            if _is_crop_edge_mask(crop_mask, crop_box, image.size):
                continue
            mask = np.zeros((height, width), dtype=bool)
            mask[y:bottom, x:right] = crop_mask
            collected.append(mask)
    return collected


def _components(
    mask: np.ndarray, min_pixels: int, *, fill_holes: bool = False
) -> list[np.ndarray]:
    """Return traceable mask components without inventing filled regions.

    AMG already performs its configured small-region postprocessing.  Filling
    every remaining hole changes an eye, ear, or gap between hairs into a
    solid region, and keeping disconnected pieces in one SVG path turns them
    into one optimisation unit.  SAMVG traces each component, so impact
    filtering must receive those components separately.
    """
    from scipy.ndimage import binary_fill_holes, label

    labels, count = label(mask)
    components = []
    for index in range(1, count + 1):
        component = labels == index
        if int(component.sum()) < min_pixels:
            continue
        if fill_holes:
            component = binary_fill_holes(component)
        components.append(np.asarray(component, dtype=bool))
    return components


def _render_layers(
    shape: tuple[int, int], layers: list[MaskLayer]
) -> tuple[np.ndarray, np.ndarray]:
    """Render opaque flat-colour layers and return their alpha coverage."""
    height, width = shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    coverage = np.zeros((height, width), dtype=bool)
    for layer in layers:
        canvas[layer.mask] = layer.colour
        coverage |= layer.mask
    return canvas, coverage


def recolour_visible_layers(
    image: Image.Image, layers: list[MaskLayer]
) -> list[MaskLayer]:
    """Estimate every flat fill from the pixels it remains visible over.

    A layer's initial mask mean includes regions that later opaque layers hide.
    For a portrait this mixes skin into hair and foreground into background.
    Re-estimating in reverse painter order is the least-squares colour for the
    actual visible portion of each fixed mask.
    """
    target = np.asarray(image.convert("RGB"), dtype=np.uint8)
    covered_above = np.zeros(target.shape[:2], dtype=bool)
    revised: list[MaskLayer] = []
    for layer in reversed(layers):
        visible = layer.mask & ~covered_above
        colour = layer.colour
        if visible.any():
            colour = tuple(
                int(value) for value in np.rint(target[visible].mean(axis=0))
            )
        revised.append(
            MaskLayer(layer.mask, colour, layer.impact, layer.overlap_pixels)
        )
        covered_above |= layer.mask
    return list(reversed(revised))


def _impact_error(
    target: np.ndarray, canvas: np.ndarray, coverage: np.ndarray
) -> float:
    """SAMVG's blank-canvas error, charging uncovered pixels maximally."""
    error = ((target.astype(np.float32) - canvas.astype(np.float32)) / 255.0) ** 2
    error[~coverage] = 1.0
    return float(error.mean())


def filter_by_impact(
    image: Image.Image,
    masks: list[np.ndarray],
    *,
    existing: list[MaskLayer] | None = None,
    initial_canvas: np.ndarray | None = None,
    initial_coverage: np.ndarray | None = None,
    min_pixels: int = 32,
    min_impact: float = 1e-5,
    max_layers: int = 128,
    fill_holes: bool = False,
) -> list[MaskLayer]:
    """Keep masks that lower blank-canvas reconstruction error.

    Masks are sorted largest first; smaller retained masks overwrite their
    parent regions. *existing* makes a prompted second pass use the current
    composite as its starting canvas, as SAMVG does.
    """
    target = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width, _ = target.shape
    accepted = list(existing or [])
    canvas, coverage = _render_layers((height, width), accepted)
    if initial_canvas is not None:
        if initial_canvas.shape != canvas.shape:
            raise ValueError("initial canvas does not match the target size")
        canvas = initial_canvas.astype(np.uint8, copy=True)
    if initial_coverage is not None:
        if initial_coverage.shape != coverage.shape:
            raise ValueError("initial coverage does not match the target size")
        coverage = initial_coverage.astype(bool, copy=True)
    error = _impact_error(target, canvas, coverage)
    initial_count = len(accepted)
    candidates = [
        component
        for mask in masks
        if np.asarray(mask).shape == (height, width)
        for component in _components(
            np.asarray(mask, dtype=bool), min_pixels, fill_holes=fill_holes
        )
    ]
    candidates.sort(key=lambda mask: int(mask.sum()), reverse=True)
    for mask in candidates:
        if int(mask.sum()) < min_pixels:
            continue
        colour = tuple(int(v) for v in np.rint(target[mask].mean(axis=0)))
        next_canvas = canvas.copy()
        next_coverage = coverage | mask
        next_canvas[mask] = colour
        next_error = _impact_error(target, next_canvas, next_coverage)
        impact = error - next_error
        if impact < min_impact:
            continue
        accepted.append(MaskLayer(mask, colour, impact))
        canvas, coverage, error = next_canvas, next_coverage, next_error
        # Each SAMVG stage is allowed its own retained-mask budget.  Applying
        # this to the combined existing+new list silently limited recovery to
        # one path once the automatic stage had filled its budget.
        if len(accepted) - initial_count >= max_layers:
            break
    return accepted


def coverage_prompt_points(
    layers: list[MaskLayer],
    shape: tuple[int, int],
    *,
    radius_fraction: float = 0.06,
    max_points: int = 16,
) -> list[tuple[int, int]]:
    """Find mean-shift centres of large circles untouched by retained masks."""
    from scipy.ndimage import distance_transform_edt
    from sklearn.cluster import MeanShift

    _canvas, coverage = _render_layers(shape, layers)
    radius = max(2, round(min(shape) * radius_fraction))
    distance = distance_transform_edt(~coverage)
    ys, xs = np.nonzero(distance >= radius)
    if len(xs) == 0:
        return []
    stride = max(1, len(xs) // 2_048)
    points = np.column_stack((xs[::stride], ys[::stride]))
    centres = MeanShift(bandwidth=radius, bin_seeding=True).fit(points).cluster_centers_
    ranked = sorted(
        ((float(distance[round(y), round(x)]), round(x), round(y)) for x, y in centres),
        reverse=True,
    )
    return [(x, y) for _distance, x, y in ranked[:max_points]]


def prompted_masks(
    image: Image.Image, points: list[tuple[int, int]]
) -> list[np.ndarray]:
    """Prompt SAM at centres and return all three masks per point.

    Predicted IoU is a segmentation-confidence signal, not reconstruction
    impact: a broad candidate can fill an uncovered field while a smaller
    high-IoU candidate captures detail. Filter-by-impact chooses between them.
    """
    if not points:
        return []
    import torch
    from transformers import SamModel, SamProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("SAMVG prompted masks: using %s.", device)
    processor = SamProcessor.from_pretrained(SAMVG_MODEL)
    model = SamModel.from_pretrained(SAMVG_MODEL).to(device)
    try:
        input_points = [[[list(point)] for point in points]]
        inputs = processor(
            images=image, input_points=input_points, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            output = model(**inputs)
        post = processor.image_processor.post_process_masks(
            output.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]
        return [
            np.asarray(post[prompt, candidate], dtype=bool)
            for prompt in range(post.shape[0])
            for candidate in range(post.shape[1])
        ]
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def retrieve_layers(
    image: Image.Image,
    masks: list[np.ndarray] | None = None,
    *,
    min_pixels: int = 32,
    min_impact: float = 1e-5,
    max_layers: int = 512,
) -> list[MaskLayer]:
    """Run SAMVG's automatic-mask, coverage-prompt, filter sequence."""
    image = image.convert("RGB")
    initial = automatic_masks(image) if masks is None else masks
    layers = filter_by_impact(
        image,
        initial,
        min_pixels=min_pixels,
        min_impact=min_impact,
        max_layers=max_layers,
    )
    layers = recolour_visible_layers(image, layers)
    points = coverage_prompt_points(layers, (image.height, image.width))
    prompted = prompted_masks(image, points)
    recovered = filter_by_impact(
        image,
        prompted,
        existing=layers,
        min_pixels=min_pixels,
        min_impact=min_impact,
        max_layers=max_layers,
    )
    recovered = recolour_visible_layers(image, recovered)
    log.info(
        "SAMVG first pass: %d automatic mask(s), %d retained; %d coverage "
        "prompt(s), %d prompted mask(s), %d total retained.",
        len(initial),
        len(layers),
        len(points),
        len(prompted),
        len(recovered),
    )
    return recovered


def _loops(mask: np.ndarray) -> list[list[tuple[float, float]]]:
    """Trace pixel-boundary loops, retaining exterior and hole contours."""
    edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask), strict=True):
        if y == 0 or not mask[y - 1, x]:
            edges[(x, y)].append((x + 1, y))
        if x == width - 1 or not mask[y, x + 1]:
            edges[(x + 1, y)].append((x + 1, y + 1))
        if y == height - 1 or not mask[y + 1, x]:
            edges[(x + 1, y + 1)].append((x, y + 1))
        if x == 0 or not mask[y, x - 1]:
            edges[(x, y + 1)].append((x, y))
    loops: list[list[tuple[float, float]]] = []
    while edges:
        start = next(iter(edges))
        current, loop = start, [tuple(map(float, start))]
        while current in edges:
            following = edges[current].pop()
            if not edges[current]:
                del edges[current]
            current = following
            if current == start:
                break
            loop.append(tuple(map(float, current)))
        if current == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def _corners(loop: list[tuple[float, float]], count: int) -> list[int]:
    """Global curvature maxima with the local exclusion SAMVG describes."""
    points = np.asarray(loop, dtype=np.float32)
    size = len(points)
    count = min(count, size)
    step = max(1, size // 12)
    before = points - np.roll(points, step, axis=0)
    after = np.roll(points, -step, axis=0) - points
    denom = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    score = np.divide(
        (before * after).sum(axis=1), denom, out=np.ones(size), where=denom > 0
    )
    blocked = np.zeros(size, dtype=bool)
    chosen: list[int] = []
    exclusion = max(1, size // (count * 2))
    for _ in range(count):
        available = np.where(~blocked)[0]
        if len(available) == 0:
            break
        index = int(available[np.argmin(score[available])])
        chosen.append(index)
        offsets = (np.arange(index - exclusion, index + exclusion + 1) % size).astype(
            int
        )
        blocked[offsets] = True
    return sorted(chosen)


def _fit_cubic(
    points: np.ndarray, *, reparameterize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Fit fixed-endpoint cubic controls, refining the samples' parameters.

    SAMVG starts with uniformly spaced ``t`` values, then applies the
    Newton--Raphson reparameterisation in dissertation equation 3.5 before a
    final least-squares control-point fit.  Pixel contours have highly uneven
    arc-length samples around corners, so this matters even with a fixed number
    of curves.
    """
    start, end = points[0], points[-1]
    t = np.linspace(0.0, 1.0, len(points), dtype=np.float64)

    def solve(parameters: np.ndarray) -> np.ndarray:
        matrix = np.column_stack(
            (
                3 * (1 - parameters) ** 2 * parameters,
                3 * (1 - parameters) * parameters**2,
            )
        )
        base = (
            (1 - parameters)[:, None] ** 3 * start
            + parameters[:, None] ** 3 * end
        )
        controls, *_ = np.linalg.lstsq(matrix, points - base, rcond=None)
        return controls

    controls = solve(t)
    if reparameterize and len(points) > 2:
        # The endpoints must remain exactly 0 and 1.  Keeping interior values
        # ordered avoids a folded parameterisation on jagged raster contours.
        epsilon = 1e-5
        for _iteration in range(8):
            p0, p1 = controls
            omt = 1 - t
            curve = (
                omt[:, None] ** 3 * start
                + 3 * omt[:, None] ** 2 * t[:, None] * p0
                + 3 * omt[:, None] * t[:, None] ** 2 * p1
                + t[:, None] ** 3 * end
            )
            first = (
                3 * omt[:, None] ** 2 * (p0 - start)
                + 6 * omt[:, None] * t[:, None] * (p1 - p0)
                + 3 * t[:, None] ** 2 * (end - p1)
            )
            second = (
                6 * omt[:, None] * (p1 - 2 * p0 + start)
                + 6 * t[:, None] * (end - 2 * p1 + p0)
            )
            offset = curve - points
            numerator = (offset * first).sum(axis=1)
            denominator = (first * first).sum(axis=1) + (offset * second).sum(axis=1)
            updated = t.copy()
            valid = np.abs(denominator[1:-1]) > 1e-10
            # Raster corners can make an unconstrained Newton step enormous;
            # a short, damped step retains the convergence benefit without
            # collapsing several samples onto one parameter value.
            delta = np.clip(
                numerator[1:-1][valid] / denominator[1:-1][valid], -0.05, 0.05
            )
            interior = updated[1:-1]
            interior[valid] -= delta
            updated[1:-1] = interior
            updated[0], updated[-1] = 0.0, 1.0
            updated[1:-1] = np.clip(updated[1:-1], epsilon, 1 - epsilon)
            updated = np.maximum.accumulate(updated)
            updated[-1] = 1.0
            if np.max(np.abs(updated - t)) < 1e-4:
                break
            t = updated
            controls = solve(t)
    return controls[0], controls[1]


def _cubic_loop(loop: list[tuple[float, float]], segments: int) -> str | None:
    size = len(loop)
    if size < 3:
        return None
    corners = _corners(loop, segments)
    if len(corners) < 3:
        return None
    points = np.asarray(loop, dtype=np.float32)
    parts = [f"M {points[corners[0], 0]:.2f} {points[corners[0], 1]:.2f}"]
    for first, second in zip(corners, [*corners[1:], corners[0]], strict=True):
        indices = (
            np.arange(first, second + 1 if second >= first else second + size + 1)
            % size
        )
        sample = np.vstack((points[indices], points[second]))
        control_a, control_b = _fit_cubic(sample)
        end = points[second]
        parts.append(
            f"C {control_a[0]:.2f} {control_a[1]:.2f} "
            f"{control_b[0]:.2f} {control_b[1]:.2f} {end[0]:.2f} {end[1]:.2f}"
        )
    return " ".join(parts) + " Z"


def mask_path(
    mask: np.ndarray, *, segments: int = 8, overlap_pixels: int = 0
) -> str | None:
    """Fit every mask contour as a fixed-count cubic Bezier SVG path."""
    if overlap_pixels:
        from scipy.ndimage import binary_dilation

        mask = binary_dilation(mask, iterations=overlap_pixels)
    parts = [piece for loop in _loops(mask) if (piece := _cubic_loop(loop, segments))]
    return " ".join(parts) or None


def generate_svg(
    image: Image.Image,
    masks: list[np.ndarray] | None = None,
    *,
    min_pixels: int = 32,
    min_impact: float = 1e-5,
    max_layers: int = 512,
    segments: int = 8,
) -> str:
    """Generate SAMVG's traced, pre-optimisation SVG from a target image."""
    image = image.convert("RGB")
    layers = (
        filter_by_impact(
            image,
            masks,
            min_pixels=min_pixels,
            min_impact=min_impact,
            max_layers=max_layers,
        )
        if masks is not None
        else retrieve_layers(
            image,
            min_pixels=min_pixels,
            min_impact=min_impact,
            max_layers=max_layers,
        )
    )
    paths = []
    for layer in layers:
        data = mask_path(
            layer.mask, segments=segments, overlap_pixels=layer.overlap_pixels
        )
        if data:
            colour = f"#{layer.colour[0]:02x}{layer.colour[1]:02x}{layer.colour[2]:02x}"
            paths.append(f'<path d="{data}" fill="{colour}" fill-rule="evenodd" />')
    width, height = image.size
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">' + "".join(paths) + "</svg>"
    )


def residual_prompt_points(
    target: Image.Image,
    rendered: Image.Image,
    *,
    radius_fraction: float = 0.06,
    threshold: float = 0.784,
    max_points: int = 16,
) -> list[tuple[int, int]]:
    """Locate SAMVG's convolved, thresholded residual components."""
    from scipy.ndimage import label
    from scipy.signal import fftconvolve

    target_pixels = np.asarray(target.convert("RGB"), dtype=np.float32) / 255.0
    rendered_pixels = np.asarray(rendered.convert("RGB"), dtype=np.float32) / 255.0
    # SAMVG sums RGB-channel difference before applying its 0.784 threshold.
    # Averaging here hides a strongly wrong but uniformly coloured face/body.
    difference = np.abs(target_pixels - rendered_pixels).sum(axis=2)
    height, width = difference.shape
    radius = max(2, round(min(height, width) * radius_fraction))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (xx * xx + yy * yy <= radius * radius).astype(np.float32)
    # Reflected padding preserves the prior symmetric-boundary definition;
    # FFT convolution keeps the full-resolution recovery pass practical.
    padded = np.pad(difference, radius, mode="symmetric")
    smoothed = fftconvolve(padded, kernel / kernel.sum(), mode="valid")
    labels, count = label(smoothed >= threshold)
    points: list[tuple[float, int, int]] = []
    for index in range(1, count + 1):
        ys, xs = np.nonzero(labels == index)
        if len(xs):
            points.append(
                (float(smoothed[ys, xs].mean()), round(xs.mean()), round(ys.mean()))
            )
    return [(x, y) for _score, x, y in sorted(points, reverse=True)[:max_points]]


def _append_layers(svg: str, layers: list[MaskLayer], segments: int) -> str:
    """Add newly prompted paths to an already optimised SVG."""
    root = ET.fromstring(svg)
    for layer in layers:
        data = mask_path(
            layer.mask, segments=segments, overlap_pixels=layer.overlap_pixels
        )
        if not data:
            continue
        colour = f"#{layer.colour[0]:02x}{layer.colour[1]:02x}{layer.colour[2]:02x}"
        ET.SubElement(
            root,
            "{http://www.w3.org/2000/svg}path",
            {
                "d": data,
                "fill": colour,
                "fill-rule": "evenodd",
            },
        )
    return ET.tostring(root, encoding="unicode")


def _render_svg(svg: str, image: Image.Image, rasterize) -> Image.Image:
    return Image.open(
        io.BytesIO(rasterize(svg, image.width, image.height))
    ).convert("RGB")


def _mse(image: Image.Image, rendered: Image.Image) -> float:
    target = np.asarray(image.convert("RGB"), dtype=np.float32)
    candidate = np.asarray(rendered.convert("RGB"), dtype=np.float32)
    return float(((target - candidate) ** 2).mean())


def _accepted_fit(
    svg: str, image: Image.Image, *, rasterize, steps: int
) -> tuple[str, Image.Image]:
    """Keep a differentiable fit only when the actual SVG renderer improves."""
    from vectrify.refine.paths import fit_filled_svg

    before = _render_svg(svg, image, rasterize)
    fitted = fit_filled_svg(svg, image, steps=steps)
    after = _render_svg(fitted, image, rasterize)
    if _mse(image, after) <= _mse(image, before):
        return fitted, after
    log.info("SAMVG fit rejected: the exported SVG MSE increased.")
    return svg, before


def vectorize_svg(
    image: Image.Image,
    *,
    rasterize,
    steps: int = 500,
    min_pixels: int = 32,
    min_impact: float = 1e-5,
    max_layers: int = 512,
    segments: int = 8,
) -> str:
    """Run SAMVG's two 500-step optimise-and-recover phases.

    ``rasterize`` is the format backend's renderer, used solely to form the
    residual map after the first pass. The actual differentiable fit is the
    built-in filled-path optimiser so SAMVG has no external renderer dependency.
    """
    image = image.convert("RGB")
    layers = retrieve_layers(
        image,
        min_pixels=min_pixels,
        min_impact=min_impact,
        max_layers=max_layers,
    )
    initial = _append_layers(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{image.width}" '
        f'height="{image.height}" viewBox="0 0 {image.width} {image.height}"></svg>',
        layers,
        segments,
    )
    first, first_render = _accepted_fit(
        initial, image, rasterize=rasterize, steps=steps
    )
    points = residual_prompt_points(image, first_render)
    _canvas, coverage = _render_layers((image.height, image.width), layers)
    added = filter_by_impact(
        image,
        prompted_masks(image, points),
        existing=layers,
        initial_canvas=np.asarray(first_render, dtype=np.uint8),
        initial_coverage=coverage,
        min_pixels=min_pixels,
        min_impact=min_impact,
        max_layers=max_layers,
    )[len(layers) :]
    log.info(
        "SAMVG residual pass: %d prompt(s), %d accepted added path(s).",
        len(points),
        len(added),
    )
    return _accepted_fit(
        _append_layers(first, added, segments), image, rasterize=rasterize, steps=steps
    )[0]
