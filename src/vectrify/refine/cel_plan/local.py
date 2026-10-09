"""Native crop updates to the fixed policy, with immutable shared raster history.

Blur and ink distances have finite support. A proposal supplies complete old
and new paint bounds; rendering the surrounding SVG preserves visibility and
opacity groups. Changed contributions retain the full reference denominators.
Complete checkpoints independently verify this incremental score.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, replace
from xml.etree import ElementTree as ET

import cairosvg
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from vectrify.refine.cel_plan.model import StageInterruptedError, Work
from vectrify.refine.cel_plan.policy import Evaluation, Feature, Policy, _ink, _robust
from vectrify.refine.cel_plan.score import composite, premultiplied, render

MAX_CROP_PIXELS = 256 * 1024
MAX_RASTER_BYTES = 16 * 1024 * 1024
MAX_TILES = 32
HALO = 16  # Largest Gaussian radius (sigma 4, truncate 4); ink needs 8 + 4.
SCORE_TOLERANCE = 2e-7


class LocalLimitError(ValueError):
    """Work exceeds this stage's independent native-buffer allowance."""


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    right: int
    bottom: int

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y, self.bottom), slice(self.x, self.right)

    @property
    def area(self) -> int:
        return max(0, self.right - self.x) * max(0, self.bottom - self.y)

    def expand(self, amount: int, shape: tuple[int, ...]) -> Box:
        return Box(
            max(0, self.x - amount),
            max(0, self.y - amount),
            min(shape[1], self.right + amount),
            min(shape[0], self.bottom + amount),
        )

    def intersection(self, other: Box) -> Box:
        return Box(
            max(self.x, other.x),
            max(self.y, other.y),
            min(self.right, other.right),
            min(self.bottom, other.bottom),
        )

    def within(self, outer: Box) -> tuple[slice, slice]:
        return (
            slice(self.y - outer.y, self.bottom - outer.y),
            slice(self.x - outer.x, self.right - outer.x),
        )

    def chunks(self, pixels: int):
        """Nonoverlapping row-major chunks with a bounded pixel count."""
        if not self.area:
            return
        width = min(self.right - self.x, max(1, pixels))
        height = max(1, pixels // width)
        for y in range(self.y, self.bottom, height):
            for x in range(self.x, self.right, width):
                yield Box(
                    x, y, min(x + width, self.right), min(y + height, self.bottom)
                )


def tile_boxes(changed: Box, shape: tuple[int, ...]) -> tuple[Box, ...]:
    """Disjoint output ownership with bounded input halos for every tile."""
    changed = changed.expand(0, shape)
    if not changed.area:
        raise ValueError("A proposal needs visible native bounds")
    output = changed.expand(HALO, shape)
    if output.expand(HALO, shape).area <= MAX_CROP_PIXELS:
        return (output,)
    side = math.isqrt(MAX_CROP_PIXELS) - 2 * HALO
    if side <= 0:
        raise LocalLimitError("Native proposal crop limit cannot hold filter halos")
    boxes = []
    for x in range(output.x, output.right, side):
        right = min(x + side, output.right)
        width = min(shape[1], right - x + 2 * HALO)
        height = MAX_CROP_PIXELS // width - 2 * HALO
        if height <= 0:
            raise LocalLimitError("Native proposal crop limit cannot hold filter halos")
        for y in range(output.y, output.bottom, height):
            boxes.append(Box(x, y, right, min(y + height, output.bottom)))
            if len(boxes) > MAX_TILES:
                raise LocalLimitError("Native proposal tile limit")
    return tuple(boxes)


@dataclass(frozen=True)
class Patch:
    box: Box
    pixels: np.ndarray  # Read-only uint8 RGBA, avoiding a float canvas per state.


@dataclass(frozen=True)
class Canvas:
    root: np.ndarray
    patches: tuple[Patch, ...] = ()

    def values(self, box: Box) -> np.ndarray:
        """Owned uint8 pixels, including all immutable patches."""
        result = self.root[box.slices].copy()
        for patch in self.patches:
            overlap = box.intersection(patch.box)
            if overlap.area:
                result[overlap.within(box)] = patch.pixels[overlap.within(patch.box)]
        return result

    def crop(self, box: Box) -> np.ndarray:
        return self.values(box).astype(np.float32) / 255

    def changed(self, box: Box, pixels: np.ndarray) -> Canvas:
        values = np.rint(pixels * 255).astype(np.uint8)
        values.flags.writeable = False
        return Canvas(self.root, (*self.patches, Patch(box, values)))

    def samples(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bounded proposal-estimate samples from the same immutable history."""
        result = self.root[y, x].copy()
        for patch in self.patches:
            inside = (
                (x >= patch.box.x)
                & (x < patch.box.right)
                & (y >= patch.box.y)
                & (y < patch.box.bottom)
            )
            result[inside] = patch.pixels[
                y[inside] - patch.box.y, x[inside] - patch.box.x
            ]
        return result.astype(np.float32) / 255

    def matches(self, actual: np.ndarray) -> bool:
        """Verify complete pixels without materializing another float canvas."""
        if actual.shape != self.root.shape:
            return False
        for y in range(0, self.root.shape[0], 128):
            box = Box(0, y, self.root.shape[1], min(y + 128, self.root.shape[0]))
            if not np.array_equal(self.crop(box), actual[box.slices]):
                return False
        return True


@dataclass(frozen=True)
class Snapshot:
    canvas: Canvas
    evaluation: Evaluation
    features: tuple[float, ...]
    holes: tuple[float, ...]
    retained: tuple[float, ...]
    opacity: tuple[float, ...]
    observed: int
    missing: float
    unsupported: float


def _render_tree(svg: str, visible_ids: frozenset[str] | None):
    inner = ET.fromstring(svg)
    if visible_ids is not None:

        def cull(node, definition=False):
            definition |= node.tag.rsplit("}", 1)[-1] in {
                "defs",
                "clipPath",
                "mask",
                "pattern",
            }
            for child in list(node):
                if (
                    not definition
                    and child.tag.rsplit("}", 1)[-1] == "path"
                    and child.get("id")
                    and child.get("id") not in visible_ids
                ):
                    node.remove(child)
                else:
                    cull(child, definition)

        cull(inner)
    return inner


def _native_raster(inner, size: tuple[int, int]) -> Canvas:
    if size[0] * size[1] * 4 > MAX_RASTER_BYTES:
        raise LocalLimitError("Native local-search raster limit")
    png = cairosvg.svg2png(
        bytestring=ET.tostring(inner), output_width=size[0], output_height=size[1]
    )
    if png is None:
        raise ValueError("The native renderer returned no pixels")
    with Image.open(io.BytesIO(png)) as image:
        values = np.asarray(image.convert("RGBA"))
        values.flags.writeable = False
        return Canvas(values)


def _render_crop(
    svg: str,
    box: Box,
    size: tuple[int, int],
    visible_ids: frozenset[str] | None = None,
) -> tuple[np.ndarray, bool]:
    inner = _render_tree(svg, visible_ids)
    native = any(
        node.tag.rsplit("}", 1)[-1] in {"linearGradient", "pattern", "mask"}
        or (len(node) and float(node.get("opacity", "1")) < 1)
        for node in inner.iter()
    )
    if native:
        # Changing an offscreen group or gradient's viewport can change Cairo's
        # intermediate rounding. Preserve the original native rasterization;
        # crop before converting the bounded buffer to floating-point pixels.
        return _native_raster(inner, size).crop(box), True
    root = ET.Element(
        "{http://www.w3.org/2000/svg}svg",
        {
            "width": str(box.right - box.x),
            "height": str(box.bottom - box.y),
            "viewBox": f"{box.x} {box.y} {box.right - box.x} {box.bottom - box.y}",
        },
    )
    # Match render(svg, size), including percentage-based original paint.
    inner.set("width", str(size[0]))
    inner.set("height", str(size[1]))
    root.append(inner)
    png = cairosvg.svg2png(bytestring=ET.tostring(root))
    if png is None:
        raise ValueError("The crop renderer returned no pixels")
    with Image.open(io.BytesIO(png)) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.float32) / 255, False


def render_crop(
    svg: str,
    box: Box,
    size: tuple[int, int],
    *,
    visible_ids: frozenset[str] | None = None,
) -> np.ndarray:
    """Native pixels in a bounded crop, retaining all paint and layer context."""
    return _render_crop(svg, box, size, visible_ids)[0]


def _distance(observed: np.ndarray) -> np.ndarray:
    # scipy's all-True EDT has an implicit zero outside the array. That zero
    # must not invent nearby ink when the actual crop contains none.
    return (
        np.minimum(np.asarray(distance_transform_edt(~observed)) / 4, 1)
        if observed.any()
        else np.ones(observed.shape, dtype=np.float64)
    )


def _support_delta(
    supports: tuple[Feature, ...],
    box: Box,
    before: np.ndarray,
    after: np.ndarray,
    *,
    mean: bool = False,
) -> tuple[float, ...]:
    result = []
    for feature in supports:
        x, y, width, height = feature.box
        feature_box = Box(x, y, x + width, y + height)
        overlap = box.intersection(feature_box)
        change = 0.0
        if overlap.area:
            own = feature.support[overlap.within(feature_box)]
            difference = after[overlap.within(box)] - before[overlap.within(box)]
            change = float(difference[own].sum(dtype=np.float64))
            if mean:
                change /= max(1, int(feature.support.sum()))
        result.append(change)
    return tuple(result)


def _add(before: tuple[float, ...], delta: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(a + b for a, b in zip(before, delta, strict=True))


class LocalPolicy:
    """Share evidence; each beam state owns scalar statistics and raster patches."""

    def __init__(self, policy: Policy):
        self.policy = policy
        self.native_context_renders = 0
        self.tiles_scored = 0
        self.observed_support = binary_dilation(policy.mask, iterations=2)
        self.ink_count = int(policy.ink.sum())
        self.mask_count = max(1, int(policy.mask.sum()))
        self.component_alpha = []
        self.component_retention = []
        for feature, fraction in zip(
            policy.opacity_components, policy.component_retention, strict=True
        ):
            x, y, width, height = feature.box
            box = np.s_[y : y + height, x : x + width]
            self.component_alpha.append(
                float(policy.truth[box][..., 3][feature.support].sum())
            )
            self.component_retention.append(fraction)

    def start(self, svg: str, evaluation: Evaluation) -> Snapshot:
        shape = self.policy.truth.shape
        if shape[0] * shape[1] * 4 > MAX_RASTER_BYTES:
            raise LocalLimitError("Native local-search raster limit")
        actual = render(svg, (shape[1], shape[0]))
        root = np.rint(actual * 255).astype(np.uint8)
        root.flags.writeable = False
        observed = _ink(actual) & self.observed_support
        missing = float(_distance(observed)[self.policy.ink].sum())
        unsupported = float(np.minimum(self.policy.ink_distance[observed] / 4, 1).sum())
        predicted = premultiplied(actual)

        def samples(supports, values, *, mean=False):
            result = []
            for feature in supports:
                x, y, width, height = feature.box
                selected = values[y : y + height, x : x + width][feature.support]
                result.append(
                    float(selected.mean() if mean else selected.sum(dtype=np.float64))
                )
            return tuple(result)

        alpha = actual[..., 3]
        return Snapshot(
            Canvas(root),
            evaluation,
            samples(
                self.policy.features,
                _robust(predicted - self.policy.targets[0]).mean(axis=-1),
                mean=True,
            ),
            samples(self.policy.holes, alpha, mean=True),
            samples(
                self.policy.opacity_components,
                np.minimum(alpha, self.policy.truth[..., 3]),
            ),
            samples(self.policy.opacity_components, alpha),
            int(observed.sum()),
            missing,
            unsupported,
        )

    def update(
        self,
        before: Snapshot,
        svg: str,
        changed: Box,
        structure: dict,
        *,
        visible_ids: frozenset[str] | None = None,
        work: Work | None = None,
    ) -> Snapshot:
        shape = self.policy.truth.shape
        changed = changed.expand(0, shape)
        outputs = tile_boxes(changed, shape)

        def check():
            if work is not None and work.interrupted:
                raise StageInterruptedError("Native tile evaluation interrupted")

        check()
        paint = (
            _native_raster(_render_tree(svg, visible_ids), (shape[1], shape[0]))
            if len(outputs) > 1
            else None
        )
        self.native_context_renders += int(paint is not None)
        check()
        stats = before
        patches = []
        for output in outputs:
            check()
            input_box = output.expand(HALO, shape)
            old = before.canvas.crop(input_box)
            if paint is not None:
                new = paint.crop(input_box)
            else:
                new, native = _render_crop(
                    svg, input_box, (shape[1], shape[0]), visible_ids
                )
                self.native_context_renders += int(native)
            check()
            outside = np.ones(old.shape[:2], dtype=bool)
            overlap = changed.intersection(input_box)
            if overlap.area:
                outside[overlap.within(input_box)] = False
            if not np.array_equal(old[outside], new[outside]):
                raise ValueError(
                    "Proposal changed paint outside its declared native bounds"
                )
            own = changed.intersection(output)
            if not own.area:
                own = Box(output.x, output.y, output.x, output.y)
            stats = self._tile(stats, own, structure, output, input_box, old, new)
            self.tiles_scored += 1
            if own.area:
                values = np.rint(new[own.within(input_box)] * 255).astype(np.uint8)
                values.flags.writeable = False
                patches.append(Patch(own, values))
            check()
        return replace(
            stats,
            canvas=Canvas(before.canvas.root, (*before.canvas.patches, *patches)),
        )

    def _tile(
        self,
        before: Snapshot,
        changed: Box,
        structure: dict,
        output: Box,
        input_box: Box,
        old: np.ndarray,
        new: np.ndarray,
    ) -> Snapshot:
        policy = self.policy
        selection = output.within(input_box)
        old_predicted, new_predicted = premultiplied(old), premultiplied(new)
        color = before.evaluation.terms["color"]
        native = 0.35 * (1 - 0.8 * policy.texture[output.slices])
        for sigma, weight, target in zip(
            (0, 2, 4),
            (native, 0.8 - native, 0.2),
            policy.targets,
            strict=True,
        ):
            old_blur = (
                gaussian_filter(old_predicted, (sigma, sigma, 0))
                if sigma
                else old_predicted
            )
            new_blur = (
                gaussian_filter(new_predicted, (sigma, sigma, 0))
                if sigma
                else new_predicted
            )
            expected = target[output.slices][..., :3]
            delta = _robust(new_blur[selection][..., :3] - expected).mean(
                axis=-1
            ) - _robust(old_blur[selection][..., :3] - expected).mean(axis=-1)
            color += (
                float(
                    (delta * weight)[policy.mask[output.slices]].sum(dtype=np.float64)
                )
                / self.mask_count
            )

        local = changed.within(input_box)
        old_local, new_local = old[local], new[local]
        truth = policy.truth[changed.slices]
        alpha = (
            before.evaluation.terms["alpha"]
            + float(
                (
                    _robust(new_local[..., 3] - truth[..., 3])
                    - _robust(old_local[..., 3] - truth[..., 3])
                ).sum(dtype=np.float64)
            )
            / policy.area
        )
        for background in (0.0, 1.0):
            expected = composite(truth, background)
            delta = _robust(composite(new_local, background) - expected).mean(
                axis=-1
            ) - _robust(composite(old_local, background) - expected).mean(axis=-1)
            alpha += float(delta.sum(dtype=np.float64)) / policy.area * 0.125

        old_ink = _ink(old) & self.observed_support[input_box.slices]
        new_ink = _ink(new) & self.observed_support[input_box.slices]
        own = policy.ink[output.slices]
        missing = before.missing + float(
            (_distance(new_ink)[selection] - _distance(old_ink)[selection])[own].sum()
        )
        distances = np.minimum(policy.ink_distance[output.slices] / 4, 1)
        unsupported = before.unsupported + float(
            distances[new_ink[selection]].sum() - distances[old_ink[selection]].sum()
        )
        observed = (
            before.observed
            + int(new_ink[selection].sum())
            - int(old_ink[selection].sum())
        )
        edges = (
            (missing + unsupported) / (2 * self.ink_count)
            if self.ink_count and observed
            else float(bool(self.ink_count or observed))
        )
        features = _add(
            before.features,
            _support_delta(
                policy.features,
                changed,
                _robust(old_predicted[local] - policy.targets[0][changed.slices]).mean(
                    axis=-1
                ),
                _robust(new_predicted[local] - policy.targets[0][changed.slices]).mean(
                    axis=-1
                ),
                mean=True,
            ),
        )
        feature_error = (
            (float(np.mean(features)) + max(features)) / 2 if features else 0.0
        )
        terms = {
            **before.evaluation.terms,
            "color": color,
            "alpha": alpha,
            "edges": edges,
            "features": feature_error,
        }
        for support, threshold, term in (
            (
                policy.inside[changed.slices],
                old_local[..., 3] < 0.5,
                "interior_missing_pixels",
            ),
            (
                policy.outside[changed.slices],
                old_local[..., 3] > 0.5 / 255,
                "outside_spill_pixels",
            ),
            (
                policy.opacity_inside[changed.slices],
                old_local[..., 3] - truth[..., 3]
                < -policy.opacity_tolerance[changed.slices],
                "opacity_missing_pixels",
            ),
            (
                policy.opacity_inside[changed.slices],
                old_local[..., 3] - truth[..., 3]
                > policy.opacity_tolerance[changed.slices],
                "opacity_excess_pixels",
            ),
        ):
            if term == "interior_missing_pixels":
                new_threshold = new_local[..., 3] < 0.5
            elif term == "outside_spill_pixels":
                new_threshold = new_local[..., 3] > 0.5 / 255
            else:
                difference = new_local[..., 3] - truth[..., 3]
                tolerance = policy.opacity_tolerance[changed.slices]
                new_threshold = (
                    difference < -tolerance
                    if term == "opacity_missing_pixels"
                    else difference > tolerance
                )
            terms[term] += int((support & new_threshold).sum()) - int(
                (support & threshold).sum()
            )
        terms["visual"] = (
            policy.weights.color * color
            + policy.weights.alpha * alpha
            + policy.weights.edges * edges
            + policy.weights.features * feature_error
        )
        holes = _add(
            before.holes,
            _support_delta(
                policy.holes, changed, old_local[..., 3], new_local[..., 3], mean=True
            ),
        )
        retained = _add(
            before.retained,
            _support_delta(
                policy.opacity_components,
                changed,
                np.minimum(old_local[..., 3], truth[..., 3]),
                np.minimum(new_local[..., 3], truth[..., 3]),
            ),
        )
        opacity = _add(
            before.opacity,
            _support_delta(
                policy.opacity_components, changed, old_local[..., 3], new_local[..., 3]
            ),
        )
        rejected = self._rejections(terms, holes, retained, opacity)
        return Snapshot(
            before.canvas,
            Evaluation(terms, structure, rejected),
            features,
            holes,
            retained,
            opacity,
            observed,
            missing,
            unsupported,
        )

    def _rejections(self, terms, holes, retained, opacity) -> tuple[str, ...]:
        policy = self.policy
        allowance = max(4, round(policy.area * 0.0005))
        rejected = []
        for term, reason in (
            ("interior_missing_pixels", "opaque-interior-gap"),
            ("outside_spill_pixels", "silhouette-spill"),
            ("opacity_missing_pixels", "translucent-interior-gap"),
            ("opacity_excess_pixels", "translucent-opacity-excess"),
        ):
            ceiling = policy.baseline.terms[term] if policy.baseline else 0
            if terms[term] > ceiling + allowance:
                rejected.append(reason)
        if any(
            value > ceiling
            for value, ceiling in zip(holes, policy.hole_ceilings, strict=True)
        ):
            rejected.append("protected-hole-lost")
        if any(
            value < expected * fraction
            for value, expected, fraction in zip(
                retained, self.component_alpha, self.component_retention, strict=True
            )
        ):
            rejected.append("translucent-component-lost")
        if any(
            value > expected * 1.25 + int(feature.support.sum()) / 255
            for value, expected, feature in zip(
                opacity, self.component_alpha, policy.opacity_components, strict=True
            )
        ):
            rejected.append("translucent-component-opacity-excess")
        return tuple(rejected)
