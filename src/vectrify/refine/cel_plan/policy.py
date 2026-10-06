"""One fixed evidence score and exact checkpoint validator for CEL planning.

All denominators and protected features belong to the reference. Removing a
feature or adding canvas padding therefore cannot improve the normalization.
These initial weights are versioned engineering values; corpus calibration
and release tolerances remain separate benchmark artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
)

from vectrify.document import import_svg
from vectrify.refine.cel_plan.model import Evidence, Graph
from vectrify.refine.cel_plan.score import (
    SCORE_VERSION,
    composite,
    foreground_mask,
    masked_mean,
    premultiplied,
    render,
    svg_metrics,
)


@dataclass(frozen=True)
class Weights:
    color: float = 1.0
    alpha: float = 0.5
    edges: float = 0.15
    features: float = 0.5
    detail: float = 0.04


@dataclass(frozen=True)
class Feature:
    """A local mask, stored only inside its source-coordinate bounding box."""

    box: tuple[int, int, int, int]
    support: np.ndarray


@dataclass(frozen=True)
class Evaluation:
    terms: dict[str, float]
    structure: dict
    rejections: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.rejections

    @property
    def visual(self) -> float:
        return self.terms["visual"]

    @property
    def cost(self) -> float:
        return self.structure["representation_cost"]

    def objective(self, complexity: int, normalizer: float, detail: float = 0.04):
        if not 0 <= complexity <= 100 or not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError(
                "Objective requires complexity 0-100 and a positive cost scale"
            )
        return (
            self.visual
            + detail * 2 ** ((50 - complexity) / 25) * self.cost / normalizer
        )

    def metrics(self) -> dict:
        return {
            "score_version": SCORE_VERSION,
            "score_terms": self.terms,
            "validation_rejections": list(self.rejections),
            **self.structure,
        }


def _robust(difference: np.ndarray, delta: float = 0.1) -> np.ndarray:
    absolute = np.abs(difference)
    return np.where(absolute <= delta, difference**2, 2 * delta * absolute - delta**2)


def source_field(field: np.ndarray, evidence: Evidence) -> np.ndarray:
    """Map scalar analysis evidence back through the actual rounded scales."""
    sx, sy = evidence.scale
    size = (round(field.shape[1] / sx), round(field.shape[0] / sy))
    resized = np.asarray(
        Image.fromarray(field.astype(np.float32)).resize(
            size, Image.Resampling.BILINEAR
        )
    )
    result = np.zeros(evidence.rgba.shape[:2], dtype=np.float32)
    x, y = evidence.offset
    result[y : y + size[1], x : x + size[0]] = resized
    return result


def _feature(mask: np.ndarray) -> Feature | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x, y, right, bottom = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )
    support = mask[y:bottom, x:right].copy()
    support.flags.writeable = False
    return Feature((x, y, right - x, bottom - y), support)


def graph_features(evidence: Evidence, graph: Graph) -> tuple[Feature, ...]:
    """Stable, high-contrast small regions are local score supports, never labels."""
    total = max(1, sum(region.area for region in graph.regions))
    eligible = sorted(
        (
            region
            for region in graph.regions
            if region.id not in graph.hidden
            and region.feature >= 0.25
            and region.texture < 0.6
            and 3 <= region.area <= total * 0.05
        ),
        key=lambda region: (-region.feature, region.id),
    )[:64]
    features = []
    for region in eligible:
        mask = source_field((graph.labels == region.id).astype(float), evidence) >= 0.5
        feature = _feature(mask)
        if feature is not None:
            features.append(feature)
    return tuple(features)


def _ink(rgba: np.ndarray) -> np.ndarray:
    luminance = composite(rgba) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    contrast = gaussian_filter(luminance, 2) - luminance
    return (contrast >= 0.035) & (rgba[..., 3] >= 0.5)


class Policy:
    """Fixed multi-scale evidence with baseline-relative hard safeguards.

    The initial candidate must itself pass document, crossing and hole checks.
    Its measured antialias/coverage residual sets the ceiling for subsequent
    edits, with an explicit small raster allowance. No candidate establishes
    its own ceiling, and no human geometry is available here.
    """

    def __init__(
        self,
        truth: np.ndarray,
        *,
        features: tuple[Feature, ...] = (),
        ink: np.ndarray | None = None,
        texture: np.ndarray | None = None,
        weights: Weights | None = None,
    ):
        if truth.ndim != 3 or truth.shape[2] != 4 or not np.isfinite(truth).all():
            raise ValueError("Score evidence must be finite RGBA")
        if np.any((truth < 0) | (truth > 1)):
            raise ValueError("Score evidence must lie between zero and one")
        self.truth = np.array(truth, dtype=np.float32, copy=True)
        self.truth.flags.writeable = False
        self.weights = weights or Weights()
        self.mask = foreground_mask(truth)
        self.alpha = truth[..., 3] >= 0.5
        self.area = max(1, int(self.alpha.sum()))
        self.inside = binary_erosion(truth[..., 3] >= 0.99, iterations=2)
        self.outside = ~binary_dilation(truth[..., 3] > 0.01, iterations=2)
        holes = np.asarray(binary_fill_holes(self.alpha)) & (truth[..., 3] < 0.01)
        components, count = label(holes)
        self.holes = tuple(
            feature
            for index in range(1, count + 1)
            if (components == index).sum() >= 4
            if (feature := _feature(components == index)) is not None
        )
        self.features = features
        self.ink = _ink(truth) if ink is None else np.array(ink, dtype=bool, copy=True)
        self.ink &= self.alpha
        self.texture = (
            np.zeros(truth.shape[:2], dtype=np.float32)
            if texture is None
            else np.array(texture, copy=True)
        )
        if self.ink.shape != truth.shape[:2] or self.texture.shape != truth.shape[:2]:
            raise ValueError(
                "Score support fields must match native reference dimensions"
            )
        if not np.isfinite(self.texture).all() or np.any(
            (self.texture < 0) | (self.texture > 1)
        ):
            raise ValueError(
                "Texture confidence must be finite and lie between zero and one"
            )
        for feature in (*self.features, *self.holes):
            x, y, width, height = feature.box
            if (
                x < 0
                or y < 0
                or width < 1
                or height < 1
                or x + width > truth.shape[1]
                or y + height > truth.shape[0]
                or feature.support.shape != (height, width)
            ):
                raise ValueError("Feature support must lie inside the native reference")
        self.targets = tuple(
            gaussian_filter(premultiplied(self.truth), (sigma, sigma, 0))
            for sigma in (0, 2, 4)
        )
        self.ink_distance = np.asarray(distance_transform_edt(~self.ink))
        self.baseline: Evaluation | None = None

    @classmethod
    def from_evidence(cls, evidence: Evidence, graph: Graph) -> Policy:
        return cls(
            evidence.rgba,
            features=graph_features(evidence, graph),
            ink=source_field(evidence.drawn.astype(float), evidence) >= 0.5,
            texture=np.clip(source_field(evidence.texture, evidence), 0, 1),
        )

    def _terms(self, actual: np.ndarray) -> dict[str, float]:
        predicted = premultiplied(actual)
        # Texture can reduce the native color weight, but never alpha or local
        # feature support. The fixed weights sum to one at every reference pixel.
        native = 0.35 * (1 - 0.8 * self.texture)
        scales = (native, 0.45 + 0.35 - native, 0.2)
        color = 0.0
        for sigma, weight, target in zip((0, 2, 4), scales, self.targets, strict=True):
            blurred = (
                gaussian_filter(predicted, (sigma, sigma, 0)) if sigma else predicted
            )
            error = _robust(blurred[..., :3] - target[..., :3]).mean(axis=-1)
            color += masked_mean(error * weight, self.mask)
        # Both contrasting backdrops detect partial-alpha/color compensation.
        alpha_error = _robust(actual[..., 3] - self.truth[..., 3]).sum() / self.area
        backgrounds = (
            sum(
                _robust(
                    composite(actual, background) - composite(self.truth, background)
                )
                .mean(axis=-1)
                .sum()
                / self.area
                for background in (0.0, 1.0)
            )
            / 2
        )
        alpha_error += backgrounds * 0.25
        observed = _ink(actual) & binary_dilation(self.mask, iterations=2)
        if self.ink.any() and observed.any():
            to_actual = np.asarray(distance_transform_edt(~observed))
            missing = float(np.minimum(to_actual[self.ink] / 4, 1).mean())
            unsupported = float(
                np.minimum(self.ink_distance[observed] / 4, 1).sum()
            ) / max(1, int(self.ink.sum()))
            edge = (missing + unsupported) / 2
        else:
            edge = float(self.ink.any() or observed.any())
        features = []
        for feature in self.features:
            x, y, width, height = feature.box
            box = np.s_[y : y + height, x : x + width]
            features.append(
                masked_mean(
                    _robust(predicted[box] - self.targets[0][box]), feature.support
                )
            )
        feature_error = (
            (float(np.mean(features)) + max(features)) / 2 if features else 0.0
        )
        inside_missing = int((self.inside & (actual[..., 3] < 0.5)).sum())
        outside_spill = int((self.outside & (actual[..., 3] >= 0.5)).sum())
        visual = (
            self.weights.color * color
            + self.weights.alpha * float(alpha_error)
            + self.weights.edges * edge
            + self.weights.features * feature_error
        )
        return {
            "color": color,
            "alpha": float(alpha_error),
            "edges": edge,
            "features": feature_error,
            "interior_missing_pixels": float(inside_missing),
            "outside_spill_pixels": float(outside_spill),
            "visual": visual,
        }

    def evaluate(self, svg: str, *, pixels: np.ndarray | None = None) -> Evaluation:
        document = import_svg(svg)
        document.validate()
        structure = svg_metrics(svg, include_crossings=True)
        actual = (
            render(svg, (self.truth.shape[1], self.truth.shape[0]))
            if pixels is None
            else pixels
        )
        if (
            actual.shape != self.truth.shape
            or not np.isfinite(actual).all()
            or np.any((actual < 0) | (actual > 1))
        ):
            raise ValueError(
                "Candidate raster must be finite native-size RGBA in [0, 1]"
            )
        terms = self._terms(actual)
        rejected = []
        crossing_limit = (
            self.baseline.structure["self_crossings"] if self.baseline else 0
        )
        if structure["self_crossings"] > crossing_limit:
            rejected.append("new-self-crossing")
        # Permit subpixel boundary reconstruction, not unexplained opaque gaps.
        allowance = max(4, round(self.area * 0.0005))
        for term, reason in (
            ("interior_missing_pixels", "opaque-interior-gap"),
            ("outside_spill_pixels", "silhouette-spill"),
        ):
            ceiling = self.baseline.terms[term] if self.baseline else 0
            if terms[term] > ceiling + allowance:
                rejected.append(reason)
        for feature in self.holes:
            x, y, width, height = feature.box
            opacity = actual[y : y + height, x : x + width, 3]
            if float(opacity[feature.support].mean()) > 0.1:
                rejected.append("protected-hole-lost")
                break
        return Evaluation(terms, structure, tuple(rejected))

    def establish(self, evaluation: Evaluation) -> None:
        if self.baseline is not None:
            raise ValueError("A planning run's baseline cannot change")
        if not evaluation.valid:
            raise ValueError("The initial checkpoint did not pass hard validation")
        self.baseline = evaluation

    def metadata(self) -> dict:
        return {
            "score_version": SCORE_VERSION,
            "score_weights": asdict(self.weights),
            "feature_supports": len(self.features),
        }
