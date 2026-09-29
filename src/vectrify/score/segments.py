"""Edge-aware Voronoi masks for retaining locally good candidates."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from vectrify.score.compare import Comparison
from vectrify.score.edges import edge_map, overlap_distance
from vectrify.score.utils import clamp01


@dataclass(frozen=True)
class Segment:
    """One soft, edge-centred attention field at scoring resolution."""

    index: int
    mask: np.ndarray

    @property
    def metric_name(self) -> str:
        return f"segment_{self.index}"


def _balanced_region_labels(masks: list[np.ndarray]) -> np.ndarray:
    """Expand small SAM regions into adjacent background territory."""
    from scipy.ndimage import distance_transform_edt

    regions = [~np.logical_or.reduce(masks), *masks]
    distances = np.stack(
        [np.asarray(distance_transform_edt(~mask)) for mask in regions]
    )
    labels = np.argmin(distances, axis=0)
    areas = np.bincount(labels.ravel(), minlength=len(regions)).astype(float)
    target = areas + 0.35 * (areas.mean() - areas)
    bias = np.zeros(len(regions))
    for _ in range(40):
        labels = np.argmax(-distances + bias[:, None, None], axis=0)
        areas = np.bincount(labels.ravel(), minlength=len(regions))
        bias += 0.08 * (target - areas) / np.maximum(target, 1)
    return labels


def _merge_indistinguishable(labels: np.ndarray, image: Image.Image) -> np.ndarray:
    """Merge touching regions with a weak, low-contrast shared boundary."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    edges = edge_map(image, tolerance=0)
    changed = True
    while changed:
        changed = False
        for first, second in (
            (labels[:, :-1], labels[:, 1:]),
            (labels[:-1], labels[1:]),
        ):
            different = first != second
            for left, right in zip(first[different], second[different], strict=True):
                a, b = sorted((int(left), int(right)))
                if a == b:
                    continue
                mask_a, mask_b = labels == a, labels == b
                colour = float(
                    np.abs(rgb[mask_a].mean(axis=0) - rgb[mask_b].mean(axis=0)).mean()
                )
                boundary = (labels == a) & np.asarray(
                    Image.fromarray(mask_b.astype(np.uint8) * 255).filter(
                        ImageFilter.MaxFilter(3)
                    )
                ).astype(bool)
                edge_strength = float(edges[boundary].mean()) if boundary.any() else 0.0
                if colour < 0.035 and edge_strength < 0.12:
                    labels[labels == b] = a
                    changed = True
                    break
            if changed:
                break
    return labels


def _split_regions_and_absorb_background(
    labels: np.ndarray, image: Image.Image
) -> list[np.ndarray]:
    """Separate disconnected islands and fold plain background fragments together."""
    from scipy.ndimage import label

    parts: list[np.ndarray] = []
    for region in np.unique(labels):
        components, count = label(labels == region)
        parts.extend(components == index for index in range(1, count + 1))
    background_index = max(range(len(parts)), key=lambda index: int(parts[index].sum()))
    background = parts.pop(background_index)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    edges = edge_map(image, tolerance=0)
    background_colour = rgb[background].mean(axis=0)
    retained: list[np.ndarray] = []
    for part in parts:
        colour_delta = float(np.abs(rgb[part].mean(axis=0) - background_colour).mean())
        edge_density = float(edges[part].mean())
        if colour_delta < 0.035 and edge_density < 0.08:
            background |= part
        else:
            retained.append(part)
    return [background, *retained]


def _merge_compact_features(parts: list[np.ndarray]) -> list[np.ndarray]:
    """Keep the pieces of a tiny outlined feature in one local region."""
    changed = True
    while changed:
        changed = False
        for first, mask_a in enumerate(parts):
            if int(mask_a.sum()) > 1_200:
                continue
            grown = np.asarray(
                Image.fromarray(mask_a.astype(np.uint8) * 255).filter(
                    ImageFilter.MaxFilter(13)
                )
            ).astype(bool)
            for second in range(first + 1, len(parts)):
                mask_b = parts[second]
                if int(mask_b.sum()) > 1_200 or not (grown & mask_b).any():
                    continue
                ys, xs = np.nonzero(mask_a | mask_b)
                if xs.max() - xs.min() > 30 or ys.max() - ys.min() > 30:
                    continue
                parts[first] = mask_a | mask_b
                del parts[second]
                changed = True
                break
            if changed:
                break
    return parts


def segment_target(image: Image.Image, *, max_regions: int = 8) -> list[Segment]:
    """Return meaningful SAM regions after merging indistinguishable neighbours."""
    if max_regions < 1:
        return []
    from transformers import pipeline

    generated = pipeline("mask-generation", model="facebook/sam-vit-base", device=0)(
        image, points_per_batch=32, points_per_crop=16
    )["masks"]
    masks = [np.asarray(mask, dtype=bool) for mask in generated]
    labels = _merge_indistinguishable(_balanced_region_labels(masks), image)
    candidates = _merge_compact_features(
        _split_regions_and_absorb_background(labels, image)
    )
    edges = edge_map(image, tolerance=0)
    candidates.sort(
        key=lambda mask: float(edges[mask].sum()) + 0.05 * np.sqrt(mask.sum()),
        reverse=True,
    )
    return [
        Segment(index=index, mask=mask)
        for index, mask in enumerate(candidates[:max_regions])
    ]


def segment_error(comparison: Comparison, mask: np.ndarray) -> float:
    """Colour-and-structure error weighted by one local attention field."""
    weights = mask.astype(np.float32)
    total_weight = float(weights.sum())
    if total_weight == 0.0:
        return 1.0
    colour = float((comparison.colour * weights).sum() / total_weight)
    structure = overlap_distance(
        comparison.reference_edges * weights, comparison.candidate_edges * weights
    )
    return clamp01(0.5 * structure + 0.5 * colour)
