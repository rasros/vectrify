"""Find one reduction from a diff map to a scalar, usable everywhere.

The score, the region metrics and the structural term all reduce a per-pixel
difference to one number, and all three currently take the mean. The mean is
what makes small features invisible: a deleted numeral is a few pixels of large
error, and averaging over the canvas divides it away.

This screens reductions on the labelled damage set. Whatever wins is meant to
be used for all three -- whole canvas, one region, and the edge map -- so it is
scored on the pixel diff here and confirmed on the edge diff afterwards.

    uv run python scripts/reduction_screen.py
"""

import argparse
import io
import statistics
from collections.abc import Callable

import numpy as np
from PIL import Image
from score_screen import REPO, build_pairs, plugin

from vectrify.image_utils import resize_long_side
from vectrify.score.base import DEFAULT_CONFIG
from vectrify.score.edges import edge_map
from vectrify.score.utils import lab_array


def power(exponent: float) -> Callable[[np.ndarray], float]:
    """Generalised mean. Above 1 the worst pixels dominate, below 1 the diffuse
    ones do."""

    def reduce(diff: np.ndarray) -> float:
        return float(np.power(np.power(diff, exponent).mean(), 1.0 / exponent))

    return reduce


def top_fraction(fraction: float) -> Callable[[np.ndarray], float]:
    """Mean of the worst *fraction* of pixels: worst_region, at pixel scale."""

    def reduce(diff: np.ndarray) -> float:
        flat = diff.ravel()
        keep = max(1, int(flat.size * fraction))
        return float(np.partition(flat, -keep)[-keep:].mean())

    return reduce


def over_threshold(level: float) -> Callable[[np.ndarray], float]:
    """Share of pixels that are wrong at all, ignoring by how much."""

    def reduce(diff: np.ndarray) -> float:
        return float((diff > level).mean())

    return reduce


def region_weighted(reference_lab: np.ndarray, damp: float) -> Callable:
    """Average per target region, each region weighted by its area ** damp.

    Plain averaging weights a region by its area, so a numeral covering 0.2% of
    the canvas can only ever move the score by 0.2%. Damping the weight lets a
    small feature matter without letting a one-pixel speck dominate.
    """
    quantised = (reference_lab // 24).astype(np.int32)
    keys = quantised[:, :, 0] * 10000 + quantised[:, :, 1] * 100 + quantised[:, :, 2]
    _unique, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse)
    weights = np.power(counts, damp)

    def reduce(diff: np.ndarray) -> float:
        sums = np.bincount(inverse, weights=diff.ravel(), minlength=len(counts))
        per_region = sums / np.maximum(counts, 1)
        return float((per_region * weights).sum() / weights.sum())

    return reduce


def deviation(diff: np.ndarray) -> float:
    """Standard deviation of the local error, GMSD's pooling strategy.

    Xue et al. 2013 found the spread of local quality predicts perceived
    quality better than its average: a viewer notices that one part is wrong,
    not that the average is slightly raised. A deleted numeral is exactly a
    localised dip that a mean divides away.
    """
    return float(diff.std())


def mean_plus_deviation(diff: np.ndarray) -> float:
    return float(diff.mean() + diff.std())


REDUCTIONS: dict[str, Callable[[np.ndarray], float]] = {
    "std (GMSD-style)": deviation,
    "mean + std": mean_plus_deviation,
    "mean (now)": power(1.0),
    "power 1.5": power(1.5),
    "power 2": power(2.0),
    "power 3": power(3.0),
    "power 4": power(4.0),
    "top 1%": top_fraction(0.01),
    "top 5%": top_fraction(0.05),
    "top 20%": top_fraction(0.20),
    "over 0.05": over_threshold(0.05),
    "over 0.15": over_threshold(0.15),
}


def pixel_diff(reference: Image.Image, png: bytes) -> np.ndarray:
    candidate = Image.open(io.BytesIO(png)).convert("RGB")
    if candidate.size != reference.size:
        candidate = candidate.resize(reference.size, Image.Resampling.BILINEAR)
    return np.abs(lab_array(reference) - lab_array(candidate)).mean(axis=2) / 255.0


def edge_diff(reference: Image.Image, png: bytes) -> np.ndarray:
    candidate = Image.open(io.BytesIO(png)).convert("RGB")
    if candidate.size != reference.size:
        candidate = candidate.resize(reference.size, Image.Resampling.BILINEAR)
    return np.abs(edge_map(reference) - edge_map(candidate))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-case", type=int, default=2, dest="per_case")
    parser.add_argument("--on", choices=["pixels", "edges"], default="pixels")
    args = parser.parse_args()

    pairs = build_pairs(256, args.per_case, 0)
    diff_of = pixel_diff if args.on == "pixels" else edge_diff
    print(f"{len(pairs)} labelled pairs, reduced over {args.on}\n")

    hits: dict[str, dict[str, list[bool]]] = {name: {} for name in REDUCTIONS}
    margins: dict[str, list[float]] = {name: [] for name in REDUCTIONS}

    for case_name, kind, intact, hurt in pairs:
        case = REPO / "bench" / "cases" / case_name
        target = resize_long_side(Image.open(case / "target.png").convert("RGB"), 384)
        reference = resize_long_side(target, DEFAULT_CONFIG.target_long_side)
        good = diff_of(reference, plugin.rasterize(intact, out_w=384, out_h=384))
        bad = diff_of(reference, plugin.rasterize(hurt, out_w=384, out_h=384))

        reference_lab = lab_array(reference)
        for damp in (0.0, 0.5):
            name = f"regions damp {damp:g}"
            reduce = region_weighted(reference_lab, damp)
            a, b = reduce(good), reduce(bad)
            hits.setdefault(name, {}).setdefault(kind, []).append(a < b)
            margins.setdefault(name, []).append((b - a) / max(a, 1e-9))

        for name, reduce in REDUCTIONS.items():
            a, b = reduce(good), reduce(bad)
            hits[name].setdefault(kind, []).append(a < b)
            margins[name].append((b - a) / max(a, 1e-9))

    kinds = ("delete", "recolour", "inflate")
    header = "".join(f"{k:>10}" for k in kinds)
    print(f"{'reduction':<18}{header}{'overall':>9}{'median penalty':>16}")
    for name in hits:
        row = hits[name]
        cells = "".join(
            f"{sum(row.get(k, [])) / max(len(row.get(k, [])), 1):>9.0%} " for k in kinds
        )
        every = [ok for values in row.values() for ok in values]
        overall = sum(every) / max(len(every), 1)
        print(
            f"{name:<18}{cells}{overall:>8.0%}{statistics.median(margins[name]):>15.1%}"
        )
    print("\nshare of pairs ranked correctly, and how much damage costs")


if __name__ == "__main__":
    main()
