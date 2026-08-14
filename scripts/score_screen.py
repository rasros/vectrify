"""Screen candidate scoring functions against damage they must be able to see.

Every scoring change in this project has been justified on one or two examples
and then failed on the next. This builds a labelled set instead, large enough
to falsify a proposal, and labelled without consulting any scorer.

The labels come from damaging a drawing on purpose. An element that already
draws its part of the target well -- low error over the pixels it owns -- is
deleted, recoloured, or resized. The damaged candidate is worse than the intact
one by construction, so a scorer that ranks it better has a hole in it, and the
kind of damage says which hole.

    uv run python scripts/score_screen.py
"""

import argparse
import copy
import io
import random
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from vectrify.formats.svg.ownership import drawable_elements, owner_labels
from vectrify.formats.svg.plugin import SvgPlugin
from vectrify.image_utils import resize_long_side
from vectrify.score.base import DEFAULT_CONFIG
from vectrify.score.compare import compare, prepare
from vectrify.score.edges import edge_score
from vectrify.score.utils import color_score, lab_array

REPO = Path(__file__).resolve().parent.parent

# Kept here rather than in the package: the product no longer scores by region,
# and these exist only so region-based measures stay screenable against the ones
# that replaced them.
WORST_FRACTION = 0.01
MIN_WORST_REGIONS = 4


def grid_boxes(size: tuple[int, int], cells: int) -> list[tuple[int, int, int, int]]:
    """*cells* x *cells* boxes tiling the image."""
    width, height = size
    step_x, step_y = max(1, width // cells), max(1, height // cells)
    return [
        (x, y, min(x + step_x, width), min(y + step_y, height))
        for y in range(0, height - step_y + 1, step_y)
        for x in range(0, width - step_x + 1, step_x)
    ]


def worst_region_score(grid: np.ndarray) -> float:
    """Mean over the worst k regions -- not the single maximum, which is noisy."""
    flat = np.asarray(grid, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    keep = min(max(MIN_WORST_REGIONS, round(flat.size * WORST_FRACTION)), flat.size)
    value = float(np.partition(flat, -keep)[-keep:].mean())
    return value if np.isfinite(value) else 0.0
plugin = SvgPlugin()


def render(svg: str, size: int) -> Image.Image:
    return Image.open(
        io.BytesIO(plugin.rasterize(svg, out_w=size, out_h=size))
    ).convert("RGB")


def well_drawn(svg: str, target: Image.Image, size: int) -> list[int]:
    """Indices of elements whose own pixels already match the target closely.

    Damaging one of these is unambiguously a regression. Damaging an element
    that was drawing the wrong thing anyway would not be.
    """
    root = ET.fromstring(svg)
    labels = owner_labels(root, size=size)
    candidate = np.asarray(render(svg, size), dtype=np.int16)
    reference = np.asarray(target.resize((size, size)), dtype=np.int16)
    error = np.abs(candidate - reference).mean(axis=2)

    good = []
    for index in range(len(drawable_elements(root))):
        mask = labels == index
        area = int(mask.sum())
        # Ignore the ground and anything too small to survive rasterising.
        if area < 12 or area > 0.4 * labels.size:
            continue
        if float(error[mask].mean()) < 20.0:
            good.append(index)
    return good


def damage(svg: str, index: int, kind: str, rng: random.Random) -> str | None:
    """Delete, recolour or inflate the element at *index*."""
    root = ET.fromstring(svg)
    units = drawable_elements(root)
    if index >= len(units):
        return None
    _chain, element = units[index]

    if kind == "delete":
        for parent in root.iter():
            if element in list(parent):
                parent.remove(element)
                break
    elif kind == "recolour":
        for node in element.iter():
            if node.get("fill") and node.get("fill") != "none":
                node.set("fill", rng.choice(["#ff00ff", "#00ff00", "#ff8800"]))
            if node.get("stroke") and node.get("stroke") != "none":
                node.set("stroke", rng.choice(["#ff00ff", "#00ff00"]))
    elif kind == "inflate":
        copied = copy.deepcopy(element)
        for parent in root.iter():
            if element in list(parent):
                index_in_parent = list(parent).index(element)
                parent.remove(element)
                copied.set("transform", "scale(1.6)")
                parent.insert(index_in_parent, copied)
                break

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    return ET.tostring(root, encoding="unicode", method="xml")


def build_pairs(size: int, per_case: int, seed: int) -> list[tuple[str, str, str, str]]:
    """(case, damage kind, intact svg, damaged svg) with intact always better."""
    rng = random.Random(seed)
    pairs = []
    for case in sorted((REPO / "bench" / "cases").iterdir()):
        if not (case / "target.png").is_file():
            continue
        target = resize_long_side(Image.open(case / "target.png").convert("RGB"), 384)
        for path in sorted((case / "seeds").glob("*.svg")):
            svg = path.read_text()
            good = well_drawn(svg, target, size)
            if not good:
                continue
            for kind in ("delete", "recolour", "inflate"):
                for _ in range(per_case):
                    hurt = damage(svg, rng.choice(good), kind, rng)
                    if hurt and hurt != svg:
                        valid, _ = plugin.validate(hurt)
                        if valid:
                            pairs.append((case.name, kind, svg, hurt))
    return pairs


def fine_regions(cells: int) -> Callable:
    def scored(reference: Image.Image, png: bytes) -> float:
        candidate = Image.open(io.BytesIO(png)).convert("RGB")
        if candidate.size != reference.size:
            candidate = candidate.resize(reference.size, Image.Resampling.BILINEAR)
        diff = np.abs(lab_array(reference) - lab_array(candidate)).mean(axis=2) / 255.0
        values = np.array(
            [
                float(diff[y0:y1, x0:x1].mean())
                for x0, y0, x1, y1 in grid_boxes(reference.size, cells)
            ]
        )
        return worst_region_score(values)

    return scored


# Xue et al. 2013 use 170 for images on a 0-255 scale.
_GMSD_C = 170.0


def _gradient_magnitude(grey: np.ndarray) -> np.ndarray:
    """Prewitt gradient magnitude, as GMSD specifies."""
    padded = np.pad(grey, 1, mode="edge")
    gx = (
        padded[:-2, 2:]
        + padded[1:-1, 2:]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - padded[1:-1, :-2]
        - padded[2:, :-2]
    ) / 3.0
    gy = (
        padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - padded[:-2, 1:-1]
        - padded[:-2, 2:]
    ) / 3.0
    return np.sqrt(gx * gx + gy * gy)


def _downsample(grey: np.ndarray) -> np.ndarray:
    """Average 2x2 then subsample, the paper's preprocessing step."""
    height = grey.shape[0] - grey.shape[0] % 2
    width = grey.shape[1] - grey.shape[1] % 2
    block = grey[:height, :width].reshape(height // 2, 2, width // 2, 2)
    return block.mean(axis=(1, 3))


def gmsd(reference: Image.Image, png: bytes) -> float:
    """Gradient Magnitude Similarity Deviation (Xue et al. 2013).

    Two gradient convolutions and a standard deviation: no network, and the
    deviation pooling is the paper's central claim -- the spread of local
    quality tracks perceived quality better than its average.
    """
    candidate = Image.open(io.BytesIO(png)).convert("RGB")
    if candidate.size != reference.size:
        candidate = candidate.resize(reference.size, Image.Resampling.BILINEAR)
    ref = _downsample(np.asarray(reference.convert("L"), dtype=np.float64))
    cand = _downsample(np.asarray(candidate.convert("L"), dtype=np.float64))
    gm_ref = _gradient_magnitude(ref)
    gm_cand = _gradient_magnitude(cand)
    similarity = (2 * gm_ref * gm_cand + _GMSD_C) / (gm_ref**2 + gm_cand**2 + _GMSD_C)
    return float(similarity.std())


def gms_mean(reference: Image.Image, png: bytes) -> float:
    """The same similarity map, averaged instead -- isolates the pooling claim."""
    candidate = Image.open(io.BytesIO(png)).convert("RGB")
    if candidate.size != reference.size:
        candidate = candidate.resize(reference.size, Image.Resampling.BILINEAR)
    ref = _downsample(np.asarray(reference.convert("L"), dtype=np.float64))
    cand = _downsample(np.asarray(candidate.convert("L"), dtype=np.float64))
    gm_ref = _gradient_magnitude(ref)
    gm_cand = _gradient_magnitude(cand)
    similarity = (2 * gm_ref * gm_cand + _GMSD_C) / (gm_ref**2 + gm_cand**2 + _GMSD_C)
    return float(1.0 - similarity.mean())


def coverage_score(reference: Image.Image, png: bytes) -> float:
    """Error averaged per target region, every region counted equally.

    Area-weighted error is why deleting a numeral is cheap: it is a fraction of
    a percent of the pixels. Giving each region of the target the same weight
    makes a wrong numeral cost as much as a wrong background.
    """
    candidate = Image.open(io.BytesIO(png)).convert("RGB")
    if candidate.size != reference.size:
        candidate = candidate.resize(reference.size, Image.Resampling.BILINEAR)
    reference_lab = lab_array(reference)
    diff = np.abs(reference_lab - lab_array(candidate)).mean(axis=2) / 255.0

    # Quantised target colour as a stand-in for "region": cheap, and it groups
    # the glyphs and the ground into different buckets, which is the split that
    # matters here.
    quantised = (reference_lab // 24).astype(np.int32)
    keys = quantised[:, :, 0] * 10000 + quantised[:, :, 1] * 100 + quantised[:, :, 2]
    unique, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.ravel()
    sums = np.bincount(inverse, weights=diff.ravel(), minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    per_region = sums / np.maximum(counts, 1)
    weights = np.sqrt(counts)  # a one-pixel speck should not outweigh a shape
    return float((per_region * weights).sum() / weights.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--per-case", type=int, default=2, dest="per_case")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pairs = build_pairs(args.size, args.per_case, args.seed)
    print(f"{len(pairs)} labelled pairs, intact always better\n")

    scorers: dict[str, Callable] = {
        "round score (now)": None,  # filled per case below
        "colour": color_score,
        "edge": edge_score,
        "worst region 4": fine_regions(4),
        "worst region 16": fine_regions(16),
        "worst region 32": fine_regions(32),
        "equal-weight regions": coverage_score,
        "GMSD (std pooling)": gmsd,
        "GMS mean pooling": gms_mean,
    }

    results = {name: {} for name in scorers}
    for case_name, kind, intact, hurt in pairs:
        case = REPO / "bench" / "cases" / case_name
        target = resize_long_side(Image.open(case / "target.png").convert("RGB"), 384)
        reference = resize_long_side(target, DEFAULT_CONFIG.target_long_side)
        prepared = prepare(reference)
        png_intact = plugin.rasterize(intact, out_w=384, out_h=384)
        png_hurt = plugin.rasterize(hurt, out_w=384, out_h=384)

        for name, fn in scorers.items():
            if fn is None:
                ok = (
                    compare(prepared, png_intact).blend()
                    < compare(prepared, png_hurt).blend()
                )
            else:
                ok = fn(reference, png_intact) < fn(reference, png_hurt)
            results[name].setdefault(kind, []).append(ok)

    kinds = ("delete", "recolour", "inflate")
    print(f"{'scorer':<22}" + "".join(f"{k:>11}" for k in kinds) + f"{'overall':>10}")
    for name in scorers:
        row = results[name]
        cells = "".join(
            f"{sum(row.get(k, [])) / max(len(row.get(k, [])), 1):>10.0%} "
            for k in kinds
        )
        every = [ok for values in row.values() for ok in values]
        print(f"{name:<22}{cells}{sum(every) / max(len(every), 1):>9.0%}")
    print("\nshare of pairs where the scorer prefers the intact drawing")


if __name__ == "__main__":
    main()
