"""Sweep the structure weight and the region damping against labelled damage.

Both were fitted before the other existed: the 0.5/0.5 blend was chosen when
the structural term was an overlap and the reduction was a mean. Changing the
reduction changed which weight is right, so they have to be screened together.

    uv run python scripts/blend_screen.py
"""

import argparse
import io

import numpy as np
from PIL import Image
from score_screen import REPO, build_pairs, plugin

from vectrify.image_utils import resize_long_side
from vectrify.score.base import DEFAULT_CONFIG
from vectrify.score.edges import edge_map
from vectrify.score.utils import lab_array

DAMPS = (0.25, 0.5, 0.75, 1.0)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def diffs(reference: Image.Image, png: bytes) -> tuple[np.ndarray, np.ndarray]:
    candidate = Image.open(io.BytesIO(png)).convert("RGB")
    if candidate.size != reference.size:
        candidate = candidate.resize(reference.size, Image.Resampling.BILINEAR)
    colour = np.abs(lab_array(reference) - lab_array(candidate)).mean(axis=2) / 255.0
    structure = np.abs(edge_map(reference) - edge_map(candidate))
    return colour, structure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-case", type=int, default=2, dest="per_case")
    args = parser.parse_args()

    pairs = build_pairs(256, args.per_case, 0)
    print(f"{len(pairs)} labelled pairs\n")

    hits = {
        (d, w): {"delete": [], "recolour": [], "inflate": []}
        for d in DAMPS
        for w in WEIGHTS
    }

    for case_name, kind, intact, hurt in pairs:
        case = REPO / "bench" / "cases" / case_name
        target = resize_long_side(Image.open(case / "target.png").convert("RGB"), 384)
        reference = resize_long_side(target, DEFAULT_CONFIG.target_long_side)

        lab = lab_array(reference)
        quantised = (lab // 24).astype(np.int64)
        keys = (
            quantised[:, :, 0] * 10000 + quantised[:, :, 1] * 100 + quantised[:, :, 2]
        )
        _unique, inverse = np.unique(keys, return_inverse=True)
        flat = inverse.ravel()
        counts = np.bincount(flat)

        good = diffs(reference, plugin.rasterize(intact, out_w=384, out_h=384))
        bad = diffs(reference, plugin.rasterize(hurt, out_w=384, out_h=384))

        def reduce(diff: np.ndarray, damp: float, flat=flat, counts=counts) -> float:
            sums = np.bincount(flat, weights=diff.ravel(), minlength=len(counts))
            weights = np.power(counts, damp)
            return float(
                ((sums / np.maximum(counts, 1)) * weights).sum() / weights.sum()
            )

        for damp in DAMPS:
            colour_good, structure_good = (reduce(good[0], damp), reduce(good[1], damp))
            colour_bad, structure_bad = (reduce(bad[0], damp), reduce(bad[1], damp))
            for weight in WEIGHTS:
                a = weight * structure_good + (1 - weight) * colour_good
                b = weight * structure_bad + (1 - weight) * colour_bad
                hits[(damp, weight)][kind].append(a < b)

    print(f"{'damp':>6}" + "".join(f"{f'w={w:g}':>10}" for w in WEIGHTS))
    for damp in DAMPS:
        row = ""
        for weight in WEIGHTS:
            every = [ok for values in hits[(damp, weight)].values() for ok in values]
            row += f"{sum(every) / len(every):>9.0%} "
        print(f"{damp:>6.2f}{row}")

    print(f"\n{'best cells, by damage kind':<30}")
    best = max(hits, key=lambda k: sum(sum(v) for v in hits[k].values()))
    for kind, values in hits[best].items():
        print(f"  {kind:<10}{sum(values) / len(values):>6.0%}")
    print(f"  damp {best[0]:g}, structure weight {best[1]:g}")


if __name__ == "__main__":
    main()
