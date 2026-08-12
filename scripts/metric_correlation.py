"""Rank-correlate cheap candidate metrics against the vision score.

The round optimises cheap metrics because the vision model is ~300x their cost,
which is only sound if the cheap metric orders candidates the way the model
would. This measures that, so a proposed objective is adopted on evidence
rather than on the story told about it.

Candidates are generated the way the search generates them -- random mutation
chains from each seed -- so the spread covers the quality range a real run
moves through rather than a synthetic one.

    uv run python scripts/metric_correlation.py --per-case 200
"""

import argparse
import io
import random
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from vectrify.formats.svg.operations import apply_mutation
from vectrify.formats.svg.plugin import SvgPlugin
from vectrify.image_utils import resize_long_side
from vectrify.score.edges import edge_score
from vectrify.score.utils import color_score
from vectrify.score.vision import VisionScorer

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CASES = REPO / "bench" / "cases"


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, ties averaged."""

    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(values)
        out = np.empty(len(values), dtype=np.float64)
        out[order] = np.arange(len(values), dtype=np.float64)
        # Average the ranks of tied values, or ties bias the correlation by
        # however the sort happened to break them.
        array = np.asarray(values, dtype=np.float64)
        for value in np.unique(array):
            mask = array == value
            if mask.sum() > 1:
                out[mask] = out[mask].mean()
        return out

    a, b = ranks(xs), ranks(ys)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator else 0.0


def thumbnail_score(reference_rgb: Image.Image, candidate_png: bytes) -> float:
    """Colour distance at 32px: the squint test.

    A live hypothesis rather than a filler control -- the micro-search proxy
    measurably preferred 32px to 128px, so low frequency tracked the objective
    better than pixel accuracy did.
    """
    small = resize_long_side(reference_rgb, 32)
    candidate = Image.open(io.BytesIO(candidate_png)).convert("RGB")
    buffer = io.BytesIO()
    candidate.resize(small.size, Image.Resampling.LANCZOS).save(buffer, format="PNG")
    return color_score(small, buffer.getvalue())


def blend(structure_weight: float):
    """Structure and colour mixed the way the vision score mixes them.

    The model's own score is 0.85 embedding cosine plus 0.15 colour distance,
    so a cheap stand-in for it should be composed the same way -- the open
    question is only what the weight should be, which is what this measures.
    """

    def scored(reference, candidate_png: bytes) -> float:
        return structure_weight * edge_score(reference, candidate_png) + (
            1.0 - structure_weight
        ) * color_score(reference, candidate_png)

    return scored


METRICS = {
    "l1": color_score,
    "edge": edge_score,
    "thumb32": thumbnail_score,
    "mix.50": blend(0.50),
    "mix.70": blend(0.70),
    "mix.85": blend(0.85),
}


def candidates(case: Path, count: int, chain: int, rng: random.Random) -> list[str]:
    """Mutation chains from every seed, sampled along their whole length.

    Sampling mid-chain as well as at the end is what gives a spread: a chain's
    early candidates sit near the seed and its late ones far from it, and a
    correlation measured only on near-converged candidates says nothing about
    how the metric orders the rest of a run.
    """
    seeds = sorted((case / "seeds").glob("*.svg"))
    out: list[str] = []
    while len(out) < count:
        content = rng.choice(seeds).read_text(encoding="utf-8")
        for _ in range(rng.randrange(1, chain)):
            content = apply_mutation(content)[0]
        out.append(content)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", default=str(DEFAULT_CASES), metavar="DIR")
    parser.add_argument("--per-case", type=int, default=200, dest="per_case")
    parser.add_argument("--chain", type=int, default=60, metavar="N")
    parser.add_argument("--resolution", type=int, default=384, metavar="PX")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    plugin = SvgPlugin()
    scorer = VisionScorer()
    rng = random.Random(args.seed)

    cases = sorted(
        d
        for d in Path(args.cases).iterdir()
        if (d / "target.png").is_file() and (d / "seeds").is_dir()
    )

    names = list(METRICS)
    per_case: dict[str, list[float]] = {name: [] for name in names}
    print(f"{'case':<14} " + " ".join(f"{name:>9}" for name in names))

    for case in cases:
        target = resize_long_side(
            Image.open(case / "target.png").convert("RGB"), args.resolution
        )
        reference = scorer.prepare_reference(target)
        width, height = target.size

        vision: list[float] = []
        measured: dict[str, list[float]] = {name: [] for name in names}
        for content in candidates(case, args.per_case, args.chain, rng):
            try:
                png = plugin.rasterize(content, out_w=width, out_h=height)
            except Exception:
                continue
            vision.append(scorer.score(reference, png))
            for name, fn in METRICS.items():
                measured[name].append(fn(target, png))

        for name in names:
            per_case[name].append(spearman(measured[name], vision))
        print(
            f"{case.name:<14} "
            + " ".join(f"{per_case[name][-1]:>9.3f}" for name in names)
        )

    print(
        f"\n{'MEAN':<14} "
        + " ".join(f"{statistics.fmean(per_case[name]):>9.3f}" for name in names)
    )
    print("\nrank correlation with the vision score; 1.0 would be a perfect proxy")


if __name__ == "__main__":
    main()
