#!/usr/bin/env python3
"""A synthetic evaluation set: known damage, known order.

Every measurement of a scorer here has otherwise used another scorer as the
reference, which cannot say which of the two is right, or human labels, which
are expensive and few. This needs neither. A clean drawing is its own target,
and damaging it by a growing amount produces candidates whose true order is
known by construction. A scorer either reproduces that order or does not.

Damage comes in two kinds, because the two ask different questions.

*Vector* damage is produced by the search's own mutation operators, applied
cumulatively, so each level is the level below plus one more edit. That is
exactly the damage a run can inflict, which makes it the fair test of whether
the objective can see what the search does. It includes deleting an element --
an operator kept out of the search table precisely because nothing can undo it.

*Raster* damage is the standard image-quality repertoire, following the seven
KADID-10k categories: blur, colour, compression, noise, brightness, spatial
distortion, and sharpness/contrast. These are damage the search cannot itself
produce, and they test whether a scorer is measuring the picture or only the
particular things our operators happen to change.

Only the clean drawings are committed, under bench/distortions. Everything else
is generated here, so the set cannot drift from the code that reads it.

    uv run --extra vision scripts/distortion_screen.py

What it cannot answer: whether a scorer weights the families correctly against
each other -- whether a level-3 recolour really is worse than a level-3 blur.
That is a question about human preference and needs a human.
"""

import contextlib
import io
import random
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from vectrify.formats.svg import operations as ops
from vectrify.formats.svg.plugin import SvgPlugin

REPO = Path(__file__).resolve().parent.parent
SOURCES = REPO / "bench" / "distortions"
SIZE = 384
LEVELS = 5

# The search's own edits. Each is applied repeatedly, so level k is level k-1
# with one more edit of the same kind on top.
VECTOR_OPERATORS = {
    "move": ops.mutate_translate,
    "resize-attr": ops.mutate_numeric,
    "recolour": ops.mutate_color,
    "path-nudge": ops.mutate_path,
    "stroke": ops.mutate_stroke,
    "restack": ops.mutate_reorder,
    "drop-style": ops.mutate_drop_style_property,
    "delete": ops.mutate_remove_node,
}


def vector_levels(clean: str, operator, seed: int = 1000) -> list[str]:
    """Apply one operator repeatedly, keeping every intermediate."""
    rng = random.Random(seed)
    out, current = [clean], clean
    for _ in range(LEVELS):
        random.seed(rng.randrange(1 << 30))
        # An operator with nothing to edit leaves the drawing alone, and that
        # level is dropped later for rendering identically to the one below.
        with contextlib.suppress(Exception):
            current = operator(current)
        out.append(current)
    return out


def _np(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def _img(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def gaussian_blur(image: Image.Image, k: int) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(0.6 * k))


def motion_blur(image: Image.Image, k: int) -> Image.Image:
    """Average the picture with copies of itself shifted along one direction.

    Written out rather than convolved: Pillow's Kernel filter takes 3x3 and 5x5
    only, and the point of a motion blur is that its length grows with level.
    """
    array = _np(image)
    span = 2 * k + 1
    stack = np.zeros_like(array)
    for offset in range(span):
        stack += np.roll(array, offset - k, axis=1)
    return _img(stack / span)


def white_noise(image: Image.Image, k: int) -> Image.Image:
    rng = np.random.default_rng(17)
    return _img(_np(image) + rng.normal(0.0, 4.0 * k, _np(image).shape))


def impulse_noise(image: Image.Image, k: int) -> Image.Image:
    rng = np.random.default_rng(23)
    array = _np(image).copy()
    mask = rng.random(array.shape[:2]) < 0.01 * k
    array[mask] = np.where(rng.random((mask.sum(), 1)) < 0.5, 0.0, 255.0)
    return _img(array)


def jpeg(image: Image.Image, k: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=max(2, 40 - 7 * k))
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def desaturate(image: Image.Image, k: int) -> Image.Image:
    return ImageEnhance.Color(image).enhance(max(0.0, 1.0 - 0.2 * k))


def colour_shift(image: Image.Image, k: int) -> Image.Image:
    array = _np(image).copy()
    array[:, :, 0] += 9.0 * k
    array[:, :, 2] -= 9.0 * k
    return _img(array)


def brighten(image: Image.Image, k: int) -> Image.Image:
    return _img(_np(image) + 9.0 * k)


def contrast(image: Image.Image, k: int) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(max(0.05, 1.0 - 0.16 * k))


def block_jitter(image: Image.Image, k: int) -> Image.Image:
    """Spatial distortion: displace scattered blocks of the picture."""
    rng = np.random.default_rng(31)
    array = _np(image).copy()
    height, width = array.shape[:2]
    for _ in range(6 * k):
        size = 24
        y = int(rng.integers(0, height - size))
        x = int(rng.integers(0, width - size))
        dy = int(rng.integers(-3 * k, 3 * k + 1))
        dx = int(rng.integers(-3 * k, 3 * k + 1))
        ty, tx = np.clip([y + dy, x + dx], 0, [height - size, width - size])
        array[ty : ty + size, tx : tx + size] = array[y : y + size, x : x + size]
    return _img(array)


def sharpen(image: Image.Image, k: int) -> Image.Image:
    return ImageEnhance.Sharpness(image).enhance(1.0 + 1.6 * k)


RASTER_DISTORTIONS = {
    "gaussian-blur": gaussian_blur,
    "motion-blur": motion_blur,
    "white-noise": white_noise,
    "impulse-noise": impulse_noise,
    "jpeg": jpeg,
    "desaturate": desaturate,
    "colour-shift": colour_shift,
    "brighten": brighten,
    "contrast": contrast,
    "block-jitter": block_jitter,
    "sharpen": sharpen,
}


def monotonic(scores: list[float]) -> float:
    """Share of level pairs a scorer puts in the right order.

    Pairs whose renders came out identical are skipped by the caller, so a tie
    counted here is a scorer failing to separate two pictures that do differ.
    """
    pairs = ok = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            pairs += 1
            ok += scores[i] < scores[j]
    return 100.0 * ok / pairs if pairs else float("nan")


def _distinct(pngs: list[bytes]) -> list[int]:
    """Indices to keep: the first, then any that differ from the one before."""
    keep = [0]
    for i in range(1, len(pngs)):
        if pngs[i] != pngs[keep[-1]]:
            keep.append(i)
    return keep


def main() -> None:
    from vectrify.score.compare import compare, prepare
    from vectrify.score.edges import overlap_distance
    from vectrify.score.embedding import EmbeddingScorer
    from vectrify.score.ensemble import PANEL_MODELS, EnsembleScorer

    plugin = SvgPlugin()
    panel = EnsembleScorer()
    members = [EmbeddingScorer(model_name=m) for m in PANEL_MODELS]
    results: dict[str, dict[str, list[float]]] = {}

    def record(family: str, pngs: list[bytes], refs) -> None:
        keep = _distinct(pngs)
        if len(keep) < 2:
            return
        pngs = [pngs[i] for i in keep]
        panel_ref, member_refs, pixel_ref = refs
        row = results.setdefault(family, {})
        row.setdefault("panel", []).append(monotonic(panel.score_many(panel_ref, pngs)))
        row.setdefault("dinov2-small", []).append(
            monotonic(members[0].score_many(member_refs[0], pngs))
        )
        comparisons = [compare(pixel_ref, p) for p in pngs]
        row.setdefault("edge overlap", []).append(
            monotonic(
                [
                    overlap_distance(c.reference_edges, c.candidate_edges)
                    for c in comparisons
                ]
            )
        )
        row.setdefault("colour", []).append(
            monotonic([float(c.colour.mean()) for c in comparisons])
        )

    for source in sorted(SOURCES.glob("*.svg")):
        clean = source.read_text(encoding="utf-8")
        clean_png = plugin.rasterize(clean, SIZE, SIZE)
        target = Image.open(io.BytesIO(clean_png)).convert("RGB")
        refs = (
            panel.prepare_reference(target),
            [m.prepare_reference(target) for m in members],
            prepare(target),
        )

        for name, operator in VECTOR_OPERATORS.items():
            variants = vector_levels(clean, operator)
            record(
                f"svg: {name}",
                [plugin.rasterize(v, SIZE, SIZE) for v in variants],
                refs,
            )

        for name, distort in RASTER_DISTORTIONS.items():
            pngs = []
            for level in range(LEVELS + 1):
                image = target if level == 0 else distort(target, level)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                pngs.append(buffer.getvalue())
            record(f"raster: {name}", pngs, refs)

        print(f"{source.stem} done", flush=True)

    scorers = list(next(iter(results.values())).keys())
    header = f"{'family':<22}{'cases':>6}" + "".join(f"{s:>16}" for s in scorers)
    print(f"\nshare of level pairs ordered correctly\n{header}")
    print("-" * len(header))
    for family in sorted(results):
        row = results[family]
        line = f"{family:<22}{len(row[scorers[0]]):>6}"
        for name in scorers:
            line += f"{statistics.fmean(row[name]):>15.0f}%"
        print(line)
    print("-" * len(header))
    for kind in ("svg:", "raster:"):
        rows = [r for f, r in results.items() if f.startswith(kind)]
        line = f"{'MEAN ' + kind:<22}{'':>6}"
        for name in scorers:
            line += f"{statistics.fmean([v for r in rows for v in r[name]]):>15.0f}%"
        print(line)


if __name__ == "__main__":
    main()
