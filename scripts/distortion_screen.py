#!/usr/bin/env python3
"""A synthetic evaluation set: known damage, known order.

Every measurement of a scorer here has otherwise used another scorer as the
reference, which cannot say which of the two is right, or human labels, which
are expensive and few. This needs neither. A clean drawing is its own target,
and damaging it by a growing amount produces candidates whose true order is
known by construction. A scorer either reproduces that order or does not.

Damage comes in two kinds, because the two ask different questions.

*Vector* damage comes from two places. Most of it is the search's own mutation
operators, applied cumulatively, so each level is the level below plus one more
edit. That is exactly the damage a run can inflict, which makes it the fair
test of whether the objective can see what the search does. It includes
deleting an element -- an operator kept out of the search table precisely
because nothing can undo it. The rest is written here: mistakes a model makes
when it authors the SVG rather than ones an operator makes when it edits one,
such as drawing the right word at the wrong size or the right colour in the
wrong shape. The search cannot produce those, so a scorer blind to them still
scores well on the operator families while a run that made them goes unpunished.

*Raster* damage is the standard image-quality repertoire, following the seven
KADID-10k categories: blur, colour, compression, noise, brightness, spatial
distortion, and sharpness/contrast. These are damage the search cannot itself
produce, and they test whether a scorer is measuring the picture or only the
particular things our operators happen to change.

The base drawings are the benchmark corpus's own first seeds, read straight
from bench/cases rather than copied here, so the two cannot drift apart. Clean
means undamaged, not correct: a seed is one of five deliberately-off attempts
at its target, which costs the screen nothing, because the known order comes
from damaging whatever it starts from. Damage is generated at run time, so the
set cannot drift from the code that reads it either.

    uv run --extra vision scripts/distortion_screen.py
    uv run --extra vision --with datasets scripts/distortion_screen.py --hf 8

Six drawings is a narrow base, and it shows: a family only runs on the cases
that have something for it to damage, so gradients reach one case and strokes
four. ``--hf`` widens it by streaming drawings from public collections on
HuggingFace -- icons, emoji and sketches, the last being nearly all stroke,
which is the weakest family we measure. They are pulled at run time and never
copied into the repository, so nothing here redistributes them.

What it cannot answer: whether a scorer weights the families correctly against
each other -- whether a level-3 recolour really is worse than a level-3 blur.
That is a question about human preference and needs a human.
"""

import argparse
import contextlib
import functools
import io
import math
import random
import re
import statistics
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from vectrify.formats.svg import operations as ops
from vectrify.formats.svg.plugin import SvgPlugin

REPO = Path(__file__).resolve().parent.parent
CASES = REPO / "bench" / "cases"

# Streamed rather than downloaded whole: these run to hundreds of thousands of
# drawings and the screen wants a handful from each.
HF_SETS = (
    ("starvector/svg-icons", "Svg"),
    ("starvector/svg-emoji", "Svg"),
    ("kmewhort/sketchy-svgs", "svg"),
    # Diagrams for their labels and gradients: the families that damage text or
    # a colour ramp otherwise run on the two or three of our own drawings that
    # happen to have any, which is too few to read anything from.
    ("starvector/svg-diagrams", "Svg"),
)

# A drawing too simple has nothing to damage and one too intricate is slow to
# render at every level; this brackets the corpus we already have.
HF_ELEMENT_RANGE = (8, 120)
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


def first_seed(case: Path) -> Path:
    """The case's lowest-numbered seed, by the same glob the corpus tests use."""
    return sorted((case / "seeds").glob("*.svg"))[0]


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


# Damage written here rather than borrowed from the operators. Each is a
# function of the level rather than a step applied on top of the level below,
# because what grows is a quantity -- how many characters are wrong, how far
# the shear goes -- and reapplying a step cannot express that. Every one reads
# the clean drawing, so level k is level k-1 plus more by construction.
SvgDamage = Callable[[str, int], str]

# Elements that paint something. Gradient stops and other <defs> content are
# left out by not appearing here.
_SHAPES = ("rect", "circle", "ellipse", "line", "path", "polygon", "polyline", "text")


def _tag(element: ET.Element) -> str:
    return element.tag.split("}")[-1]


def _shapes(root: ET.Element, tags: tuple[str, ...] = _SHAPES) -> list[ET.Element]:
    return [element for element in root.iter() if _tag(element) in tags]


def svg_damage(fn: Callable[[ET.Element, int], None]) -> SvgDamage:
    """Turn an in-place edit of the parsed drawing into a damage family."""

    @functools.wraps(fn)
    def wrapper(svg: str, level: int) -> str:
        root = ET.fromstring(svg)
        fn(root, level)
        ET.register_namespace("", ops.SVG_NS)
        return ET.tostring(root, encoding="unicode", method="xml")

    return wrapper


def _growing(items: list, level: int, seed: int, share: float = 0.2) -> list:
    """The first slice of one fixed shuffle, longer at every level.

    Shuffled rather than taken in document order, which would damage one end of
    the drawing first and leave the other end clean until the top level. The
    shuffle is seeded once and reused across levels: draw a fresh order per
    level and each level is a different subset instead of a larger one, which
    is a different picture rather than a worse one.
    """
    order = list(items)
    random.Random(seed).shuffle(order)
    return order[: math.ceil(len(order) * share * level)]


def _confuse(char: str) -> str:
    """A different character of the same kind, so the damage is the string and
    not the layout: digits stay digits and letters keep their case and width."""
    if char.isdigit():
        return str((int(char) + 5) % 10)
    base = ord("a") if char.islower() else ord("A")
    return chr(base + (ord(char) - base + 13) % 26)


@svg_damage
def relabel_text(root: ET.Element, level: int) -> None:
    """Change what a <text> says without changing anything else about it.

    Counted in characters rather than elements because half the corpus draws
    one or two <text> elements, and stepping an element at a time would give
    the family two levels there.
    """
    texts = [element for element in _shapes(root, ("text",)) if element.text]
    positions = [
        (element, index)
        for element in texts
        for index, char in enumerate(element.text or "")
        if char.isalnum()
    ]
    hit: dict[int, set[int]] = {}
    for element, index in _growing(positions, level, seed=8101, share=0.12):
        hit.setdefault(id(element), set()).add(index)
    for element in texts:
        chosen = hit.get(id(element), set())
        element.text = "".join(
            _confuse(char) if index in chosen else char
            for index, char in enumerate(element.text or "")
        )


@svg_damage
def font_size(root: ET.Element, level: int) -> None:
    """Draw the right words at the wrong size.

    Here because it was seen for real: a run swapped the two lines of a wordmark
    between the sizes they were drawn at, 13 and 28, and the panel preferred the
    result. Every sized element scales together, so the drawing stays plausible
    and only its proportions are wrong -- the case a scorer is most likely to
    wave through.
    """
    for element in root.iter():
        size = element.get("font-size")
        if size is not None:
            with contextlib.suppress(ValueError):
                element.set("font-size", f"{float(size) * (1.0 + 0.15 * level):.2f}")


def _rgb(colour: str) -> tuple[float, float, float] | None:
    text = colour.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    if len(text) != 6:
        return None
    with contextlib.suppress(ValueError):
        red, green, blue = (float(int(text[i : i + 2], 16)) for i in (0, 2, 4))
        return red, green, blue
    return None


def _away(colour: str, other: str, amount: float) -> str:
    """*colour*, pushed the same distance again in the direction away from
    *other* -- a flat colour that keeps the hue of the stop it came from."""
    near, far = _rgb(colour), _rgb(other)
    if near is None or far is None:
        return colour
    channels = (
        min(255, max(0, round(a + (a - b) * amount)))
        for a, b in zip(near, far, strict=True)
    )
    return "#" + "".join(f"{value:02x}" for value in channels)


_URL_REF = re.compile(r"url\(#([^)]+)\)")


@svg_damage
def flatten_gradient(root: ET.Element, level: int) -> None:
    """Paint a gradient-filled element in one flat colour from its first stop.

    The corpus references at most two gradients, so flattening one more per
    level would run out after three. What grows instead is how far the flat
    colour sits from the gradient's other end, which keeps the family growing
    once every reference is already flat.
    """
    stops: dict[str, list[str]] = {}
    for gradient in root.iter():
        if _tag(gradient) in ("linearGradient", "radialGradient"):
            colours = [
                stop.get("stop-color", "#000000")
                for stop in gradient
                if _tag(stop) == "stop"
            ]
            identifier = gradient.get("id")
            if colours and identifier:
                stops[identifier] = colours

    for element in root.iter():
        for attribute in ("fill", "stroke"):
            match = _URL_REF.fullmatch(element.get(attribute, "").strip())
            if match is None or match.group(1) not in stops:
                continue
            colours = stops[match.group(1)]
            element.set(attribute, _away(colours[0], colours[-1], 0.2 * level))


# Geometry, which the swap replaces outright. Everything else -- paint,
# transforms, identifiers -- carries over to the new shape.
_GEOMETRY = frozenset({"x", "y", "width", "height", "rx", "ry", "cx", "cy", "r"})


def _bounds(element: ET.Element) -> tuple[float, float, float, float] | None:
    def number(name: str) -> float:
        with contextlib.suppress(ValueError, TypeError):
            return float(element.get(name, "0"))
        return 0.0

    kind = _tag(element)
    if kind == "rect":
        return number("x"), number("y"), number("width"), number("height")
    radius_x = number("r") if kind == "circle" else number("rx")
    radius_y = number("r") if kind == "circle" else number("ry")
    if radius_x <= 0 or radius_y <= 0:
        return None
    return (
        number("cx") - radius_x,
        number("cy") - radius_y,
        2 * radius_x,
        2 * radius_y,
    )


@svg_damage
def shape_swap(root: ET.Element, level: int) -> None:
    """Redraw a round element as its bounding box, and a box as the ellipse
    inside it.

    Position, size and paint all survive, so the picture keeps its colour
    histogram and very nearly its silhouette; only what the elements *are*
    changes. No operator can do this -- they edit attributes and never the tag.
    """
    for element in _growing(_shapes(root, ("rect", "circle", "ellipse")), level, 8202):
        box = _bounds(element)
        if box is None:
            continue
        x, y, width, height = box
        kept = {k: v for k, v in element.attrib.items() if k not in _GEOMETRY}
        element.attrib.clear()
        element.attrib.update(kept)
        if _tag(element) == "rect":
            element.tag = f"{{{ops.SVG_NS}}}ellipse"
            element.set("cx", f"{x + width / 2:.2f}")
            element.set("cy", f"{y + height / 2:.2f}")
            element.set("rx", f"{width / 2:.2f}")
            element.set("ry", f"{height / 2:.2f}")
        else:
            element.tag = f"{{{ops.SVG_NS}}}rect"
            element.set("x", f"{x:.2f}")
            element.set("y", f"{y:.2f}")
            element.set("width", f"{width:.2f}")
            element.set("height", f"{height:.2f}")


@svg_damage
def fade(root: ET.Element, level: int) -> None:
    """Let the background through elements that were meant to be solid.

    Scaled from whatever opacity the element already carried rather than set to
    a fixed value, so an element the drawing deliberately made faint is damaged
    by as much as the solid ones rather than left where it was.
    """
    for element in _growing(_shapes(root), level, 8303, share=0.12):
        current = element.get("fill-opacity") or element.get("opacity") or "1"
        with contextlib.suppress(ValueError):
            element.set("opacity", f"{float(current) * (1.0 - 0.12 * level):.3f}")


@svg_damage
def skew(root: ET.Element, level: int) -> None:
    """Shear elements where they stand.

    The operators write transforms too, but only translations: they move an
    element without bending it, and nothing in the search can put a straight
    edge on a slant.
    """
    for element in _growing(_shapes(root), level, 8404):
        existing = element.get("transform", "").strip()
        shear = f"skewX({3.0 * level:.1f})"
        element.set("transform", f"{shear} {existing}" if existing else shear)


@svg_damage
def dash(root: ET.Element, level: int) -> None:
    """Break solid outlines into dashes, with the gaps growing faster than the
    dashes shrink, so an outline ends up as a row of ticks."""
    for element in root.iter():
        if element.get("stroke", "none") != "none":
            element.set(
                "stroke-dasharray", f"{max(2.0, 12.0 - 2.0 * level)} {3.0 * level}"
            )


VECTOR_FAMILIES: dict[str, SvgDamage] = {
    "relabel-text": relabel_text,
    "font-size": font_size,
    "flatten-gradient": flatten_gradient,
    "shape-swap": shape_swap,
    "opacity": fade,
    "skew": skew,
    "dash": dash,
}


def damage_levels(clean: str, damage: SvgDamage) -> list[str]:
    """Every level of one family, the clean drawing first."""
    return [clean] + [damage(clean, level) for level in range(1, LEVELS + 1)]


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


def mask_patches(image: Image.Image, k: int) -> Image.Image:
    """Occlude scattered patches: information removed rather than corrupted.

    The one damage type where the picture stays perfectly sharp and correct
    everywhere it is still visible, so a scorer reading global statistics can
    miss it entirely.
    """
    rng = np.random.default_rng(41)
    out = image.copy()
    width, height = out.size
    for _ in range(3 * k):
        size = 28
        x = int(rng.integers(0, width - size))
        y = int(rng.integers(0, height - size))
        out.paste((128, 128, 128), (x, y, x + size, y + size))
    return out


def pixelate(image: Image.Image, k: int) -> Image.Image:
    factor = 1 + 2 * k
    small = image.resize(
        (max(1, image.width // factor), max(1, image.height // factor)),
        Image.Resampling.BOX,
    )
    return small.resize(image.size, Image.Resampling.NEAREST)


def quantise_colour(image: Image.Image, k: int) -> Image.Image:
    levels = max(2, 2 ** (8 - k))
    step = 256 / levels
    return _img(np.floor(_np(image) / step) * step)


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
    "mask-patches": mask_patches,
    "pixelate": pixelate,
    "quantise-colour": quantise_colour,
    "sharpen": sharpen,
}


def hf_sources(per_set: int) -> list[tuple[str, str]]:
    """Stream a few usable drawings from each public collection.

    Rejects what cannot be rendered, and what renders to a single flat colour:
    a blank picture is unaffected by most damage, so it would contribute levels
    that never differ and quietly shrink the sample.
    """
    from datasets import load_dataset

    plugin = SvgPlugin()
    out: list[tuple[str, str]] = []
    for name, key in HF_SETS:
        taken = 0
        for row in load_dataset(name, split="train", streaming=True):
            if taken >= per_set:
                break
            svg = str(row[key])
            count = len(re.findall(r"<[a-zA-Z]", svg))
            if not HF_ELEMENT_RANGE[0] <= count <= HF_ELEMENT_RANGE[1]:
                continue
            try:
                render = plugin.rasterize(svg, SIZE, SIZE)
                grey = Image.open(io.BytesIO(render)).convert("L")
            except Exception:
                continue
            low, high = grey.getextrema()
            if low == high:
                continue
            taken += 1
            out.append((f"{name.split('/')[-1]}-{taken}", svg))
        print(f"pulled {taken} drawings from {name}", flush=True)
    return out


MIXED_LEVELS = 10
MIXED_SAMPLES = 6


def mixed_chain(clean: str, seed: int) -> list[str]:
    """A candidate as the search actually builds one: a chain of varied edits.

    Every other family walks one axis -- level 3 of a recolour against level 5
    of the same recolour -- and selection almost never faces that. It faces one
    candidate with a displaced element against another with the wrong colour,
    damaged several ways at once. Measured on single axes every scorer looks
    competent, which says the single-axis question is the easy one.

    Ground truth is the number of edits. From a clean drawing essentially any
    random edit makes it worse, so more edits is worse, and unlike a severity
    dial the axes are mixed exactly as a real chain of mutations mixes them.
    """
    rng = random.Random(seed)
    operators = list(VECTOR_OPERATORS.values()) + list(VECTOR_FAMILIES.values())
    out, current = [clean], clean
    for _ in range(MIXED_LEVELS):
        operator = rng.choice(operators)
        random.seed(rng.randrange(1 << 30))
        with contextlib.suppress(Exception):
            try:
                current = operator(current)
            except TypeError:
                # The parameterised families take a level; the search's own
                # operators take only the drawing.
                current = operator(current, rng.randint(1, 3))
        out.append(current)
    return out


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


def main(per_set: int = 0) -> None:
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

    drawings = [
        (case.name, first_seed(case).read_text(encoding="utf-8"))
        for case in sorted(d for d in CASES.iterdir() if d.is_dir())
    ]
    if per_set:
        drawings += hf_sources(per_set)

    for label, clean in drawings:
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

        for name, damage in VECTOR_FAMILIES.items():
            variants = damage_levels(clean, damage)
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

        print(f"{label} done", flush=True)

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf",
        type=int,
        default=0,
        metavar="N",
        help="also stream N drawings from each public collection (needs `datasets`)",
    )
    main(parser.parse_args().hf)
