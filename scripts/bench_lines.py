"""Benchmark the cel tracer's lines against vector originals.

Each reference is an SVG drawing; it is rendered, traced with Generate cel
through the same code the editor uses, and the trace is scored against the
original's own geometry rather than its pixels. Every reference is traced
three ways:

- clean: the rendering as it is;
- noisy: a small rotation and wave warp, blur, noise and JPEG;
- stretched: a smooth random stretch of 0.55-1.45x, so line widths vary.

A distorted input is scored against the original warped the same way.

Truth ink is the original with its near-black, neutral paint (brightest
channel below 60, channel spread below 25) black and every other colour
white; the trace's ink is picked by the same rule. The scores:

- line_p / line_r / line_f: precision, recall and F of the ink's centrelines
  (thinned as cel thins), each within 2 px of the other's;
- width: the median ratio of traced ink width to true ink width along the
  true centrelines the trace found;
- edge_f: F of the colour edges of the full drawings, within 2 px;
- mse: mean squared difference of the renderings, in 0-255 RGB;
- paths, points, strokes, and ink_fills (ink drawn as filled shapes).

    uv run python scripts/bench_lines.py --out runs/lines.jsonl
    uv run python scripts/bench_lines.py --inputs clean --renders runs/r
    uv run python scripts/bench_lines.py --set regions=80 --out runs/b.jsonl
    uv run python scripts/bench_lines.py --compare runs/lines.jsonl runs/b.jsonl

The distortions are seeded, so one run per case compares settings. Keep the
machine cool: `nice -n 19 taskset -c 12-19` with `OMP_NUM_THREADS=2`.

References, from Wikimedia Commons, downloaded once into
~/.cache/vectrify-bench (not in the repository):

- https://commons.wikimedia.org/wiki/Special:FilePath/Wikipe-tan_full_length.svg
- https://commons.wikimedia.org/wiki/Special:FilePath/Wikipe-tan_face.svg
- https://commons.wikimedia.org/wiki/Special:FilePath/Wikipe-tan_sorceress_color.svg
- https://commons.wikimedia.org/wiki/Special:FilePath/Adult_Wikipe-tan.svg

Wikipe-tan is by Kasuga (Kasuga~jawiki) and other Wikimedia contributors;
the files are licensed CC BY-SA (see each file's page on Commons for its
exact version and authors).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageFilter

CACHE = Path.home() / ".cache" / "vectrify-bench"
COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath/"
REFERENCES = (
    "Wikipe-tan_full_length",
    "Wikipe-tan_face",
    "Wikipe-tan_sorceress_color",
    "Adult_Wikipe-tan",
)
INPUTS = ("clean", "noisy", "stretched")
HEIGHT = 1000
# How near, in pixels, a traced line or edge must be to a true one.
TOLERANCE = 2.0
HEX = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
NAMED = (
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "gray",
    "grey",
    "silver",
    "orange",
    "pink",
    "purple",
    "brown",
    "navy",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--references", nargs="+", default=list(REFERENCES))
    parser.add_argument("--inputs", nargs="+", default=list(INPUTS), choices=INPUTS)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a Generate cel setting",
    )
    parser.add_argument("--out", type=Path, help="Write one JSON line per case")
    parser.add_argument(
        "--renders",
        type=Path,
        help="Save each case's input, trace (SVG and PNG) and truth ink here",
    )
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    settings = dict(_setting(item) for item in args.set)
    rows = []
    for name in args.references:
        truth = reference(name)
        width, height = size(truth)
        clean = Image.fromarray(render(truth, width, height))
        for kind in args.inputs:
            image = {"clean": lambda i: i, "noisy": distort, "stretched": stretch}[
                kind
            ](clean)
            svg, seconds = trace(image, settings)
            if args.renders:
                args.renders.mkdir(parents=True, exist_ok=True)
                stem = args.renders / f"{name}-{kind}"
                image.save(f"{stem}-input.png")
                Path(f"{stem}-cel.svg").write_text(svg)
                Image.fromarray(render(svg, width, height)).save(f"{stem}-cel.png")
            row = {
                "reference": name,
                "input": kind,
                "settings": settings,
                "seconds": round(seconds, 1),
                **structure(svg),
                **score(truth, svg, width, height, kind),
            }
            rows.append(row)
            print(_line(row), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")


def reference(name: str) -> str:
    """The SVG source of reference *name*, downloaded once into the cache."""
    path = CACHE / f"{name}.svg"
    if not path.exists():
        request = urllib.request.Request(
            COMMONS + f"{name}.svg",
            headers={"User-Agent": "vectrify-bench/1.0 (line accuracy benchmark)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path.read_text()


def size(svg: str) -> tuple[int, int]:
    """The rendering's size: HEIGHT tall, at the drawing's own aspect."""
    import cairosvg

    png = cairosvg.svg2png(bytestring=svg.encode(), output_height=64)
    width, height = Image.open(io.BytesIO(cast(bytes, png))).size
    return round(HEIGHT * width / height), HEIGHT


def render(svg: str, width: int, height: int) -> np.ndarray:
    import cairosvg

    png = cairosvg.svg2png(
        bytestring=svg.encode(),
        output_width=width,
        output_height=height,
        background_color="white",
    )
    return np.asarray(Image.open(io.BytesIO(cast(bytes, png))).convert("RGB"))


def is_ink(rgb) -> bool:
    return max(rgb) < 60 and max(rgb) - min(rgb) < 25


def _rgb(code: str) -> tuple[int, ...]:
    code = code if len(code) == 6 else "".join(c * 2 for c in code)
    return tuple(int(code[i : i + 2], 16) for i in (0, 2, 4))


def ink_only(svg: str) -> str:
    """*svg* with its ink paint black and every other colour white."""

    def functional(match: re.Match) -> str:
        values = [float(v) for v in match.group(1).replace("%", "").split(",")[:3]]
        return "#000000" if is_ink(values) else "#ffffff"

    svg = HEX.sub(lambda m: "#000000" if is_ink(_rgb(m.group(1))) else "#ffffff", svg)
    svg = re.sub(r"rgba?\(([^)]*)\)", functional, svg)
    for name in NAMED:
        svg = re.sub(rf'([:="\s]){name}([;"\s])', r"\1#ffffff\2", svg)
    return svg


def warp(image: Image.Image) -> Image.Image:
    """The noisy input's geometry: a small rotation and a gentle wave."""
    from scipy.ndimage import map_coordinates

    rotated = image.rotate(1.5, resample=Image.Resampling.BICUBIC, fillcolor="white")
    a = np.asarray(rotated).astype(np.float32)
    h, w, _ = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = 1.5 * np.sin(yy / 37) + 0.8 * np.sin(xx / 53)
    dy = 1.5 * np.sin(xx / 41) + 0.8 * np.cos(yy / 61)
    return _resampled(a, yy + dy, xx + dx, map_coordinates)


def distort(image: Image.Image, seed: int = 7) -> Image.Image:
    """*image* warped, blurred, with noise, and saved as a poor JPEG."""
    rng = np.random.default_rng(seed)
    blurred = warp(image).filter(ImageFilter.GaussianBlur(0.8))
    a = np.asarray(blurred).astype(np.float32)
    a = a + rng.normal(0, 8, a.shape)
    buffer = io.BytesIO()
    Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(buffer, "JPEG", quality=55)
    return Image.open(buffer).convert("RGB")


def stretch(image: Image.Image, seed: int = 11) -> Image.Image:
    """*image* stretched by a smooth random field, 0.55-1.45x locally: sums
    of sinusoids 90-260 px long, so lines come out thicker and thinner."""
    from scipy.ndimage import map_coordinates

    a = np.asarray(image.convert("RGB")).astype(np.float32)
    h, w, _ = a.shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    for _ in range(6):
        for d in (dx, dy):
            length = rng.uniform(90, 260)
            angle = rng.uniform(0, np.pi)
            amplitude = rng.uniform(0.25, 0.45) * length / (2 * np.pi) / 1.2
            phase = rng.uniform(0, 2 * np.pi)
            along = xx * np.cos(angle) + yy * np.sin(angle)
            d += amplitude * np.sin(along * 2 * np.pi / length + phase)
    return _resampled(a, yy + dy, xx + dx, map_coordinates)


def _resampled(a: np.ndarray, ys, xs, map_coordinates) -> Image.Image:
    channels = [
        map_coordinates(a[..., c], [ys, xs], order=1, mode="nearest") for c in range(3)
    ]
    return Image.fromarray(np.stack(channels, -1).clip(0, 255).astype(np.uint8))


def trace(image: Image.Image, settings: dict) -> tuple[str, float]:
    """*image* traced with Generate cel and *settings*, as SVG, and the
    seconds it took."""
    from vectrify.document import Editor, Selection, export_svg, import_svg
    from vectrify.operations import Budget, Job, OperationRequest, Permissions, method

    width, height = image.size
    editor = Editor(
        import_svg(
            f'<svg width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"></svg>'
        ),
        selection=Selection(whole_document=True),
    )
    request = OperationRequest(
        "generate",
        "cel",
        editor.snapshot,
        editor,
        Permissions(geometry=True, structure=True, paint=True),
        settings=settings,
        budget=Budget(),
        reference=image,
    )
    started = time.perf_counter()
    job = Job(method("generate", "cel"), request)
    job.run()
    seconds = time.perf_counter() - started
    job.apply()
    return export_svg(editor.snapshot.document), seconds


def structure(svg: str) -> dict:
    """How heavy the trace is, and how it draws its ink."""
    paths = re.findall(r"<path\b[^>]*>", svg)
    points = 0
    strokes = ink_fills = 0
    for path in paths:
        data = re.search(r'\sd="([^"]*)"', path)
        if data:
            points += len(re.findall(r"[MLCQSTHVAmlcqsthva]", data.group(1)))
        strokes += bool(re.search(r'stroke="(?!none)', path))
        fill = re.search(r'fill="#([0-9a-fA-F]{6})"', path)
        ink_fills += bool(fill and is_ink(_rgb(fill.group(1))))
    return {
        "paths": len(paths),
        "points": points,
        "strokes": strokes,
        "ink_fills": ink_fills,
    }


def score(truth: str, svg: str, width: int, height: int, kind: str) -> dict:
    """The trace *svg* of input *kind* scored against the original *truth*."""
    from vectrify.refine.cel import thin
    from vectrify.refine.colour_regions import distance_transform_edt, nearest_indices

    bend: Callable[[np.ndarray], np.ndarray] = {
        "noisy": lambda a: np.asarray(warp(Image.fromarray(a))),
        "stretched": lambda a: np.asarray(stretch(Image.fromarray(a))),
    }.get(kind, lambda a: a)
    full_truth = bend(render(truth, width, height))
    full_trace = render(svg, width, height)
    ink_truth = bend(render(ink_only(truth), width, height)).min(-1) < 128
    ink_trace = render(ink_only(svg), width, height).min(-1) < 128
    line_truth, line_trace = thin(ink_truth), thin(ink_trace)
    precision = _near(line_trace, line_truth)
    recall = _near(line_truth, line_trace)
    # Widths along the true centrelines the trace found, each against the
    # traced width at the traced centreline nearest it.
    nearest = nearest_indices(~line_trace)
    found = line_truth & (distance_transform_edt(~line_trace) <= TOLERANCE)
    width_truth = 2 * distance_transform_edt(ink_truth)
    width_trace = 2 * distance_transform_edt(ink_trace)
    ratio = width_trace[nearest[0][found], nearest[1][found]] / np.maximum(
        width_truth[found], 1
    )
    edges_truth, edges_trace = _edges(full_truth), _edges(full_trace)
    edge_p = _near(edges_trace, edges_truth)
    edge_r = _near(edges_truth, edges_trace)
    mse = float(((full_truth.astype(float) - full_trace) ** 2).mean())
    return {
        "mse": round(mse, 1),
        "line_p": round(precision, 3),
        "line_r": round(recall, 3),
        "line_f": round(_f(precision, recall), 3),
        "width": round(float(np.median(ratio)), 2) if ratio.size else None,
        "edge_p": round(edge_p, 3),
        "edge_r": round(edge_r, 3),
        "edge_f": round(_f(edge_p, edge_r), 3),
    }


def _near(a: np.ndarray, b: np.ndarray) -> float:
    """The share of *a*'s pixels within TOLERANCE of one of *b*'s."""
    from vectrify.refine.colour_regions import distance_transform_edt

    if not a.any():
        return float("nan")
    if not b.any():
        return 0.0
    return float((distance_transform_edt(~b)[a] <= TOLERANCE).mean())


def _f(precision: float, recall: float) -> float:
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


def _edges(rgb: np.ndarray) -> np.ndarray:
    """Where neighbouring pixels differ by more than 40 in a channel, thinned."""
    from vectrify.refine.cel import thin

    a = rgb.astype(np.int16)
    edges = np.zeros(a.shape[:2], bool)
    edges[:, :-1] |= np.abs(a[:, 1:] - a[:, :-1]).max(-1) > 40
    edges[:-1] |= np.abs(a[1:] - a[:-1]).max(-1) > 40
    return thin(edges)


def _setting(item: str) -> tuple[str, object]:
    key, _, value = item.partition("=")
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return key, lowered == "true"
    if re.fullmatch(r"-?\d+", value):
        return key, int(value)
    try:
        return key, float(value)
    except ValueError:
        return key, value


def _line(row: dict) -> str:
    return (
        f"{row['reference']} [{row['input']}]: "
        f"lines {row['line_p']}/{row['line_r']}/{row['line_f']}, "
        f"width {row['width']}, edges {row['edge_f']}, mse {row['mse']}, "
        f"{row['paths']} paths, {row['points']} points, {row['strokes']} strokes, "
        f"{row['ink_fills']} ink fills, {row['seconds']} s"
    )


def compare(before: Path, after: Path) -> None:
    """Print the two runs' cases side by side, paired by reference and input."""

    def load(path: Path) -> dict[tuple[str, str], dict]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        return {(r["reference"], r["input"]): r for r in rows}

    old, new = load(before), load(after)
    fields = (
        "line_p",
        "line_r",
        "line_f",
        "width",
        "edge_f",
        "mse",
        "paths",
        "points",
        "strokes",
        "ink_fills",
    )
    print("| case | " + " | ".join(fields) + " |")
    print("|---|" + "---|" * len(fields))
    for key in sorted(old.keys() & new.keys()):
        a, b = old[key], new[key]
        cells = " | ".join(f"{a.get(f)} → {b.get(f)}" for f in fields)
        print(f"| {key[0]} [{key[1]}] | {cells} |")
    shared = sorted(old.keys() & new.keys())
    if shared:
        means = " | ".join(
            f"{_mean(old, shared, f)} → {_mean(new, shared, f)}" for f in fields
        )
        print(f"| mean | {means} |")


def _mean(rows: dict, keys: list, field: str):
    values = [rows[k].get(field) for k in keys]
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 3) if values else None


if __name__ == "__main__":
    main()
