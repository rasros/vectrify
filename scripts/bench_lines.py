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
channel below 60, channel spread below 25) black and every other colour white.
The trace's ink adds any neutral stroke paint darker than 90 (its brightest
channel), so a line drawn a shade lighter than the original's ink still counts
as found; its colour error is the mse's. Each is rendered and kept where
darker than 128 (192 for bent truth, since resampling spreads a one-pixel line
over two at half its darkness). The scores:

- line_p / line_r / line_f: precision, recall and F of the ink's centrelines
  (thinned as cel thins), each within 2 px of the other's;
- stroke_r: the share of the true ink centrelines within 2 px of a traced
  stroke of any colour, which tells a line not found from one drawn too
  light to count as ink;
- width: the median ratio of traced ink width to true ink width along the
  true centrelines the trace found;
- edge_f: F of the colour edges of the full drawings, within 2 px;
- mse: mean squared difference of the renderings, in 0-255 RGB;
- paths, points, strokes, and ink_fills (ink drawn as filled shapes).

    uv run python scripts/bench_lines.py --out runs/lines.jsonl
    uv run python scripts/bench_lines.py --inputs clean --renders runs/r
    uv run python scripts/bench_lines.py --set regions=80 --out runs/b.jsonl
    uv run python scripts/bench_lines.py --compare runs/lines.jsonl runs/b.jsonl
    uv run python scripts/bench_lines.py --rescore runs/r --out runs/r.jsonl
    uv run python scripts/bench_lines.py --tidy snap,simplify --out runs/t.jsonl
    uv run python scripts/bench_lines.py --compare runs/lines.jsonl runs/t.jsonl

`--tidy STEPS` also runs Tidy (improve/nodes, the steps a comma list of
snap, simplify, detail and shape) on the trace's `--tidy-paths` largest
paths (20; -1 for all) against the input, as bench_trace does, with
`--nodes` setting its other settings, and scores the tidied trace too: the
row's own scores stay the trace's, and `tidy` holds what Tidy did and, under
`after`, the same scores for the tidied trace. `--compare` then also
prints each run's cases before and after Tidy.

The distortions are seeded, so one run per case compares settings. Keep the
machine cool: `nice -n 19 taskset -c 12-19` with `OMP_NUM_THREADS=2`.

References, from Wikimedia Commons, downloaded once into
~/.cache/vectrify-bench (not in the repository):

- https://commons.wikimedia.org/wiki/Special:FilePath/Wikipe-tan_full_length.svg
- https://commons.wikimedia.org/wiki/Special:FilePath/Wikipe-tan_face.svg
- https://commons.wikimedia.org/wiki/Special:FilePath/Wikipe-tan_sorceress_color.svg
- https://commons.wikimedia.org/wiki/Special:FilePath/Adult_Wikipe-tan.svg

Those four are in the tuning set: settings are chosen on them. `--heldout`
runs the held-out set (HELDOUT) instead, drawings never used for tuning,
so a change tuned on the first is checked once on the second. Its Commons
drawings are downloaded once into ~/.cache/vectrify-bench/heldout:

- https://commons.wikimedia.org/wiki/Special:FilePath/Neko_Wikipe-tan.svg
  (CC BY-SA 3.0; Kasuga, vectorised by Malyszkz)
- https://commons.wikimedia.org/wiki/Special:FilePath/Angry_Wikipe-tan.svg
  (CC BY-SA 3.0; Kasuga, Mikael Häggström, Esby and Antonsusi)

    uv run python scripts/bench_lines.py --heldout --out runs/heldout.jsonl

Both sets also hold drawings made for this bench (svg-*), in
scripts/bench_data/svg, made to be hard: gradient fills (linear and
radial) and soft-edged shading, calligraphic ink drawn as filled shapes
that swell and taper to points, from hairline to very bold in one drawing,
outlines broken into overlapping or gapped strokes, hatching, speed lines,
dense small details, and ink over and against dark navy and gradient
areas. Gradients never use near-black neutral stops, so the ink rule still
picks only the ink. The tuning set has svg-anime-girl (anime cel, full
body), svg-anime-face (anime close-up, navy hair), svg-western-park
(Western TV cartoon) and svg-rubberhose-band (1930s rubber-hose); the
held-out set has svg-anime-runner (anime action, navy jacket, speed
lines), svg-manga-swordsman (manga, bold inking), svg-chibi-kitchen
(chibi, many small objects) and svg-game-mech (flat-shaded game art, navy
panels). The trace and shadow benches use them as raster references too.

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
# Drawings made for the bench, in the repository.
DRAWINGS = Path(__file__).resolve().parent / "bench_data" / "svg"
REFERENCES = (
    "Wikipe-tan_full_length",
    "Wikipe-tan_face",
    "Wikipe-tan_sorceress_color",
    "Adult_Wikipe-tan",
    "svg-anime-girl",
    "svg-anime-face",
    "svg-western-park",
    "svg-rubberhose-band",
)
# Never tune on these.
HELDOUT = (
    "Neko_Wikipe-tan",
    "Angry_Wikipe-tan",
    "svg-anime-runner",
    "svg-manga-swordsman",
    "svg-chibi-kitchen",
    "svg-game-mech",
)
HELDOUT_DIR = CACHE / "heldout"
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
    parser.add_argument("--references", nargs="+")
    parser.add_argument(
        "--heldout",
        action="store_true",
        help="Run the held-out drawings (HELDOUT), not the tuning set",
    )
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
        help="Save each case's input, truth (bent like it) and trace here",
    )
    parser.add_argument(
        "--rescore",
        type=Path,
        metavar="DIR",
        help="Score the traces a run saved with --renders DIR, without tracing",
    )
    parser.add_argument(
        "--tidy",
        metavar="STEPS",
        help="Tidy the trace too, with these steps (a comma list of snap, "
        "simplify, detail and shape), and score it before and after",
    )
    parser.add_argument(
        "--tidy-paths",
        type=int,
        default=20,
        help="How many of the largest paths to Tidy (-1: all)",
    )
    parser.add_argument(
        "--nodes",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a Tidy setting; rounds=N sets the rounds",
    )
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if args.tidy and args.rescore:
        parser.error("--tidy needs the inputs, so it cannot --rescore")
    settings = dict(_setting(item) for item in args.set)
    nodes = dict(_setting(item) for item in args.nodes)
    rows = []
    names = args.references or list(HELDOUT if args.heldout else REFERENCES)
    for name in names:
        truth = reference(name, HELDOUT_DIR if args.heldout else CACHE)
        width, height = size(truth)
        clean = Image.fromarray(render(truth, width, height))
        for kind in args.inputs:
            seconds = image = None
            if args.rescore:
                svg = (args.rescore / f"{name}-{kind}-cel.svg").read_text()
            else:
                image = {"clean": lambda i: i, "noisy": distort, "stretched": stretch}[
                    kind
                ](clean)
                svg, seconds = trace(image, settings)
                if args.renders:
                    args.renders.mkdir(parents=True, exist_ok=True)
                    stem = args.renders / f"{name}-{kind}"
                    image.save(f"{stem}-input.png")
                    Image.fromarray(bent(kind)(np.asarray(clean))).save(
                        f"{stem}-truth.png"
                    )
                    Path(f"{stem}-cel.svg").write_text(svg)
                    Image.fromarray(render(svg, width, height)).save(f"{stem}-cel.png")
            row = {
                "reference": name,
                "input": kind,
                "settings": settings,
                "seconds": None if seconds is None else round(seconds, 1),
                **structure(svg),
                **score(truth, svg, width, height, kind),
            }
            if args.tidy and image is not None:
                tidied, row["tidy"] = tidy(
                    svg, image, args.tidy, args.tidy_paths, nodes
                )
                row["tidy"] |= {
                    "after": {
                        **structure(tidied),
                        **score(truth, tidied, width, height, kind),
                    }
                }
                if args.renders:
                    stem = args.renders / f"{name}-{kind}"
                    Path(f"{stem}-tidy.svg").write_text(tidied)
                    Image.fromarray(render(tidied, width, height)).save(
                        f"{stem}-tidy.png"
                    )
            rows.append(row)
            print(_line(row), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")


def reference(name: str, folder: Path = CACHE) -> str:
    """The SVG source of reference *name*: one of DRAWINGS, or else
    downloaded once into *folder*."""
    if (DRAWINGS / f"{name}.svg").exists():
        return (DRAWINGS / f"{name}.svg").read_text()
    path = folder / f"{name}.svg"
    if not path.exists():
        request = urllib.request.Request(
            COMMONS + f"{name}.svg",
            headers={"User-Agent": "vectrify-bench/1.0 (line accuracy benchmark)"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        folder.mkdir(parents=True, exist_ok=True)
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


def is_ink(rgb, brightest: float = 60) -> bool:
    return max(rgb) < brightest and max(rgb) - min(rgb) < 25


# A trace's lines are its strokes: any neutral stroke paint darker than
# TRACE_INK_MOST is ink, so a line drawn a shade lighter than the original's
# ink still counts as found (its colour error is the mse's). Navy and other
# coloured darks stay out by the spread rule, fills keep the truth's rule,
# and mid-grey shading stays out by the cap.
TRACE_INK_MOST = 90
STROKE = re.compile(r'stroke="#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})"')


def _rgb(code: str) -> tuple[int, ...]:
    code = code if len(code) == 6 else "".join(c * 2 for c in code)
    return tuple(int(code[i : i + 2], 16) for i in (0, 2, 4))


def trace_ink_limit(svg: str) -> float:
    """The brightest channel below which *svg*'s neutral paint is its ink."""
    strokes = [
        max(rgb)
        for rgb in (_rgb(m.group(1)) for m in STROKE.finditer(svg))
        if max(rgb) - min(rgb) < 25 and max(rgb) < TRACE_INK_MOST
    ]
    return max([60.0, *(v + 1 for v in strokes)])


def ink_only(svg: str, brightest: float = 60) -> str:
    """*svg* with its ink paint (by `is_ink` with *brightest*) black and every
    other colour white."""

    def functional(match: re.Match) -> str:
        values = [float(v) for v in match.group(1).replace("%", "").split(",")[:3]]
        return "#000000" if is_ink(values, brightest) else "#ffffff"

    svg = HEX.sub(
        lambda m: "#000000" if is_ink(_rgb(m.group(1)), brightest) else "#ffffff",
        svg,
    )
    svg = re.sub(r"rgba?\(([^)]*)\)", functional, svg)
    for name in NAMED:
        svg = re.sub(rf'([:="\s]){name}([;"\s])', r"\1#ffffff\2", svg)
    return svg


def strokes_only(svg: str) -> str:
    """*svg* with only its stroked paths, each drawn black."""
    svg = re.sub(r"<rect\b[^>]*>", "", svg)
    svg = re.sub(r'<path\b(?![^>]*stroke="#)[^>]*>', "", svg)
    return re.sub(r'stroke="#[0-9a-fA-F]{3,6}"', 'stroke="#000000"', svg)


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


def tidy(
    svg: str, image: Image.Image, steps: str, count: int, nodes: dict
) -> tuple[str, dict]:
    """(*svg* with Tidy run on its *count* largest paths, -1 all, with the
    comma list *steps* and the settings *nodes*, against *image*; what Tidy
    did, as bench_trace's optimize reports it, less the per-path list)."""
    from bench_trace import OPTIMIZE, _steps, optimize

    from vectrify.document import Editor, Selection, export_svg, import_svg

    editor = Editor(import_svg(svg), selection=Selection(whole_document=True))
    report = optimize(editor, image, count, OPTIMIZE | nodes | _steps(steps))
    report.pop("each", None)
    report.pop("each_s", None)
    return export_svg(editor.snapshot.document), report


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


def bent(kind: str) -> Callable[[np.ndarray], np.ndarray]:
    """What input *kind* does to a drawing's geometry, on an RGB array."""
    return {
        "noisy": lambda a: np.asarray(warp(Image.fromarray(a))),
        "stretched": lambda a: np.asarray(stretch(Image.fromarray(a))),
    }.get(kind, lambda a: a)


def score(truth: str, svg: str, width: int, height: int, kind: str) -> dict:
    """The trace *svg* of input *kind* scored against the original *truth*."""
    from vectrify.refine.cel import thin
    from vectrify.refine.colour_regions import distance_transform_edt, nearest_indices

    bend = bent(kind)
    full_truth = bend(render(truth, width, height))
    full_trace = render(svg, width, height)
    # Resampling spreads a one-pixel line over two at half its darkness, so
    # bent truth counts as ink at a lighter level, or its thin lines go.
    ink_truth = bend(render(ink_only(truth), width, height)).min(-1) < (
        128 if kind == "clean" else 192
    )
    ink_trace = render(ink_only(svg, trace_ink_limit(svg)), width, height).min(-1) < 128
    line_truth, line_trace = thin(ink_truth), thin(ink_trace)
    precision = _near(line_trace, line_truth)
    recall = _near(line_truth, line_trace)
    # Found at all: a hairline stroke drawn black renders grey.
    stroked = render(strokes_only(svg), width, height).min(-1) < 192
    stroke_recall = _near(line_truth, thin(stroked))
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
        "stroke_r": round(stroke_recall, 3),
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
    text = (
        f"{row['reference']} [{row['input']}]: "
        f"lines {row['line_p']}/{row['line_r']}/{row['line_f']}, "
        f"strokes found {row.get('stroke_r')}, "
        f"width {row['width']}, edges {row['edge_f']}, mse {row['mse']}, "
        f"{row['paths']} paths, {row['points']} points, {row['strokes']} strokes, "
        f"{row['ink_fills']} ink fills, {row['seconds']} s"
    )
    if "tidy" in row:
        t, after = row["tidy"], row["tidy"]["after"]
        text += (
            f"; tidy {t['paths']} paths ({','.join(t['steps'])}): "
            f"lines F {after['line_f']}, width {after['width']}, "
            f"edges {after['edge_f']}, mse {after['mse']}, "
            f"{after['points']} points, {t['changed']} changed, {t['seconds']} s"
        )
    return text


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
        "stroke_r",
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
    # A run with --tidy: each case before and after Tidy, run by run.
    for label, rows in (("BEFORE", old), ("AFTER", new)):
        tidied = {k: r for k, r in rows.items() if "tidy" in r}
        if not tidied:
            continue
        scored = {k: r["tidy"]["after"] for k, r in tidied.items()}
        keys = sorted(tidied)
        print(f"\n{label} run, before → after Tidy:\n")
        print("| case | " + " | ".join(fields) + " |")
        print("|---|" + "---|" * len(fields))
        for key in keys:
            a, b = tidied[key], scored[key]
            cells = " | ".join(f"{a.get(f)} → {b.get(f)}" for f in fields)
            print(f"| {key[0]} [{key[1]}] | {cells} |")
        means = " | ".join(
            f"{_mean(tidied, keys, f)} → {_mean(scored, keys, f)}" for f in fields
        )
        print(f"| mean | {means} |")


def _mean(rows: dict, keys: list, field: str):
    values = [rows[k].get(field) for k in keys]
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 3) if values else None


if __name__ == "__main__":
    main()
