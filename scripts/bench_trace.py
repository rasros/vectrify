"""Benchmark tracing and Tidy (improve/nodes) on fixed references.

For each reference and preset, this runs a Generate method (cel unless
`--method` names another) through the same code the editor uses, then
Tidy on the largest traced paths, and records how close the
result is to the reference, how heavy it is and how long it took.

    uv run python scripts/bench_trace.py --out runs/base.jsonl
    uv run python scripts/bench_trace.py --set regions=200 --out runs/b.jsonl
    uv run python scripts/bench_trace.py --compare runs/base.jsonl runs/b.jsonl
    uv run python scripts/bench_trace.py --method cel --paths 0 --out runs/cel.jsonl
    uv run python scripts/bench_trace.py --nodes shape=true --nodes seconds=60
    uv run python scripts/bench_trace.py --method cel --paths 0 --heldout
    uv run python scripts/bench_trace.py --photos --paths 0 --renders runs/p

The default references are the tuning set: settings are chosen on them.
`--heldout` runs the held-out set instead, images never used for tuning,
so a change tuned on the first is checked once on the second. Each set
holds:

- generated cartoon images (gen-*.png) in several styles, anime cel most
  of all, then Western TV cartoon, manga with bold inking, children's-book
  flat vector, 1930s rubber-hose, chibi and flat-shaded game art. They are
  a fixed dataset, not in the repository but in
  ~/.cache/vectrify-bench/generated; scripts/bench_data/generated.json
  records each one's set, style and the prompt it was made from. Missing
  ones are left out with a note.
- the line bench's own drawings (svg-*, see bench_lines), rendered
  1000 px tall.

`--photos` runs a set of photographs instead of the cartoons, the tuning
set's six or with `--heldout` the held-out set's six: openly licensed
photos from Wikimedia Commons (a portrait, animals, landscapes, food, a
night street, an interior), downscaled to 1600 px on the long side and
kept outside the repository in ~/.cache/vectrify-bench/photos;
scripts/bench_data/photos.json records each one's set, source, licence
and author. Photos have no vector truth, so only this bench's pixel error
applies to them, not the line bench's scores.

The error is the mean squared difference to the reference in 0-255 RGB.
Some references also have small facial features marked (FEATURES: eyes
and mouths, boxes in the reference's pixels), and `features` is the same
error over just those boxes, which a whole-image error hardly notices.
`--repeat` runs each case more than once, to judge timings. Keep the machine
cool: `nice -n 19 taskset -c 12-19` with `OMP_NUM_THREADS=2`.

Tidy as a touch-up after the trace:

    uv run python scripts/bench_trace.py --tidy --out runs/tidy.jsonl
    uv run python scripts/bench_trace.py --tidy --tidy-steps snap,simplify \\
        --tidy-steps shape --tidy-crops runs/crops --out runs/shape.jsonl
    uv run python scripts/bench_trace.py --summary runs/tidy.jsonl runs/shape.jsonl

`--paths N` tidies the N largest traced paths (by painted area) one at a
time, each kept before the next is tidied, as when touching up a trace by
hand. `--tidy` makes that the point of the run: it tidies the 20 largest
unless `--paths` says otherwise, or every path with `--tidy-all`. Each
`--tidy-steps` (a comma list of snap, simplify, detail and shape; detail is
Snap adding points, so it needs snap) tidies the same trace once more, a row
each marked `tidy_steps`, so configurations compare on one trace; without
it Tidy runs its own defaults, snap and simplify. `--nodes` sets its other
settings. The row's `optimize` field then also holds, before and after Tidy:

- whole: the whole image's error, both drawings rendered the same way;
- local: the error over the tidied paths' area only (their painted areas
  before and after, widened by 2 px), where Tidy acts and which the whole
  image's error dilutes; `features` too, when the reference has any;
- points in the tidied paths, Tidy's seconds per path (each_s), and how many
  paths it changed, left unchanged (no step helped), refused (the job
  failed) and stopped at its time limit (out_of_time);
- crossings: the tidied paths' self-crossings (refine.crossings) summed
  before and after, and crossed: how many paths cross themselves more;
- gaps and overlaps: pixels where Tidy acted that no fill covers (the
  drawing beneath shows through) or two fills cover, before and after;
- each: all of that per path, its local error over its own area included.

`--trace-cache DIR` keeps each trace in DIR under a hash of the image, the
method, its settings and the source of the code the method imports, and
reuses it while they stay the same, so Tidy is benched on one trace without
tracing again. `--tidy-crops DIR` saves, per case, the reference, the trace
and the tidied trace side by side around the paths Tidy helped and hurt
most. `--summary
RUN...` prints a line per Tidy configuration over the runs' rows, with sign
tests of the local error across images and across paths.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

from PIL import Image

from vectrify.image_utils import on_white

# Generate settings on top of each method's defaults, by method and preset.
PRESETS: dict[str, dict[str, dict]] = {
    "cel": {
        "defaults": {},
        "regions-100": {"regions": 100},
        "regions-200": {"regions": 200},
        "filled-lines": {"strokes": False},
    },
    "colour-regions": {
        "defaults": {},
        "clean-outlines": {"preserve_outlines": True, "outline_style": "clean"},
    },
}
# Small facial features of some references, as (name, x, y, width, height)
# boxes in the reference's pixels: the eyes and mouths a trace must keep.
FEATURES: dict[str, tuple[tuple[str, int, int, int, int], ...]] = {
    "gen-anime-knight.png": (
        ("left eye", 440, 166, 36, 36),
        ("right eye", 508, 152, 52, 34),
    ),
    "gen-anime-closeup.png": (
        ("left eye", 273, 440, 127, 133),
        ("right eye", 553, 520, 194, 140),
        ("mouth", 367, 753, 53, 40),
    ),
    "gen-chibi-picnic.png": (
        ("boy left eye", 553, 440, 60, 73),
        ("boy right eye", 653, 440, 60, 73),
        ("boy mouth", 607, 507, 46, 40),
        ("girl left eye", 853, 440, 54, 73),
        ("girl right eye", 947, 440, 53, 73),
        ("girl mouth", 896, 507, 37, 33),
    ),
}
# Optimize nodes runs with its own defaults, a quick tidy, on one worker;
# `--nodes` overrides them.
OPTIMIZE = {"workers": 1}
CACHE = Path.home() / ".cache" / "vectrify-bench"
GENERATED_DIR = CACHE / "generated"
GENERATED_DATA = Path(__file__).resolve().parent / "bench_data" / "generated.json"
PHOTOS_DIR = CACHE / "photos"
PHOTOS_DATA = Path(__file__).resolve().parent / "bench_data" / "photos.json"


def generated(heldout: bool = False) -> list[Path]:
    """The generated images of the tuning set, or with *heldout* the
    held-out set, that are on disk."""
    return _listed(GENERATED_DATA, "images", GENERATED_DIR, heldout, "generated images")


def photos(heldout: bool = False) -> list[Path]:
    """The photographs of the tuning set, or with *heldout* the held-out
    set, that are on disk."""
    return _listed(PHOTOS_DATA, "photos", PHOTOS_DIR, heldout, "photos")


def _listed(data: Path, key: str, folder: Path, heldout: bool, kind: str) -> list[Path]:
    """The files *data* lists under *key* in the tuning set, or with
    *heldout* the held-out set, that are in *folder*."""
    items = json.loads(data.read_text())[key]
    wanted = "heldout" if heldout else "tuning"
    paths = [folder / i["file"] for i in items if i["set"] == wanted]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(
            f"left out {len(missing)} {kind} not in {folder}: " + ", ".join(missing),
            flush=True,
        )
    return [p for p in paths if p.exists()]


def references(heldout: bool = False, photographs: bool = False) -> list[Path]:
    """The tuning set's references, or with *heldout* the held-out set's:
    the generated images, then the line bench's own drawings; with
    *photographs*, the set's photos instead."""
    from bench_lines import DRAWINGS, HELDOUT, REFERENCES

    if photographs:
        return photos(heldout)
    names = HELDOUT if heldout else REFERENCES
    drawings = [DRAWINGS / f"{n}.svg" for n in names if n.startswith("svg-")]
    return generated(heldout) + drawings


def load(path: Path) -> Image.Image:
    """Reference *path* as the editor shows it, transparency over white (not
    black); an SVG drawing rendered as the line bench renders it."""
    if path.suffix == ".svg":
        from bench_lines import render, size

        svg = path.read_text()
        return Image.fromarray(render(svg, *size(svg)))
    return on_white(Image.open(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--references", nargs="+", type=Path)
    parser.add_argument(
        "--heldout",
        action="store_true",
        help="Run the held-out references, not the tuning set",
    )
    parser.add_argument(
        "--photos",
        action="store_true",
        help="Run the set's photographs, not its cartoons",
    )
    parser.add_argument("--method", default="cel", choices=sorted(PRESETS))
    parser.add_argument("--preset", nargs="+", help="Presets of the method to run")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a Generate setting for every preset",
    )
    parser.add_argument(
        "--paths",
        type=int,
        help="Largest paths to Tidy (0: none; 3, or 20 with --tidy)",
    )
    parser.add_argument(
        "--tidy",
        action="store_true",
        help="Tidy the trace as a touch-up, the 20 largest paths unless --paths",
    )
    parser.add_argument("--tidy-all", action="store_true", help="Tidy every path")
    parser.add_argument(
        "--tidy-steps",
        action="append",
        metavar="STEPS",
        help="Tidy's steps, a comma list of snap, simplify, detail and shape; "
        "repeat it to tidy the same trace with each",
    )
    parser.add_argument(
        "--tidy-crops",
        type=Path,
        metavar="DIR",
        help="Save before/after crops of the paths Tidy changed most here",
    )
    parser.add_argument(
        "--nodes",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an Optimize nodes setting; rounds=N sets the rounds",
    )
    parser.add_argument(
        "--trace-cache",
        type=Path,
        metavar="DIR",
        help="Keep each trace here, keyed by its image, settings and tracing "
        "code, and reuse it: Tidy configurations then bench one trace",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", type=Path, help="Write one JSON line per case")
    parser.add_argument(
        "--renders", type=Path, help="Save each case's traced drawing here as PNG"
    )
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    parser.add_argument(
        "--summary",
        nargs="+",
        type=Path,
        metavar="RUN",
        help="Summarise the Tidy configurations of these runs",
    )
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if args.summary:
        summary(args.summary)
        return
    try:
        configs = [_steps(item) for item in args.tidy_steps or ()] or [{}]
    except ValueError as exc:
        parser.error(str(exc))
    count = args.paths if args.paths is not None else (20 if args.tidy else 3)
    if args.tidy_all:
        count = -1
    presets = PRESETS[args.method]
    unknown = set(args.preset or ()) - set(presets)
    if unknown:
        parser.error(
            f"no {args.method} preset {', '.join(sorted(unknown))}; "
            f"choose from {', '.join(sorted(presets))}"
        )
    overrides = dict(_setting(item) for item in args.set)
    nodes = OPTIMIZE | dict(_setting(item) for item in args.nodes)
    rows = []
    for path in args.references or references(args.heldout, args.photos):
        image = load(path)
        for preset in args.preset or list(presets):
            settings = {**presets[preset], **overrides}
            features = FEATURES.get(path.name, ())
            name = f"{path.stem}-{args.method}-{preset}"
            for _ in range(args.repeat):
                traced, editor = generate(
                    image,
                    args.method,
                    settings,
                    features=features,
                    render=args.renders / f"{name}.png" if args.renders else None,
                    cache=args.trace_cache,
                )
                case = {
                    "reference": path.name,
                    "method": args.method,
                    "preset": preset,
                    "settings": settings,
                    **traced,
                }
                if not count or editor is None:
                    rows.append(case)
                    print(_line(case), flush=True)
                    continue
                for config in configs:
                    row = dict(case)
                    if config:
                        row["tidy_steps"] = _label(config)
                    crops = None
                    if args.tidy_crops:
                        label = (
                            "-" + _label(config).replace(",", "+").replace(" ", "-")
                            if config
                            else ""
                        )
                        crops = args.tidy_crops / f"{name}{label}"
                    row["optimize"] = optimize(
                        _copy(editor),
                        image,
                        count,
                        nodes | config,
                        features=features,
                        crops=crops,
                    )
                    rows.append(row)
                    print(_line(row), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")


def trace(
    image: Image.Image,
    name: str,
    settings: dict,
    paths: int,
    *,
    nodes: dict | None = None,
    features: tuple[tuple[str, int, int, int, int], ...] = (),
    render: Path | None = None,
) -> dict:
    """Generate from *image* with method *name* and *settings*, then Tidy
    its *paths* largest paths (-1: all) with the settings *nodes*. With
    *render*, the traced drawing is saved there."""
    row, editor = generate(image, name, settings, features=features, render=render)
    if paths and editor is not None:
        row["optimize"] = optimize(
            editor, image, paths, nodes or OPTIMIZE, features=features
        )
    return row


def generate(
    image: Image.Image,
    name: str,
    settings: dict,
    *,
    features: tuple[tuple[str, int, int, int, int], ...] = (),
    render: Path | None = None,
    cache: Path | None = None,
):
    """(the row, the editor holding the trace) for Generate with method
    *name* and *settings* on *image*; the editor is None when it failed.
    With *render*, the traced drawing is saved there. With the folder
    *cache*, the trace is kept there under a key of the image, the method,
    its settings and the code it runs (`trace_key`), and read back from
    there when the key matches, so Tidy configurations bench one trace."""
    from vectrify.document import Editor, Selection, export_svg, import_svg
    from vectrify.operations import Budget, Job, OperationRequest, Permissions, method

    if cache is not None:
        cache = cache / trace_key(image, name, settings)
        svg, saved = cache.with_suffix(".svg"), cache.with_suffix(".json")
        if svg.exists() and saved.exists() and not render:
            return json.loads(saved.read_text()), Editor(
                import_svg(svg.read_text()),
                selection=Selection(whole_document=True),
            )
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
        name,
        editor.snapshot,
        editor,
        Permissions(geometry=True, structure=True, paint=True),
        settings=settings,
        budget=Budget(),
        reference=image,
    )
    started = time.perf_counter()
    job = Job(method("generate", name), request)
    job.run()
    total = time.perf_counter() - started
    state = job.state()
    if "result" not in state:
        return {"failed": state.get("error", state.get("status"))}, None
    metrics = state["result"]["metrics"]
    job.apply()
    document = editor.snapshot.document
    if render:
        import cairosvg

        render.parent.mkdir(parents=True, exist_ok=True)
        cairosvg.svg2png(
            bytestring=export_svg(document).encode(),
            write_to=str(render),
            output_width=width,
            output_height=height,
            background_color="white",
        )
    feature_error = None
    if features:
        rendered = _render(document, width, height)
        feature_error = _feature_error(rendered, image, features)
    data = " ".join(
        e.get("d") or "" for e in _parse(export_svg(document)) if e.tag.endswith("path")
    )
    drawn = [e for e in document.elements() if e.tag == "path"]
    row = {
        "error": round(metrics["after"]["error"] * 255**2, 2),
        "paths": len(drawn),
        "curves": data.count("C"),
        # Every node, the moves that start each contour included.
        "points": sum(
            len(s.nodes) for e in drawn for s in document.geometry_for(e.id).subpaths
        ),
        # Seams Generate snapped together.
        "snapped": metrics.get("snapped", 0),
        "total_s": round(total, 1),
    }
    if feature_error is not None:
        row["features"] = feature_error
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.with_suffix(".svg").write_text(export_svg(document))
        cache.with_suffix(".json").write_text(json.dumps(row))
    return row, editor


def trace_key(image: Image.Image, name: str, settings: dict) -> str:
    """A key for the trace of *image* by method *name* with *settings*: a
    hash of the image's pixels, the method and settings, and the source of
    every vectrify module its method and tracer import, so a change to the
    tracing code traces again while one to Tidy does not."""
    import ast
    import hashlib

    import vectrify

    root = Path(vectrify.__file__).parent
    pending = [
        root / "operations" / "methods" / f"{name.replace('-', '_')}.py",
        root / "refine" / f"{name.replace('-', '_')}.py",
    ]
    seen: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in seen or not source.exists():
            continue
        seen.add(source)
        for node in ast.walk(ast.parse(source.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module] + [
                    f"{node.module}.{a.name}" for a in node.names
                ]
            elif isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            for module in modules:
                if module.startswith("vectrify."):
                    parts = module.split(".")[1:]
                    pending += [
                        root.joinpath(*parts).with_suffix(".py"),
                        root.joinpath(*parts, "__init__.py"),
                    ]
    digest = hashlib.sha256()
    rgb = image.convert("RGB")
    digest.update(f"{rgb.size}".encode())
    digest.update(rgb.tobytes())
    digest.update(json.dumps([name, settings], sort_keys=True).encode())
    for source in sorted(seen):
        digest.update(source.relative_to(root).as_posix().encode())
        digest.update(source.read_bytes())
    return f"{name}-{digest.hexdigest()[:16]}"


def optimize(
    editor,
    image: Image.Image,
    count: int,
    settings: dict,
    *,
    features: tuple[tuple[str, int, int, int, int], ...] = (),
    crops: Path | None = None,
) -> dict:
    """Tidy (Optimize nodes) each of the *count* largest paths (-1: all), one
    at a time, each kept before the next, with *settings*, where `rounds` is
    the budget's steps; and how the drawing changed. With *crops*, crops of
    the paths it changed most are saved there."""
    import numpy as np

    from vectrify.document import Selection
    from vectrify.document.hit_test import HitIndex
    from vectrify.operations import Budget, Job, OperationRequest, Permissions, method
    from vectrify.refine.crossings import crossings

    start = editor.snapshot.document
    index = HitIndex(start)

    def painted(oid: str) -> float:
        area = index.area(oid)
        return float(area.area) if area is not None else 0.0

    largest = sorted(
        (e.id for e in start.elements() if e.tag == "path"),
        key=lambda oid: -painted(oid),
    )
    if count >= 0:
        largest = largest[:count]
    before = after = seconds = 0.0
    points_before = points_after = 0
    each: list[float] = []
    # Paths whose run the time limit ended.
    late = 0
    paths: dict[str, dict] = {}
    for oid in largest:
        editor.select(Selection(object_ids=frozenset({oid})))
        request = OperationRequest(
            "improve",
            "nodes",
            editor.snapshot,
            editor,
            Permissions(geometry=True, structure=True, paint=True),
            settings={k: v for k, v in settings.items() if k != "rounds"},
            budget=Budget(steps=settings.get("rounds")),
            reference=image,
        )
        started = time.perf_counter()
        job = Job(method("improve", "nodes"), request)
        job.run()
        spent = time.perf_counter() - started
        seconds += spent
        each.append(round(spent, 1))
        state = job.state()
        result = state.get("result")
        record = paths[oid] = {"id": oid, "s": round(spent, 2)}
        if result is None:
            record["status"] = "refused"
            record["why"] = state.get("error") or state.get("status")
            continue
        metrics = result["metrics"]
        before += metrics["before"]["difference"]
        after += metrics["after"]["difference"]
        points_before += metrics["before"]["nodes"]
        points_after += metrics["after"]["nodes"]
        late += bool(metrics.get("out_of_time"))
        record |= {
            "status": "changed" if result["changed"] else "unchanged",
            "points": [metrics["before"]["nodes"], metrics["after"]["nodes"]],
            "steps": metrics.get("steps", []),
            "out_of_time": bool(metrics.get("out_of_time")),
        }
        if result["changed"]:
            job.apply()
    end = editor.snapshot.document
    width, height = image.size
    first, last = _render(start, width, height), _render(end, width, height)
    truth = np.asarray(image.convert("RGB"), float)
    off_first = ((first - truth) ** 2).mean(-1)
    off_last = ((last - truth) ** 2).mean(-1)
    end_index = HitIndex(end)
    union = np.zeros((height, width), bool)
    traced = np.zeros((height, width), bool)
    for oid, record in paths.items():
        mask = _area_mask((index.area(oid), end_index.area(oid)), width, height, 2)
        own = _area_mask((index.area(oid),), width, height, 4)
        union |= mask
        traced |= own
        record |= {
            "area": round(painted(oid)),
            "local": _masked(off_first, off_last, mask),
            # Over the traced path's own area only, widened by 4 px: the
            # same pixels whatever Tidy did, so configurations compare.
            "trace": _masked(off_first, off_last, own),
            "crossings": [crossings(d.geometry_for(oid)) for d in (start, end)],
        }
    statuses = [r["status"] for r in paths.values()]
    # Seams between fills where Tidy acted, before and after: pixels no fill
    # covers (a gap the drawing beneath shows through) and pixels two fills
    # cover (an overlap).
    seams = [_seams(_fill_cover(d, width, height), union) for d in (start, end)]
    row = {
        "settings": settings,
        "paths": len(largest),
        # Each path's own region, summed: the share left says how much it fixed.
        "error_left": round(after / before, 3) if before else None,
        "points": [points_before, points_after],
        "seconds": round(seconds, 1),
        "each_s": each,
        "out_of_time": late,
        "steps": [s for s in TIDY_DEFAULTS if settings.get(s, TIDY_DEFAULTS[s])],
        "whole": [round(float(off_first.mean()), 2), round(float(off_last.mean()), 2)],
        "local": _masked(off_first, off_last, union),
        "local_pixels": int(union.sum()),
        "trace_local": _masked(off_first, off_last, traced),
        "changed": statuses.count("changed"),
        "unchanged": statuses.count("unchanged"),
        "refused": statuses.count("refused"),
        "crossings": [
            sum(r["crossings"][0] for r in paths.values()),
            sum(r["crossings"][1] for r in paths.values()),
        ],
        "crossed": sum(r["crossings"][1] > r["crossings"][0] for r in paths.values()),
        "gaps": [seams[0][0], seams[1][0]],
        "overlaps": [seams[0][1], seams[1][1]],
        "each": list(paths.values()),
    }
    if features:
        row["features"] = [
            _feature_error(first, image, features),
            _feature_error(last, image, features),
        ]
    if crops is not None:
        _save_crops(crops, truth, first, last, paths, (index, end_index))
    return row


# Tidy's steps and which are on by default, as operations/methods/nodes.py
# has them; detail is Snap adding points.
TIDY_DEFAULTS = {"snap": True, "simplify": True, "detail": False, "shape": False}


def _steps(item: str) -> dict:
    """Tidy's settings for *item*: a comma list of its steps, then any other
    of its settings as KEY=VALUE, apart by spaces."""
    first, *rest = item.split()
    chosen = {s.strip() for s in first.split(",") if s.strip()}
    if not chosen or chosen - set(TIDY_DEFAULTS):
        raise ValueError(f"--tidy-steps {item}: choose from {', '.join(TIDY_DEFAULTS)}")
    if "detail" in chosen and "snap" not in chosen:
        raise ValueError(f"--tidy-steps {item}: detail is Snap's, so add snap")
    return {step: step in chosen for step in TIDY_DEFAULTS} | dict(
        _setting(other) for other in rest
    )


def _label(config: dict) -> str:
    """Tidy's steps and other settings in *config*, as `--tidy-steps`
    takes them."""
    steps = ",".join(s for s in TIDY_DEFAULTS if config.get(s))
    others = [f"{k}={v}" for k, v in config.items() if k not in TIDY_DEFAULTS]
    return " ".join([steps, *others])


def _copy(editor):
    """A new editor on *editor*'s drawing, so each Tidy configuration
    starts from the same trace."""
    from vectrify.document import Editor, Selection

    return Editor(editor.snapshot.document, selection=Selection(whole_document=True))


def _render(document, width: int, height: int):
    """*document* rendered over white at *width* x *height*, as float RGB."""
    import io

    import cairosvg
    import numpy as np

    from vectrify.document import export_svg

    png = cairosvg.svg2png(
        bytestring=export_svg(document).encode(),
        output_width=width,
        output_height=height,
        background_color="white",
    )
    assert png is not None
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), float)


def _fill_cover(document, width: int, height: int):
    """How many fills cover each pixel of *document*, roughly: its filled
    paths drawn black at half opacity, without strokes or basic shapes (a
    cel trace's backing rectangle), over white, as 0-1 darkness."""
    import io
    import xml.etree.ElementTree as ET

    import cairosvg
    import numpy as np

    from vectrify.document import export_svg

    root = ET.fromstring(export_svg(document))
    for parent in list(root.iter()):
        for child in list(parent):
            tag = child.tag.rsplit("}", 1)[-1]
            if tag in {"rect", "circle", "ellipse", "line", "polyline", "polygon"}:
                parent.remove(child)
            elif tag == "path":
                if child.get("fill", "black") == "none":
                    parent.remove(child)
                    continue
                for name in ("stroke", "stroke-width", "opacity", "fill-opacity"):
                    child.attrib.pop(name, None)
                child.set("fill", "#000000")
                child.set("fill-opacity", "0.5")
                child.set("stroke", "none")
    png = cairosvg.svg2png(
        bytestring=ET.tostring(root),
        output_width=width,
        output_height=height,
        background_color="white",
    )
    assert png is not None
    grey = np.asarray(Image.open(io.BytesIO(png)).convert("L"), float) / 255
    return 1 - grey


def _seams(cover, mask) -> list[int]:
    """[gap pixels, overlap pixels] within *mask* of the fill cover *cover*:
    almost none of it, or two fills' worth (over two thirds)."""
    return [int((cover[mask] < 0.1).sum()), int((cover[mask] > 0.67).sum())]


def _feature_error(rendered, image: Image.Image, features) -> float:
    """The error of the render *rendered* over the *features* boxes."""
    import numpy as np

    truth = np.asarray(image.convert("RGB"), float)
    squared = [
        ((rendered - truth)[y : y + h, x : x + w] ** 2).ravel()
        for _, x, y, w, h in features
    ]
    return round(float(np.concatenate(squared).mean()), 2)


def _masked(first, last, mask) -> list:
    """The per-pixel errors *first* and *last*, each averaged over *mask*."""
    if not mask.any():
        return [None, None]
    return [round(float(first[mask].mean()), 2), round(float(last[mask].mean()), 2)]


def _area_mask(shapes, width: int, height: int, widen: int = 0):
    """The pixels that any of *shapes* (shapely areas in pixels, or None)
    covers, widened by *widen* pixels."""
    import numpy as np
    import shapely
    from PIL import ImageDraw
    from scipy import ndimage

    mask = np.zeros((height, width), bool)
    for shape in shapes:
        if shape is None or shape.is_empty:
            continue
        for part in shapely.get_parts(shape):
            if part.geom_type != "Polygon" or part.is_empty:
                continue
            # Each polygon on its own canvas over its bounds, its holes cut
            # out, so one polygon's hole never erases another inside it.
            x0, y0, x1, y1 = part.bounds
            left, top = max(0, int(x0)), max(0, int(y0))
            right, bottom = min(width, int(x1) + 2), min(height, int(y1) + 2)
            if right <= left or bottom <= top:
                continue
            canvas = Image.new("1", (right - left, bottom - top), 0)
            draw = ImageDraw.Draw(canvas)

            def moved(coords, left=left, top=top):
                return [(x - left, y - top) for x, y in coords]

            draw.polygon(moved(part.exterior.coords), fill=1, outline=1)
            for hole in part.interiors:
                draw.polygon(moved(hole.coords), fill=0)
            mask[top:bottom, left:right] |= np.asarray(canvas, bool)
    if widen and mask.any():
        mask = ndimage.binary_dilation(mask, iterations=widen)
    return mask


def _save_crops(folder: Path, truth, first, last, paths: dict, indexes) -> None:
    """The reference, the trace and the tidied trace side by side, at twice
    the size, around the two paths whose local error Tidy lowered most and
    the two it raised most, named by how much: over the CROP pixels square
    of each path's area where the error changed most that way."""
    import numpy as np
    from scipy import ndimage

    off = ((last - truth) ** 2).mean(-1) - ((first - truth) ** 2).mean(-1)
    changed = sorted(
        (r["local"][1] - r["local"][0], oid)
        for oid, r in paths.items()
        if r["status"] == "changed" and r["local"][0] is not None
    )
    chosen = [("better", d, o) for d, o in changed[:2] if d < 0]
    chosen += [("worse", d, o) for d, o in changed[::-1][:2] if d > 0]
    if not chosen:
        return
    folder.mkdir(parents=True, exist_ok=True)
    height, width = truth.shape[:2]
    for kind, change, oid in chosen:
        mask = _area_mask([i.area(oid) for i in indexes], width, height, 2)
        # The CROP square, within the image, whose summed change is most
        # negative (better) or positive (worse).
        signed = np.where(mask, off if kind == "worse" else -off, 0.0)
        summed = ndimage.uniform_filter(signed, CROP, mode="constant")
        y, x = np.unravel_index(int(np.argmax(summed)), summed.shape)
        left = int(min(max(0, x - CROP // 2), max(0, width - CROP)))
        top = int(min(max(0, y - CROP // 2), max(0, height - CROP)))
        window = (slice(top, top + CROP), slice(left, left + CROP))
        parts = [a[window] for a in (truth, first, last)]
        gap = np.full((parts[0].shape[0], 4, 3), 255.0)
        strip = np.concatenate([parts[0], gap, parts[1], gap, parts[2]], axis=1)
        image = Image.fromarray(strip.astype(np.uint8))
        image = image.resize(
            (image.width * 2, image.height * 2), Image.Resampling.NEAREST
        )
        image.save(folder / f"{kind}-{oid}-{change:+.0f}-at-{left},{top}.png")


# The side of a Tidy crop, in reference pixels.
CROP = 160


def _parse(svg: str):
    import xml.etree.ElementTree as ET

    return ET.fromstring(svg).iter()


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
    case = f"{row['reference']} [{_case(row)}]"
    if "failed" in row:
        return f"{case}: failed: {row['failed']}"
    text = (
        f"{case}: error {row['error']}, "
        f"{row['paths']} paths, {row.get('points')} points, {row['curves']} curves, "
        f"{row['snapped']} snapped, "
        + (f"features {row['features']}, " if "features" in row else "")
        + f"{row['total_s']} s"
    )
    if "optimize" in row:
        o = row["optimize"]
        text += (
            f"; tidy {o['paths']} paths ({','.join(o.get('steps', []))}): "
            f"{o['error_left']} of the error left, "
            f"points {o['points'][0]} -> {o['points'][1]}, {o['seconds']} s"
        )
        if "whole" in o:
            text += (
                f", whole {o['whole'][0]} -> {o['whole'][1]}, "
                f"local {o['local'][0]} -> {o['local'][1]}, "
                f"{o['changed']} changed, {o['unchanged']} unchanged, "
                f"{o['refused']} refused, {o['out_of_time']} out of time, "
                f"crossings {o['crossings'][0]} -> {o['crossings'][1]}"
                + (
                    f", gaps {o['gaps'][0]} -> {o['gaps'][1]}, "
                    f"overlaps {o['overlaps'][0]} -> {o['overlaps'][1]}"
                    if "gaps" in o
                    else ""
                )
            )
    return text


def _case(row: dict) -> str:
    """The method and preset of *row*, and its Tidy steps if it names them."""
    steps = f" tidy {row['tidy_steps']}" if row.get("tidy_steps") else ""
    return f"{row['method']} {row['preset']}{steps}"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def compare(before: Path, after: Path) -> None:
    """Print the two runs' cases side by side.

    Cases pair by reference and preset, and by method too when both runs
    used the same one; runs of two methods pair up preset by preset. Rows
    of several Tidy configurations (`tidy_steps`) pair by configuration
    when both runs have them, and each with the other run's one row when
    only one does.
    """
    old_rows, new_rows = _rows(before), _rows(after)
    methods = {r["method"] for r in old_rows + new_rows}

    def case(row: dict) -> tuple[str, str]:
        return row["reference"], (
            row["preset"] if len(methods) > 1 else f"{row['method']} {row['preset']}"
        )

    old: dict[tuple, list[dict]] = {}
    new: dict[tuple, list[dict]] = {}
    for rows, into in ((old_rows, old), (new_rows, new)):
        for r in rows:
            into.setdefault(case(r), []).append(r)
    print(
        "| case | error | paths | points | points/path | curves | total s "
        "| optimize error left | features | tidy whole | tidy local Δ% "
        "| tidy points | tidy s/path | tidy crossed |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(old.keys() & new.keys()):
        for a, b in _pairs(old[key], new[key]):
            steps = a.get("tidy_steps") or b.get("tidy_steps")
            name = f"{key[0]} [{key[1]}{f' tidy {steps}' if steps else ''}]"

            def pair(field, a=a, b=b):
                return f"{a.get(field)} → {b.get(field)}"

            def tidy(row, field):
                return (row.get("optimize") or {}).get(field)

            def tidied(field, at, a=a, b=b):
                values = [tidy(r, field) for r in (a, b)]
                shown = [v[at] if isinstance(v, list) else v for v in values]
                return f"{shown[0]} → {shown[1]}"

            def local(row):
                v = tidy(row, "local")
                if not v or not v[0]:
                    return None
                return round(100 * (v[1] - v[0]) / v[0], 1)

            def per_path(row):
                if row.get("points") is None or not row.get("paths"):
                    return None
                return round(row["points"] / row["paths"], 1)

            def per_tidied(row):
                o = row.get("optimize") or {}
                return round(o["seconds"] / o["paths"], 1) if o.get("paths") else None

            print(
                f"| {name} | {pair('error')} | {pair('paths')} | "
                f"{pair('points')} | {per_path(a)} → {per_path(b)} | "
                f"{pair('curves')} | {pair('total_s')} | "
                f"{tidy(a, 'error_left')} → {tidy(b, 'error_left')} | "
                f"{pair('features')} | {tidied('whole', 1)} | "
                f"{local(a)} → {local(b)} | {tidied('points', 1)} | "
                f"{per_tidied(a)} → {per_tidied(b)} | "
                f"{tidy(a, 'crossed')} → {tidy(b, 'crossed')} |"
            )


def _pairs(old: list[dict], new: list[dict]) -> list[tuple[dict, dict]]:
    """The rows of one case to show side by side: by Tidy configuration
    when both runs name them, else every row with the other run's first."""
    if all(r.get("tidy_steps") for r in old + new):
        by_steps = {r["tidy_steps"]: r for r in new}
        return [
            (r, by_steps[r["tidy_steps"]]) for r in old if r["tidy_steps"] in by_steps
        ]
    if len(old) == 1:
        return [(old[0], r) for r in new]
    return [(r, new[0]) for r in old]


def sign_test(better: int, worse: int) -> float:
    """The two-sided sign test's p for *better* against *worse*, ties left
    out: how likely a split at least this uneven is from coin flips."""
    n = better + worse
    if not n:
        return 1.0
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def summary(runs: list[Path]) -> None:
    """One line per Tidy configuration over the rows of *runs* that tidied.

    Per image: the mean whole-image and local error before and after Tidy,
    and how many images it lowered or raised the local error of, with the
    sign test's p. Per path: how many paths' own local error it lowered or
    raised, with p. Then points, seconds per path, what became of the
    paths, and self-crossings. Last, each configuration against Tidy's
    default (snap,simplify) on the same traces: images whose whole error,
    and paths whose error over their traced area, Tidy leaves lower or
    higher than the default does.
    """
    rows = [
        r for path in runs for r in _rows(path) if "whole" in (r.get("optimize") or {})
    ]
    configs: dict[str, list[dict]] = {}
    for r in rows:
        steps = r.get("tidy_steps") or ",".join(r["optimize"]["steps"])
        configs.setdefault(steps, []).append(r)

    def mean(values) -> float:
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 2) if values else float("nan")

    def change(a, b) -> str:
        return f"{a} → {b} ({100 * (b - a) / a:+.1f}%)" if a else f"{a} → {b}"

    def split(pairs) -> str:
        better = sum(b < a for a, b in pairs)
        worse = sum(b > a for a, b in pairs)
        return f"{better}/{worse} p={sign_test(better, worse):.2g}"

    print(
        "| tidy steps | images | whole | local | images better/worse "
        "| paths better/worse | points | s/path (max) | changed/unchanged/refused "
        "| out of time | crossings (paths crossing more) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for steps, group in configs.items():
        tidies = [r["optimize"] for r in group]
        paths = [
            p
            for o in tidies
            for p in o["each"]
            if p.get("local", [None])[0] is not None
        ]
        images = [o["local"] for o in tidies if o["local"][0] is not None]
        points = [sum(o["points"][i] for o in tidies) for i in (0, 1)]
        each = [s for o in tidies for s in o["each_s"]]

        def total(field, at=None, tidies=tidies):
            return sum(o[field] if at is None else o[field][at] for o in tidies)

        whole = [mean(o["whole"][i] for o in tidies) for i in (0, 1)]
        print(
            f"| {steps} | {len(group)} | {change(*whole)} "
            f"| {change(mean(v[0] for v in images), mean(v[1] for v in images))} "
            f"| {split(images)} | {split([p['local'] for p in paths])} "
            f"| {change(*points)} | {mean(each)} ({max(each, default=0)}) "
            f"| {total('changed')}/{total('unchanged')}/{total('refused')} "
            f"| {total('out_of_time')} "
            f"| {total('crossings', 0)} → {total('crossings', 1)} "
            f"({total('crossed')}) |"
        )
    default = {
        (r["reference"], r["preset"]): r for r in configs.get("snap,simplify", [])
    }
    if not default or len(configs) < 2:
        return
    print()
    # Local errors here are over the traced paths' own areas (trace,
    # trace_local): the same pixels in both configurations.
    print(
        "| tidy steps against snap,simplify | images | whole after "
        "| traced area after | images lower/higher (whole) "
        "| paths lower/higher (traced area) |"
    )
    print("|---|---|---|---|---|---|")
    for steps, group in configs.items():
        if steps == "snap,simplify":
            continue
        matched = [
            (default[(r["reference"], r["preset"])]["optimize"], r["optimize"])
            for r in group
            if (r["reference"], r["preset"]) in default
            and default[(r["reference"], r["preset"])]["optimize"]["whole"][0]
            == r["optimize"]["whole"][0]
        ]
        if not matched:
            continue
        paths = []
        for base, other in matched:
            ours = {p["id"]: p["trace"] for p in base["each"] if p.get("trace")}
            paths += [
                (ours[p["id"]][1], p["trace"][1])
                for p in other["each"]
                if p.get("trace") and ours.get(p["id"], [None])[0] == p["trace"][0]
            ]
        whole = [(a["whole"][1], b["whole"][1]) for a, b in matched]
        local = [(a["trace_local"][1], b["trace_local"][1]) for a, b in matched]
        print(
            f"| {steps} | {len(matched)} "
            f"| {change(mean(a for a, _ in whole), mean(b for _, b in whole))} "
            f"| {change(mean(a for a, _ in local), mean(b for _, b in local))} "
            f"| {split(whole)} | {split(paths)} |"
        )


if __name__ == "__main__":
    main()
