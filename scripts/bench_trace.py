"""Benchmark tracing and Tidy (improve/nodes) on fixed references.

For each reference and preset, this runs a Generate method (SAMVG unless
`--method` names another) through the same code the editor uses, then
Tidy on the largest traced paths, and records how close the
result is to the reference, how heavy it is and how long it took. SAM's
masks are cached on disk per image and model setting, so after the first
run only the steps after SAM are timed and tuning them takes seconds;
`--no-cache` segments again.

    uv run python scripts/bench_trace.py --out runs/base.jsonl
    uv run python scripts/bench_trace.py --set max_side=1024 --out runs/b.jsonl
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
error over just those boxes, which a whole-image error hardly notices. SAM
and the steps after it are deterministic, so one run per case compares
settings; repeat with `--repeat` only to judge timings. Keep the machine
cool: `nice -n 19 taskset -c 12-19` with `OMP_NUM_THREADS=2`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import time
from pathlib import Path

from PIL import Image

from vectrify.image_utils import on_white

# Generate settings on top of each method's defaults, by method and preset.
PRESETS: dict[str, dict[str, dict]] = {
    "samvg": {
        "defaults": {},
    },
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
    parser.add_argument("--method", default="samvg", choices=sorted(PRESETS))
    parser.add_argument("--preset", nargs="+", help="Presets of the method to run")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a Generate setting for every preset",
    )
    parser.add_argument(
        "--paths", type=int, default=3, help="Largest paths to Optimize (0: none)"
    )
    parser.add_argument(
        "--nodes",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an Optimize nodes setting; rounds=N sets the rounds",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", type=Path, help="Write one JSON line per case")
    parser.add_argument(
        "--renders", type=Path, help="Save each case's traced drawing here as PNG"
    )
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
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
            for _ in range(args.repeat):
                row = {
                    "reference": path.name,
                    "method": args.method,
                    "preset": preset,
                    "settings": settings,
                    **trace(
                        image,
                        args.method,
                        settings,
                        args.paths,
                        nodes=nodes,
                        cache=not args.no_cache,
                        features=FEATURES.get(path.name, ()),
                        render=(
                            args.renders / f"{path.stem}-{args.method}-{preset}.png"
                            if args.renders
                            else None
                        ),
                    ),
                }
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
    cache: bool,
    features: tuple[tuple[str, int, int, int, int], ...] = (),
    render: Path | None = None,
) -> dict:
    """Generate from *image* with method *name* and *settings*, then Optimize
    its largest paths with the settings *nodes*. With *render*, the traced
    drawing is saved there."""
    from vectrify.document import Editor, Selection, export_svg, import_svg
    from vectrify.operations import Budget, Job, OperationRequest, Permissions, method

    timing = (
        _cached_segmentation(cache)
        if name == "samvg"
        else {"segment": 0.0, "cached": False, "spent": 0.0}
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
        return {"failed": state.get("error", state.get("status"))}
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
        import io

        import cairosvg
        import numpy as np

        png = cairosvg.svg2png(
            bytestring=export_svg(document).encode(),
            output_width=width,
            output_height=height,
            background_color="white",
        )
        assert png is not None
        traced = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), float)
        truth = np.asarray(image.convert("RGB"), float)
        squared = [
            ((traced - truth)[y : y + h, x : x + w] ** 2).ravel()
            for _, x, y, w, h in features
        ]
        feature_error = round(float(np.concatenate(squared).mean()), 2)
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
        # Seams Generate snapped together; SAMVG snaps none.
        "snapped": metrics.get("snapped", 0),
        "total_s": round(total, 1),
    }
    if feature_error is not None:
        row["features"] = feature_error
    if name == "samvg":
        row |= {
            "segment_s": round(timing["segment"], 1),
            "cached": timing["cached"],
            "after_sam_s": round(total - timing["spent"], 1),
        }
    if paths:
        row["optimize"] = optimize(editor, image, paths, nodes or OPTIMIZE)
    return row


def optimize(editor, image: Image.Image, count: int, settings: dict) -> dict:
    """Optimize nodes on each of the *count* largest paths, one at a time,
    with *settings*, where `rounds` is the budget's steps."""
    from vectrify.document import Selection
    from vectrify.document.hit_test import HitIndex
    from vectrify.operations import Budget, Job, OperationRequest, Permissions, method

    document = editor.snapshot.document
    index = HitIndex(document)
    largest = sorted(
        (e.id for e in document.elements() if e.tag == "path"),
        key=lambda oid: -(index.area(frozenset({oid})) or 0),
    )[:count]
    before = after = seconds = 0.0
    points_before = points_after = 0
    each: list[float] = []
    # Paths whose run the time limit ended.
    late = 0
    for oid in largest:
        editor.select(Selection(object_ids=frozenset({oid})))
        request = OperationRequest(
            "improve",
            "nodes",
            editor.snapshot,
            editor,
            Permissions(geometry=True, structure=True),
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
        result = job.state().get("result")
        if result is None:
            continue
        metrics = result["metrics"]
        before += metrics["before"]["difference"]
        after += metrics["after"]["difference"]
        points_before += metrics["before"]["nodes"]
        points_after += metrics["after"]["nodes"]
        late += bool(metrics.get("out_of_time"))
    return {
        "settings": settings,
        "paths": len(largest),
        # Each path's own region, summed: the share left says how much it fixed.
        "error_left": round(after / before, 3) if before else None,
        "points": [points_before, points_after],
        "seconds": round(seconds, 1),
        "each_s": each,
        "out_of_time": late,
    }


def _cached_segmentation(enabled: bool) -> dict:
    """Keep SAM's layers on disk per image and model setting, and time SAM."""
    import vectrify.refine.samvg as samvg

    # What SAM took (then or now), and what this run spent getting its masks.
    timing = {"segment": 0.0, "cached": False, "spent": 0.0}
    original = getattr(samvg.retrieve_layers, "__wrapped__", samvg.retrieve_layers)

    def retrieve(image, **kwargs):
        started = time.perf_counter()
        try:
            return _retrieve(image, **kwargs)
        finally:
            timing["spent"] = time.perf_counter() - started

    def _retrieve(image, **kwargs):
        options = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        digest = hashlib.sha1(image.tobytes() + repr(image.size).encode())
        digest.update(json.dumps(options, sort_keys=True, default=str).encode())
        path = CACHE / f"{digest.hexdigest()}.pkl"
        if enabled and path.exists():
            timing["cached"] = True
            with path.open("rb") as file:
                cached = pickle.load(file)
            timing["segment"] = cached["seconds"]
            layers = [_unpacked(layer) for layer in cached["layers"]]
            if any(not isinstance(layer, dict) for layer in cached["layers"]):
                # Written before masks were packed: pack it now.
                with path.open("wb") as file:
                    pickle.dump(
                        {
                            "layers": [_packed(layer) for layer in layers],
                            "seconds": cached["seconds"],
                        },
                        file,
                    )
            return layers
        started = time.perf_counter()
        layers = original(image, **kwargs)
        timing["segment"] = time.perf_counter() - started
        if enabled:
            CACHE.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as file:
                pickle.dump(
                    {
                        "layers": [_packed(layer) for layer in layers],
                        "seconds": timing["segment"],
                    },
                    file,
                )
        return layers

    retrieve.__wrapped__ = original  # type: ignore[attr-defined]
    samvg.retrieve_layers = retrieve
    return timing


def _packed(layer):
    """A layer with its mask as bits: hundreds of full-size masks are large."""
    import dataclasses

    import numpy as np

    fields = dataclasses.asdict(layer)
    mask = fields.pop("mask")
    return {"bits": np.packbits(mask), "shape": mask.shape, **fields}


def _unpacked(stored):
    import numpy as np

    from vectrify.refine.samvg_types import MaskLayer

    if not isinstance(stored, dict):
        return stored
    stored = dict(stored)
    # Fields layers no longer have, from caches written before they went.
    stored.pop("overlap_pixels", None)
    stored.pop("stroke", None)
    shape = stored.pop("shape")
    count = shape[0] * shape[1]
    mask = np.unpackbits(stored.pop("bits"), count=count).astype(bool).reshape(shape)
    return MaskLayer(mask=mask, **stored)


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
    if "after_sam_s" in row:
        text += (
            f" ({row['after_sam_s']} s after SAM"
            f"{', SAM cached' if row['cached'] else ''})"
        )
    if "optimize" in row:
        o = row["optimize"]
        text += (
            f"; optimize {o['paths']} paths: {o['error_left']} of the error left, "
            f"points {o['points'][0]} -> {o['points'][1]}, {o['seconds']} s"
        )
    return text


def _case(row: dict) -> str:
    """The method and preset of *row*; runs from before `--method` are SAMVG."""
    method = row.get("method", "samvg")
    return row["preset"] if method == "samvg" else f"{method} {row['preset']}"


def compare(before: Path, after: Path) -> None:
    """Print the two runs' cases side by side.

    Cases pair by reference and preset, and by method too when both runs
    used the same one; runs of two methods pair up preset by preset.
    """

    def load(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    old_rows, new_rows = load(before), load(after)
    methods = {r.get("method", "samvg") for r in old_rows + new_rows}

    def case(row: dict) -> tuple[str, str]:
        return row["reference"], (row["preset"] if len(methods) > 1 else _case(row))

    old = {case(r): r for r in old_rows}
    new = {case(r): r for r in new_rows}
    print(
        "| case | error | paths | points | points/path | curves | total s "
        "| optimize error left | features |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(old.keys() & new.keys()):
        a, b = old[key], new[key]

        def pair(field, a=a, b=b):
            return f"{a.get(field)} → {b.get(field)}"

        def per_path(row):
            if row.get("points") is None or not row.get("paths"):
                return None
            return round(row["points"] / row["paths"], 1)

        left = (
            (a.get("optimize") or {}).get("error_left"),
            (b.get("optimize") or {}).get("error_left"),
        )
        print(
            f"| {key[0]} [{key[1]}] | {pair('error')} | {pair('paths')} | "
            f"{pair('points')} | {per_path(a)} → {per_path(b)} | "
            f"{pair('curves')} | {pair('total_s')} | {left[0]} → {left[1]} | "
            f"{pair('features')} |"
        )


if __name__ == "__main__":
    main()
