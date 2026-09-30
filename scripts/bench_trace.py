"""Benchmark tracing and Optimize nodes on fixed references.

For each reference and preset, this runs Generate with SAMVG through the
same method the editor uses, then Optimize nodes on the largest traced
paths, and records how close the result is to the reference, how heavy it
is and how long it took. SAM's masks are cached on disk per image and model
setting, so after the first run only the steps after SAM are timed and
tuning them takes seconds; `--no-cache` segments again.

    uv run python scripts/bench_trace.py --out runs/base.jsonl
    uv run python scripts/bench_trace.py --set merge=false --out runs/b.jsonl
    uv run python scripts/bench_trace.py --compare runs/base.jsonl runs/b.jsonl

The error is the mean squared difference to the reference in 0-255 RGB. SAM
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

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = (
    "ChatGPT Image Sep 29, 2026, 10_40_22 PM.png",
    "chest-clothing-bold-v2.png",
)
# Generate settings on top of the method's defaults.
PRESETS: dict[str, dict] = {
    "defaults": {},
    "flattened": {"flatten": True},
}
# Optimize nodes: which steps, and how many rounds per path.
OPTIMIZE = {"shape": True, "snap": True, "detail": False, "simplify": True}
ROUNDS = 4
CACHE = Path.home() / ".cache" / "vectrify-bench"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--references", nargs="+", type=Path)
    parser.add_argument("--preset", nargs="+", choices=sorted(PRESETS))
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
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", type=Path, help="Write one JSON line per case")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    overrides = dict(_setting(item) for item in args.set)
    references = args.references or [ROOT / name for name in REFERENCES]
    rows = []
    for path in references:
        image = Image.open(path).convert("RGB")
        for preset in args.preset or list(PRESETS):
            settings = {**PRESETS[preset], **overrides}
            for _ in range(args.repeat):
                row = {
                    "reference": path.name,
                    "preset": preset,
                    "settings": settings,
                    **trace(image, settings, args.paths, cache=not args.no_cache),
                }
                rows.append(row)
                print(_line(row), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")


def trace(image: Image.Image, settings: dict, paths: int, *, cache: bool) -> dict:
    """Generate from *image* with *settings*, then Optimize its largest paths."""
    from vectrify.document import Editor, Selection, export_svg, import_svg
    from vectrify.operations import Budget, Job, OperationRequest, Permissions, method

    timing = _cached_segmentation(cache)
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
        "samvg",
        editor.snapshot,
        editor,
        Permissions(geometry=True, structure=True, paint=True),
        settings=settings,
        budget=Budget(),
        reference=image,
    )
    started = time.perf_counter()
    job = Job(method("generate", "samvg"), request)
    job.run()
    total = time.perf_counter() - started
    state = job.state()
    if "result" not in state:
        return {"failed": state.get("error", state.get("status"))}
    metrics = state["result"]["metrics"]
    job.apply()
    document = editor.snapshot.document
    data = " ".join(
        e.get("d") or "" for e in _parse(export_svg(document)) if e.tag.endswith("path")
    )
    row = {
        "error": round(metrics["after"]["error"] * 255**2, 2),
        "paths": sum(1 for e in document.elements() if e.tag == "path"),
        "curves": data.count("C"),
        "linked": metrics.get("linked", 0),
        "segment_s": round(timing["segment"], 1),
        "cached": timing["cached"],
        "total_s": round(total, 1),
        "after_sam_s": round(total - timing["segment"], 1),
    }
    if paths:
        row["optimize"] = optimize(editor, image, paths)
    return row


def optimize(editor, image: Image.Image, count: int) -> dict:
    """Optimize nodes on each of the *count* largest paths, one at a time."""
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
    for oid in largest:
        editor.select(Selection(object_ids=frozenset({oid})))
        request = OperationRequest(
            "improve",
            "nodes",
            editor.snapshot,
            editor,
            Permissions(geometry=True, structure=True),
            settings={"workers": 1, **OPTIMIZE},
            budget=Budget(steps=ROUNDS),
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
    return {
        "paths": len(largest),
        # Each path's own region, summed: the share left says how much it fixed.
        "error_left": round(after / before, 3) if before else None,
        "points": [points_before, points_after],
        "seconds": round(seconds, 1),
        "each_s": each,
    }


def _cached_segmentation(enabled: bool) -> dict:
    """Keep SAM's layers on disk per image and model setting, and time SAM."""
    import vectrify.refine.samvg as samvg

    timing = {"segment": 0.0, "cached": False}
    original = getattr(samvg.retrieve_layers, "__wrapped__", samvg.retrieve_layers)

    def retrieve(image, masks=None, **kwargs):
        options = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        digest = hashlib.sha1(image.tobytes() + repr(image.size).encode())
        digest.update(json.dumps(options, sort_keys=True, default=str).encode())
        path = CACHE / f"{digest.hexdigest()}.pkl"
        if enabled and masks is None and path.exists():
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
        layers = original(image, masks, **kwargs)
        timing["segment"] = time.perf_counter() - started
        if enabled and masks is None:
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
    if "failed" in row:
        return f"{row['reference']} [{row['preset']}]: failed: {row['failed']}"
    text = (
        f"{row['reference']} [{row['preset']}]: error {row['error']}, "
        f"{row['paths']} paths, {row['curves']} curves, {row['linked']} links, "
        f"{row['total_s']} s ({row['after_sam_s']} s after SAM"
        f"{', SAM cached' if row['cached'] else ''})"
    )
    if "optimize" in row:
        o = row["optimize"]
        text += (
            f"; optimize {o['paths']} paths: {o['error_left']} of the error left, "
            f"points {o['points'][0]} -> {o['points'][1]}, {o['seconds']} s"
        )
    return text


def compare(before: Path, after: Path) -> None:
    """Print the two runs' cases side by side."""

    def load(path: Path) -> dict:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        return {(r["reference"], r["preset"]): r for r in rows}

    old, new = load(before), load(after)
    print("| case | error | paths | curves | after SAM s | optimize error left |")
    print("|---|---|---|---|---|---|")
    for key in sorted(old.keys() & new.keys()):
        a, b = old[key], new[key]

        def pair(field, a=a, b=b):
            return f"{a.get(field)} → {b.get(field)}"

        left = (
            (a.get("optimize") or {}).get("error_left"),
            (b.get("optimize") or {}).get("error_left"),
        )
        print(
            f"| {key[0]} [{key[1]}] | {pair('error')} | {pair('paths')} | "
            f"{pair('curves')} | {pair('after_sam_s')} | {left[0]} → {left[1]} |"
        )


if __name__ == "__main__":
    main()
