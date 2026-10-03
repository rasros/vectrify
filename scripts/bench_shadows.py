"""Benchmark how much shading the cel tracer leaves out.

A shadow the trace drops shows as the trace drawn lighter than the
reference there. The reference is smoothed (gaussian, sigma 1.5), so its
grain does not count, and both are taken as luminance (0.299, 0.587,
0.114); pixels where the trace is lighter by more
than 18 levels, opened twice so thin line misses go, form blobs, and blobs
of at least 150 px count. The report is their pixels, blobs and largest.

The references are the trace bench's raster images (over white, as the
editor shows them) and the line bench's SVG drawings, rendered clean:

    uv run python scripts/bench_shadows.py --out runs/shadows.jsonl
    uv run python scripts/bench_shadows.py --set outline=true
    uv run python scripts/bench_shadows.py --renders runs/shadows
    uv run python scripts/bench_shadows.py --rescore runs/shadows
    uv run python scripts/bench_shadows.py --compare runs/a.jsonl runs/b.jsonl
    uv run python scripts/bench_shadows.py --heldout

`--heldout` runs both benches' held-out sets instead, never used for
tuning (see bench_trace and bench_lines). Each set includes its generated
images (gen-*, see bench_trace) and drawings made for the bench (svg-*).

Cel is deterministic, so one run per case compares settings. Keep the
machine cool: `nice -n 19 taskset -c 12-19` with `OMP_NUM_THREADS=2`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from bench_lines import CACHE, _setting, reference, render, size, trace
from bench_lines import HELDOUT as LINE_HELDOUT
from bench_lines import HELDOUT_DIR as LINE_HELDOUT_DIR
from bench_lines import REFERENCES as LINE_REFERENCES
from bench_trace import ROOT, load, references
from PIL import Image
from scipy.ndimage import binary_opening, gaussian_filter, label

# The measure's constants, as the brief for it gives them.
SIGMA = 1.5
LIGHTER = 18
OPENING = 2
BLOB = 150
LUMA = np.array([0.299, 0.587, 0.114])


def missing_shadows(reference: np.ndarray, traced: np.ndarray) -> dict:
    """Where *traced* is lighter than *reference*, both RGB: the pixels of
    blobs at least BLOB large, their count and the largest."""
    smooth = gaussian_filter(np.asarray(reference, dtype=np.float64), (SIGMA, SIGMA, 0))
    lighter = np.asarray(traced, dtype=np.float64) @ LUMA - smooth @ LUMA > LIGHTER
    lighter = binary_opening(lighter, iterations=OPENING)
    blobs, count = label(lighter)
    sizes = np.bincount(blobs.ravel(), minlength=count + 1)[1:]
    kept = sizes[sizes >= BLOB]
    return {
        "shadow_px": int(kept.sum()),
        "shadow_blobs": len(kept),
        "shadow_largest": int(kept.max()) if kept.size else 0,
        "mask": np.isin(blobs, 1 + np.flatnonzero(sizes >= BLOB)),
    }


def cases(images: Path, heldout: bool = False) -> list[tuple[str, Image.Image]]:
    """Each reference's name and image, the raster ones from *images*; with
    *heldout*, the held-out ones instead."""
    found = [(path.name, load(path)) for path in references(heldout, images)]
    if heldout:
        drawings, folder = LINE_HELDOUT, LINE_HELDOUT_DIR
    else:
        drawings, folder = LINE_REFERENCES, CACHE
    for name in drawings:
        svg = reference(name, folder)
        found.append((name, Image.fromarray(render(svg, *size(svg)))))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a Generate cel setting",
    )
    parser.add_argument(
        "--images", type=Path, default=ROOT, help="Where the raster references are"
    )
    parser.add_argument(
        "--heldout",
        action="store_true",
        help="Run the held-out references, not the tuning set",
    )
    parser.add_argument("--out", type=Path, help="Write one JSON line per case")
    parser.add_argument(
        "--renders",
        type=Path,
        help="Save each trace and its missing-shadow mask here as PNG",
    )
    parser.add_argument(
        "--rescore",
        type=Path,
        metavar="DIR",
        help="Score the traces a run saved with --renders DIR, without tracing",
    )
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    settings = dict(_setting(item) for item in args.set)
    rows = []
    for name, image in cases(args.images, args.heldout):
        seconds = None
        if args.rescore:
            svg = (args.rescore / f"{Path(name).stem}-cel.svg").read_text()
        else:
            svg, seconds = trace(image, settings)
        traced = render(svg, *image.size)
        found = missing_shadows(np.asarray(image), traced)
        mask = found.pop("mask")
        if args.renders:
            args.renders.mkdir(parents=True, exist_ok=True)
            stem = args.renders / Path(name).stem
            if not args.rescore:
                Image.fromarray(traced).save(f"{stem}-cel.png")
                Path(f"{stem}-cel.svg").write_text(svg)
            Image.fromarray(mask.astype(np.uint8) * 255).save(f"{stem}-missing.png")
        row = {
            "reference": name,
            "settings": settings,
            "seconds": None if seconds is None else round(seconds, 1),
            **found,
        }
        rows.append(row)
        print(
            f"{name}: {row['shadow_px']} px missing in {row['shadow_blobs']} "
            f"blobs, largest {row['shadow_largest']}, {row['seconds']} s",
            flush=True,
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")


def compare(before: Path, after: Path) -> None:
    """Print the two runs' cases side by side, paired by reference."""

    def load(path: Path) -> dict[str, dict]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        return {r["reference"]: r for r in rows}

    old, new = load(before), load(after)
    fields = ("shadow_px", "shadow_blobs", "shadow_largest")
    print("| reference | " + " | ".join(fields) + " |")
    print("|---|" + "---|" * len(fields))
    for key in sorted(old.keys() & new.keys()):
        cells = " | ".join(f"{old[key][f]} → {new[key][f]}" for f in fields)
        print(f"| {key} | {cells} |")


if __name__ == "__main__":
    main()
