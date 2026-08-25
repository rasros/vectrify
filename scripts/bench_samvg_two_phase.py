"""Run and record SAMVG's two-phase segmentation, fit, and recovery process.

Examples:
    uv run python scripts/bench_samvg_two_phase.py --cat
    uv run python scripts/bench_samvg_two_phase.py --all

The native CUDA extension must be available for the 1024px cat workload.  The
script deliberately uses the regular SAMVG masks and fixed-16-segment tracer;
it only bounds the differentiable fit to one spatial fill group at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from vectrify.formats.svg.plugin import SvgPlugin
from vectrify.refine.paths import fit_filled_svg_bounded
from vectrify.refine.samvg import (
    _append_layers,
    _mse,
    _render_svg,
    _sam_runtime,
    filter_by_impact,
    prompted_masks,
    residual_prompt_points,
    retrieve_layers,
)

ROOT = Path(__file__).resolve().parents[1]


def _path_count(svg: str) -> int:
    return sum(
        element.tag.split("}")[-1] == "path" for element in ET.fromstring(svg).iter()
    )


def _l1(target: Image.Image, rendered: Image.Image) -> float:
    return float(
        np.abs(
            np.asarray(target.convert("RGB"), dtype=np.float32) / 255.0
            - np.asarray(rendered.convert("RGB"), dtype=np.float32) / 255.0
        ).mean()
    )


def _write_gallery(images: list[tuple[str, Image.Image]], destination: Path) -> None:
    width = max(image.width for _name, image in images)
    height = max(image.height for _name, image in images)
    gallery = Image.new("RGB", (width * len(images), height + 28), "white")
    labels = ImageDraw.Draw(gallery)
    for index, (name, image) in enumerate(images):
        gallery.paste(image.convert("RGB"), (index * width, 28))
        labels.text((index * width + 4, 6), name, fill="black")
    gallery.save(destination)


def _fit_if_improved(
    svg: str,
    target: Image.Image,
    plugin: SvgPlugin,
    steps: int,
    learn_alpha: bool,
) -> tuple[str, Image.Image, list[dict[str, int | float]], bool]:
    before = _render_svg(svg, target, plugin.rasterize)
    measurements: list[dict[str, int | float]] = []
    candidate = fit_filled_svg_bounded(
        svg,
        target,
        rasterize=plugin.rasterize,
        steps=steps,
        measurements=measurements,
        learn_alpha=learn_alpha,
    )
    after = _render_svg(candidate, target, plugin.rasterize)
    if _mse(target, after) <= _mse(target, before):
        return candidate, after, measurements, True
    return svg, before, measurements, False


def run_target(
    target_path: Path,
    output: Path,
    *,
    steps: int,
    reference_svg: Path | None = None,
    learn_alpha: bool = False,
    curvature_threshold: float | None = None,
) -> None:
    target = Image.open(target_path).convert("RGB")
    plugin = SvgPlugin()
    destination = output / target_path.stem
    destination.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    runtime = _sam_runtime()
    layers = retrieve_layers(target, _runtime=runtime)
    initial = _append_layers(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{target.width}" '
        f'height="{target.height}" viewBox="0 0 {target.width} {target.height}"></svg>',
        layers,
        16,
        hybrid_strokes=False,
        curvature_threshold=curvature_threshold,
    )
    _render_svg(initial, target, plugin.rasterize).save(destination / "first-seed.png")
    (destination / "first-seed.svg").write_text(initial)
    first, first_render, first_measurements, first_accepted = _fit_if_improved(
        initial, target, plugin, steps, learn_alpha
    )
    first_render.save(destination / "first-fit.png")
    (destination / "first-fit.svg").write_text(first)
    points = residual_prompt_points(target, first_render)
    added = filter_by_impact(
        target,
        prompted_masks(target, points, _runtime=runtime),
        existing=layers,
        initial_canvas=np.asarray(first_render, dtype=np.uint8),
        # The residual pass starts from the first fitted raster.  It is not an
        # uncovered-mask pass, so all pixels must use their actual raster MSE.
        initial_coverage=np.ones((target.height, target.width), dtype=bool),
    )[len(layers) :]
    recovery = _append_layers(
        first,
        added,
        16,
        hybrid_strokes=False,
        curvature_threshold=curvature_threshold,
    )
    _render_svg(recovery, target, plugin.rasterize).save(
        destination / "residual-recovery.png"
    )
    (destination / "residual-recovery.svg").write_text(recovery)
    final, final_render, final_measurements, final_accepted = _fit_if_improved(
        recovery, target, plugin, steps, learn_alpha
    )
    stages = [
        ("target", target, None),
        ("first-seed", _render_svg(initial, target, plugin.rasterize), initial),
        ("first-fit", first_render, first),
        (
            "residual-recovery",
            _render_svg(recovery, target, plugin.rasterize),
            recovery,
        ),
        ("final-fit", final_render, final),
    ]
    if reference_svg is not None:
        reference = _render_svg(reference_svg.read_text(), target, plugin.rasterize)
        stages.append(("reference-svg", reference, reference_svg.read_text()))
    rows = []
    for name, rendered, svg in stages:
        rendered.save(destination / f"{name}.png")
        if svg is not None:
            (destination / f"{name}.svg").write_text(svg)
        rows.append(
            {
                "stage": name,
                "l1": _l1(target, rendered),
                "mse": _mse(target, rendered),
                "paths": _path_count(svg) if svg is not None else 0,
            }
        )
    _write_gallery(
        [(name, image) for name, image, _svg in stages], destination / "gallery.png"
    )
    with (destination / "stages.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "l1", "mse", "paths"])
        writer.writeheader()
        writer.writerows(rows)
    measurements = [
        {"phase": "first-fit", **measurement} for measurement in first_measurements
    ] + [{"phase": "final-fit", **measurement} for measurement in final_measurements]
    (destination / "fit-groups.json").write_text(json.dumps(measurements, indent=2))
    (destination / "summary.json").write_text(
        json.dumps(
            {
                "target": str(target_path),
                "first_fit_accepted": first_accepted,
                "final_fit_accepted": final_accepted,
                "initial_layers": len(layers),
                "residual_layers": len(added),
                "wall_seconds": perf_counter() - started,
                "stages": rows,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, action="append", default=[])
    parser.add_argument(
        "--reference-svg",
        type=Path,
        help="Reference SVG to Cairo-rasterize alongside a single target.",
    )
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        help="Use the dissertation's variable-segment tracing variation.",
    )
    parser.add_argument("--cat", action="store_true")
    parser.add_argument(
        "--all", action="store_true", help="Run cat, duck, and all bench targets."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "bench/results/samvg-two-phase"
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--learn-alpha",
        action="store_true",
        help="Use the dissertation's SAMVG+alpha fitter variation.",
    )
    args = parser.parse_args()
    targets = list(args.target)
    if args.cat or args.all:
        targets.append(Path("/tmp/SAMVG_thesis/cat1024.jpg"))
    if args.all:
        targets.extend(sorted((ROOT / "bench/cases").glob("*/target.png")))
        targets.append(ROOT / "connect-the-dots-little-duck.png")
    if not targets:
        parser.error("give --target, --cat, or --all")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.reference_svg is not None and len(targets) != 1:
        parser.error("--reference-svg requires exactly one target")
    for target in targets:
        reference_svg = args.reference_svg
        if target == Path("/tmp/SAMVG_thesis/cat1024.jpg"):
            candidate = target.with_suffix(".svg")
            if candidate.exists():
                reference_svg = candidate
        run_target(
            target,
            args.output,
            steps=args.steps,
            reference_svg=reference_svg,
            learn_alpha=args.learn_alpha,
            curvature_threshold=args.curvature_threshold,
        )


if __name__ == "__main__":
    main()
