"""Benchmark the colour-region vectorizer on one image.

Writes the SVG, its full-size and 1024 px renders, and a JSON metrics file.
Run from the repository with PYTHONPATH=src and the vision dependencies installed.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cairosvg
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from vectrify.refine.colour_regions import vectorize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--colours", type=int, default=32)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1.25)
    parser.add_argument("--smooth-sigma", type=float, default=1.5)
    parser.add_argument(
        "--texture-tolerance",
        type=float,
        default=5.0,
        help="Clean mode texture tolerance in source pixels; preserves contours",
    )
    parser.add_argument(
        "--preserve-outlines",
        action="store_true",
        help="Preserve thin dark linework as a separate vector layer",
    )
    parser.add_argument(
        "--outline-radius",
        type=int,
        default=3,
        help="Local dark-line detection radius in source pixels",
    )
    parser.add_argument(
        "--outline-style", choices=("preserve", "clean"), default="preserve"
    )
    parser.add_argument(
        "--outline-regions",
        type=int,
        default=3,
        help="Broad colour regions for clean outlines",
    )
    parser.add_argument(
        "--outline-width",
        type=float,
        default=1.5,
        help="Clean stroke width in source pixels",
    )
    parser.add_argument(
        "--outline-contrast",
        type=float,
        default=6,
        help="Minimum strong dark-line contrast on the 0-255 luminance scale",
    )
    parser.add_argument(
        "--no-geometry-cleanup",
        action="store_true",
        help="Skip the final geometry simplification and path merging stage",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    image = Image.open(args.image).convert("RGB")
    svg, metrics = vectorize(
        image,
        colours=args.colours,
        steps=args.steps,
        min_pixels=args.min_pixels,
        tolerance=args.tolerance,
        smooth_sigma=args.smooth_sigma,
        preserve_outlines=args.preserve_outlines,
        outline_radius=args.outline_radius,
        outline_contrast=args.outline_contrast,
        outline_style=args.outline_style,
        outline_regions=args.outline_regions,
        outline_width=args.outline_width,
        texture_tolerance=args.texture_tolerance,
        geometry_cleanup=not args.no_geometry_cleanup,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)
    png = cairosvg.svg2png(bytestring=svg.encode())
    args.output.with_suffix(".png").write_bytes(png)
    rendered = Image.open(io.BytesIO(png)).convert("RGBA")
    alpha = np.asarray(rendered)[:, :, 3]
    metrics["uncovered_percent"] = float((1 - alpha.astype(float) / 255).mean() * 100)
    metrics["mse_4k"] = float(
        np.square(
            np.asarray(rendered.convert("RGB"), dtype=float)
            - np.asarray(image, dtype=float)
        ).mean()
    )
    size = (1024, round(1024 * image.height / image.width))
    preview = cairosvg.svg2png(
        bytestring=svg.encode(), output_width=size[0], output_height=size[1]
    )
    args.output.with_suffix(".preview.png").write_bytes(preview)
    small = np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=float)
    actual = np.asarray(Image.open(io.BytesIO(preview)).convert("RGB"), dtype=float)
    metrics["mse_1024"] = float(np.square(actual - small).mean())
    metrics["mse_blur_2_1024"] = float(
        np.square(
            gaussian_filter(actual, (2, 2, 0)) - gaussian_filter(small, (2, 2, 0))
        ).mean()
    )
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
