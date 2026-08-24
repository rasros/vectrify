"""Measure one steady SAMVG filled-path optimisation step.

Example:
    uv run python scripts/bench_samvg_renderer.py /tmp/cat.svg /tmp/cat.jpg
    uv run python scripts/bench_samvg_renderer.py /tmp/cat.svg /tmp/cat.jpg \
        --torch-fallback
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from PIL import Image

from vectrify.refine.paths import fit_filled_svg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--long-side", type=int, default=64)
    parser.add_argument("--torch-fallback", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least one")

    import torch

    if args.torch_fallback:
        from vectrify.refine import cuda_renderer

        cuda_renderer._extension = lambda: None
    svg = args.svg.read_text()
    target = Image.open(args.target)
    # Setup, CUDA allocator warm-up, and any Torch compilation happen outside
    # the timed region so the result is a steady optimisation step.
    fit_filled_svg(svg, target, steps=1, optimisation_long_side=args.long_side)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = perf_counter()
    fit_filled_svg(
        svg, target, steps=args.steps, optimisation_long_side=args.long_side
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"{(perf_counter() - started) / args.steps:.6f} seconds/step")


if __name__ == "__main__":
    main()
