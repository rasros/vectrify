"""Run one reproducible CUDA renderer timing without downloading a model."""

from __future__ import annotations

from time import perf_counter

from PIL import Image

from vectrify.refine import cuda_renderer
from vectrify.refine.paths import fit_filled_svg

SVG = """<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"32\" height=\"32\">
<path d=\"M 2 2 C 20 0 32 12 30 30 C 10 32 0 20 2 2 Z\" fill=\"#4080c0\"/>
</svg>"""


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SAMVG CUDA benchmark requires a GPU runner")
    if cuda_renderer._extension() is None:
        raise RuntimeError("SAMVG CUDA extension was not built")
    target = Image.new("RGB", (32, 32), "#4080c0")
    fit_filled_svg(SVG, target, steps=1, optimisation_long_side=32)
    torch.cuda.synchronize()
    started = perf_counter()
    fit_filled_svg(SVG, target, steps=1, optimisation_long_side=32)
    torch.cuda.synchronize()
    print(f"samvg-cuda renderer step: {perf_counter() - started:.4f}s")


if __name__ == "__main__":
    main()
