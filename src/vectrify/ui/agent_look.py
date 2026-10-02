"""What an agent's looking calls work out from pixels: how an image maps onto
the drawing, a labelled grid to read coordinates off it, colours under a
spot, and the outlines of the reference's dark or same-coloured areas."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

Box = tuple[float, float, float, float]

# Grid lines and labels: magenta reads over most drawings and references.
GRID = (230, 0, 160)


def _number(value: float) -> float | int:
    rounded = round(value, 3)
    return int(rounded) if rounded == int(rounded) else rounded


def mapping(box: Box, size: tuple[int, int], *, gap: int = 0) -> dict[str, Any]:
    """How an image of *box* at *size* pixels maps onto the document.

    With *gap*, the image is two of them side by side, the second starting
    *size[0] + gap* pixels in.
    """
    x, y, w, h = box
    ux, uy = w / size[0], h / size[1]
    text = (
        f"Pixel (px, py) is document ({_number(x)} + px * {ux:.6g}, "
        f"{_number(y)} + py * {uy:.6g}): the image's top-left corner is "
        f"document ({_number(x)}, {_number(y)}), its bottom-right "
        f"({_number(x + w)}, {_number(y + h)}); {1 / ux:.4g} pixels per unit."
    )
    if gap:
        text += (
            f" The reference starts {size[0] + gap} pixels in, with the same "
            "mapping from there."
        )
    return {
        "region": [_number(v) for v in box],
        "pixels": list(size),
        "units_per_pixel": [float(f"{ux:.6g}"), float(f"{uy:.6g}")],
        "text": text,
    }


def grid_step(span: float, lines: int = 8) -> float:
    """A round step that puts about *lines* lines across *span*."""
    raw = span / lines
    power = 10 ** math.floor(math.log10(raw))
    return next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw)


def draw_grid(
    image: Image.Image, box: Box, left: int = 0, width: int | None = None
) -> Image.Image:
    """*image* with lines at round document coordinates over the part of it
    from pixel *left*, *width* wide, that shows *box*, labelled with their
    values."""
    x, y, w, h = box
    height = image.height
    width = image.width - left if width is None else width
    step = grid_step(max(w, h))
    out = image.convert("RGBA")
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = ImageFont.load_default()
    sx, sy = width / w, height / h
    value = math.ceil(x / step) * step
    while value <= x + w:
        px = left + (value - x) * sx
        draw.line([(px, 0), (px, height)], fill=(*GRID, 110), width=1)
        draw.text((px + 2, 1), f"{_number(value)}", fill=(*GRID, 255), font=font)
        value += step
    value = math.ceil(y / step) * step
    while value <= y + h:
        py = (value - y) * sy
        draw.line([(left, py), (left + width, py)], fill=(*GRID, 110), width=1)
        draw.text((left + 2, py + 1), f"{_number(value)}", fill=(*GRID, 255), font=font)
        value += step
    return Image.alpha_composite(out, layer).convert("RGB")


def disc_mean(image: Image.Image) -> tuple[float, float, float]:
    """The mean colour of the disc inscribed in *image*, 0 to 1."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    height, width = pixels.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    inside = ((xs + 0.5 - width / 2) / (width / 2)) ** 2 + (
        (ys + 0.5 - height / 2) / (height / 2)
    ) ** 2 <= 1
    chosen = pixels[inside] if inside.any() else pixels.reshape(-1, 3)
    r, g, b = chosen.mean(axis=0)
    return float(r), float(g), float(b)


def hex_colour(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(min(1, max(0, v)) * 255):02x}" for v in rgb)


def area_mask(
    image: Image.Image,
    *,
    colour: tuple[float, float, float] | None,
    tolerance: float,
) -> np.ndarray:
    """Where *image* is dark (luminance at most *tolerance*), or within
    *tolerance* (RGB distance, 0 to 1) of *colour*."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    if colour is None:
        luminance = pixels @ np.array([0.299, 0.587, 0.114])
        return luminance <= tolerance
    distance = np.linalg.norm(pixels - np.asarray(colour), axis=2) / math.sqrt(3)
    return distance <= tolerance


def _mapped_path(d: str, box: Box, size: tuple[int, int]) -> str:
    """Path data in mask pixels, mapped into document units over *box*."""
    x, y, w, h = box
    sx, sy = w / size[0], h / size[1]
    out = []
    numbers: list[float] = []

    def flush() -> None:
        for i in range(0, len(numbers) - 1, 2):
            out.append(f"{x + numbers[i] * sx:.2f} {y + numbers[i + 1] * sy:.2f}")
        numbers.clear()

    for token in re.findall(r"[MCLZ]|-?[\d.]+(?:e-?\d+)?", d):
        if token in "MCLZ":
            flush()
            out.append(token)
        else:
            numbers.append(float(token))
    flush()
    return " ".join(out)


def trace_areas(
    mask: np.ndarray,
    box: Box,
    *,
    min_pixels: int = 6,
    limit: int = 40,
    budget: int = 16000,
) -> dict[str, Any]:
    """The connected areas of *mask* over *box* as closed path data in
    document units, largest first: each with its holes, its area and bounds.

    At most *limit* areas and about *budget* characters of path data; the
    rest are counted, not given.
    """
    from vectrify.refine.samvg import mask_path

    size = (mask.shape[1], mask.shape[0])
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    order = [i for i in np.argsort(-sizes[1:]) + 1 if sizes[i] >= min_pixels]
    x, y, w, h = box
    unit = (w / size[0]) * (h / size[1])
    shapes = []
    used = 0
    skipped = 0
    for index in order:
        part = labels == index
        if len(shapes) >= limit:
            skipped += 1
            continue
        d = mask_path(part, smooth=1.0, density=4)
        if not d:
            continue
        mapped = _mapped_path(d, box, size)
        if used + len(mapped) > budget and shapes:
            skipped += 1
            continue
        used += len(mapped)
        rows, cols = np.nonzero(part)
        shapes.append(
            {
                "d": mapped,
                "area": round(float(sizes[index]) * unit, 3),
                "bounds": [
                    round(x + cols.min() * w / size[0], 2),
                    round(y + rows.min() * h / size[1], 2),
                    round((cols.max() + 1 - cols.min()) * w / size[0], 2),
                    round((rows.max() + 1 - rows.min()) * h / size[1], 2),
                ],
            }
        )
    return {
        "shapes": shapes,
        "more": skipped,
        "covered": round(float(mask.mean()), 4),
    }
