"""Reproducible clean-vector/degraded-raster pairs split by original artwork."""

from __future__ import annotations

import hashlib
import io
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from vectrify.refine.cel_plan.score import render

DATA = Path(__file__).parent / "bench_data"


def cases(heldout: bool = False) -> list[dict]:
    manifest = json.loads((DATA / "planned_pairs.json").read_text())
    split = "heldout" if heldout else "tuning"
    return [case for case in manifest["cases"] if case["set"] == split]


def degrade(image: Image.Image, kind: str, seed: int) -> Image.Image:
    """Corrupt the input only; the clean vector remains the independent oracle."""
    image = image.convert("RGBA")
    if kind == "clean":
        return image
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(1.25))
    if kind == "resized":
        small = (max(1, image.width // 2), max(1, image.height // 2))
        return image.resize(small, Image.Resampling.BILINEAR).resize(
            image.size, Image.Resampling.BICUBIC
        )
    if kind == "jpeg":
        stream = io.BytesIO()
        image.convert("RGB").save(stream, format="JPEG", quality=55)
        stream.seek(0)
        with Image.open(stream) as compressed:
            result = compressed.convert("RGBA")
        result.putalpha(image.getchannel("A"))
        return result
    pixels = np.asarray(image).copy()
    rng = np.random.default_rng(seed)
    if kind == "noise":
        noise = rng.normal(0, 7, pixels[..., :3].shape)
        pixels[..., :3] = np.clip(pixels[..., :3] + noise, 0, 255).round()
    elif kind == "alpha":
        alpha = pixels[..., 3].astype(float)
        partial = (alpha > 0) & (alpha < 255)
        alpha[partial] += rng.normal(0, 18, int(partial.sum()))
        pixels[..., 3] = np.clip(alpha, 0, 255).round()
    else:
        raise ValueError(f"Unknown raster degradation: {kind}")
    return Image.fromarray(pixels)


def composition(source: str, opacity: float = 1) -> str:
    """Derive a uniform-opacity artwork before both clean and input renders.

    The repository-authored fixtures use explicit paint attributes. Isolating
    their rendering children applies opacity once to the complete composition,
    including overlaps; definitions and original root attributes remain intact.
    This is a target variant, not input corruption or a new artwork family.
    """
    if not math.isfinite(opacity) or not 0 < opacity <= 1:
        raise ValueError("Composition opacity must be finite and between zero and one")
    if opacity == 1:
        return source
    root = ET.fromstring(source)
    namespace = root.tag.partition("}")[0] + "}" if root.tag.startswith("{") else ""
    group = ET.Element(f"{namespace}g", {"opacity": str(opacity)})
    for child in list(root):
        if child.tag.rsplit("}", 1)[-1] in {
            "defs",
            "style",
            "title",
            "desc",
            "metadata",
        }:
            continue
        root.remove(child)
        group.append(child)
    root.append(group)
    return ET.tostring(root, encoding="unicode")


def pair(case: dict, kind: str, long_side: int = 1000, composition_opacity: float = 1):
    source = (DATA / case["file"]).read_text()
    source = composition(source, composition_opacity)
    # Use the existing line benchmark's aspect-ratio calculation.
    from scripts.bench_lines import size

    width, height = size(source)
    factor = long_side / max(width, height)
    dimensions = (max(1, round(width * factor)), max(1, round(height * factor)))
    pixels = render(source, dimensions)
    clean = Image.fromarray((pixels * 255).round().astype(np.uint8))
    seed = int.from_bytes(hashlib.sha256(case["family"].encode()).digest()[:4], "big")
    return source, clean, degrade(clean, kind, seed)
