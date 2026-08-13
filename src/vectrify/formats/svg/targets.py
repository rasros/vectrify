"""Where the error in a candidate lives, element by element.

Mutation picks an element to work on. Picking uniformly spends a run in
proportion to how many elements there are rather than to where the drawing is
wrong: on an emblem seed the background answers for 57% of the error and the
star for 3.5%, and uniform choice gives them the same attention forever.
"""

import io
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from vectrify.formats.svg.ownership import MASK_SIZE, element_error
from vectrify.image_utils import rasterize_svg_to_png_bytes


def _as_array(png: bytes, size: int) -> np.ndarray:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def element_targets(
    content: str, reference_png: bytes, size: int = MASK_SIZE
) -> dict[int, float]:
    """Error per drawable element, keyed by position, normalised to sum to 1.

    Costs two renders at *size*, so callers should cache it against the parent
    rather than recompute it for every task the parent is used for.
    """
    try:
        root = ET.fromstring(content)
        render = _as_array(
            rasterize_svg_to_png_bytes(content, out_w=size, out_h=size), size
        )
    except Exception:
        return {}

    errors = element_error(root, _as_array(reference_png, size), render, size=size)
    total = sum(errors)
    if total <= 0.0:
        return {}
    return {index: value / total for index, value in enumerate(errors)}
