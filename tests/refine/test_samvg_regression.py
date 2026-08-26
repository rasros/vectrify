"""Deterministic, model-free SAMVG seed regression coverage."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from vectrify.refine.samvg import generate_svg


@pytest.fixture
def two_band_target() -> tuple[Image.Image, list[np.ndarray]]:
    """A fixed seed image and its segmentation masks, with no model download."""
    pixels = np.array(
        [
            [[20, 30, 40]] * 6,
            [[20, 30, 40]] * 6,
            [[220, 210, 200]] * 6,
            [[220, 210, 200]] * 6,
        ],
        dtype=np.uint8,
    )
    upper = np.zeros((4, 6), dtype=bool)
    upper[:2] = True
    lower = np.zeros((4, 6), dtype=bool)
    lower[2:] = True
    return Image.fromarray(pixels), [upper, lower]


def test_fixed_seed_image_exports_two_editable_coloured_paths(two_band_target):
    image, masks = two_band_target

    svg = generate_svg(
        image,
        masks,
        min_pixels=1,
        min_impact=0,
        segments=4,
        hybrid_strokes=False,
        ocr=False,
    )

    root = ET.fromstring(svg)
    paths = list(root)
    assert root.attrib["viewBox"] == "0 0 6 4"
    assert [path.attrib["fill"] for path in paths] == ["#141e28", "#dcd2c8"]
    assert all(path.attrib["fill-rule"] == "evenodd" for path in paths)
    assert all(path.attrib["d"].endswith(" Z") for path in paths)
