"""Paired data has deterministic corruption and no artwork-family leakage."""

import numpy as np
import pytest
from PIL import Image

from scripts.cel_pairs import cases, degrade


def test_training_and_heldout_families_do_not_overlap():
    training = {case["family"] for case in cases()}
    heldout = {case["family"] for case in cases(True)}
    assert training
    assert heldout
    assert training.isdisjoint(heldout)


@pytest.mark.parametrize("kind", ["clean", "blur", "resized", "jpeg", "noise", "alpha"])
def test_degradations_are_reproducible_and_preserve_image_extent(kind):
    image = Image.new("RGBA", (32, 24), (100, 50, 20, 128))
    first = degrade(image, kind, 19)
    second = degrade(image, kind, 19)
    assert first.size == image.size
    np.testing.assert_array_equal(first, second)
    assert first.mode == "RGBA"


def test_noise_does_not_corrupt_clean_alpha_or_input_pixels():
    image = Image.new("RGBA", (12, 12), (100, 50, 20, 128))
    result = degrade(image, "noise", 2)
    np.testing.assert_array_equal(np.asarray(result)[..., 3], 128)
    assert image.getpixel((0, 0)) == (100, 50, 20, 128)
