"""Paired data has deterministic corruption and no artwork-family leakage."""

import numpy as np
import pytest
from PIL import Image

from scripts.cel_pairs import cases, composition, degrade
from vectrify.refine.cel_plan.score import render


def test_training_and_heldout_families_do_not_overlap():
    training = {case["family"] for case in cases()}
    heldout = {case["family"] for case in cases(True)}
    assert training
    assert heldout
    assert training.isdisjoint(heldout)


def test_uniform_opacity_changes_the_composition_once_and_retains_holes_and_defs():
    source = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="24" opacity="0.8">'
        '<defs><linearGradient id="r"><stop stop-color="#804020" stop-opacity="0.4"/>'
        '<stop offset="1" stop-color="#408020"/></linearGradient></defs>'
        '<path fill="url(#r)" fill-rule="evenodd" d="M4 4H28V20H4Z M12 8H20V16H12Z"/>'
        '<path fill="#804020" d="M22 4H28V20H22Z"/></svg>'
    )
    assert composition(source) == source
    original = render(source, (32, 24))
    half = render(composition(source, 0.5), (32, 24))
    np.testing.assert_allclose(half[..., 3], original[..., 3] * 0.5, atol=1 / 255)
    assert half[10:14, 14:18, 3].max() == 0
    assert half[6:18, 24:26, 3].max() <= 0.4 + 1 / 255


@pytest.mark.parametrize("opacity", [0, -0.5, 1.5, float("nan"), float("inf")])
def test_invalid_composition_opacity_is_rejected(opacity):
    with pytest.raises(ValueError, match="Composition opacity must"):
        composition('<svg width="32" height="24"/>', opacity)


def test_pair_variant_precedes_clean_render_and_input_only_alpha_corruption(
    tmp_path, monkeypatch
):
    from scripts import cel_pairs

    source = (
        '<svg width="32" height="24"><path fill="#804020" fill-rule="evenodd" '
        'd="M4 4H28V20H4Z M12 8H20V16H12Z"/></svg>'
    )
    (tmp_path / "case.svg").write_text(source)
    monkeypatch.setattr(cel_pairs, "DATA", tmp_path)
    case = {"file": "case.svg", "family": "original-family"}
    default_svg, original, _ = cel_pairs.pair(case, "clean", 32)
    assert default_svg == source
    variant_svg, clean, corrupted = cel_pairs.pair(case, "alpha", 32, 0.5)
    assert variant_svg != source
    np.testing.assert_array_equal(
        np.asarray(clean),
        (render(variant_svg, clean.size) * 255).round().astype(np.uint8),
    )
    np.testing.assert_allclose(
        np.asarray(clean)[..., 3], np.asarray(original)[..., 3] * 0.5, atol=1
    )
    assert not np.array_equal(np.asarray(clean), np.asarray(corrupted))
    np.testing.assert_array_equal(np.asarray(corrupted)[10:14, 14:18, 3], 0)


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
