import io

import numpy as np
import pytest
from PIL import Image

from svgizer.score.vision import VisionScorer, _apply_hot_colormap


class _TinyVisionScorer(VisionScorer):
    def _load_dependencies(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                SiglipConfig,
                SiglipImageProcessor,
                SiglipModel,
                SiglipTextConfig,
                SiglipVisionConfig,
            )

            vision_config = SiglipVisionConfig()
            vision_config.hidden_size = 8
            vision_config.intermediate_size = 16
            vision_config.num_hidden_layers = 1
            vision_config.num_attention_heads = 2
            vision_config.image_size = 16
            vision_config.patch_size = 16

            text_config = SiglipTextConfig()
            text_config.hidden_size = 8
            text_config.intermediate_size = 16
            text_config.num_hidden_layers = 1
            text_config.num_attention_heads = 2
            text_config.vocab_size = 32

            config = SiglipConfig()
            config.vision_config = vision_config
            config.text_config = text_config
            model = SiglipModel(config)
            model.eval()
            processor = SiglipImageProcessor(size={"height": 16, "width": 16})

            self._model = model
            self._processor = processor
            self._torch = torch
            self._device_str = "cpu"
        except ImportError as e:
            raise ImportError(f"transformers or torch not available: {e}") from e


@pytest.fixture(scope="module")
def scorer():
    s = _TinyVisionScorer(device="cpu")
    try:
        s.validate_environment()
    except ImportError as e:
        pytest.skip(f"Vision scorer unavailable: {e}")
    return s


def _png(color: str, size: int = 32) -> bytes:
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_identical_image_scores_zero(scorer):
    ref_img = Image.new("RGB", (32, 32), color="red")
    ref = scorer.prepare_reference(ref_img)
    score = scorer.score(ref, _png("red"))
    assert score == pytest.approx(0.0, abs=0.05)


def test_different_image_scores_higher(scorer):
    ref_img = Image.new("RGB", (32, 32), color="red")
    ref = scorer.prepare_reference(ref_img)
    score_same = scorer.score(ref, _png("red"))
    score_diff = scorer.score(ref, _png("blue"))
    assert score_diff > score_same


def test_score_is_in_unit_range(scorer):
    ref_img = Image.new("RGB", (32, 32), color="green")
    ref = scorer.prepare_reference(ref_img)
    for color in ("green", "red", "blue", "white", "black"):
        score = scorer.score(ref, _png(color))
        assert 0.0 <= score <= 1.0, f"Score out of range for {color}: {score}"


def test_score_handles_size_mismatch(scorer):
    ref_img = Image.new("RGB", (64, 64), color="red")
    ref = scorer.prepare_reference(ref_img)
    score = scorer.score(ref, _png("red", size=16))
    assert 0.0 <= score <= 1.0


def test_validate_environment_does_not_raise(scorer):
    scorer.validate_environment()


def test_load_is_idempotent(scorer):
    ref_img = Image.new("RGB", (32, 32), color="white")
    ref1 = scorer.prepare_reference(ref_img)
    ref2 = scorer.prepare_reference(ref_img)
    s1 = scorer.score(ref1, _png("white"))
    s2 = scorer.score(ref2, _png("white"))
    assert s1 == pytest.approx(s2, abs=1e-6)


def test_prepare_reference_includes_patch_embeddings(scorer):
    ref_img = Image.new("RGB", (32, 32), color="red")
    ref = scorer.prepare_reference(ref_img)
    assert ref.patch_embeddings is not None
    assert ref.patch_embeddings.shape[0] == 1  # 1 patch
    assert ref.grid_hw == (1, 1)


def test_patch_embeddings_are_unit_norm(scorer):
    ref_img = Image.new("RGB", (32, 32), color="blue")
    ref = scorer.prepare_reference(ref_img)
    assert ref.patch_embeddings is not None
    norms = ref.patch_embeddings.norm(dim=-1)
    assert norms == pytest.approx(1.0, abs=1e-5)


def test_diff_heatmap_returns_valid_png(scorer):
    ref_img = Image.new("RGB", (32, 32), color="red")
    ref = scorer.prepare_reference(ref_img)
    png = scorer.diff_heatmap(ref, _png("blue"), long_side=64)
    assert png is not None
    img = Image.open(io.BytesIO(png))
    assert img.mode == "RGB"
    assert img.size[0] > 0
    assert img.size[1] > 0


def test_diff_heatmap_identical_images_are_dark(scorer):
    ref_img = Image.new("RGB", (32, 32), color="green")
    ref = scorer.prepare_reference(ref_img)
    png = scorer.diff_heatmap(ref, _png("green"), long_side=64)
    assert png is not None
    arr = np.array(Image.open(io.BytesIO(png)))
    assert arr.mean() < 30.0


def test_diff_heatmap_different_images_are_brighter(scorer):
    ref_img = Image.new("RGB", (32, 32), color="red")
    ref = scorer.prepare_reference(ref_img)
    same_png = scorer.diff_heatmap(ref, _png("red"), long_side=64)
    diff_png = scorer.diff_heatmap(ref, _png("blue"), long_side=64)
    assert same_png is not None
    assert diff_png is not None
    mean_same = np.array(Image.open(io.BytesIO(same_png))).mean()
    mean_diff = np.array(Image.open(io.BytesIO(diff_png))).mean()
    assert mean_diff >= mean_same


def test_diff_heatmap_respects_long_side(scorer):
    ref_img = Image.new("RGB", (32, 32), color="red")
    ref = scorer.prepare_reference(ref_img)
    png = scorer.diff_heatmap(ref, _png("blue"), long_side=32)
    assert png is not None
    img = Image.open(io.BytesIO(png))
    assert max(img.size) == 32


def test_hot_colormap_zero_is_black():
    arr = np.zeros((1, 1), dtype=np.float32)
    rgb = _apply_hot_colormap(arr)
    assert rgb[0, 0].tolist() == [0, 0, 0]


def test_hot_colormap_one_is_white():
    arr = np.ones((1, 1), dtype=np.float32)
    rgb = _apply_hot_colormap(arr)
    assert rgb[0, 0].tolist() == [255, 255, 255]


def test_hot_colormap_third_is_red():
    arr = np.full((1, 1), 1.0 / 3.0, dtype=np.float32)
    rgb = _apply_hot_colormap(arr)
    r, g, b = rgb[0, 0]
    assert r == 255
    assert g < 10
    assert b < 10


def test_hot_colormap_output_shape_matches_input():
    arr = np.random.rand(10, 20).astype(np.float32)
    rgb = _apply_hot_colormap(arr)
    assert rgb.shape == (10, 20, 3)
    assert rgb.dtype == np.uint8


def test_diff_heatmap_returns_none_when_patch_embeddings_none(scorer):
    from svgizer.score.vision import VisionReference

    ref_img = Image.new("RGB", (32, 32), color="red")
    ref = scorer.prepare_reference(ref_img)

    # Manually strip patch embeddings to simulate a model without vision_model
    ref_no_patches = VisionReference(
        image=ref.image,
        embedding=ref.embedding,
        patch_embeddings=None,
        grid_hw=None,
    )
    result = scorer.diff_heatmap(ref_no_patches, _png("blue"), long_side=32)
    assert result is None


def test_diff_heatmap_returns_none_on_grid_mismatch(scorer):
    import torch

    from svgizer.score.vision import VisionReference

    ref_img = Image.new("RGB", (32, 32), color="red")
    ref = scorer.prepare_reference(ref_img)

    if ref.patch_embeddings is None:
        pytest.skip("Patch embeddings not available")

    fake_patch_embs = torch.zeros(4, ref.patch_embeddings.shape[-1])
    ref_wrong_grid = VisionReference(
        image=ref.image,
        embedding=ref.embedding,
        patch_embeddings=fake_patch_embs,
        grid_hw=(2, 2),
    )
    result = scorer.diff_heatmap(ref_wrong_grid, _png("blue"), long_side=32)
    assert result is None
