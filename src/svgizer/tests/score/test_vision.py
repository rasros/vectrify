import io

import pytest
from PIL import Image

from svgizer.score.vision import VisionScorer


class _TinyVisionScorer(VisionScorer):
    """VisionScorer backed by a tiny SiglipModel with random weights — no download."""

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
