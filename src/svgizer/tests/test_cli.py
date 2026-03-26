import pytest

from svgizer.cli import (
    DEFAULT_EPOCH_DIVERSITY,
    DEFAULT_LLM_RATE,
    DEFAULT_POOL_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    parse_args,
)
from svgizer.search import StrategyType


def test_parse_args_basic():
    args = parse_args(["input.png", "--workers", "4"])
    assert args.image == "input.png"
    assert args.workers == 4
    assert args.strategy == StrategyType.NSGA.value


def test_max_wall_seconds_zero_becomes_none():
    args = parse_args(["img.png", "--max-wall-seconds", "0"])
    assert args.max_wall_seconds is None


def test_max_wall_seconds_negative_becomes_none():
    args = parse_args(["img.png", "--max-wall-seconds", "-10"])
    assert args.max_wall_seconds is None


def test_max_wall_seconds_positive_kept():
    args = parse_args(["img.png", "--max-wall-seconds", "120"])
    assert args.max_wall_seconds == 120.0


def test_workers_zero_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--workers", "0"])


def test_pool_size_zero_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--pool-size", "0"])


def test_image_long_side_negative_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--image-long-side", "-1"])


def test_default_pool_size():
    args = parse_args(["img.png"])
    assert args.pool_size == DEFAULT_POOL_SIZE


def test_default_llm_rate():
    args = parse_args(["img.png"])
    assert args.llm_rate == DEFAULT_LLM_RATE


def test_default_similarity_thresholds():
    args = parse_args(["img.png"])
    assert args.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
    assert args.epoch_diversity == DEFAULT_EPOCH_DIVERSITY


def test_default_epoch_patience_zero():
    args = parse_args(["img.png"])
    assert args.epoch_patience == 0


def test_max_epochs_parsed():
    args = parse_args(["img.png", "--max-epochs", "5"])
    assert args.max_epochs == 5


def test_max_epochs_unlimited():
    args = parse_args(["img.png", "--max-epochs", "-1"])
    assert args.max_epochs == -1


def test_max_epochs_zero_means_one_epoch():
    args = parse_args(["img.png", "--max-epochs", "0"])
    assert args.max_epochs == 0


def test_max_epochs_below_minus_one_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--max-epochs", "-2"])
