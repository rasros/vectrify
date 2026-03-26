import pytest

from svgizer.cli import (
    DEFAULT_DIVERSITY_BOOST_THRESHOLD,
    DEFAULT_DIVERSITY_THRESHOLD,
    DEFAULT_LLM_RATE,
    DEFAULT_POOL_SIZE,
    parse_args,
)
from svgizer.search import StrategyType


def test_parse_args_basic():
    args = parse_args(["input.png", "--workers", "4", "--max-accepts", "10"])
    assert args.image == "input.png"
    assert args.workers == 4
    assert args.max_accepts == 10
    assert args.strategy == StrategyType.NSGA.value


# ---------------------------------------------------------------------------
# max_wall_seconds conversion
# ---------------------------------------------------------------------------


def test_max_wall_seconds_zero_becomes_none():
    args = parse_args(["img.png", "--max-wall-seconds", "0"])
    assert args.max_wall_seconds is None


def test_max_wall_seconds_negative_becomes_none():
    args = parse_args(["img.png", "--max-wall-seconds", "-10"])
    assert args.max_wall_seconds is None


def test_max_wall_seconds_positive_kept():
    args = parse_args(["img.png", "--max-wall-seconds", "120"])
    assert args.max_wall_seconds == 120.0


# ---------------------------------------------------------------------------
# Boundary validation errors
# ---------------------------------------------------------------------------


def test_max_accepts_zero_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--max-accepts", "0"])


def test_workers_zero_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--workers", "0"])


def test_pool_size_zero_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--pool-size", "0"])


def test_image_long_side_negative_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--image-long-side", "-1"])


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_pool_size():
    args = parse_args(["img.png"])
    assert args.pool_size == DEFAULT_POOL_SIZE


def test_default_llm_rate():
    args = parse_args(["img.png"])
    assert args.llm_rate == DEFAULT_LLM_RATE


def test_default_diversity_thresholds():
    args = parse_args(["img.png"])
    assert args.diversity_threshold == DEFAULT_DIVERSITY_THRESHOLD
    assert args.diversity_boost_threshold == DEFAULT_DIVERSITY_BOOST_THRESHOLD


def test_default_patience_zero():
    args = parse_args(["img.png"])
    assert args.patience == 0
