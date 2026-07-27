import pytest

from vectrify.cli import parse_args
from vectrify.search import StrategyType


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


def test_image_long_side_zero_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--image-long-side", "0"])


@pytest.mark.parametrize("rate", ["-0.1", "1.5"])
def test_llm_rate_out_of_range_raises(rate):
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--llm-rate", rate])


@pytest.mark.parametrize("rate", ["0.0", "1.0", "0.5"])
def test_llm_rate_in_range_accepted(rate):
    assert parse_args(["img.png", "--llm-rate", rate]).llm_rate == float(rate)


@pytest.mark.parametrize("keep", ["0", "0.0", "-0.5", "1.1"])
def test_cull_keep_out_of_range_raises(keep):
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--strategy", "beam", "--cull-keep", keep])


def test_cull_keep_upper_bound_accepted():
    args = parse_args(["img.png", "--strategy", "beam", "--cull-keep", "1.0"])
    assert args.cull_keep == 1.0


# Defaults are pinned as literals so a change to any default is a visible,
# deliberate edit here rather than silently tracking the constant.
def test_defaults_pinned():
    args = parse_args(["img.png"])
    assert args.pool_size == 100
    assert args.epoch_diversity == 0.0
    assert args.epoch_patience == 20
    assert args.max_epochs == 4


def test_default_llm_rate_tracks_workers():
    # Small worker count: 2/4 = 0.5 clamped to the 0.2 cap.
    assert parse_args(["img.png", "--workers", "4"]).llm_rate == 0.2
    # Larger worker count derives below the cap and scales with --workers.
    assert parse_args(["img.png", "--workers", "40"]).llm_rate == 2 / 40


def test_explicit_llm_rate_overrides_workers_derivation():
    args = parse_args(["img.png", "--workers", "40", "--llm-rate", "0.5"])
    assert args.llm_rate == 0.5


@pytest.mark.parametrize(
    ("flag", "attr", "default", "flagged"),
    [
        ("--no-dashboard", "dashboard", True, False),
        ("--debug", "debug", False, True),
    ],
)
def test_boolean_flags(flag, attr, default, flagged):
    assert getattr(parse_args(["img.png"]), attr) is default
    assert getattr(parse_args(["img.png", flag]), attr) is flagged


def test_max_epochs_parsed():
    args = parse_args(["img.png", "--max-epochs", "10"])
    assert args.max_epochs == 10


def test_max_epochs_zero_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--max-epochs", "0"])


def test_max_epochs_negative_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--max-epochs", "-1"])
