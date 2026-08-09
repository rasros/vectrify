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


def test_resolution_negative_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--resolution", "-1"])


def test_resolution_zero_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--resolution", "0"])


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
    # Patience counts tasks, not LLM calls: 200 tasks is roughly what 20 LLM
    # calls came to at the default rate, but no longer moves with --llm-rate.
    assert args.epoch_patience == 200
    # Epoch 0 produced 82% of the total gain on the reference image and
    # epochs 2-3 produced 3.7% for a third of the wall clock.
    assert args.max_epochs == 2


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


def test_tournament_size_defaults_to_two():
    from vectrify.cli import DEFAULT_TOURNAMENT_SIZE

    args = parse_args(["in.png"])
    assert args.tournament_size == DEFAULT_TOURNAMENT_SIZE == 2


def test_tournament_size_is_accepted():
    args = parse_args(["in.png", "--tournament-size", "4"])
    assert args.tournament_size == 4


def test_tournament_size_below_two_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["in.png", "--tournament-size", "1"])


def test_tournament_size_is_nsga_only():
    with pytest.raises(SystemExit):
        parse_args(["in.png", "--strategy", "beam", "--tournament-size", "4"])


def test_default_tournament_size_does_not_trip_the_beam_check():
    """The nsga-only guard must compare against the default, not against zero --
    a default of 2 would otherwise look 'set' and break every beam run.
    """
    args = parse_args(["in.png", "--strategy", "beam"])
    assert args.strategy == "beam"


@pytest.mark.parametrize("value", ["-1", "0"])
def test_resolution_llm_must_be_positive(value):
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--resolution-llm", value])


def test_resolution_llm_defaults_below_the_vision_tile_boundary():
    """Vision pricing tiles at 512px, so anything above triples image cost for
    detail that never reaches the scorer."""
    assert parse_args(["img.png"]).resolution_llm <= 512
