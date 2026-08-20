import pytest

from vectrify.cli import parse_args


def test_parse_args_basic():
    args = parse_args(["input.png", "--workers", "4"])
    assert args.image == "input.png"
    assert args.workers == 4


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


def test_seeds_defaults_to_a_fixed_number_not_a_share_of_the_pool():
    """It derived from --pool-size, which tied the LLM budget to a number chosen
    for entirely separate reasons: widening the pool silently bought more LLM
    calls."""
    assert parse_args(["img.png"]).seeds == 5
    assert parse_args(["img.png", "--pool-size", "400"]).seeds == 5


def test_seeds_is_accepted():
    assert parse_args(["img.png", "--seeds", "20"]).seeds == 20


def test_negative_seeds_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--seeds", "-1"])


def test_seeds_zero_requires_resume():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--seeds", "0"])
    assert parse_args(["img.png", "--seeds", "0", "--resume"]).seeds == 0


# Defaults are pinned as literals so a change to any default is a visible,
# deliberate edit here rather than silently tracking the constant.
def test_defaults_pinned():
    args = parse_args(["img.png"])
    assert args.pool_size == 100
    # Off: a pool collapses into agreement long before it stops improving, so
    # a threshold that looks safe ends search while it is still working.
    assert args.epoch_max_tasks is None
    # Unset: it was binding before the limits that describe the search.
    assert args.max_total_tasks is None
    # Patience counts local tasks only; a seed batch is not a hill-climb and
    # cannot go stale. Raised from 200 once the gap between improvements was
    # measured: 142 tasks at the 95th percentile and 497 at the longest seen.
    assert args.epoch_patience == 500
    # A ceiling, not a budget: what ends a run is an epoch that stops
    # improving the evaluator's best. At 4 this bound two measured runs of the
    # same image inside 13 minutes of a one-hour wall, one of them still
    # improving on the last node it produced.
    assert args.epochs == 50
    # Any improvement at all counts, and one epoch that buys none ends the run.
    assert args.epoch_improvement == 0.0
    assert args.epoch_improvement_patience == 2
    # The evaluator is asked every 2000 tasks and five consecutive refusals end
    # the epoch, so an epoch is roughly 10,000 tasks of local search. Armed
    # because leaving it unset let one run drift for 145,000 tasks past the
    # evaluator's best while the front it was shown degraded 40%.
    assert args.epoch_eval_interval == 2000
    assert args.epoch_eval_patience == 5


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


def test_epochs_parsed():
    args = parse_args(["img.png", "--epochs", "10"])
    assert args.epochs == 10


def test_epochs_zero_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--epochs", "0"])


def test_epochs_negative_raises():
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--epochs", "-1"])


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


@pytest.mark.parametrize("value", ["-1", "0"])
def test_resolution_llm_must_be_positive(value):
    with pytest.raises(SystemExit):
        parse_args(["img.png", "--resolution-llm", value])


def test_resolution_llm_defaults_below_the_vision_tile_boundary():
    """Vision pricing tiles at 512px, so anything above triples image cost for
    detail that never reaches the scorer."""
    assert parse_args(["img.png"]).resolution_llm <= 512
