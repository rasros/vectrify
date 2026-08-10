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


def test_seeds_defaults_to_none_so_the_runner_derives_it():
    assert parse_args(["img.png"]).seeds is None


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
    assert args.epoch_diversity == 0.0
    # Patience counts local tasks only; a seed batch is not a hill-climb and
    # cannot go stale.
    assert args.epoch_patience == 200
    # Epoch 0 produced 82% of the total gain on the reference image and
    # epochs 2-3 produced 3.7% for a third of the wall clock.
    assert args.epochs == 2


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
