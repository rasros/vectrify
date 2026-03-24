import sys

from svgizer.cli import parse_args
from svgizer.search import StrategyType


def test_parse_args(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["svgizer", "input.png", "--workers", "4", "--max-accepts", "10"]
    )
    args = parse_args()
    assert args.image == "input.png"
    assert args.workers == 4
    assert args.max_accepts == 10
    assert args.strategy == StrategyType.GENETIC.value
