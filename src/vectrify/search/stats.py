import dataclasses
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping

from vectrify.search.models import INVALID_SCORE


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _valid_llm_calls(c: Mapping[str, float]) -> float:
    return c.get("llm_call_count", 0.0) - c.get("llm_invalid_count", 0.0)


# Derived rate name -> (numerator from raw counters, denominator counter).
# Single source of truth: SearchStats reads live counters through it, and
# scripts/plot_run.py reads stats.csv rows through it, so the two cannot drift.
RATE_SPECS: dict[str, tuple[Callable[[Mapping[str, float]], float], str]] = {
    "accept_rate": (lambda c: c.get("accepted_count", 0.0), "tasks_completed"),
    "pool_rejected_rate": (
        lambda c: c.get("pool_rejected_count", 0.0),
        "tasks_completed",
    ),
    "invalid_rate": (lambda c: c.get("invalid_count", 0.0), "tasks_completed"),
    "llm_valid_rate": (_valid_llm_calls, "llm_call_count"),
    "llm_accept_rate": (lambda c: c.get("llm_accepted_count", 0.0), "llm_call_count"),
    "mutation_accept_rate": (
        lambda c: c.get("mutation_accepted_count", 0.0),
        "mutation_call_count",
    ),
}


def derived_rates(counts: Mapping[str, float]) -> dict[str, float]:
    """Compute every derived rate from a mapping of raw counters."""
    return {
        name: _rate(numerator(counts), counts.get(denominator, 0.0))
        for name, (numerator, denominator) in RATE_SPECS.items()
    }


def score_std(scores: list[float]) -> float:
    """Population standard deviation; 0.0 for fewer than two samples."""
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    return math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))


@dataclasses.dataclass
class SearchStats:
    strategy_name: str = ""
    model_name: str = ""
    epoch_patience: int = 0

    epoch: int = 0
    epoch_no_improve: int = 0
    # "seed" while the epoch's LLM batch is running, "local" while its children
    # are being refined by mutation and crossover.
    phase: str = "seed"
    seeds_completed: int = 0
    seeds_target: int = 0
    pool_diversity: float = 1.0
    epoch_diversity: float = 0.0

    tasks_completed: int = 0
    accepted_count: int = 0
    pool_rejected_count: int = 0
    invalid_count: int = 0

    llm_call_count: int = 0
    llm_calls_in_flight: int = 0
    llm_invalid_count: int = 0
    llm_accepted_count: int = 0

    mutation_call_count: int = 0
    mutation_accepted_count: int = 0

    shutting_down: bool = False
    pool_score_std: float = 0.0

    best_score: float = INVALID_SCORE
    score_history: deque = dataclasses.field(default_factory=lambda: deque(maxlen=80))
    recent_events: deque = dataclasses.field(default_factory=lambda: deque(maxlen=8))

    start_time: float = dataclasses.field(default_factory=time.monotonic)
    _lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def _rate_of(self, name: str) -> float:
        numerator, denominator = RATE_SPECS[name]
        counts = vars(self)
        return _rate(numerator(counts), counts.get(denominator, 0.0))

    def derived_rates(self) -> dict[str, float]:
        return derived_rates(vars(self))

    def accept_rate(self) -> float:
        return self._rate_of("accept_rate")

    def pool_rejected_rate(self) -> float:
        return self._rate_of("pool_rejected_rate")

    def invalid_rate(self) -> float:
        return self._rate_of("invalid_rate")

    def llm_valid_rate(self) -> float:
        return self._rate_of("llm_valid_rate")

    def llm_accept_rate(self) -> float:
        return self._rate_of("llm_accept_rate")

    def mutation_accept_rate(self) -> float:
        return self._rate_of("mutation_accept_rate")

    def seed_fraction(self) -> float:
        """Progress through the current epoch's LLM seed batch, in [0, 1]."""
        if self.seeds_target <= 0:
            return 1.0
        return min(1.0, self.seeds_completed / self.seeds_target)

    def stagnation_fraction(self) -> float:
        if self.epoch_patience <= 0:
            return 0.0
        return min(1.0, self.epoch_no_improve / self.epoch_patience)
