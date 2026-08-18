import dataclasses
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping


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


@dataclasses.dataclass
class SearchStats:
    strategy_name: str = ""
    model_name: str = ""
    epoch_patience: int = 0

    epoch: int = 0
    epochs: int = 0
    epoch_no_improve: int = 0
    # "seed" while the epoch's LLM batch is running, "local" while its children
    # are being refined by mutation and crossover.
    phase: str = "seed"
    seeds_completed: int = 0
    seeds_target: int = 0
    pool_diversity: float = 1.0
    epoch_max_tasks: int = 0

    tasks_completed: int = 0
    accepted_count: int = 0
    pool_rejected_count: int = 0
    invalid_count: int = 0
    # Rejected for measuring exactly as its parent: the markup changed and the
    # drawing did not. Counted apart from invalid because they mean opposite
    # things -- one candidate was broken, the other was never new -- and one
    # run spent 58% of itself on the second kind.
    unchanged_count: int = 0

    llm_call_count: int = 0
    llm_calls_in_flight: int = 0
    llm_invalid_count: int = 0
    llm_accepted_count: int = 0

    mutation_call_count: int = 0
    mutation_accepted_count: int = 0

    shutting_down: bool = False

    # The evaluator's verdict, and what it has been doing. It is the only score
    # a run has, and it speaks a few dozen times rather than once per task, so
    # how long since it last approved anything is as much of the picture as the
    # number itself.
    best_score: float = math.inf
    eval_checks: int = 0
    eval_checks_without_gain: int = 0
    eval_patience: int = 0
    epoch_tasks: int = 0
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

    def eval_patience_fraction(self) -> float:
        """Progress toward the evaluator ending the epoch, in [0, 1]."""
        if self.eval_patience <= 0:
            return 0.0
        return min(1.0, self.eval_checks_without_gain / self.eval_patience)

    def epoch_budget_fraction(self) -> float:
        """Progress toward the epoch's task ceiling, in [0, 1]."""
        if self.epoch_max_tasks <= 0:
            return 0.0
        return min(1.0, self.epoch_tasks / self.epoch_max_tasks)

    def nearest_epoch_end(self) -> tuple[str, float]:
        """Which criterion is closest to ending the epoch, and how close.

        A run has three that can fire and no way to tell which is doing the
        work: across four runs the answer changed from the evaluator to
        staleness with no setting changed, only the acceptance rate moving.
        """
        live = [
            ("staleness", self.stagnation_fraction()),
            ("evaluator", self.eval_patience_fraction()),
            ("budget", self.epoch_budget_fraction()),
        ]
        return max(live, key=lambda pair: pair[1])
