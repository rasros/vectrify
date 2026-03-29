import dataclasses
import threading
import time
from collections import deque

from svgizer.search.models import INVALID_SCORE


@dataclasses.dataclass
class SearchStats:
    strategy_name: str = ""
    model_name: str = ""
    epoch_patience: int = 0

    epoch: int = 0
    epoch_no_improve: int = 0
    llm_pressure: float = 0.0
    pool_diversity: float = 1.0  # mean pairwise Jaccard distance in active pool (0-1)
    epoch_diversity: float = 0.0  # --epoch-diversity threshold (0 = disabled)

    tasks_completed: int = 0
    accepted_count: int = 0
    pool_rejected_count: int = 0
    invalid_count: int = 0

    llm_rate: float = 0.0  # configured llm_rate, used to compute effective call rate

    llm_call_count: int = 0
    llm_calls_in_flight: int = 0  # LLM calls currently being processed by workers
    llm_invalid_count: int = 0  # LLM calls that produced invalid/unparseable SVG
    llm_accepted_count: int = 0

    mutation_call_count: int = 0
    mutation_accepted_count: int = 0

    shutting_down: bool = False

    pool_score_std: float = 0.0  # std dev of scores in active pool (0 = converged)
    epoch_variance: float = 0.0  # --epoch-variance threshold (0 = disabled)

    best_score: float = INVALID_SCORE
    # (elapsed_seconds, score) on each new-best event; kept for seeding but not graphed
    score_history: deque = dataclasses.field(default_factory=lambda: deque(maxlen=80))

    # Captured from logging by DashboardLogHandler
    recent_events: deque = dataclasses.field(default_factory=lambda: deque(maxlen=8))

    start_time: float = dataclasses.field(default_factory=time.monotonic)
    _lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def accept_rate(self) -> float:
        return (
            self.accepted_count / self.tasks_completed if self.tasks_completed else 0.0
        )

    def pool_rejected_rate(self) -> float:
        return (
            self.pool_rejected_count / self.tasks_completed
            if self.tasks_completed
            else 0.0
        )

    def invalid_rate(self) -> float:
        return (
            self.invalid_count / self.tasks_completed if self.tasks_completed else 0.0
        )

    def llm_valid_rate(self) -> float:
        valid = self.llm_call_count - self.llm_invalid_count
        return valid / self.llm_call_count if self.llm_call_count else 0.0

    def llm_accept_rate(self) -> float:
        return (
            self.llm_accepted_count / self.llm_call_count
            if self.llm_call_count
            else 0.0
        )

    def effective_llm_rate(self) -> float:
        """Actual fraction of tasks that call the LLM (pressure * llm_rate)."""
        return self.llm_pressure * self.llm_rate

    def mutation_accept_rate(self) -> float:
        return (
            self.mutation_accepted_count / self.mutation_call_count
            if self.mutation_call_count
            else 0.0
        )

    def stagnation_fraction(self) -> float:
        if self.epoch_patience <= 0:
            return 0.0
        return min(1.0, self.epoch_no_improve / self.epoch_patience)
