import csv
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from vectrify.search.models import INVALID_SCORE
from vectrify.search.stats import score_std

if TYPE_CHECKING:
    from vectrify.search.models import Result, SearchNode
    from vectrify.search.stats import SearchStats

log = logging.getLogger(__name__)


def _best_score(s: "SearchStats") -> float | str:
    """Blank rather than 'inf' so the CSV stays numeric for plotting."""
    return "" if s.best_score >= INVALID_SCORE else s.best_score


def _rounded(field: str, digits: int) -> Callable[["SearchStats"], float]:
    return lambda s: round(getattr(s, field), digits)


# Column name -> how to read it off SearchStats. The CSV header and every row
# are both derived from this, so they cannot fall out of alignment.
STATS_FIELDS: dict[str, Callable[["SearchStats"], object]] = {
    "elapsed": lambda s: round(s.elapsed(), 2),
    "tasks_completed": lambda s: s.tasks_completed,
    "accepted_count": lambda s: s.accepted_count,
    "pool_rejected_count": lambda s: s.pool_rejected_count,
    "invalid_count": lambda s: s.invalid_count,
    "best_score": _best_score,
    "llm_call_count": lambda s: s.llm_call_count,
    "llm_accepted_count": lambda s: s.llm_accepted_count,
    "llm_invalid_count": lambda s: s.llm_invalid_count,
    "llm_in_flight": lambda s: s.llm_calls_in_flight,
    "mutation_call_count": lambda s: s.mutation_call_count,
    "mutation_accepted_count": lambda s: s.mutation_accepted_count,
    "epoch": lambda s: s.epoch,
    "epoch_no_improve": lambda s: s.epoch_no_improve,
    "phase": lambda s: s.phase,
    "seeds_completed": lambda s: s.seeds_completed,
    "seeds_target": lambda s: s.seeds_target,
    "pool_diversity": _rounded("pool_diversity", 4),
    "pool_score_std": _rounded("pool_score_std", 6),
    "epoch_patience": lambda s: s.epoch_patience,
    "epoch_diversity": _rounded("epoch_diversity", 4),
    "pool_dominated": _rounded("pool_dominated", 4),
    "epoch_dominated": _rounded("epoch_dominated", 4),
}

STATS_COLUMNS = list(STATS_FIELDS)


class StatCollector:
    """Translates engine events into SearchStats mutations and appends rows to
    a wide-format stats.csv (one column per metric, one row per flush event).

    Flush events:
    - Every LLM call completion.
    - Every 100th task completion.
    - Every new-best score.
    - Every epoch transition and every seed/local phase change.
    - On shutdown.
    """

    def __init__(self, stats: "SearchStats", run_dir: Path | None = None) -> None:
        self._stats = stats
        self._run_dir = run_dir
        self._csv_ready = False  # True once the header has been written

    def configure_run(
        self,
        *,
        epoch_diversity: float,
        epoch_dominated: float,
    ) -> None:
        s = self._stats
        s.epoch_diversity = epoch_diversity
        s.epoch_dominated = epoch_dominated

    def seed_initial_score(self, best_score: float) -> None:
        s = self._stats
        s.best_score = best_score
        # The seed score is the run's t=0 datapoint by definition.
        s.score_history.append((0.0, best_score))

    def on_run_start(self, *, start_time: float, epoch_patience: int) -> None:
        s = self._stats
        s.start_time = start_time
        s.epoch_patience = epoch_patience

    def on_shutdown(self) -> None:
        self._stats.shutting_down = True
        self._flush_row()

    def on_phase(self, phase: str, seeds_target: int) -> None:
        s = self._stats
        s.phase = phase
        s.seeds_target = seeds_target
        if phase == "seed":
            s.seeds_completed = 0
        self._flush_row()

    def on_result(
        self,
        res: "Result",
        *,
        tasks_completed: int,
        epoch_no_improve: int,
        seeds_completed: int,
        llm_in_flight: int,
    ) -> None:
        """Called for every completed result (before accept/reject decision)."""
        s = self._stats
        s.tasks_completed = tasks_completed
        s.epoch_no_improve = epoch_no_improve
        s.seeds_completed = seeds_completed
        s.llm_calls_in_flight = llm_in_flight
        if res.llm_type:
            s.llm_call_count += 1
        else:
            s.mutation_call_count += 1

    def on_invalid(self, res: "Result") -> None:
        s = self._stats
        s.invalid_count += 1
        if res.llm_type:
            s.llm_invalid_count += 1
        self._maybe_flush(is_llm=bool(res.llm_type))

    def on_pool_rejected(self, *, is_llm: bool = False) -> None:
        self._stats.pool_rejected_count += 1
        self._maybe_flush(is_llm=is_llm)

    def on_accepted(
        self,
        node: "SearchNode",
        *,
        is_new_best: bool,
        elapsed: float,
        llm_type: str | None,
    ) -> None:
        s = self._stats
        s.accepted_count += 1
        if llm_type:
            s.llm_accepted_count += 1
        else:
            s.mutation_accepted_count += 1
        if is_new_best:
            s.best_score = node.score
            s.score_history.append((elapsed, node.score))
            self._flush_row()
        else:
            self._maybe_flush(is_llm=bool(llm_type))

    def on_no_improve_reset(self) -> None:
        self._stats.epoch_no_improve = 0

    # ── Pool state events ─────────────────────────────────────────────────────

    def on_pool_state(
        self, *, diversity: float, score_std: float, dominated: float
    ) -> None:
        s = self._stats
        s.pool_diversity = diversity
        s.pool_score_std = score_std
        s.pool_dominated = dominated

    def on_epoch_transition(self, epoch: int) -> None:
        s = self._stats
        s.epoch = epoch
        s.epoch_no_improve = 0
        self._flush_row()

    def on_idle(self, *, llm_in_flight: int, valid_scores: list[float]) -> None:
        """Called ~every 0.2 s when the result queue is empty."""
        s = self._stats
        s.llm_calls_in_flight = llm_in_flight
        if len(valid_scores) >= 2:
            s.pool_score_std = score_std(valid_scores)

    def _maybe_flush(self, *, is_llm: bool) -> None:
        """Flush if this is an LLM call or a task-count milestone."""
        s = self._stats
        if is_llm or s.tasks_completed % 100 == 0:
            self._flush_row()

    def _flush_row(self) -> None:
        if self._run_dir is None:
            return
        s = self._stats
        path = self._run_dir / "stats.csv"
        try:
            write_header = not self._csv_ready and not path.exists()
            with path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=STATS_COLUMNS)
                if write_header:
                    writer.writeheader()
                self._csv_ready = True
                writer.writerow({k: read(s) for k, read in STATS_FIELDS.items()})
        except Exception as e:
            log.warning(f"Failed to write stats row: {e}")
