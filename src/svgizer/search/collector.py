import csv
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from svgizer.search.models import Result, SearchNode
    from svgizer.search.stats import SearchStats

log = logging.getLogger(__name__)

# Wide-format columns written to stats.csv (one row per flush event).
STATS_COLUMNS = [
    "elapsed",
    "tasks_completed",
    "accepted_count",
    "pool_rejected_count",
    "invalid_count",
    "best_score",
    "llm_call_count",
    "llm_accepted_count",
    "llm_invalid_count",
    "llm_in_flight",
    "mutation_call_count",
    "mutation_accepted_count",
    "epoch",
    "epoch_no_improve",
    "llm_pressure",
    "llm_rate",
    "pool_diversity",
    "pool_score_std",
    "epoch_patience",
    "epoch_diversity",
    "epoch_variance",
]


class StatCollector:
    """Translates engine events into SearchStats mutations and appends rows to
    a wide-format stats.csv (one column per metric, one row per flush event).

    Flush events:
    - Every LLM call completion.
    - Every 100th task completion.
    - Every new-best score.
    - Every epoch transition.
    - On shutdown.
    """

    def __init__(self, stats: "SearchStats", run_dir: Path | None = None) -> None:
        self._stats = stats
        self._run_dir = run_dir
        self._csv_ready = False  # True once the header has been written

    @property
    def stats(self) -> "SearchStats":
        return self._stats

    # ── Pre-run configuration (called by runner before engine.run()) ──────────

    def configure_run(
        self,
        *,
        llm_rate: float,
        epoch_diversity: float,
        epoch_variance: float,
    ) -> None:
        s = self._stats
        s.llm_rate = llm_rate
        s.epoch_diversity = epoch_diversity
        s.epoch_variance = epoch_variance

    def seed_initial_score(self, best_score: float, elapsed: float = 0.0) -> None:
        s = self._stats
        s.best_score = best_score
        s.score_history.append((elapsed, best_score))

    # ── Engine lifecycle ──────────────────────────────────────────────────────

    def on_run_start(self, *, start_time: float, epoch_patience: int) -> None:
        s = self._stats
        s.start_time = start_time
        s.epoch_patience = epoch_patience

    def on_shutdown(self) -> None:
        self._stats.shutting_down = True
        self._flush_row()

    # ── Per-task events ───────────────────────────────────────────────────────

    def on_llm_pressure(self, pressure: float) -> None:
        self._stats.llm_pressure = pressure

    def on_result(
        self,
        res: "Result",
        *,
        tasks_completed: int,
        epoch_no_improve: int,
        llm_in_flight: int,
    ) -> None:
        """Called for every completed result (before accept/reject decision)."""
        s = self._stats
        s.tasks_completed = tasks_completed
        s.epoch_no_improve = epoch_no_improve
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

    def on_pool_state(self, *, diversity: float, score_std: float) -> None:
        s = self._stats
        s.pool_diversity = diversity
        s.pool_score_std = score_std

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
            mean = sum(valid_scores) / len(valid_scores)
            s.pool_score_std = math.sqrt(
                sum((v - mean) ** 2 for v in valid_scores) / len(valid_scores)
            )

    # ── CSV helpers ───────────────────────────────────────────────────────────

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
                bs = s.best_score
                writer.writerow(
                    {
                        "elapsed": round(s.elapsed(), 2),
                        "tasks_completed": s.tasks_completed,
                        "accepted_count": s.accepted_count,
                        "pool_rejected_count": s.pool_rejected_count,
                        "invalid_count": s.invalid_count,
                        "best_score": "" if bs == float("inf") else bs,
                        "llm_call_count": s.llm_call_count,
                        "llm_accepted_count": s.llm_accepted_count,
                        "llm_invalid_count": s.llm_invalid_count,
                        "llm_in_flight": s.llm_calls_in_flight,
                        "mutation_call_count": s.mutation_call_count,
                        "mutation_accepted_count": s.mutation_accepted_count,
                        "epoch": s.epoch,
                        "epoch_no_improve": s.epoch_no_improve,
                        "llm_pressure": round(s.llm_pressure, 4),
                        "llm_rate": round(s.llm_rate, 4),
                        "pool_diversity": round(s.pool_diversity, 4),
                        "pool_score_std": round(s.pool_score_std, 6),
                        "epoch_patience": s.epoch_patience,
                        "epoch_diversity": round(s.epoch_diversity, 4),
                        "epoch_variance": round(s.epoch_variance, 6),
                    }
                )
        except Exception as e:
            log.warning(f"Failed to write stats row: {e}")
