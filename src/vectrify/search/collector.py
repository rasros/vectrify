import csv
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from vectrify.score.metrics import FRONT_SCORE

if TYPE_CHECKING:
    from vectrify.search.models import Result, SearchNode
from vectrify.search.stats import SearchStats

log = logging.getLogger(__name__)


def _best_score(s: "SearchStats") -> float | str:
    """Blank rather than 'inf' so the CSV stays numeric for plotting."""
    return "" if s.best_score >= math.inf else s.best_score


def _rounded(field: str, digits: int) -> Callable[["SearchStats"], float]:
    return lambda s: round(getattr(s, field), digits)


# Column name -> how to read it off SearchStats. The CSV header and every row
# are both derived from this, so they cannot fall out of alignment.
# One column per thing a reader of the finished file can use.
#
# Configuration does not belong here: --epoch-patience, --epoch-max-tasks and
# the seed batch size were each repeated identically on every row, 2112 of them
# in one run, and a reader cannot tell a setting that never moved from a
# measurement that happened not to.
#
# Nor does momentary state. Calls in flight is a live gauge -- it says what the
# workers are doing right now -- and in a finished file it is a sample of a
# quantity that was oscillating, which invites reading a trend into noise.
#
# Nor does anything another column already says. Seeds completed is
# llm_call_count within an epoch, counted a second way, and pairing it with a
# constant target was two columns describing one thing. All of these remain on
# SearchStats, where the dashboard reads them to draw progress against.
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
    "mutation_call_count": lambda s: s.mutation_call_count,
    "mutation_accepted_count": lambda s: s.mutation_accepted_count,
    "epoch": lambda s: s.epoch,
    "epoch_no_improve": lambda s: s.epoch_no_improve,
    "phase": lambda s: s.phase,
    "pool_diversity": _rounded("pool_diversity", 4),
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
        epoch_max_tasks: int | None,
    ) -> None:
        s = self._stats
        s.epoch_max_tasks = epoch_max_tasks or 0

    def on_evaluator_best(self, score: float, *, elapsed: float) -> None:
        """The evaluator found a candidate better than anything it had seen.

        The only event that moves the run's best, because the evaluator's score
        is the only score. Accepting a candidate used to move it, on a blended
        proxy that nothing ranks by any more -- and once that went, nothing was
        left to record a best at all: `best_score` came out empty on all 2112
        rows of a real run.
        """
        s = self._stats
        s.best_score = score
        s.score_history.append((elapsed, score))
        self._flush_row()

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
            panel = node.metrics.get(FRONT_SCORE)
            if panel is not None:
                s.best_score = panel
                s.score_history.append((elapsed, panel))
            self._flush_row()
        else:
            self._maybe_flush(is_llm=bool(llm_type))

    def on_no_improve_reset(self) -> None:
        self._stats.epoch_no_improve = 0

    # ── Pool state events ─────────────────────────────────────────────────────

    def on_pool_state(self, *, diversity: float) -> None:
        self._stats.pool_diversity = diversity

    def on_epoch_transition(self, epoch: int) -> None:
        s = self._stats
        s.epoch = epoch
        s.epoch_no_improve = 0
        self._flush_row()

    def on_idle(self, *, llm_in_flight: int) -> None:
        """Called ~every 0.2 s when the result queue is empty."""
        self._stats.llm_calls_in_flight = llm_in_flight

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
