"""Hill climbing over the selected paths' nodes, as a callable.

``run_search`` has worker processes apply one move at a time to the current
paths, scores every child in this process, and lets a child replace the
current paths when it scores no worse. With simplify on, a share of the tasks
try removing a point instead; a removal is kept while the score stays within
``tolerance`` of where the run started. That budget is shared by every removal
in the run, so the result is never worse than the start by more than it, and
nudges that improve the fit earn room for further removals.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass

from vectrify.vector.nodes import Paths
from vectrify.vector.operators import (
    Exp3Policy,
    FixedWeightPolicy,
    GradedReward,
    OperatorPolicy,
)
from vectrify.vector.worker import (
    Mutant,
    Renderer,
    WorkerContext,
    init_worker,
    mutate,
)

# How much a split must improve the score by, as a share of it, to be kept.
SPLIT_GAIN = 0.01
# Starting weights for the moves the policy chooses between.
MOVE_WEIGHTS = {"shape": 0.6, "detail": 0.2, "position": 0.1, "strokes": 0.1}


@dataclass(frozen=True)
class SearchSettings:
    """What bounds and shapes one search."""

    # The moves the policy chooses between: shape, detail, position, strokes.
    moves: tuple[str, ...] = ("shape",)
    simplify: bool = False
    # Score the result may lose against the start, spent by removals.
    tolerance: float = 0.0
    # With simplify on, the share of tasks that try removing a point.
    removal_share: float = 0.2
    workers: int = 1
    adaptive_operators: bool = True
    max_total_tasks: int | None = None
    max_wall_seconds: float | None = None
    # Set it and a one-worker run repeats exactly.
    random_seed: int | None = None


@dataclass(frozen=True)
class Candidate:
    state: Paths
    score: float


@dataclass(frozen=True)
class SearchProgress:
    tasks_completed: int
    score: float
    nodes: int
    elapsed: float


@dataclass(frozen=True)
class SearchOutcome:
    start: Candidate
    best: Candidate
    tasks_completed: int
    # Children that replaced the paths they were made from.
    accepted: int


def operator_policy(moves: tuple[str, ...], adaptive: bool) -> OperatorPolicy:
    weights = {move: MOVE_WEIGHTS.get(move, 0.1) for move in moves}
    return Exp3Policy(weights) if adaptive else FixedWeightPolicy(weights)


def run_search(
    start: Paths,
    score: Callable[[bytes], float],
    context: WorkerContext,
    settings: SearchSettings,
    *,
    stop: threading.Event | None = None,
    progress: Callable[[SearchProgress], None] | None = None,
) -> SearchOutcome:
    """Climb from *start*; *score* maps a render to lower-is-better."""
    if not settings.moves and not settings.simplify:
        raise ValueError("run_search needs at least one move")
    if settings.random_seed is not None:
        # The operator policy draws from the module generator.
        random.seed(settings.random_seed)
    draws = random.Random(settings.random_seed)
    policy = operator_policy(settings.moves, settings.adaptive_operators)
    reward = GradedReward()

    first = Candidate(start, score(Renderer(context)(start)))
    current = first
    ceiling = first.score + settings.tolerance
    budget = settings.max_total_tasks
    started = time.monotonic()
    dispatched = completed = accepted = 0

    def next_move() -> str:
        if settings.simplify and (
            not settings.moves or draws.random() < settings.removal_share
        ):
            return "simplify"
        move = policy.select()
        assert move is not None
        return move

    def out_of_time() -> bool:
        return (
            settings.max_wall_seconds is not None
            and time.monotonic() - started >= settings.max_wall_seconds
        )

    workers = max(1, settings.workers)
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=init_worker,
        initargs=(context,),
    )
    # Each task remembers the paths it was made from. A child of paths that
    # have since been replaced is dropped rather than scored: accepting it
    # would quietly undo whatever replaced them.
    pending: dict[Future[Mutant], Candidate] = {}
    try:
        while True:
            halted = (stop is not None and stop.is_set()) or out_of_time()
            while (
                not halted
                and len(pending) < workers * 2
                and (budget is None or dispatched < budget)
            ):
                future = pool.submit(
                    mutate, current.state, next_move(), draws.getrandbits(32)
                )
                pending[future] = current
                dispatched += 1
            if halted or not pending:
                break
            finished, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in finished:
                parent = pending.pop(future)
                completed += 1
                mutant = future.result()
                removal = mutant.move == "simplify"
                if mutant.state is None or mutant.png is None:
                    if not removal:
                        policy.update(mutant.move, 0.0)
                    continue
                if parent is not current:
                    continue
                child = Candidate(mutant.state, score(mutant.png))
                if removal:
                    keep = child.score <= ceiling
                elif mutant.move == "detail":
                    # A point has to pay for itself: a split that barely helps
                    # adds a point every later move must work around.
                    keep = child.score <= current.score * (1 - SPLIT_GAIN)
                else:
                    keep = child.score <= current.score
                if not removal:
                    policy.update(
                        mutant.move,
                        reward({"score": parent.score}, {"score": child.score})
                        if keep
                        else 0.0,
                    )
                if keep:
                    current = child
                    accepted += 1
            if progress is not None and finished:
                progress(
                    SearchProgress(
                        completed,
                        current.score,
                        current.state.nodes(),
                        time.monotonic() - started,
                    )
                )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    return SearchOutcome(
        start=first, best=current, tasks_completed=completed, accepted=accepted
    )
