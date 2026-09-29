"""Hill climbing over a drawing, as a callable.

``run_search`` starts from the best of the drawings it is given, has worker
processes mutate the current drawing, scores every child in this process, and
lets a child replace the current drawing when it scores no worse. It returns
the best few distinct drawings it saw, the current one first.

One score decides everything: which child replaces its parent, which operator
earned credit and which drawings come back. There is no pool and no front, so
nothing downstream has to re-rank what the search already ranked.
"""

from __future__ import annotations

import bisect
import multiprocessing as mp
import random
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass

from vectrify.image_utils import rasterize_svg_to_png_bytes
from vectrify.svg.operations import mutation_weights
from vectrify.svg.selection import MutationScope
from vectrify.vector.operators import (
    Exp3Policy,
    FixedWeightPolicy,
    GradedReward,
    OperatorPolicy,
)
from vectrify.vector.worker import Mutant, WorkerContext, init_worker, mutate


@dataclass(frozen=True)
class SearchSettings:
    """What bounds and shapes one search."""

    workers: int = 1
    adaptive_operators: bool = True
    max_total_tasks: int | None = None
    max_wall_seconds: float | None = None
    # How many of the best distinct drawings come back.
    keep: int = 4
    # Set it and a one-worker run repeats exactly. Above one worker the order
    # results arrive in varies, and so does which child each task mutates.
    random_seed: int | None = None


@dataclass(frozen=True)
class Candidate:
    content: str
    score: float


@dataclass(frozen=True)
class SearchProgress:
    tasks_completed: int
    best: float
    elapsed: float


@dataclass(frozen=True)
class SearchOutcome:
    # The seed the climb started from: the best-scoring one.
    start: Candidate
    best: Candidate
    # The best distinct drawings seen, best first; best is the first of them.
    ranked: list[Candidate]
    tasks_completed: int
    # Children that replaced the drawing they were mutated from.
    accepted: int


def operator_policy(scope: MutationScope | None, adaptive: bool) -> OperatorPolicy:
    weights = mutation_weights(scope)
    return Exp3Policy(weights) if adaptive else FixedWeightPolicy(weights)


class _Best:
    """The *size* lowest-scoring distinct drawings seen so far."""

    def __init__(self, size: int):
        self.size = max(1, size)
        self.items: list[Candidate] = []
        self._seen: set[str] = set()

    def add(self, candidate: Candidate) -> None:
        if candidate.content in self._seen:
            return
        if len(self.items) >= self.size and candidate.score >= self.items[-1].score:
            return
        self._seen.add(candidate.content)
        bisect.insort(self.items, candidate, key=lambda c: c.score)
        if len(self.items) > self.size:
            self._seen.discard(self.items.pop().content)


def run_search(
    seeds: Sequence[str],
    score: Callable[[bytes], float],
    context: WorkerContext,
    settings: SearchSettings,
    *,
    policy: OperatorPolicy | None = None,
    stop: threading.Event | None = None,
    progress: Callable[[SearchProgress], None] | None = None,
) -> SearchOutcome:
    """Climb from the best of *seeds*; *score* maps a render to lower-is-better."""
    if not seeds:
        raise ValueError("run_search needs at least one starting drawing")
    if settings.random_seed is not None:
        # The operator policy draws from the module generator.
        random.seed(settings.random_seed)
    task_seeds = random.Random(settings.random_seed)
    policy = policy or operator_policy(context.scope, settings.adaptive_operators)
    reward = GradedReward()

    best = _Best(settings.keep)
    for content in seeds:
        png = rasterize_svg_to_png_bytes(
            content, out_w=context.original_w, out_h=context.original_h
        )
        best.add(Candidate(content, score(png)))
    current = start = best.items[0]

    budget = settings.max_total_tasks
    started = time.monotonic()
    dispatched = completed = accepted = 0

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
    # Each task remembers the score of the drawing it mutates: by the time it
    # comes back the current drawing may have moved on, and the operator
    # earns what it improved on its own parent.
    pending: dict[Future[Mutant], float] = {}
    try:
        while True:
            halted = (stop is not None and stop.is_set()) or out_of_time()
            while (
                not halted
                and len(pending) < workers * 2
                and (budget is None or dispatched < budget)
            ):
                future = pool.submit(
                    mutate, current.content, policy.select(), task_seeds.getrandbits(32)
                )
                pending[future] = current.score
                dispatched += 1
            if halted or not pending:
                break
            finished, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in finished:
                parent_score = pending.pop(future)
                completed += 1
                mutant = future.result()
                if mutant.content is None or mutant.png is None:
                    policy.update(mutant.operator, 0.0)
                    continue
                child = Candidate(mutant.content, score(mutant.png))
                best.add(child)
                if child.score <= current.score:
                    current = child
                    accepted += 1
                    policy.update(
                        mutant.operator,
                        reward({"score": parent_score}, {"score": child.score}),
                    )
                else:
                    policy.update(mutant.operator, 0.0)
            if progress is not None and finished:
                progress(
                    SearchProgress(completed, current.score, time.monotonic() - started)
                )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    # Accepting ties keeps current at the lowest score seen, but a tie may have
    # displaced an equal drawing from the head of the list.
    ranked = [current] + [c for c in best.items if c.content != current.content]
    return SearchOutcome(
        start=start,
        best=current,
        ranked=ranked[: best.size],
        tasks_completed=completed,
        accepted=accepted,
    )
