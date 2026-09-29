"""Beam search over the selected paths' nodes, as a callable.

``run_search`` keeps a beam of the best states found so far. Each generation
every beam member is expanded into a few children, one move each, across the
worker processes; every child is scored in this process; and the next beam is
the best distinct states among the members and their children. A tie keeps
the older state, so a move nothing can measure never displaces what the beam
already holds. Children of one parent usually improve different points, so
each generation also merges those improvements into one state that takes
them all at once, which is what moves a whole shape rather than one point.

With simplify on, some children try removing a point. The beam then prefers
fewer points among states within ``tolerance`` of where the run started; that
budget is shared by every removal, so the result is never worse than the
start by more than it, and nudges that improve the fit earn room for more
removals. A split has to improve its parent by ``SPLIT_GAIN`` to be kept.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as NotYet
from dataclasses import dataclass, field, replace

import numpy as np

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
    # With simplify on, the share of children that try removing a point.
    removal_share: float = 0.2
    # States kept per generation, and children made from each.
    # Measured over 600 tries on a moved pine tree and river, 3 seeds each:
    # 2 x 4 with merging ended at 0.053 and 0.040 against hill climbing's
    # 0.066 and 0.057, better on every seed; 6 x 4 spread too few generations
    # over too many lineages and ended at 0.156 on the pine.
    beam_width: int = 2
    children: int = 4
    workers: int = 1
    adaptive_operators: bool = True
    max_total_tasks: int | None = None
    max_wall_seconds: float | None = None
    # Set it and a run repeats exactly, with any number of workers.
    random_seed: int | None = None


@dataclass(frozen=True)
class Candidate:
    state: Paths
    score: float
    # The generation that found it: ties go to the older state.
    born: int = 0
    parent: Candidate | None = field(default=None, repr=False, compare=False)


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
    # Children that made it into the beam.
    accepted: int
    generations: int


def merge_improvements(base: Paths, children: list[Paths]) -> Paths | None:
    """*base* with every child's changes that touch points no other took.

    Children of one parent usually improve different points, and apart they
    each move the shape one point per generation. Merged, a generation moves
    every point that found a better place. Only same-topology children merge,
    in the order given, and a child is skipped whole if it touches a point an
    earlier one changed; strokes merge the same way.
    """
    geometries = dict(base.geometries)
    strokes = dict(base.strokes)
    taken: set[str] = set()
    merged = 0
    for child in children:
        changes: dict[str, dict[str, tuple[float, ...]]] = {}
        compatible = True
        for oid, geometry in child.geometries.items():
            original = base.geometries[oid]
            if geometry == original:
                continue
            old = [n for s in original.subpaths for n in s.nodes]
            new = [n for s in geometry.subpaths for n in s.nodes]
            if [(n.id, n.command) for n in old] != [(n.id, n.command) for n in new]:
                compatible = False
                break
            changes[oid] = {
                n.id: n.values for o, n in zip(old, new, strict=True) if o != n
            }
        stroke_changes = {
            oid: w for oid, w in child.strokes.items() if w != base.strokes[oid]
        }
        touched = {f"{oid}:{nid}" for oid, c in changes.items() for nid in c}
        touched |= {f"{oid}:stroke" for oid in stroke_changes}
        if not compatible or not touched or touched & taken:
            continue
        taken |= touched
        merged += 1
        for oid, values in changes.items():
            geometry = geometries[oid]
            geometries[oid] = replace(
                geometry,
                subpaths=tuple(
                    replace(
                        s,
                        nodes=tuple(
                            replace(n, values=values[n.id]) if n.id in values else n
                            for n in s.nodes
                        ),
                    )
                    for s in geometry.subpaths
                ),
            )
        strokes.update(stroke_changes)
    if merged < 2:
        return None
    return Paths(geometries, strokes)


def operator_policy(moves: tuple[str, ...], adaptive: bool) -> OperatorPolicy:
    weights = {move: MOVE_WEIGHTS.get(move, 0.1) for move in moves}
    return Exp3Policy(weights) if adaptive else FixedWeightPolicy(weights)


def run_search(
    start: Paths,
    score: Callable[[np.ndarray], float],
    context: WorkerContext,
    settings: SearchSettings,
    *,
    stop: threading.Event | None = None,
    progress: Callable[[SearchProgress], None] | None = None,
) -> SearchOutcome:
    """Search from *start*; *score* maps an RGB render to lower-is-better."""
    if not settings.moves and not settings.simplify:
        raise ValueError("run_search needs at least one move")
    if settings.random_seed is not None:
        # The operator policy draws from the module generator.
        random.seed(settings.random_seed)
    draws = random.Random(settings.random_seed)
    policy = operator_policy(settings.moves, settings.adaptive_operators)
    reward = GradedReward()

    render = Renderer(context)
    first = Candidate(start, score(render(start)))
    ceiling = first.score + settings.tolerance

    def rank(candidate: Candidate) -> tuple:
        if settings.simplify:
            return (candidate.state.nodes(), candidate.score, candidate.born)
        return (candidate.score, candidate.born)

    def next_move() -> str:
        if settings.simplify and (
            not settings.moves or draws.random() < settings.removal_share
        ):
            return "simplify"
        move = policy.select()
        assert move is not None
        return move

    def halted() -> bool:
        return (stop is not None and stop.is_set()) or (
            settings.max_wall_seconds is not None
            and time.monotonic() - started >= settings.max_wall_seconds
        )

    beam = [first]
    seen = {start.key()}
    budget = settings.max_total_tasks
    started = time.monotonic()
    completed = accepted = generation = 0
    workers = max(1, settings.workers)
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=init_worker,
        initargs=(context,),
    )
    try:
        while not halted() and (budget is None or completed < budget):
            generation += 1
            width = settings.children * len(beam)
            if budget is not None:
                width = min(width, budget - completed)
            # All of a generation's tasks go out together and come back in the
            # order they went out, so which parent a child belongs to, and so
            # the whole run, does not depend on which worker finished first.
            tasks: list[tuple[Future[Mutant], Candidate, str]] = []
            for index in range(width):
                parent = beam[index % len(beam)]
                move = next_move()
                future = pool.submit(mutate, parent.state, move, draws.getrandbits(32))
                tasks.append((future, parent, move))
            children: list[Candidate] = []
            for future, parent, move in tasks:
                mutant: Mutant | None = None
                while mutant is None:
                    try:
                        mutant = future.result(timeout=0.25)
                    except NotYet:
                        if halted():
                            break
                if mutant is None:
                    break
                completed += 1
                removal = move == "simplify"
                if mutant.state is None or mutant.image is None:
                    if not removal:
                        policy.update(move, 0.0)
                    continue
                key = mutant.state.key()
                if key in seen:
                    continue
                seen.add(key)
                child = Candidate(mutant.state, score(mutant.image), generation, parent)
                if removal:
                    keep = child.score <= ceiling
                elif move == "detail":
                    # A point has to pay for itself: a split that barely helps
                    # adds a point every later move must work around.
                    keep = child.score <= parent.score * (1 - SPLIT_GAIN)
                else:
                    keep = not settings.simplify or child.score <= ceiling
                if not removal:
                    better = child.score < parent.score
                    policy.update(
                        move,
                        reward({"score": parent.score}, {"score": child.score})
                        if better
                        else 0.0,
                    )
                if keep:
                    children.append(child)
            # Coordinate: per parent, merge the children that improved on it
            # into one state that takes all of their gains at once.
            for parent in beam:
                gains = sorted(
                    (
                        c
                        for c in children
                        if c.parent is parent and c.score < parent.score
                    ),
                    key=lambda c: c.score,
                )
                combined = merge_improvements(parent.state, [c.state for c in gains])
                if combined is not None and combined.key() not in seen:
                    seen.add(combined.key())
                    candidate = Candidate(combined, score(render(combined)), generation)
                    if not settings.simplify or candidate.score <= ceiling:
                        children.append(candidate)
            beam = sorted(beam + children, key=rank)[: settings.beam_width]
            accepted += sum(1 for c in beam if c.born == generation)
            if progress is not None:
                progress(
                    SearchProgress(
                        completed,
                        beam[0].score,
                        beam[0].state.nodes(),
                        time.monotonic() - started,
                    )
                )
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    return SearchOutcome(
        start=first,
        best=beam[0],
        tasks_completed=completed,
        accepted=accepted,
        generations=generation,
    )
