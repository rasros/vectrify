"""The NSGA-II vector search as a callable.

``run_search`` takes a prepared Reference, starting candidates and a worker
context, runs the multiprocess engine until a budget, a stop event or
convergence ends it, and returns the best candidate and the final pool.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from vectrify.image_utils import rasterize_svg_to_png_bytes
from vectrify.score.metrics import FRONT_SCORE
from vectrify.score.utils import MAX_SCORE
from vectrify.search import (
    ChainState,
    MultiprocessSearchEngine,
    NsgaStrategy,
    Result,
    SearchNode,
)
from vectrify.search.diversity import simhash
from vectrify.search.engine import SearchOutcome, SearchProgress
from vectrify.search.operators import Exp3Policy, FixedWeightPolicy, OperatorPolicy
from vectrify.svg.operations import mutation_weights
from vectrify.svg.selection import MutationScope
from vectrify.vector.payloads import VectorStatePayload
from vectrify.vector.reference import Reference
from vectrify.vector.worker import WorkerContext, worker_loop

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchSettings:
    """What bounds and shapes one search."""

    workers: int = 1
    pool_size: int = 20
    tournament_size: int = 3
    crossover_distance: int = 12
    adaptive_operators: bool = True
    epochs: int | None = None
    epoch_patience: int | None = None
    epoch_max_tasks: int | None = None
    epoch_eval_interval: int | None = None
    epoch_eval_patience: int | None = None
    epoch_improvement: float = 0.0
    epoch_improvement_patience: int = 1
    max_total_tasks: int | None = None
    max_wall_seconds: float | None = None


def to_state(result: Result) -> ChainState[VectorStatePayload]:
    """Pool state for a scored result: its drawing, not its render."""
    return ChainState(VectorStatePayload(result.payload.content, result.payload.origin))


def seed_node(
    reference: Reference,
    content: str,
    png: bytes,
    *,
    node_id: int,
    origin: str,
) -> SearchNode:
    """A measured starting candidate, e.g. an imported or generated drawing."""
    return SearchNode(
        valid=True,
        id=node_id,
        parent_id=0,
        metrics=reference.measure(png),
        signature=simhash(content),
        state=ChainState(VectorStatePayload(content=content, origin=origin)),
    )


def evaluate_front(
    nodes: list[SearchNode],
    *,
    front_scorer: Callable[[], tuple[Any, Any]],
    out_w: int,
    out_h: int,
) -> list[SearchNode]:
    """Order *nodes* by the evaluator, best first, scoring only what is new.

    *front_scorer* is called for (scorer, reference) and only when there is
    something to score, so a call the cache answers in full never builds a
    model.

    Re-rasterises rather than reading a node's stored render, which is only
    kept when --write-lineage or --save-raster is on.
    """
    renders: list[tuple[bytes, SearchNode]] = []
    for node in nodes:
        # Already judged, and the judgement travels: the panel's score is a
        # calibrated distance to the target, so it means the same thing in
        # every call. Re-rasterising and re-embedding a node the evaluator has
        # already seen would buy an identical number at full price -- and a run
        # asks about the same pool members repeatedly.
        if FRONT_SCORE in node.metrics:
            continue
        content = getattr(node.state.payload, "content", None)
        if not content:
            continue
        try:
            renders.append(
                (rasterize_svg_to_png_bytes(content, out_w=out_w, out_h=out_h), node)
            )
        except Exception as exc:
            log.debug(f"Front evaluation skipped node {node.id}: {exc}")

    if renders:
        scorer, ref = front_scorer()
        pngs = [png for png, _ in renders]
        try:
            values = scorer.rank(ref, pngs)
        except AttributeError:
            values = [scorer.score(ref, png) for png in pngs]
        except Exception as exc:
            log.warning(f"Front evaluation failed, keeping rank order: {exc}")
            return nodes

        for value, (_png, node) in zip(values, renders, strict=True):
            node.metrics[FRONT_SCORE] = value

    # Every node the panel has ever scored, freshly measured or recalled.
    scored = [
        (node.metrics[FRONT_SCORE], node)
        for node in nodes
        if FRONT_SCORE in node.metrics
    ]
    if not scored:
        return nodes
    scored.sort(key=lambda pair: pair[0])
    log.info(
        f"Front evaluated: {len(scored)} candidate(s) "
        f"({len(renders)} newly scored), "
        f"best {scored[0][0]:.6f}, worst {scored[-1][0]:.6f}"
    )
    return [node for _value, node in scored]


def pixel_scorer(
    reference: Reference, pool: ThreadPoolExecutor
) -> Callable[[list], None]:
    """Measure each result's render against the reference, in parallel."""

    def measure(res) -> None:
        png = res.payload.raster_png
        if not png:
            return
        try:
            res.metrics.update(reference.measure(png))
            # Measured, so valid. `score` carries no magnitude any more: the
            # measures are ranked by dominance and the only score in the run is
            # the evaluator's, recorded as FRONT_SCORE on the nodes it sees.
            res.measured = True
        except Exception as exc:
            log.debug(f"Pixel objectives skipped: {exc}")

    def score(results) -> None:
        # One scorer thread runs this, in a decode-resize-convolve pass per
        # candidate that is where a run's throughput was going: profiled,
        # workers sat idle above four of them while this serialised. The work
        # is numpy and Pillow, both of which drop the GIL, so threads overlap.
        if len(results) > 1:
            list(pool.map(measure, results))
        else:
            for res in results:
                measure(res)
        for res in results:
            if not res.measured:
                # Nothing rendered, so nothing can be measured.
                res.score = MAX_SCORE

    return score


def operator_policy(scope: MutationScope | None, adaptive: bool) -> OperatorPolicy:
    weights = mutation_weights(scope)
    return Exp3Policy(weights) if adaptive else FixedWeightPolicy(weights)


def run_search(
    reference: Reference,
    initial_nodes: list[SearchNode],
    worker_context: WorkerContext,
    settings: SearchSettings,
    *,
    rank_front: Callable[[list[SearchNode]], list[SearchNode]] | None = None,
    policy: OperatorPolicy | None = None,
    stop: threading.Event | None = None,
    progress: Callable[[SearchProgress], None] | None = None,
) -> SearchOutcome:
    """Run workers over *initial_nodes* and return the best candidate and pool."""
    policy = policy or operator_policy(
        worker_context.scope, settings.adaptive_operators
    )
    engine = MultiprocessSearchEngine(
        workers=settings.workers,
        strategy=NsgaStrategy[VectorStatePayload](
            pool_size=settings.pool_size,
            tournament_size=settings.tournament_size,
            crossover_distance_threshold=settings.crossover_distance,
        ),
        max_total_tasks=settings.max_total_tasks,
        rank_front=rank_front,
        make_state=to_state,
        elite_metric_names=tuple(s.metric_name for s in reference.segments),
    )
    # Sized against the scorer thread's own work rather than the worker count:
    # it is one batch of candidates at a time, and oversubscribing here would
    # only take cores from the workers producing them.
    with ThreadPoolExecutor(
        max_workers=min(8, os.cpu_count() or 4), thread_name_prefix="pixel"
    ) as pool:
        engine.start_workers(worker_loop, worker_context)
        return engine.run(
            initial_nodes,
            max_wall_seconds=settings.max_wall_seconds,
            epoch_patience=settings.epoch_patience,
            active_pool_size=settings.pool_size,
            score_fn=pixel_scorer(reference, pool),
            epochs=settings.epochs,
            epoch_max_tasks=settings.epoch_max_tasks,
            epoch_eval_interval=settings.epoch_eval_interval,
            epoch_eval_patience=settings.epoch_eval_patience,
            epoch_improvement=settings.epoch_improvement,
            epoch_improvement_patience=settings.epoch_improvement_patience,
            operator_policy=policy,
            stop=stop,
            progress=progress,
        )
