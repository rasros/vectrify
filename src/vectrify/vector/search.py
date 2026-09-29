"""The NSGA-II vector search as a callable, independent of the CLI's wiring.

``run_search`` takes a prepared Reference, starting candidates and a worker
context, runs the multiprocess engine until a budget, a stop event or
convergence ends it, and returns the best candidate and the final pool. Output
directories, logging setup, dashboards and resume handling stay with callers.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from vectrify.formats.svg.operations import mutation_weights
from vectrify.formats.svg.selection import MutationScope
from vectrify.image_utils import make_preview_data_url
from vectrify.score.utils import MAX_SCORE
from vectrify.search import (
    ChainState,
    MultiprocessSearchEngine,
    NsgaStrategy,
    SearchNode,
    StorageAdapter,
)
from vectrify.search.collector import StatCollector
from vectrify.search.diversity import simhash
from vectrify.search.engine import SearchOutcome, SearchProgress
from vectrify.search.operators import Exp3Policy, FixedWeightPolicy, OperatorPolicy
from vectrify.search.storage import MemoryStorage
from vectrify.vector.payloads import VectorStatePayload
from vectrify.vector.reference import Reference
from vectrify.vector.state import VectorStateBuilder
from vectrify.vector.worker import WorkerContext, worker_loop

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchSettings:
    """What bounds and shapes one search; the CLI flags map onto these."""

    workers: int = 1
    pool_size: int = 20
    tournament_size: int = 3
    crossover_distance: int = 12
    adaptive_operators: bool = True
    epoch_seeds: int = 0
    initial_seeds: int | None = None
    epochs: int | None = None
    epoch_patience: int | None = None
    epoch_max_tasks: int | None = None
    epoch_eval_interval: int | None = None
    epoch_eval_patience: int | None = None
    epoch_improvement: float = 0.0
    epoch_improvement_patience: int = 1
    max_total_tasks: int | None = None
    max_wall_seconds: float | None = None
    resolution_llm: int = 512
    write_lineage: bool = False
    save_raster: bool = False


def seed_node(
    reference: Reference,
    content: str,
    png: bytes,
    *,
    node_id: int,
    origin: str,
    resolution_llm: int,
) -> SearchNode:
    """A measured starting candidate, e.g. an imported or generated drawing."""
    return SearchNode(
        valid=True,
        id=node_id,
        parent_id=0,
        metrics=reference.measure(png),
        signature=simhash(content),
        state=ChainState(
            VectorStatePayload(
                content=content,
                raster_data_url=None,
                raster_preview_data_url=make_preview_data_url(png, resolution_llm),
                origin=origin,
            )
        ),
    )


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
    storage: StorageAdapter | None = None,
    rank_front: Callable[[list[SearchNode]], list[SearchNode]] | None = None,
    policy: OperatorPolicy | None = None,
    collector: StatCollector | None = None,
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
        storage=storage or MemoryStorage(),
        max_total_tasks=settings.max_total_tasks,
        rank_front=rank_front,
        make_state=VectorStateBuilder(
            resolution_llm=settings.resolution_llm,
            write_lineage=settings.write_lineage,
            save_raster=settings.save_raster,
        ),
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
            epoch_seeds=settings.epoch_seeds,
            initial_seeds=settings.initial_seeds,
            epochs=settings.epochs,
            epoch_max_tasks=settings.epoch_max_tasks,
            epoch_eval_interval=settings.epoch_eval_interval,
            epoch_eval_patience=settings.epoch_eval_patience,
            epoch_improvement=settings.epoch_improvement,
            epoch_improvement_patience=settings.epoch_improvement_patience,
            operator_policy=policy,
            collector=collector,
            stop=stop,
            progress=progress,
        )
