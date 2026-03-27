import contextlib
import logging
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from svgizer.search.base import SearchStrategy, StorageAdapter
from svgizer.search.models import Result, SearchNode, Task
from svgizer.search.stats import SearchStats

TState = TypeVar("TState")
log = logging.getLogger(__name__)


class MultiprocessSearchEngine(Generic[TState]):
    def __init__(
        self,
        workers: int,
        strategy: SearchStrategy[TState],
        storage: StorageAdapter[TState],
        max_total_tasks: int = 10000,
    ):
        self.workers = workers
        self.strategy = strategy
        self.storage = storage
        self.max_total_tasks = max_total_tasks

        self.ctx = mp.get_context("spawn")
        self.task_q = self.ctx.Queue(maxsize=max(64, workers * 8))
        self.unscored_q = self.ctx.Queue()
        self.result_q = queue.Queue()
        self.procs: list[Any] = []

    def start_workers(self, worker_target: Callable, worker_params: dict) -> None:
        log.info(f"Starting {self.workers} worker processes...")
        self._llm_in_flight = self.ctx.Value("i", 0)
        worker_params["llm_in_flight"] = self._llm_in_flight
        for _ in range(max(1, self.workers)):
            p = self.ctx.Process(
                target=worker_target,
                args=(self.task_q, self.unscored_q, worker_params),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

    def run(
        self,
        initial_nodes: list[SearchNode[TState]],
        max_accepts: int = 2_000_000_000,
        max_wall_seconds: float | None = None,
        epoch_patience: int | None = None,
        epoch_min_delta: float = 1e-4,
        active_pool_size: int = 20,
        score_fn: Callable[[Result], float] | None = None,
        seed_tasks: int = 0,
        max_epochs: int | None = None,
        epoch_pool_size: int | None = None,
        stats: SearchStats | None = None,
    ) -> None:
        start_time = time.monotonic()

        if stats is not None:
            stats.start_time = start_time
            stats.epoch_patience = epoch_patience or 0

        # Track seen genomes to avoid redundant evaluations
        seen_signatures: set[tuple[int, ...]] = {
            n.signature for n in initial_nodes if n.signature
        }

        def _scorer_worker():
            while True:
                res = self.unscored_q.get()
                if res is None:
                    self.result_q.put(None)
                    break

                # Duplicate rejection via MinHash signature (skips scoring)
                if res.valid and res.signature:
                    if res.signature in seen_signatures:
                        res.valid = False
                        res.invalid_msg = "Duplicate genome"
                    else:
                        seen_signatures.add(res.signature)

                try:
                    if res.valid and res.score is None and score_fn is not None:
                        res.score = score_fn(res)
                except Exception as e:
                    res.valid = False
                    res.invalid_msg = f"Scoring error: {e}"
                    res.score = float("inf")

                self.result_q.put(res)

        scorer_thread = threading.Thread(
            target=_scorer_worker, daemon=True, name="ScorerThread"
        )
        scorer_thread.start()

        node_states = {n.id: n.state for n in initial_nodes}

        sorted_initial = sorted(initial_nodes, key=lambda n: n.score)
        active_pool: list[SearchNode[TState]] = sorted_initial[:active_pool_size]

        best_node = sorted_initial[0] if sorted_initial else None

        # Epoch state
        epoch = 0
        epoch_no_improve = 0
        epoch_patience_best = best_node.score if best_node else float("inf")
        # Epoch 0: track how many seed tasks (force_llm) we've dispatched
        epoch0_seeds_dispatched = 0

        next_task_id = 1
        tasks_completed = 0
        accepted_count = 0
        in_flight = 0

        next_node_id = max(
            self.storage.max_node_id, max((n.id for n in initial_nodes), default=0)
        )

        log.info(
            f"Search started. Initial best: {best_node.score if best_node else 'N/A'}"
        )

        try:
            while True:
                if (
                    max_wall_seconds
                    and (time.monotonic() - start_time) >= max_wall_seconds
                ):
                    log.warning("Time limit reached.")
                    break
                if accepted_count >= max_accepts:
                    log.info(f"Target of {max_accepts} accepts reached.")
                    break
                if tasks_completed >= self.max_total_tasks:
                    log.warning("Max task limit reached.")
                    break
                if max_epochs is not None and epoch >= max_epochs:
                    log.info(f"Max epochs ({max_epochs}) reached.")
                    break

                while in_flight < self.workers and next_task_id <= self.max_total_tasks:
                    progress = (
                        accepted_count / float(max_accepts) if max_accepts > 0 else 0.0
                    )
                    pid1, pid2 = self.strategy.select_parent(active_pool, progress)

                    # Epoch 0: seed with force_llm until seed_tasks are dispatched
                    is_epoch0_seed = epoch == 0 and epoch0_seeds_dispatched < seed_tasks
                    if is_epoch0_seed:
                        epoch0_seeds_dispatched += 1
                    elif epoch == 0 and seed_tasks > 0 and accepted_count == 0:
                        # No seed accepted yet — don't dispatch regular tasks since
                        # the pool has no valid SVG to mutate.
                        break

                    # Ramp LLM pressure from 0 → 1 as stagnation grows
                    ramp_ref = epoch_patience or 100
                    llm_pressure = min(1.0, epoch_no_improve / ramp_ref)

                    if stats is not None:
                        stats.llm_pressure = llm_pressure

                    self.task_q.put(
                        Task(
                            task_id=next_task_id,
                            parent_id=pid1,
                            parent_state=node_states[pid1],
                            worker_slot=in_flight % self.workers,
                            secondary_parent_id=pid2,
                            secondary_parent_state=node_states[pid2] if pid2 else None,
                            force_llm=is_epoch0_seed,
                            llm_pressure=llm_pressure,
                        )
                    )
                    next_task_id += 1
                    in_flight += 1

                try:
                    res = self.result_q.get(timeout=0.2)
                    if res is None:  # Caught shutdown sentinel from ScorerThread
                        break
                except queue.Empty:
                    if not any(p.is_alive() for p in self.procs):
                        raise RuntimeError(
                            "All worker processes have exited."
                        ) from None
                    continue

                if isinstance(res, dict) and "init_error" in res:
                    raise RuntimeError(
                        f"Worker initialization failed: {res['init_error']}"
                    )

                in_flight -= 1
                tasks_completed += 1
                epoch_no_improve += 1

                if stats is not None:
                    stats.tasks_completed = tasks_completed
                    stats.epoch_no_improve = epoch_no_improve
                    if hasattr(self, "_llm_in_flight"):
                        stats.llm_calls_in_flight = self._llm_in_flight.value
                    if res.llm_type:
                        stats.llm_call_count += 1
                    else:
                        stats.mutation_call_count += 1

                if not res.valid:
                    is_duplicate = res.invalid_msg == "Duplicate genome"
                    if res.llm_type:
                        log.info(
                            f"[{res.llm_type.upper()} INVALID] "
                            f"task={res.task_id} msg={res.invalid_msg}"
                        )
                    elif is_duplicate:
                        log.debug(f"Task {res.task_id} rejected: duplicate genome.")
                    else:
                        log.debug(f"Task {res.task_id} rejected: {res.invalid_msg}")
                    if stats is not None:
                        if is_duplicate:
                            stats.duplicate_count += 1
                        else:
                            stats.invalid_count += 1
                            if res.llm_type:
                                stats.llm_invalid_count += 1
                    continue

                if res.score is None:
                    raise RuntimeError("Result has no score and no score_fn provided")

                next_node_id += 1
                new_state = self.strategy.create_new_state(res)

                new_node = SearchNode(
                    score=res.score,
                    id=next_node_id,
                    parent_id=res.parent_id,
                    state=new_state,
                    secondary_parent_id=res.secondary_parent_id,
                    complexity=res.complexity,
                    signature=res.signature,
                    epoch=epoch,
                )

                active_pool.append(new_node)
                if len(active_pool) > active_pool_size:
                    worst_idx = max(
                        range(len(active_pool)), key=lambda i: active_pool[i].score
                    )
                    evicted_node = active_pool.pop(worst_idx)
                    node_states.pop(evicted_node.id, None)

                    if evicted_node is new_node:
                        if res.llm_type:
                            log.info(
                                f"[{res.llm_type.upper()} REJECTED] "
                                f"node={new_node.id} score={new_node.score:.6f}"
                                " (worse than pool)"
                            )
                        else:
                            log.debug(
                                f"[REJECTED] node={new_node.id} "
                                f"score={new_node.score:.6f} (worse than pool)"
                            )
                        if stats is not None:
                            stats.pool_rejected_count += 1
                        continue

                node_states[new_node.id] = new_state
                accepted_count += 1
                is_new_best = best_node is None or new_node.score < best_node.score

                if stats is not None:
                    stats.accepted_count = accepted_count
                    if res.llm_type:
                        stats.llm_accepted_count += 1
                    else:
                        stats.mutation_accepted_count += 1

                if new_node.score <= epoch_patience_best - epoch_min_delta:
                    epoch_patience_best = new_node.score
                    epoch_no_improve = 0
                    if stats is not None:
                        stats.epoch_no_improve = 0

                if is_new_best:
                    best_node = new_node
                    if stats is not None:
                        stats.best_score = best_node.score
                        stats.score_history.append(
                            (time.monotonic() - start_time, best_node.score)
                        )
                    if res.llm_type:
                        log.info(
                            f"[{res.llm_type.upper()} NEW BEST] "
                            f"node={new_node.id} score={new_node.score:.6f}"
                        )
                    else:
                        log.info(
                            f"[NEW BEST] node={new_node.id} score={new_node.score:.6f}"
                        )
                else:
                    if res.llm_type:
                        log.info(
                            f"[{res.llm_type.upper()} ACCEPTED] "
                            f"node={new_node.id} score={new_node.score:.6f}"
                        )
                    else:
                        log.debug(
                            f"[ACCEPTED] node={new_node.id} score={new_node.score:.6f}"
                        )

                self.storage.save_node(new_node)

                # Check epoch-end conditions (only once epoch 0 seed phase is complete)
                past_seed_phase = epoch > 0 or epoch0_seeds_dispatched >= seed_tasks
                if past_seed_phase:
                    staleness = (
                        epoch_patience is not None
                        and epoch_no_improve >= epoch_patience
                    )
                    low_diversity, pool_diversity = self.strategy.should_diversify(
                        active_pool
                    )
                    if stats is not None:
                        stats.pool_diversity = pool_diversity

                    if staleness or low_diversity:
                        reason = "staleness" if staleness else "low diversity"
                        log.info(
                            f"Epoch {epoch} → {epoch + 1}: {reason} "
                            f"(no_improve={epoch_no_improve}, pool={len(active_pool)})"
                        )
                        epoch += 1
                        if stats is not None:
                            stats.epoch = epoch
                        n_seeds = epoch_pool_size or max(1, active_pool_size // 4)
                        seeds = self.strategy.epoch_seeds(active_pool, n_seeds)
                        if seeds:
                            active_pool = seeds
                            log.info(
                                f"Epoch {epoch}: seeded with "
                                f"{len(active_pool)} Pareto-front nodes."
                            )
                        else:
                            active_pool = list(initial_nodes[:active_pool_size])
                            log.info(f"Epoch {epoch}: restarting from initial node.")
                        epoch_no_improve = 0
                        if stats is not None:
                            stats.epoch_no_improve = 0
                        valid_scores = [
                            n.score for n in active_pool if n.score < float("inf")
                        ]
                        epoch_patience_best = (
                            min(valid_scores) if valid_scores else float("inf")
                        )

        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        log.info("Shutting down workers...")

        with contextlib.suppress(queue.Full, OSError, ValueError):
            self.unscored_q.put(None, timeout=0.5)

        for _ in self.procs:
            try:
                self.task_q.put(None, timeout=0.5)
            except (queue.Full, OSError, ValueError):
                log.debug("Task queue full during shutdown.")

        self.task_q.cancel_join_thread()
        self.unscored_q.cancel_join_thread()

        for p in self.procs:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
                p.join()
