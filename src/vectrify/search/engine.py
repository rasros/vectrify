import contextlib
import logging
import math
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from vectrify.search.base import SearchStrategy, StorageAdapter
from vectrify.search.collector import StatCollector
from vectrify.search.models import INVALID_SCORE, Result, SearchNode, Task

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

    def start_workers(self, worker_target: Callable, worker_params: Any) -> None:
        log.info(f"Starting {self.workers} worker processes...")
        self._llm_in_flight = self.ctx.Value("i", 0)
        if isinstance(worker_params, dict):
            worker_params["llm_in_flight"] = self._llm_in_flight
        else:
            worker_params.llm_in_flight = self._llm_in_flight
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
        epoch_variance: float | None = None,
        epoch_steps: int | None = None,
        max_llm_calls: int | None = None,
        collector: StatCollector | None = None,
    ) -> None:
        start_time = time.monotonic()

        if collector is not None:
            collector.on_run_start(
                start_time=start_time,
                epoch_patience=epoch_patience or 0,
            )

        def _scorer_worker():
            while True:
                res = self.unscored_q.get()
                if res is None:
                    self.result_q.put(None)
                    break

                try:
                    if res.valid and res.score is None and score_fn is not None:
                        res.score = score_fn(res)
                except Exception as e:
                    res.valid = False
                    res.invalid_msg = f"Scoring error: {e}"
                    res.score = INVALID_SCORE

                self.result_q.put(res)

        scorer_thread = threading.Thread(
            target=_scorer_worker, daemon=True, name="ScorerThread"
        )
        scorer_thread.start()

        node_states = {n.id: n.state for n in initial_nodes}
        sorted_initial = sorted(initial_nodes, key=lambda n: n.score)
        active_pool: list[SearchNode[TState]] = sorted_initial[:active_pool_size]
        best_node = sorted_initial[0] if sorted_initial else None

        epoch = 0
        epoch_no_improve = 0
        epoch_tasks = 0
        total_llm_completions = 0
        epoch_patience_best = best_node.score if best_node else INVALID_SCORE
        epoch0_seeds_dispatched = 0
        epoch0_seeds_completed = 0
        draining_epoch = False
        epoch_drain_reason = ""
        pool_refilling = (
            False  # True between epoch transition and pool reaching capacity
        )

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

        def _dispatch_tasks():
            nonlocal in_flight, next_task_id, epoch0_seeds_dispatched
            if draining_epoch:
                return
            while in_flight < self.workers and next_task_id <= self.max_total_tasks:
                progress = (
                    accepted_count / float(max_accepts) if max_accepts > 0 else 0.0
                )
                pid1, pid2 = self.strategy.select_parent(active_pool, progress)

                is_epoch0_seed = epoch == 0 and epoch0_seeds_dispatched < seed_tasks
                if is_epoch0_seed:
                    epoch0_seeds_dispatched += 1
                elif (
                    epoch == 0
                    and seed_tasks > 0
                    and (
                        epoch0_seeds_completed < seeds_ready_threshold
                        or accepted_count == 0
                    )
                ):
                    # Wait until 80% of seeds returned and pool has valid SVG
                    break

                ramp_ref = epoch_patience or 100
                llm_pressure = max(0.1, min(1.0, epoch_no_improve / ramp_ref))

                if collector is not None:
                    collector.on_llm_pressure(llm_pressure)

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

        def _fetch_result() -> tuple[bool, Result | None]:
            try:
                res = self.result_q.get(timeout=0.2)
                if res is None:
                    return False, None
                return True, res
            except queue.Empty:
                if not any(p.is_alive() for p in self.procs):
                    raise RuntimeError("All worker processes have exited.") from None
                if collector is not None and hasattr(self, "_llm_in_flight"):
                    valid_scores = [
                        n.score for n in active_pool if n.score < INVALID_SCORE
                    ]
                    collector.on_idle(
                        llm_in_flight=self._llm_in_flight.value,
                        valid_scores=valid_scores,
                    )
                return True, None

        def _process_valid_result(res: Result):
            nonlocal \
                next_node_id, \
                accepted_count, \
                epoch_patience_best, \
                epoch_no_improve, \
                best_node

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
                    if collector is not None:
                        collector.on_pool_rejected(is_llm=bool(res.llm_type))
                    return

                self.storage.record_eviction(evicted_node.id, tasks_completed)

            node_states[new_node.id] = new_state
            accepted_count += 1
            is_new_best = best_node is None or new_node.score < best_node.score

            if collector is not None:
                collector.on_accepted(
                    new_node,
                    is_new_best=is_new_best,
                    elapsed=time.monotonic() - start_time,
                    llm_type=res.llm_type,
                )

            if new_node.score <= epoch_patience_best - epoch_min_delta:
                epoch_patience_best = new_node.score
                epoch_no_improve = 0
                if collector is not None:
                    collector.on_no_improve_reset()

            if is_new_best:
                best_node = new_node
                log.info(
                    f"[{res.llm_type.upper() if res.llm_type else 'NEW BEST'}] "
                    f"node={new_node.id} score={new_node.score:.6f}"
                )
            elif res.llm_type:
                log.info(
                    f"[{res.llm_type.upper()} ACCEPTED] "
                    f"node={new_node.id} score={new_node.score:.6f}"
                )
            else:
                log.debug(f"[ACCEPTED] node={new_node.id} score={new_node.score:.6f}")

            self.storage.save_node(new_node)

        def _do_epoch_transition() -> None:
            nonlocal \
                epoch, \
                epoch_no_improve, \
                epoch_tasks, \
                active_pool, \
                epoch_patience_best, \
                draining_epoch, \
                epoch_drain_reason, \
                pool_refilling

            reason = epoch_drain_reason
            draining_epoch = False
            epoch_drain_reason = ""

            log.info(f"Epoch {epoch} → {epoch + 1}: {reason}")
            epoch += 1
            if collector is not None:
                collector.on_epoch_transition(epoch)
            epoch_no_improve = 0
            epoch_tasks = 0
            n_seeds = epoch_pool_size or max(1, active_pool_size // 4)
            seeds = self.strategy.epoch_seeds(active_pool, n_seeds)
            old_pool_ids = {n.id for n in active_pool}
            if seeds:
                active_pool = seeds
                log.info(f"Epoch {epoch}: seeded with Pareto-front nodes.")
            else:
                active_pool = list(initial_nodes[:active_pool_size])
                log.info(f"Epoch {epoch}: restarting from initial node.")
            kept_ids = {n.id for n in active_pool}
            for nid in old_pool_ids - kept_ids:
                self.storage.record_eviction(nid, tasks_completed)

            valid_scores = [n.score for n in active_pool if n.score < INVALID_SCORE]
            epoch_patience_best = min(valid_scores) if valid_scores else INVALID_SCORE
            pool_refilling = True

        def _check_epoch_end():
            nonlocal draining_epoch, epoch_drain_reason, pool_refilling

            past_seed_phase = (
                epoch > 0 or seed_tasks == 0 or epoch0_seeds_completed >= seed_tasks
            )
            if not past_seed_phase:
                return

            if pool_refilling:
                if len(active_pool) < active_pool_size:
                    return
                pool_refilling = False

            llm_in_flight = (
                self._llm_in_flight.value if hasattr(self, "_llm_in_flight") else 0
            )

            # If already draining, transition as soon as all LLM calls finish.
            if draining_epoch:
                if llm_in_flight == 0:
                    _do_epoch_transition()
                return

            staleness = (
                epoch_patience is not None and epoch_no_improve >= epoch_patience
            )
            steps_exhausted = epoch_steps is not None and epoch_tasks >= epoch_steps
            low_diversity, pool_diversity = self.strategy.should_diversify(active_pool)

            valid_scores = [n.score for n in active_pool if n.score < INVALID_SCORE]
            if len(valid_scores) >= 2:
                mean = sum(valid_scores) / len(valid_scores)
                score_std = math.sqrt(
                    sum((s - mean) ** 2 for s in valid_scores) / len(valid_scores)
                )
            else:
                score_std = 0.0

            if collector is not None:
                collector.on_pool_state(diversity=pool_diversity, score_std=score_std)

            low_variance = (
                epoch_variance is not None
                and epoch_variance > 0
                and score_std < epoch_variance
            )

            if staleness or steps_exhausted or low_diversity or low_variance:
                if staleness:
                    reason = (
                        f"staleness ({epoch_no_improve} >="
                        f" {epoch_patience} LLM calls without improvement)"
                    )
                elif steps_exhausted:
                    reason = (
                        f"steps exhausted ({epoch_tasks} >= {epoch_steps} LLM calls)"
                    )
                elif low_diversity:
                    reason = f"low diversity ({pool_diversity:.4f})"
                else:
                    reason = f"low variance ({score_std:.6f} < {epoch_variance})"

                if llm_in_flight == 0:
                    epoch_drain_reason = reason
                    _do_epoch_transition()
                else:
                    log.info(
                        f"Epoch {epoch} end condition met ({reason}),"
                        f" waiting for {llm_in_flight} LLM call(s) to finish."
                    )
                    draining_epoch = True
                    epoch_drain_reason = reason

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
                if max_llm_calls is not None and total_llm_completions >= max_llm_calls:
                    log.warning(f"Max LLM calls ({max_llm_calls}) reached. Ending run.")
                    break

                seeds_ready_threshold = (
                    math.ceil(0.8 * seed_tasks) if seed_tasks > 0 else 0
                )

                _dispatch_tasks()

                continue_loop, res = _fetch_result()
                if not continue_loop:
                    break
                if res is None:
                    continue

                if isinstance(res, dict) and "init_error" in res:
                    raise RuntimeError(
                        f"Worker initialization failed: {res['init_error']}"
                    )

                in_flight -= 1
                tasks_completed += 1
                if res.llm_type:
                    epoch_no_improve += 1
                    epoch_tasks += 1
                    total_llm_completions += 1
                if epoch == 0 and seed_tasks > 0 and res.task_id <= seed_tasks:
                    epoch0_seeds_completed += 1

                llm_in_flight = (
                    self._llm_in_flight.value if hasattr(self, "_llm_in_flight") else 0
                )
                if collector is not None:
                    collector.on_result(
                        res,
                        tasks_completed=tasks_completed,
                        epoch_no_improve=epoch_no_improve,
                        epoch_tasks=epoch_tasks,
                        llm_in_flight=llm_in_flight,
                    )

                if not res.valid:
                    if res.llm_type:
                        log.info(
                            f"[{res.llm_type.upper()} INVALID] "
                            f"task={res.task_id} msg={res.invalid_msg}"
                        )
                    else:
                        log.debug(f"Task {res.task_id} rejected: {res.invalid_msg}")
                    if collector is not None:
                        collector.on_invalid(res)
                    continue

                _process_valid_result(res)
                _check_epoch_end()

        finally:
            if collector is not None:
                collector.on_shutdown()
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
