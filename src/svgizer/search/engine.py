import logging
import multiprocessing as mp
import queue
import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from svgizer.search.base import SearchStrategy, StorageAdapter
from svgizer.search.models import Result, SearchNode, Task

TState = TypeVar("TState")
log = logging.getLogger(__name__)

_DIVERSE_INTERVAL = 5  # max one diversity seed per N tasks dispatched


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
        self.result_q = self.ctx.Queue()
        self.procs: list[Any] = []

    def start_workers(self, worker_target: Callable, worker_params: dict) -> None:
        log.info(f"Starting {self.workers} worker processes...")
        for _ in range(max(1, self.workers)):
            p = self.ctx.Process(
                target=worker_target,
                args=(self.task_q, self.result_q, worker_params),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

    def run(
        self,
        initial_nodes: list[SearchNode[TState]],
        max_accepts: int,
        max_wall_seconds: float | None,
        patience: int | None = None,
        min_delta: float = 1e-4,
        active_pool_size: int = 20,
        score_fn: Callable[[Result], float] | None = None,
        warmup_tasks: int = 0,
        warmup_diverse: int = 0,
    ) -> None:
        start_time = time.monotonic()
        node_states = {n.id: n.state for n in initial_nodes}

        # Fixed-size active pool: best N nodes by score, used for parent selection.
        sorted_initial = sorted(initial_nodes, key=lambda n: n.score)
        active_pool: list[SearchNode[TState]] = sorted_initial[:active_pool_size]

        best_node = sorted_initial[0] if sorted_initial else None
        patience_best = best_node.score if best_node else float("inf")
        no_improve_tasks = 0

        next_task_id = 1
        tasks_completed = 0
        accepted_count = 0
        in_flight = 0

        tasks_since_diverse = (
            _DIVERSE_INTERVAL  # Start at threshold to evaluate immediately
        )

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
                if patience and no_improve_tasks >= patience:
                    log.info(
                        f"Patience exhausted: no improvement >= {min_delta} "
                        f"over {patience} consecutive tasks."
                    )
                    break

                while in_flight < self.workers and next_task_id <= self.max_total_tasks:
                    progress = (
                        accepted_count / float(max_accepts) if max_accepts > 0 else 0.0
                    )
                    pid1, pid2 = self.strategy.select_parent(active_pool, progress)

                    is_warmup = next_task_id <= warmup_tasks
                    force_diverse = False

                    if next_task_id <= warmup_diverse:
                        force_diverse = True
                    elif not is_warmup and tasks_since_diverse >= _DIVERSE_INTERVAL:
                        force_diverse = self.strategy.should_diversify(active_pool)
                        if force_diverse:
                            log.warning(
                                "Pool diversity critically low. "
                                "Dispatching fresh LLM seed."
                            )

                    tasks_since_diverse = (
                        0 if force_diverse else tasks_since_diverse + 1
                    )

                    self.task_q.put(
                        Task(
                            task_id=next_task_id,
                            parent_id=pid1,
                            parent_state=node_states[pid1],
                            worker_slot=in_flight % self.workers,
                            secondary_parent_id=pid2,
                            secondary_parent_state=node_states[pid2] if pid2 else None,
                            force_llm=is_warmup,
                            force_diverse=force_diverse,
                        )
                    )
                    next_task_id += 1
                    in_flight += 1

                try:
                    res = self.result_q.get(timeout=0.2)
                except queue.Empty:
                    if not any(p.is_alive() for p in self.procs):
                        raise RuntimeError(
                            "All worker processes have exited. Check logs for "
                            "initialization errors (missing API keys, etc)."
                        ) from None
                    continue

                if isinstance(res, dict) and "init_error" in res:
                    raise RuntimeError(
                        f"Worker initialization failed: {res['init_error']}"
                    )

                in_flight -= 1
                tasks_completed += 1
                no_improve_tasks += 1

                if not res.valid:
                    log.warning(f"Task {res.task_id} rejected: {res.invalid_msg}")
                    continue

                if res.score is None:
                    if score_fn is None:
                        raise RuntimeError(
                            "Result has no score and no score_fn provided"
                        )
                    res.score = score_fn(res)
                elif score_fn is not None:
                    res.score = score_fn(res)

                next_node_id += 1
                new_state = self.strategy.create_new_state(res)

                new_node = SearchNode(
                    score=res.score,
                    id=next_node_id,
                    parent_id=res.parent_id,
                    state=new_state,
                    secondary_parent_id=res.secondary_parent_id,
                    complexity=res.complexity,
                    content=res.content,
                )

                active_pool.append(new_node)
                if len(active_pool) > active_pool_size:
                    # Evict the worst node (highest score) to keep pool bounded.
                    worst_idx = max(
                        range(len(active_pool)), key=lambda i: active_pool[i].score
                    )
                    active_pool.pop(worst_idx)
                node_states[new_node.id] = new_state
                accepted_count += 1

                if new_node.score <= patience_best - min_delta:
                    patience_best = new_node.score
                    no_improve_tasks = 0

                if best_node is None or new_node.score < best_node.score:
                    best_node = new_node
                    status = "NEW BEST"
                else:
                    status = "ACCEPTED"

                log.info(f"[{status}] node={new_node.id} score={new_node.score:.6f} ")
                self.storage.save_node(new_node)

        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        log.info("Shutting down workers...")
        for _ in self.procs:
            try:
                self.task_q.put(None, timeout=0.5)
            except queue.Full:
                log.debug("Task queue full during shutdown.")
        for p in self.procs:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
