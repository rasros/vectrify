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
    ) -> SearchNode[TState] | None:
        start_time = time.monotonic()
        node_states = {n.id: n.state for n in initial_nodes}
        accepted_nodes = list(initial_nodes)

        best_node = min(initial_nodes, key=lambda n: n.score) if initial_nodes else None

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

                while in_flight < self.workers and next_task_id <= self.max_total_tasks:
                    progress = (
                        accepted_count / float(max_accepts) if max_accepts > 0 else 0.0
                    )
                    pid1, pid2 = self.strategy.select_parent(accepted_nodes, progress)

                    self.task_q.put(
                        Task(
                            task_id=next_task_id,
                            parent_id=pid1,
                            parent_state=node_states[pid1],
                            worker_slot=in_flight % self.workers,
                            secondary_parent_id=pid2,
                            secondary_parent_state=node_states[pid2] if pid2 else None,
                        )
                    )
                    next_task_id += 1
                    in_flight += 1

                try:
                    res: Result = self.result_q.get(timeout=0.2)
                except queue.Empty:
                    if not any(p.is_alive() for p in self.procs):
                        raise RuntimeError(
                            "All worker processes have exited. Check logs for "
                            "initialization errors (missing API keys, etc)."
                        ) from None
                    continue

                in_flight -= 1
                tasks_completed += 1

                if not res.valid:
                    log.debug(f"Task {res.task_id} invalid: {res.invalid_msg}")
                    continue

                next_node_id += 1
                new_state = self.strategy.create_new_state(
                    node_states[res.parent_id], res
                )

                new_node = SearchNode(
                    score=res.score,
                    id=next_node_id,
                    parent_id=res.parent_id,
                    state=new_state,
                    secondary_parent_id=res.secondary_parent_id,
                )

                accepted_nodes.append(new_node)
                node_states[new_node.id] = new_state
                accepted_count += 1

                if best_node is None or new_node.score < best_node.score:
                    best_node = new_node
                    status = "NEW BEST"
                else:
                    status = "ACCEPTED"

                log.info(
                    f"[{status}] node={new_node.id} "
                    f"score={new_node.score:.6f} "
                    f"temp={new_node.state.model_temperature:.3f}"
                )
                self.storage.save_node(new_node)

        finally:
            self._shutdown()

        return best_node

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
