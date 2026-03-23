import logging
import multiprocessing as mp
import queue
import time
from collections.abc import Callable
from typing import Any

from .base import SearchStrategy
from .models import Result, SearchNode, Task

log = logging.getLogger(__name__)


class MultiprocessSearchEngine:
    def __init__(
        self,
        workers: int,
        strategy: SearchStrategy,
        storage: Any,
        max_total_tasks: int = 10000,
    ):
        self.workers = workers
        self.strategy = strategy
        self.storage = storage
        self.max_total_tasks = max_total_tasks

        self.ctx = mp.get_context("spawn")
        self.task_q = self.ctx.Queue(maxsize=max(64, workers * 8))
        self.result_q = self.ctx.Queue()
        self.procs = []

    def start_workers(self, worker_target: Callable, worker_params: dict):
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
        initial_nodes: list[SearchNode],
        max_accepts: int,
        max_wall_seconds: float | None,
    ):
        start_time = time.monotonic()
        node_states = {n.id: n.state for n in initial_nodes}
        accepted_nodes = list(initial_nodes)

        # Track the best score found so far
        best_node = min(initial_nodes, key=lambda n: n.score) if initial_nodes else None

        next_task_id = 1
        tasks_completed = 0
        accepted_count = 0
        in_flight = 0
        next_node_id = max((n.id for n in initial_nodes), default=0)

        log.info(
            f"Search started. Target accepts: {max_accepts}. Initial best score: {best_node.score if best_node else 'N/A'}"
        )

        try:
            while True:
                # 1. Termination Checks
                if (
                    max_wall_seconds
                    and (time.monotonic() - start_time) >= max_wall_seconds
                ):
                    log.warning("Time limit reached. Shutting down...")
                    break
                if accepted_count >= max_accepts:
                    log.info(f"Target of {max_accepts} accepts reached.")
                    break
                if tasks_completed >= self.max_total_tasks:
                    log.warning("Max task limit reached.")
                    break

                # 2. Feeding Tasks
                while in_flight < self.workers and next_task_id <= self.max_total_tasks:
                    progress = (
                        accepted_count / float(max_accepts) if max_accepts > 0 else 0.0
                    )
                    pid1, pid2 = self.strategy.select_parent(accepted_nodes, progress)
                    state2 = node_states.get(pid2) if pid2 is not None else None

                    self.task_q.put(
                        Task(
                            task_id=next_task_id,
                            parent_id=pid1,
                            parent_state=node_states.get(pid1),
                            worker_slot=in_flight % self.workers,
                            secondary_parent_id=pid2,
                            secondary_parent_state=state2,
                        )
                    )
                    next_task_id += 1
                    in_flight += 1

                # 3. Collecting Results
                try:
                    res: Result = self.result_q.get(timeout=0.2)
                except queue.Empty:
                    continue

                in_flight -= 1
                tasks_completed += 1

                if not res.valid:
                    log.debug(f"Task {res.task_id} failed: {res.invalid_msg}")
                    continue

                # 4. Evolution via Strategy
                next_node_id += 1
                new_state = self.strategy.create_new_state(
                    node_states[res.parent_id], res
                )

                new_node = SearchNode(
                    score=res.score,
                    id=next_node_id,
                    parent_id=res.parent_id,
                    state=new_state,
                )

                accepted_nodes.append(new_node)
                node_states[new_node.id] = new_state
                accepted_count += 1

                # Check for New Best
                is_new_best = False
                if best_node is None or new_node.score < best_node.score:
                    best_node = new_node
                    is_new_best = True

                # REPORT PROGRESS
                status_cross = "(CROSSOVER)" if res.secondary_parent_id else ""
                status_best = "NEW BEST" if is_new_best else "ACCEPTED"
                log.info(
                    f"[{status_best} {status_cross}] node={new_node.id} parent={res.parent_id} "
                    f"score={new_node.score:.6f} (Total: {accepted_count}/{max_accepts})"
                )

                self.storage.save_node(new_node)

        finally:
            self._shutdown()

        return best_node

    def _shutdown(self):
        log.info("Cleaning up worker processes...")
        for _ in self.procs:
            try:
                self.task_q.put_nowait(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=2.0)