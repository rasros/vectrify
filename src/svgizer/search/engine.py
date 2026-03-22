import logging
import multiprocessing as mp
import time
import queue
from typing import Dict, List, Optional, Tuple

from svgizer.models import ChainState, SearchNode, Task, Result, INVALID_SCORE
from svgizer.image_utils import png_bytes_to_data_url, make_preview_data_url
from svgizer.worker import worker_loop
from .base import SearchStrategy

log = logging.getLogger(__name__)

class MultiprocessSearchEngine:
    def __init__(
            self,
            workers: int,
            strategy: SearchStrategy,
            storage,
            scorer_type: str,
            max_total_tasks: int = 10000
    ):
        self.workers = workers
        self.strategy = strategy
        self.storage = storage
        self.scorer_type = scorer_type
        self.max_total_tasks = max_total_tasks

        self.ctx = mp.get_context("spawn")
        self.task_q = self.ctx.Queue(maxsize=max(64, workers * 8))
        self.result_q = self.ctx.Queue()
        self.procs = []

    def start_workers(self, params: dict):
        for _ in range(max(1, self.workers)):
            p = self.ctx.Process(
                target=worker_loop,
                args=(self.task_q, self.result_q, *params.values()),
                daemon=True
            )
            p.start()
            self.procs.append(p)

    def run(
            self,
            initial_nodes: List[SearchNode],
            max_accepts: int,
            max_wall_seconds: Optional[float],
            openai_image_long_side: int,
            original_dims: Tuple[int, int]
    ):
        start_time = time.monotonic()
        node_states = {n.id: n.state for n in initial_nodes}
        accepted_nodes = list(initial_nodes)
        node_info = {} # For lineage

        next_task_id = 1
        tasks_completed = 0
        accepted_count = 0
        in_flight = 0
        next_node_id = max((n.id for n in initial_nodes), default=0)

        try:
            while True:
                # 1. Termination Checks
                if (max_wall_seconds and (time.monotonic() - start_time) >= max_wall_seconds) or \
                        (accepted_count >= max_accepts) or (tasks_completed >= self.max_total_tasks):
                    break

                # 2. Feeding Tasks
                while in_flight < self.workers and next_task_id <= self.max_total_tasks:
                    progress = accepted_count / float(max_accepts) if max_accepts > 0 else 0.0
                    pid = self.strategy.select_parent(accepted_nodes, progress)

                    self.task_q.put(Task(
                        next_task_id, pid, node_states.get(pid), in_flight % self.workers
                    ))
                    next_task_id += 1
                    in_flight += 1

                # 3. Collecting Results
                try:
                    res: Result = self.result_q.get(timeout=0.2)
                except queue.Empty:
                    continue

                in_flight -= 1
                tasks_completed += 1

                if not res.valid: continue

                # 4. Evolution via Strategy
                next_node_id += 1
                new_state = self.strategy.create_new_state(node_states[res.parent_id], res)

                # Hydrate UI/Preview data (Engine responsibility)
                if self.storage.write_lineage_enabled:
                    new_state.raster_data_url = png_bytes_to_data_url(res.raster_png)
                new_state.raster_preview_data_url = make_preview_data_url(
                    res.raster_png, openai_image_long_side
                )

                new_node = SearchNode(score=res.score, id=next_node_id, parent_id=res.parent_id, state=new_state)

                accepted_nodes.append(new_node)
                node_states[new_node.id] = new_state
                accepted_count += 1

                # Save & Log
                iter_path = self.storage.save_node(new_node)
                node_info[new_node.id] = (res.parent_id, res.score, iter_path, res.change_summary)

                if accepted_count % 10 == 0:
                    self.storage.write_lineage(node_info)

        finally:
            self._shutdown()

        return min(accepted_nodes, key=lambda n: n.score)

    def _shutdown(self):
        for _ in self.procs:
            try: self.task_q.put_nowait(None)
            except: pass
        for p in self.procs: p.join(timeout=1.0)