import contextlib
import logging
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from vectrify.score.metrics import FRONT_SCORE
from vectrify.search.base import SearchStrategy
from vectrify.search.models import (
    ChainState,
    Result,
    SearchNode,
    Task,
)
from vectrify.search.operators import GradedReward, OperatorPolicy

TState = TypeVar("TState")
log = logging.getLogger(__name__)

# Cap the expensive evaluator's front.
FRONT_EVAL_CAP = 24

# Batch candidates for model-backed scoring.
SCORE_BATCH_SIZE = 32


def keep_payload(result: Result) -> ChainState:
    """Default state builder: carry the worker's payload through unchanged."""
    return ChainState(payload=result.payload)


@dataclass
class _RunState(Generic[TState]):
    """Mutable state owned by one execution of the search loop.

    Keeping this state together makes the lifetime of IDs, lineages, and the
    active pool explicit.  The engine itself is reusable: queues and worker
    processes live on it, while none of these values do.
    """

    node_states: dict[int, ChainState[TState]]
    node_metrics: dict[int, dict[str, float]]
    node_roots: dict[int, int]
    node_origins: dict[int, int]
    active_pool: list[SearchNode[TState]]
    next_node_id: int

    @classmethod
    def from_initial_nodes(
        cls,
        nodes: list[SearchNode[TState]],
        *,
        pool_size: int,
    ) -> "_RunState[TState]":
        return cls(
            node_states={node.id: node.state for node in nodes},
            node_metrics={node.id: dict(node.metrics) for node in nodes},
            node_roots={node.id: node.root_id or node.id for node in nodes},
            node_origins={node.id: node.origin_id or node.id for node in nodes},
            active_pool=list(nodes)[:pool_size],
            next_node_id=max((node.id for node in nodes), default=0),
        )


class _ScoringRelay:
    """Batch worker output, score it, then forward it to the search loop."""

    def __init__(
        self,
        unscored_q: Any,
        result_q: queue.Queue[Result | None],
        score_fn: Callable[[list[Result]], None] | None,
    ) -> None:
        self.unscored_q = unscored_q
        self.result_q = result_q
        self.score_fn = score_fn

    def _gather_batch(self) -> tuple[list[Result], bool]:
        """Block for one result, then take results that are already queued."""
        first = self.unscored_q.get()
        if first is None:
            return [], True

        batch = [first]
        while len(batch) < SCORE_BATCH_SIZE:
            try:
                result = self.unscored_q.get_nowait()
            except queue.Empty:
                break
            if result is None:
                return batch, True
            batch.append(result)
        return batch, False

    def run(self) -> None:
        while True:
            batch, done = self._gather_batch()
            pending = [
                result for result in batch if result.valid and not result.measured
            ]
            if pending and self.score_fn is not None:
                try:
                    self.score_fn(pending)
                except Exception as exc:
                    # Preserve successfully scored peers if one batch score fails.
                    for result in pending:
                        if not result.measured:
                            result.valid = False
                            result.invalid_msg = f"Scoring error: {exc}"
                            result.measured = True

            for result in batch:
                self.result_q.put(result)
            if done:
                self.result_q.put(None)
                return


@dataclass(frozen=True)
class SearchProgress:
    """A snapshot of a running search, reported after every result."""

    tasks_completed: int
    pool_size: int
    elapsed: float


@dataclass
class SearchOutcome(Generic[TState]):
    """How a search ended: the evaluator's pick and every valid pool member."""

    best: SearchNode[TState] | None
    pool: list[SearchNode[TState]]
    tasks_completed: int


class MultiprocessSearchEngine(Generic[TState]):
    """NSGA-II local search, judged by an optional evaluator.

    Workers mutate and recombine the pool; generations are merged by NSGA-II
    truncation. The evaluator is asked every ``epoch_eval_interval`` tasks and
    picks the best candidate at the end of the run.
    """

    def __init__(
        self,
        workers: int,
        strategy: SearchStrategy[TState],
        max_total_tasks: int | None = None,
        make_state: Callable[[Result], ChainState[TState]] = keep_payload,
        rank_front: Callable[[list[SearchNode[TState]]], list[SearchNode[TState]]]
        | None = None,
    ):
        self.workers = workers
        self.strategy = strategy
        self.max_total_tasks = max_total_tasks
        self.make_state = make_state
        # Orders a converged front by the run's real objective.
        self.rank_front = rank_front

        self.ctx = mp.get_context("spawn")
        self.task_q = self.ctx.Queue(maxsize=max(64, workers * 8))
        self.unscored_q = self.ctx.Queue()
        self.result_q = queue.Queue()
        self.procs: list[Any] = []

    def start_workers(self, worker_target: Callable, worker_params: Any) -> None:
        log.info(f"Starting {self.workers} worker processes...")
        for index in range(max(1, self.workers)):
            if isinstance(worker_params, dict):
                worker_params["worker_index"] = index
            elif hasattr(worker_params, "worker_index"):
                worker_params.worker_index = index
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
        max_wall_seconds: float | None = None,
        active_pool_size: int = 20,
        generation_size: int | None = None,
        score_fn: Callable[[list[Result]], None] | None = None,
        epoch_eval_interval: int | None = None,
        operator_policy: OperatorPolicy | None = None,
        stop: threading.Event | None = None,
        progress: Callable[[SearchProgress], None] | None = None,
    ) -> SearchOutcome[TState]:
        start_time = time.monotonic()

        scorer_thread = threading.Thread(
            target=_ScoringRelay(self.unscored_q, self.result_q, score_fn).run,
            daemon=True,
            name="ScorerThread",
        )
        scorer_thread.start()

        run_state = _RunState.from_initial_nodes(
            initial_nodes,
            pool_size=active_pool_size,
        )
        node_states = run_state.node_states
        # Each node's measures, kept so a child can be compared with the parent
        # it came from. A candidate measuring the same on every objective is
        # indistinguishable from its parent to everything downstream: it cannot
        # be ranked above or below it, so it is admitted wherever the parent
        # sits and reports back as a survivor. See _is_no_op.
        node_metrics = run_state.node_metrics
        # One scale for the whole run, so every operator's children are graded
        # against the same notion of how big a step currently is.
        graded_reward = GradedReward()
        # Each starting candidate is its own lineage; children inherit it.
        node_roots = run_state.node_roots
        node_origins = run_state.node_origins

        # No ordering to apply: the measures are traded off by dominance and
        # nothing ranks a candidate on its own. The pool is a set, and the cap
        # takes whatever arrived.
        active_pool = run_state.active_pool
        # Set by the evaluator, the run's only score, at each check and at the
        # end. There is deliberately no best between those points.
        best_node: SearchNode[TState] | None = None

        # Hold children back and merge them as a generation: selection is a
        # whole-population NSGA-II truncation.
        pending_children: list[SearchNode[TState]] = []
        lambda_size = max(1, generation_size or active_pool_size)

        # The evaluator's best so far. Its score is a calibrated distance to the
        # target, so a value from one check is comparable with the next.
        best_panel: float | None = None
        last_eval_at = 0

        next_task_id = 1
        tasks_completed = 0
        in_flight = 0

        log.info(f"Search started with {len(active_pool)} candidate(s) in the pool.")

        def _dispatch_tasks():
            nonlocal in_flight, next_task_id

            while in_flight < self.workers and (
                self.max_total_tasks is None or next_task_id <= self.max_total_tasks
            ):
                pid1, pid2 = self.strategy.select_parent(active_pool)
                task = Task(
                    task_id=next_task_id,
                    parent_id=pid1,
                    parent_state=node_states[pid1],
                    secondary_parent_id=pid2,
                    secondary_parent_state=node_states[pid2] if pid2 else None,
                    # Crossover ignores it, but the worker falls back to
                    # mutation when the second parent turns out unusable.
                    operator=(
                        operator_policy.select()
                        if operator_policy is not None
                        else None
                    ),
                )

                self.task_q.put(task)
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
                return True, None

        def _make_node(res: Result) -> SearchNode[TState]:
            if not res.measured:
                raise RuntimeError("Result was never measured and no score_fn ran")

            run_state.next_node_id += 1
            node_id = run_state.next_node_id
            # A child continues its parent's lineage.
            root = node_roots.get(res.parent_id, node_id)
            node_roots[node_id] = root
            origin = node_origins.get(res.parent_id) or node_id
            node_origins[node_id] = origin
            return SearchNode(
                valid=True,
                id=node_id,
                parent_id=res.parent_id,
                state=self.make_state(res),
                secondary_parent_id=res.secondary_parent_id,
                metrics=res.metrics,
                signature=res.signature,
                root_id=root,
                origin_id=origin,
                operator=res.operator,
            )

        def _close_generation() -> None:
            """Merge the finished batch of children into the pool.

            Survival is an NSGA-II truncation of parents and children by
            non-dominated rank then crowding distance.
            """
            nonlocal active_pool

            if not pending_children:
                return

            combined = active_pool + pending_children
            survivors = self.strategy.select_survivors(combined, active_pool_size)
            kept = {n.id for n in survivors}

            for child in pending_children:
                if operator_policy is not None:
                    # Surviving is necessary and not sufficient: an operator
                    # earns what its child actually improved on its parent, so
                    # one that changes nothing perceptible scores nothing even
                    # though nothing can rank it below the parent either.
                    parent = node_metrics.get(child.parent_id)
                    reward = (
                        graded_reward(parent, child.metrics)
                        if child.id in kept and parent is not None
                        else 0.0
                    )
                    operator_policy.update(child.operator, reward)
                if child.id in kept:
                    node_states[child.id] = child.state
                    node_metrics[child.id] = dict(child.metrics)
                    continue
                log.debug(f"[REJECTED] node={child.id} (dominated by the pool)")

            for node in active_pool:
                if node.id not in kept:
                    node_states.pop(node.id, None)
                    node_metrics.pop(node.id, None)

            # Keep arrival order rather than the selector's rank order: the
            # pool is an unordered set to every reader, and reshuffling it each
            # generation would churn the dashboard for nothing.
            active_pool = [n for n in combined if n.id in kept]
            run_state.active_pool = active_pool
            pending_children.clear()

        def _is_no_op(res: Result) -> bool:
            """Whether this candidate measures exactly as its parent does.

            Not the same test as rejecting an identical file, which is all the
            worker can see. An operator may rewrite the markup and leave the
            render untouched -- reordering elements that do not overlap is the
            clearest case -- and the result is a candidate that differs in bytes
            and not in anything the search can perceive.

            Identical objectives cannot be ranked
            against the parent, so the candidate survives selection wherever the
            parent does and the operator policy is told it succeeded.
            """
            parent = node_metrics.get(res.parent_id)
            if parent is None or not res.metrics:
                return False
            return all(
                name in parent and parent[name] == res.metrics[name]
                for name in res.metrics
            )

        def _process_local_result(res: Result) -> None:
            if _is_no_op(res):
                if operator_policy is not None and res.operator is not None:
                    operator_policy.update(res.operator, 0.0)
                log.debug(
                    f"Task {res.task_id} measured identically to its parent "
                    f"({res.operator})"
                )
                return

            new_node = _make_node(res)
            pending_children.append(new_node)
            log.debug(f"[ACCEPTED] node={new_node.id}")

            # Progress is decided when the generation closes, where the pool is
            # ranked -- see _close_generation. A candidate cannot be known to
            # have reached the top tier before it has been ranked against one.
            if len(pending_children) >= lambda_size:
                _close_generation()

        def _run_panel_check() -> None:
            """Put the current front to the evaluator and record its verdict.

            The field is the best-ranked distinct candidates, capped: the top
            tier can be most of the pool, and evaluating near-clones spends the
            expensive part of the run learning nothing. Whatever the evaluator
            has already scored costs nothing to include, so the cap is about
            new work, not about the size of the field.
            """
            nonlocal best_panel, last_eval_at, best_node

            # Before anything else, including ranking a field: with no
            # evaluator there is no check to run, and building the field costs
            # the strategy a pass over the pool for a verdict nobody can give.
            if self.rank_front is None:
                return

            last_eval_at = tasks_completed
            field = self.strategy.epoch_parents(active_pool, FRONT_EVAL_CAP)
            if not field or self.rank_front is None:
                return
            try:
                ranked = self.rank_front(field)
            except Exception as exc:
                log.warning(f"Evaluator check failed, continuing: {exc}")
                return

            top = next((n for n in ranked if FRONT_SCORE in n.metrics), None)
            if top is None:
                return
            value = top.metrics[FRONT_SCORE]
            if best_panel is None or value < best_panel:
                best_panel = value
                best_node = top
                log.info(f"Evaluator: node={top.id} score={value:.6f}")

        def _maybe_check_evaluator() -> None:
            # Cheap measures can improve without the drawing getting better, so
            # ask the evaluator while the run is going, not only at its end.
            if (
                self.rank_front is not None
                and epoch_eval_interval
                and tasks_completed - last_eval_at >= epoch_eval_interval
            ):
                _run_panel_check()

        def _any_top_tier() -> SearchNode[TState] | None:
            """A member of the best-ranked tier, for when the evaluator never
            ran or failed. Nothing else can name a best: with the measures
            traded off there is no scalar to sort by, so any unbeaten candidate
            is as defensible as another -- and writing one of those beats
            losing the run's artifact to a scorer error at shutdown.
            """
            valid = [n for n in active_pool if n.valid]
            if not valid:
                return None
            top = self.strategy.top_tier_ids(valid)
            return next((n for n in valid if n.id in top), valid[0])

        def _final_artifact() -> SearchNode[TState] | None:
            """The candidate to write out, chosen by the evaluator.

            best_node is whatever the evaluator chose at the last check. It is
            included in the comparison rather than replaced, so
            this cannot come out worse by the evaluator's own judgement than its
            previous pick.

            The whole pool is evaluated, not the capped front used by the
            checks, so the final choice can include any valid candidate.
            """
            fallback = best_node or _any_top_tier()
            if self.rank_front is None or not active_pool:
                return fallback

            finalists = [n for n in active_pool if n.valid]
            if best_node is not None and all(n.id != best_node.id for n in finalists):
                finalists.append(best_node)
            if not finalists:
                return fallback

            try:
                return self.rank_front(finalists)[0]
            except Exception as exc:
                log.warning(f"Final evaluation failed, keeping a top-tier node: {exc}")
                return fallback

        try:
            while True:
                if stop is not None and stop.is_set():
                    log.info("Stopped on request.")
                    break
                if (
                    max_wall_seconds
                    and (time.monotonic() - start_time) >= max_wall_seconds
                ):
                    log.warning("Time limit reached.")
                    break
                if (
                    self.max_total_tasks is not None
                    and tasks_completed >= self.max_total_tasks
                ):
                    log.warning("Max task limit reached.")
                    break

                _dispatch_tasks()

                continue_loop, res = _fetch_result()
                if not continue_loop:
                    break
                if res is None:
                    continue

                in_flight -= 1
                tasks_completed += 1

                if not res.valid:
                    # A failed result names its operator only when that operator
                    # produced nothing to score. Charge the draw: it consumed a
                    # slot and returned no candidate, which is what a zero
                    # reward means. Leaving it unreported instead would park the
                    # operator at its prior weight and let it keep drawing.
                    if res.operator is not None and operator_policy is not None:
                        operator_policy.update(res.operator, 0.0)
                    log.debug(f"Task {res.task_id} rejected: {res.invalid_msg}")
                else:
                    _process_local_result(res)

                _maybe_check_evaluator()

                if progress is not None:
                    progress(
                        SearchProgress(
                            tasks_completed=tasks_completed,
                            pool_size=len(run_state.active_pool),
                            elapsed=time.monotonic() - start_time,
                        )
                    )

        finally:
            # Merge the partial generation the run stopped in the middle of, so
            # every candidate that was paid for can be the final pick.
            with contextlib.suppress(Exception):
                _close_generation()
            with contextlib.suppress(Exception):
                best_node = _final_artifact()
            self._shutdown()
        return SearchOutcome(
            best=best_node,
            pool=[n for n in run_state.active_pool if n.valid],
            tasks_completed=tasks_completed,
        )

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
