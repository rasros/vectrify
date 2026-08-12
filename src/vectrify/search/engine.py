import contextlib
import logging
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from vectrify.search.base import SearchStrategy, StorageAdapter
from vectrify.search.collector import StatCollector
from vectrify.search.models import (
    INVALID_SCORE,
    ChainState,
    Result,
    SearchNode,
    Task,
    valid_scores,
)
from vectrify.search.operators import OperatorPolicy
from vectrify.search.stats import score_std

TState = TypeVar("TState")
log = logging.getLogger(__name__)

SEED_PHASE = "seed"
LOCAL_PHASE = "local"

# How much of a converged front to hand the evaluator. The front can be large
# and the evaluator is the expensive part of the run, so it sees the best few
# by the round's own objective rather than all of them.
FRONT_EVAL_CAP = 24


def keep_payload(result: Result) -> ChainState:
    """Default state builder: carry the worker's payload through unchanged."""
    return ChainState(score=result.score, payload=result.payload)


class MultiprocessSearchEngine(Generic[TState]):
    """Alternating LLM-seed / local-refine epochs.

    Every epoch opens with a batch of LLM calls and then runs local mutation
    and crossover only, until it converges. The operators are never mixed: an
    LLM edit degrades the median parent ~4x as much as a local mutation at
    ~1000x the cost, so it earns its keep as a restart point rather than as a
    move competing against local ones.
    """

    def __init__(
        self,
        workers: int,
        strategy: SearchStrategy[TState],
        storage: StorageAdapter[TState],
        max_total_tasks: int = 10000,
        make_state: Callable[[Result], ChainState[TState]] = keep_payload,
        rank_front: Callable[[list[SearchNode[TState]]], list[SearchNode[TState]]]
        | None = None,
    ):
        self.workers = workers
        self.strategy = strategy
        self.storage = storage
        self.max_total_tasks = max_total_tasks
        self.make_state = make_state
        # Orders a converged front by the run's real objective. The round
        # optimises pixel L1 because it is ~300x cheaper; this is where
        # perceptual judgement decides direction, once per epoch over a front
        # of tens rather than once per task over thousands.
        self.rank_front = rank_front

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
        epoch_patience: int | None = None,
        epoch_min_delta: float = 1e-4,
        active_pool_size: int = 20,
        generation_size: int | None = None,
        score_fn: Callable[[Result], float] | None = None,
        epoch_seeds: int = 0,
        initial_seeds: int | None = None,
        epochs: int | None = None,
        epoch_variance: float | None = None,
        operator_policy: OperatorPolicy | None = None,
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
        # Each starting candidate is its own lineage; children inherit it.
        node_roots = {n.id: n.root_id or n.id for n in initial_nodes}
        sorted_initial = sorted(initial_nodes, key=lambda n: n.score)
        active_pool: list[SearchNode[TState]] = sorted_initial[:active_pool_size]
        best_node = sorted_initial[0] if sorted_initial else None

        # Children are held back and merged as a generation, NSGA-II's mu+lambda
        # replacement: the truncation is a whole-population sort, so paying it
        # once per lambda children rather than once per child keeps the engine
        # from becoming the bottleneck. At the pool size runs actually use it
        # costs ~23 ms, against ~13 ms to produce a candidate -- per child that
        # would make selection, not search, the thing the run spends its time on.
        pending_children: list[SearchNode[TState]] = []
        lambda_size = max(1, generation_size or active_pool_size)

        epoch = 0
        epoch_no_improve = 0
        epoch_patience_best = best_node.score if best_node else INVALID_SCORE
        pool_refilling = False  # True until a fresh epoch's pool reaches capacity

        # Children accumulate outside active_pool: they replace it wholesale
        # once the batch lands. Epoch 0's batch is sized separately because
        # resumed candidates already count as seeds.
        first_batch = epoch_seeds if initial_seeds is None else initial_seeds
        phase = SEED_PHASE if first_batch > 0 else LOCAL_PHASE
        seed_parents: list[SearchNode[TState]] = list(active_pool)
        seed_children: list[SearchNode[TState]] = []
        seeds_target = first_batch
        seeds_dispatched = 0
        seeds_completed = 0
        # An epoch can transition with local tasks still in flight, and those
        # results land during the next seed phase. Without this they count as
        # seeds and end the batch before its LLM children arrive.
        seed_task_ids: set[int] = set()

        next_task_id = 1
        tasks_completed = 0
        in_flight = 0
        last_invalid_msg = "unknown error"

        next_node_id = max(
            self.storage.max_node_id, max((n.id for n in initial_nodes), default=0)
        )

        log.info(
            f"Search started. Initial best: {best_node.score if best_node else 'N/A'}"
        )
        if phase == SEED_PHASE:
            log.info(
                f"Epoch 0: seeding with {seeds_target} LLM call(s) "
                f"over {len(seed_parents)} parent(s)."
            )
        if collector is not None:
            collector.on_phase(phase, seeds_target)

        def _begin_seed_phase() -> None:
            """Open an epoch with a batch of LLM edits of the current front."""
            nonlocal \
                phase, \
                seed_parents, \
                seed_children, \
                seeds_target, \
                seeds_dispatched, \
                seeds_completed, \
                seed_task_ids

            parents = self.strategy.epoch_parents(
                active_pool, max(epoch_seeds, FRONT_EVAL_CAP)
            )
            if parents and self.rank_front is not None:
                try:
                    parents = self.rank_front(parents)
                except Exception as exc:
                    log.warning(f"Front evaluation failed, keeping L1 order: {exc}")
            parents = parents[:epoch_seeds]
            if not parents:
                parents = list(active_pool)

            seed_parents = parents
            seed_children = []
            pending_children.clear()
            seed_task_ids = set()
            seeds_dispatched = 0
            seeds_completed = 0
            seeds_target = epoch_seeds

            if seeds_target <= 0 or not seed_parents:
                phase = LOCAL_PHASE
                return

            phase = SEED_PHASE
            log.info(
                f"Epoch {epoch}: seeding with {seeds_target} LLM call(s) "
                f"over {len(seed_parents)} parent(s)."
            )
            if collector is not None:
                collector.on_phase(phase, seeds_target)

        def _finish_seed_phase() -> None:
            """Install the LLM children as the epoch's pool and start refining."""
            nonlocal \
                phase, \
                active_pool, \
                node_states, \
                epoch_no_improve, \
                epoch_patience_best, \
                pool_refilling

            valid_children = [c for c in seed_children if c.score < INVALID_SCORE]
            previous_ids = {n.id for n in active_pool}

            if not valid_children:
                if epoch == 0 and not any(n.score < INVALID_SCORE for n in active_pool):
                    raise RuntimeError(
                        f"All {seeds_target} epoch-0 seed task(s) failed and no "
                        f"candidate was accepted; last error: {last_invalid_msg}"
                    )
                # Not fatal mid-run: keep refining the pool the edits came from.
                log.warning(
                    f"Epoch {epoch}: every seed edit failed; "
                    "continuing from the previous pool."
                )
            else:
                if epoch == 0:
                    # Nothing to restart from yet, and clearing here would
                    # discard what --resume just restored. Later epochs do
                    # replace the pool outright.
                    carried = [n for n in active_pool if n.score < INVALID_SCORE]
                    new_pool = valid_children + carried
                else:
                    new_pool = valid_children

                active_pool = new_pool[:active_pool_size]
                node_states = {n.id: n.state for n in active_pool}
                for nid in previous_ids - set(node_states):
                    self.storage.record_eviction(nid, tasks_completed)

            phase = LOCAL_PHASE
            epoch_no_improve = 0
            scores = valid_scores(active_pool)
            epoch_patience_best = min(scores) if scores else INVALID_SCORE
            pool_refilling = True
            log.info(
                f"Epoch {epoch}: refining {len(active_pool)} candidate(s) locally."
            )
            if collector is not None:
                collector.on_phase(phase, seeds_target)

        def _dispatch_tasks():
            nonlocal in_flight, next_task_id, seeds_dispatched

            while in_flight < self.workers and next_task_id <= self.max_total_tasks:
                if phase == SEED_PHASE:
                    if seeds_dispatched >= seeds_target:
                        return
                    # Round-robin: a front smaller than the batch still gets
                    # every parent edited before any is edited twice.
                    parent = seed_parents[seeds_dispatched % len(seed_parents)]
                    task = Task(
                        task_id=next_task_id,
                        parent_id=parent.id,
                        parent_state=parent.state,
                        force_llm=True,
                    )
                    seed_task_ids.add(next_task_id)
                    seeds_dispatched += 1
                else:
                    pid1, pid2 = self.strategy.select_parent(active_pool)
                    task = Task(
                        task_id=next_task_id,
                        parent_id=pid1,
                        parent_state=node_states[pid1],
                        secondary_parent_id=pid2,
                        secondary_parent_state=node_states[pid2] if pid2 else None,
                        force_llm=False,
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
                if collector is not None and hasattr(self, "_llm_in_flight"):
                    collector.on_idle(
                        llm_in_flight=self._llm_in_flight.value,
                        valid_scores=valid_scores(active_pool),
                    )
                return True, None

        def _make_node(res: Result, *, new_lineage: bool = False) -> SearchNode[TState]:
            nonlocal next_node_id

            if res.score is None:
                raise RuntimeError("Result has no score and no score_fn provided")

            next_node_id += 1
            # An LLM seed is an independent attempt at the picture, so it opens
            # a lineage; a local child continues its parent's.
            root = (
                next_node_id
                if new_lineage
                else node_roots.get(res.parent_id, next_node_id)
            )
            node_roots[next_node_id] = root
            return SearchNode(
                score=res.score,
                id=next_node_id,
                parent_id=res.parent_id,
                state=self.make_state(res),
                secondary_parent_id=res.secondary_parent_id,
                metrics=res.metrics,
                signature=res.signature,
                epoch=epoch,
                root_id=root,
                operator=res.operator,
            )

        def _note_best(new_node: SearchNode[TState], res: Result) -> bool:
            nonlocal best_node

            is_new_best = best_node is None or new_node.score < best_node.score
            if collector is not None:
                collector.on_accepted(
                    new_node,
                    is_new_best=is_new_best,
                    elapsed=time.monotonic() - start_time,
                    llm_type=res.llm_type,
                )
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
            return is_new_best

        def _process_seed_result(res: Result) -> None:
            new_node = _make_node(res, new_lineage=True)
            seed_children.append(new_node)
            node_states[new_node.id] = new_node.state
            _note_best(new_node, res)
            self.storage.save_node(new_node)

        def _close_generation() -> None:
            """Merge the finished batch of children into the pool.

            Survival is an NSGA-II truncation of parents+children by
            non-dominated rank then crowding distance, the same comparison
            parent selection uses. Doing it per arriving child would mean a
            full sort per result, which at pool size 20 costs about as much as
            producing the candidate did.
            """
            nonlocal active_pool

            if not pending_children:
                return

            combined = active_pool + pending_children
            survivors = self.strategy.select_survivors(combined, active_pool_size)
            kept = {n.id for n in survivors}

            for child in pending_children:
                if operator_policy is not None:
                    operator_policy.update(child.operator, child.id in kept)
                if child.id in kept:
                    node_states[child.id] = child.state
                    self.storage.save_node(child)
                    continue
                # A child can be the run's best and still lose its generation on
                # another objective. Save it anyway: save_best is about to write
                # it out, and lineage.csv should not omit the winner.
                if child is best_node:
                    self.storage.save_node(child)
                log.debug(
                    f"[REJECTED] node={child.id} "
                    f"score={child.score:.6f} (dominated by the pool)"
                )
                if collector is not None:
                    collector.on_pool_rejected(is_llm=False)

            for node in active_pool:
                if node.id not in kept:
                    node_states.pop(node.id, None)
                    self.storage.record_eviction(node.id, tasks_completed)

            # Keep arrival order rather than the selector's rank order: the
            # pool is an unordered set to every reader, and reshuffling it each
            # generation would churn the dashboard for nothing.
            active_pool = [n for n in combined if n.id in kept]
            pending_children.clear()

        def _process_local_result(res: Result) -> None:
            nonlocal epoch_patience_best, epoch_no_improve

            new_node = _make_node(res)
            pending_children.append(new_node)
            _note_best(new_node, res)

            if new_node.score <= epoch_patience_best - epoch_min_delta:
                epoch_patience_best = new_node.score
                epoch_no_improve = 0
                if collector is not None:
                    collector.on_no_improve_reset()

            if len(pending_children) >= lambda_size:
                _close_generation()

        def _do_epoch_transition(reason: str) -> None:
            nonlocal epoch

            # The next seed batch edits this pool's front, so the children that
            # arrived since the last generation have to land in it first.
            _close_generation()

            log.info(f"Epoch {epoch} → {epoch + 1}: {reason}")
            epoch += 1
            if collector is not None:
                collector.on_epoch_transition(epoch)
            if epochs is not None and epoch >= epochs:
                # The run loop is about to stop; a batch opened here would be
                # paid for and discarded.
                return
            _begin_seed_phase()

        def _check_epoch_end():
            nonlocal pool_refilling

            if pool_refilling:
                if len(active_pool) < active_pool_size:
                    return
                pool_refilling = False

            staleness = (
                epoch_patience is not None and epoch_no_improve >= epoch_patience
            )
            low_diversity, pool_div = self.strategy.should_diversify(active_pool)
            pool_std = score_std(valid_scores(active_pool))

            if collector is not None:
                collector.on_pool_state(diversity=pool_div, score_std=pool_std)

            low_variance = (
                epoch_variance is not None
                and epoch_variance > 0
                and pool_std < epoch_variance
            )

            if staleness:
                reason = (
                    f"staleness ({epoch_no_improve} >="
                    f" {epoch_patience} tasks without improvement)"
                )
            elif low_diversity:
                reason = f"low diversity ({pool_div:.4f})"
            elif low_variance:
                reason = f"low variance ({pool_std:.6f} < {epoch_variance})"
            else:
                return

            _do_epoch_transition(reason)

        def _final_artifact() -> SearchNode[TState] | None:
            """The candidate to write out, chosen by the evaluator.

            best_node is the winner on the round's score, which is a cheap
            stand-in for the real objective and ranks candidates at about rho
            0.83 against it. Within one front the evaluator finds roughly a 2x
            spread, so trusting the proxy here is close to picking arbitrarily
            among the good candidates -- and the run has already paid for every
            one of them.

            best_node is included in the comparison rather than replaced, so
            this cannot do worse by the evaluator's own judgement than the
            score-based pick did.

            The whole pool is evaluated, not the capped front an epoch boundary
            gets: FRONT_EVAL_CAP exists because a boundary pays that cost every
            epoch, and this happens once. Measured on the bench, restricting it
            to the front would have left most of the gap unclaimed -- on one
            case the pool held a candidate the evaluator scored 8x better than
            the one the round score picked.
            """
            if self.rank_front is None or not active_pool:
                return best_node

            finalists = [n for n in active_pool if n.score < INVALID_SCORE]
            if best_node is not None and all(n.id != best_node.id for n in finalists):
                finalists.append(best_node)
            if not finalists:
                return best_node

            try:
                return self.rank_front(finalists)[0]
            except Exception as exc:
                log.warning(f"Final evaluation failed, keeping the best score: {exc}")
                return best_node

        try:
            while True:
                if (
                    max_wall_seconds
                    and (time.monotonic() - start_time) >= max_wall_seconds
                ):
                    log.warning("Time limit reached.")
                    break
                if tasks_completed >= self.max_total_tasks:
                    log.warning("Max task limit reached.")
                    break
                if epochs is not None and epoch >= epochs:
                    log.info(f"Max epochs ({epochs}) reached.")
                    break

                _dispatch_tasks()

                continue_loop, res = _fetch_result()
                if not continue_loop:
                    break
                if res is None:
                    continue

                in_flight -= 1
                tasks_completed += 1

                in_batch = res.task_id in seed_task_ids
                # Outlived its epoch: the pool it was measured against is gone.
                stale = phase == SEED_PHASE and not in_batch
                if in_batch:
                    seeds_completed += 1
                elif not stale:
                    # Staleness asks how long hill-climbing has stalled, so
                    # only local tasks count.
                    epoch_no_improve += 1

                llm_in_flight = (
                    self._llm_in_flight.value if hasattr(self, "_llm_in_flight") else 0
                )
                if collector is not None:
                    collector.on_result(
                        res,
                        tasks_completed=tasks_completed,
                        epoch_no_improve=epoch_no_improve,
                        seeds_completed=seeds_completed,
                        llm_in_flight=llm_in_flight,
                    )

                if stale:
                    log.debug(
                        f"Task {res.task_id} outlived epoch {epoch - 1}; dropped."
                    )
                elif not res.valid:
                    last_invalid_msg = res.invalid_msg or "unknown error"
                    if res.llm_type:
                        log.info(
                            f"[{res.llm_type.upper()} INVALID] "
                            f"task={res.task_id} msg={res.invalid_msg}"
                        )
                    else:
                        log.debug(f"Task {res.task_id} rejected: {res.invalid_msg}")
                    if collector is not None:
                        collector.on_invalid(res)
                elif in_batch:
                    _process_seed_result(res)
                else:
                    _process_local_result(res)

                if phase == SEED_PHASE:
                    if seeds_completed >= seeds_target and in_flight == 0:
                        _finish_seed_phase()
                else:
                    _check_epoch_end()

        finally:
            # Record the partial generation the run stopped in the middle of,
            # so lineage.csv covers every candidate that was paid for.
            with contextlib.suppress(Exception):
                _close_generation()
            with contextlib.suppress(Exception):
                best_node = _final_artifact()
            if best_node is not None:
                # Still swallowed so a save failure cannot mask an in-flight
                # exception during shutdown, but never silently: this is the
                # run's single most important artifact.
                try:
                    self.storage.save_best(best_node)
                except Exception as e:
                    log.error(f"Failed to write the best candidate: {e!r}")
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
