import contextlib
import logging
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from vectrify.score.metrics import FRONT_SCORE, SCORER_METRICS
from vectrify.search.base import SearchStrategy, StorageAdapter
from vectrify.search.collector import StatCollector
from vectrify.search.diversity import pool_diversity
from vectrify.search.models import (
    ChainState,
    Result,
    SearchNode,
    Task,
)
from vectrify.search.operators import GradedReward, OperatorPolicy

TState = TypeVar("TState")
log = logging.getLogger(__name__)

SEED_PHASE = "seed"
LOCAL_PHASE = "local"

# How much of a converged front to hand the evaluator. The front can be large
# and the evaluator is the expensive part of the run, so it sees the best few
# by dominance over the measures rather than all of them.
FRONT_EVAL_CAP = 24

# What share of the active pool the remembered LLM seeds may add to a front.
# They are there to give the evolved pool something to beat, so a handful is
# enough; letting them grow with the epoch count would eventually make the
# front mostly old seeds and spend the batch re-editing candidates the search
# has already moved past.
SEED_ARCHIVE_POOL_SHARE = 4

# How many candidates the scorer thread scores in one call. A model-backed
# scorer costs almost the same for a batch as for a single candidate, so
# anything already queued rides along nearly free.
SCORE_BATCH_SIZE = 32

# Replacement seed edits an epoch may ask for, as a share of the batch it
# wanted. One, so a batch of five may spend up to five retries and no more: a
# model that cannot produce a usable edit for this drawing at all would
# otherwise retry until the wall, and an epoch running on a short batch is
# still better than an epoch that never starts.
SEED_RETRY_SHARE = 1.0


def keep_payload(result: Result) -> ChainState:
    """Default state builder: carry the worker's payload through unchanged."""
    return ChainState(payload=result.payload)


def _spread_parents(
    ranked: list[SearchNode[TState]], count: int
) -> list[SearchNode[TState]]:
    """The best *count* candidates that are not near-copies of each other.

    An epoch has one LLM call per parent and a run has at most epochs x seeds
    of them, so they are the scarcest thing in the search. Taking the top
    *count* by evaluator rank spends them badly: the evaluator ranks
    near-identical drawings adjacently because they look near-identical, so the
    slots fill with variants of one candidate. Measured over one run's nine
    epochs, the five chosen parents were cousins a median three generations
    apart, and their mean pairwise distance over the measures was 0.0056 where
    five random survivors of the same pool stood 0.0543 apart -- ten times less
    variety than picking at random would have given.

    So walk the ranking in order and skip anything sitting closer to an
    already-chosen parent than the field's own median separation. The threshold
    is read off the candidates rather than fixed: what counts as near-identical
    depends on the drawing and on how far the run has converged, and a constant
    would stop meaning the same thing between the first epoch and the last. If
    too few clear it, the remainder falls back to rank order, because an LLM
    call left unspent is worse than one spent on a near-copy.
    """
    if count <= 0 or len(ranked) <= count:
        return ranked[:count]

    def vector(node: SearchNode[TState]) -> tuple[float, ...] | None:
        values: list[float] = []
        for name in SCORER_METRICS:
            value = node.metrics.get(name)
            if value is None:
                return None
            values.append(value)
        return tuple(values)

    vectors = {n.id: vector(n) for n in ranked}
    known = [v for v in vectors.values() if v is not None]
    if len(known) < 2:
        return ranked[:count]

    gaps = sorted(
        sum(abs(a - b) for a, b in zip(left, right, strict=True))
        for index, left in enumerate(known)
        for right in known[index + 1 :]
    )
    threshold = gaps[len(gaps) // 2]

    chosen: list[SearchNode[TState]] = []
    for node in ranked:
        here = vectors[node.id]
        if here is None or all(
            sum(abs(a - b) for a, b in zip(here, there, strict=True)) >= threshold
            for other in chosen
            if (there := vectors[other.id]) is not None
        ):
            chosen.append(node)
        if len(chosen) == count:
            return chosen

    for node in ranked:
        if len(chosen) == count:
            break
        if all(node.id != c.id for c in chosen):
            chosen.append(node)
    return chosen


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
        max_total_tasks: int | None = None,
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
        active_pool_size: int = 20,
        generation_size: int | None = None,
        score_fn: Callable[[list[Result]], None] | None = None,
        epoch_seeds: int = 0,
        initial_seeds: int | None = None,
        epochs: int | None = None,
        epoch_max_tasks: int | None = None,
        epoch_eval_interval: int | None = None,
        epoch_eval_patience: int | None = None,
        epoch_improvement: float = 0.0,
        epoch_improvement_patience: int = 1,
        operator_policy: OperatorPolicy | None = None,
        collector: StatCollector | None = None,
    ) -> None:
        start_time = time.monotonic()

        if collector is not None:
            collector.on_run_start(
                start_time=start_time,
                epoch_patience=epoch_patience or 0,
            )

        def _gather_batch() -> tuple[list[Result], bool]:
            """Block for one result, then take whatever else is already waiting.

            It never waits for a batch to fill. Waiting on a timeout made a
            slow producer pay that timeout on every single-candidate batch,
            which cost more than batching saved; taking only what is already
            queued means batches form exactly when scoring is the bottleneck
            and the queue is backing up, and cost nothing when it is not.

            Returns the batch and whether the shutdown sentinel arrived.
            """
            first = self.unscored_q.get()
            if first is None:
                return [], True

            batch = [first]
            while len(batch) < SCORE_BATCH_SIZE:
                try:
                    res = self.unscored_q.get_nowait()
                except queue.Empty:
                    break
                if res is None:
                    return batch, True
                batch.append(res)
            return batch, False

        def _scorer_worker():
            while True:
                batch, done = _gather_batch()

                pending = [r for r in batch if r.valid and not r.measured]
                if pending and score_fn is not None:
                    try:
                        score_fn(pending)
                    except Exception as e:
                        # One scoring failure must not discard the candidates
                        # batched alongside it, so each is marked individually
                        # and only those still unscored are lost.
                        for res in pending:
                            if not res.measured:
                                res.valid = False
                                res.invalid_msg = f"Scoring error: {e}"
                                res.measured = True

                for res in batch:
                    self.result_q.put(res)

                if done:
                    self.result_q.put(None)
                    break

        scorer_thread = threading.Thread(
            target=_scorer_worker, daemon=True, name="ScorerThread"
        )
        scorer_thread.start()

        node_states = {n.id: n.state for n in initial_nodes}
        # Each node's measures, kept so a child can be compared with the parent
        # it came from. A candidate measuring the same on every objective is
        # indistinguishable from its parent to everything downstream: it cannot
        # be ranked above or below it, so it is admitted wherever the parent
        # sits and reports back as a survivor. See _is_no_op.
        node_metrics = {n.id: dict(n.metrics) for n in initial_nodes}
        # One scale for the whole run, so every operator's children are graded
        # against the same notion of how big a step currently is.
        graded_reward = GradedReward()
        # Each starting candidate is its own lineage; children inherit it.
        node_roots = {n.id: n.root_id or n.id for n in initial_nodes}
        # Origins outlive lineages: an epoch's LLM edit opens a lineage but
        # continues the original attempt it was derived from.
        node_origins = {n.id: n.origin_id or n.id for n in initial_nodes}
        # The LLM's own output from earlier epochs, kept as a candidate for the
        # fronts later epochs are seeded from. Local refinement is not monotone
        # -- measured on the corpus it finishes behind best-of-5 seeding on half
        # the cases -- so a front drawn only from the evolved pool can hand the
        # model back a worse drawing than the one it produced itself, and the
        # next epoch then builds on the damage. Keyed by lineage, which a seed
        # opens and its descendants inherit, so this holds each seed once and
        # never the local children that came after it.
        #
        # It starts empty even on --resume: what storage restores is drawings
        # and ids, with no record of which the LLM wrote. Assuming they were
        # seeds would give a locally degraded candidate exactly the protection
        # this exists to escape, where assuming none were costs only the first
        # epoch's, which that epoch's own batch immediately restores.
        seed_archive: dict[int, SearchNode[TState]] = {}
        seed_archive_cap = max(1, active_pool_size // SEED_ARCHIVE_POOL_SHARE)

        # No ordering to apply: the measures are traded off by dominance and
        # nothing ranks a candidate on its own. The pool is a set, and the cap
        # takes whatever arrived.
        active_pool: list[SearchNode[TState]] = list(initial_nodes)[:active_pool_size]
        # Set by the evaluator, the run's only score, at each epoch boundary and
        # at the end. There is deliberately no best between those points.
        best_node: SearchNode[TState] | None = None

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
        epoch_started_at = 0
        # The evaluator's view of the epoch, kept between checks. Its score is
        # a calibrated distance to the target, so a value from one check is
        # comparable with the next -- which is the whole reason it can be
        # tracked at all.
        best_panel: float | None = None
        last_eval_at = 0
        # Counted in evaluator checks, not generations. A generation is 100
        # accepted candidates, so its size in tasks moves with the acceptance
        # rate and with --pool-size, and a threshold in generations quietly
        # depends on both -- and on --epoch-eval-interval, since a threshold
        # below one interval's worth fires before a check can ever intervene. A
        # check is the evaluator's own observation, which is the thing whose
        # failures are being counted.
        checks_without_gain = 0
        # The evaluator's best as it stood when the current epoch opened, and
        # how many consecutive epochs have failed to beat it by the required
        # margin. This is a run-level test, and deliberately separate from
        # checks_without_gain, which ends one epoch so the model can re-seed:
        # an epoch ending is a decision to try a different starting point,
        # while this is the decision that trying again is not paying for
        # itself. It also asks a different question from staleness, which
        # counts tasks since anything reached the top tier of the cheap
        # measures -- on a measured run the evaluator found a better candidate
        # 16 seconds before staleness ended the epoch, so the two disagree
        # about whether a run is still going anywhere.
        panel_at_epoch_open: float | None = None
        epochs_without_gain = 0
        # Set when the epochs stop paying, so the loop stops without the
        # transition having opened a seed batch it is about to discard.
        epochs_exhausted = False
        # Reset at every transition, so each epoch is judged against the
        # pool it opened with rather than against the first one.
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
        # Replacements left for seed edits that came back unusable. An epoch
        # asks for a batch and previously took whatever survived: measured on
        # one run, 5 of 15 edits failed and the epochs opened with 5, then 3,
        # then 2 candidates, and the last of those went stale in 14 seconds
        # because a pool grown from two seeds runs out of anything that can
        # outrank its own members. The failures are the model's formatting
        # rather than the drawing -- 3 of them were search/replace blocks that
        # matched nothing, 2 were replies with no parseable block at all -- so
        # another sample usually succeeds where the last one did not.
        #
        # Seeded here as well as in _begin_seed_phase because epoch 0 does not
        # go through it -- its batch is sized from initial_seeds and the phase
        # is set directly -- and epoch 0 is where the run's seeds come from.
        seed_retries_left = int(first_batch * SEED_RETRY_SHARE)
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

        log.info(f"Search started with {len(active_pool)} candidate(s) in the pool.")
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
                seed_task_ids, \
                seed_retries_left, \
                best_node, \
                best_panel

            # The remembered seeds enter the ranking as candidates rather than
            # being handed a reserved slot: a seed the pool has genuinely
            # improved on deserves to lose, and reserving would spend an LLM
            # call re-editing a drawing the search already beat. All that has to
            # be guaranteed is that the model's own work is still reachable when
            # local search has wandered away from it -- from there the same
            # comparison that ranks everything else can decide.
            pool_ids = {n.id for n in active_pool}
            candidates = active_pool + [
                n for n in seed_archive.values() if n.id not in pool_ids
            ]
            parents = self.strategy.epoch_parents(
                candidates, max(epoch_seeds, FRONT_EVAL_CAP)
            )
            # The standing best joins the ranked set afterwards, not before:
            # epoch_parents selects by dominance over the measures, and the
            # candidate the evaluator likes best is often dominated on those --
            # so choosing the field first would drop it, and then the boundary
            # would hand its title to something the evaluator rates lower.
            if best_node is not None and all(n.id != best_node.id for n in parents):
                parents.append(best_node)
            if parents and self.rank_front is not None:
                try:
                    parents = self.rank_front(parents)
                    # The evaluator has just spoken, which is the only occasion
                    # anything in the run is called best -- but only a candidate
                    # it rates higher takes the title. Assigning the boundary's
                    # top pick outright let a run find 0.340 and ship 0.377: the
                    # pick is the best of whatever survived selection on the
                    # measures, which need not include what the evaluator
                    # already preferred.
                    top = parents[0] if parents else None
                    value = top.metrics.get(FRONT_SCORE) if top is not None else None
                    if (
                        top is not None
                        and value is not None
                        and (best_panel is None or value < best_panel)
                    ):
                        best_panel = value
                        best_node = top
                        log.info(f"Best so far: node={top.id} evaluator={value:.6f}")
                        if collector is not None:
                            collector.on_evaluator_best(
                                value, elapsed=time.monotonic() - start_time
                            )
                except Exception as exc:
                    log.warning(f"Front evaluation failed, keeping rank order: {exc}")
            parents = _spread_parents(parents, epoch_seeds)
            if not parents:
                parents = list(active_pool)

            seed_parents = parents
            seed_children = []
            pending_children.clear()
            seed_task_ids = set()
            seeds_dispatched = 0
            seeds_completed = 0
            seeds_target = epoch_seeds
            seed_retries_left = int(epoch_seeds * SEED_RETRY_SHARE)

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
            nonlocal phase, active_pool, node_states, epoch_no_improve, pool_refilling

            valid_children = [c for c in seed_children if c.valid]
            previous_ids = {n.id for n in active_pool}

            if not valid_children:
                if epoch == 0 and not any(n.valid for n in active_pool):
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
                    carried = [n for n in active_pool if n.valid]
                    new_pool = valid_children + carried
                else:
                    new_pool = valid_children

                active_pool = new_pool[:active_pool_size]
                node_states = {n.id: n.state for n in active_pool}
                for nid in previous_ids - set(node_states):
                    self.storage.record_eviction(nid, tasks_completed)

            phase = LOCAL_PHASE
            epoch_no_improve = 0
            pool_refilling = True
            log.info(
                f"Epoch {epoch}: refining {len(active_pool)} candidate(s) locally."
            )
            if collector is not None:
                collector.on_phase(phase, seeds_target)

        def _dispatch_tasks():
            nonlocal in_flight, next_task_id, seeds_dispatched

            while in_flight < self.workers and (
                self.max_total_tasks is None or next_task_id <= self.max_total_tasks
            ):
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
                    )
                return True, None

        def _make_node(res: Result, *, new_lineage: bool = False) -> SearchNode[TState]:
            nonlocal next_node_id

            if not res.measured:
                raise RuntimeError("Result was never measured and no score_fn ran")

            next_node_id += 1
            # An LLM seed is an independent attempt at the picture, so it opens
            # a lineage; a local child continues its parent's.
            root = (
                next_node_id
                if new_lineage
                else node_roots.get(res.parent_id, next_node_id)
            )
            node_roots[next_node_id] = root
            origin = node_origins.get(res.parent_id) or next_node_id
            node_origins[next_node_id] = origin
            return SearchNode(
                valid=True,
                id=next_node_id,
                parent_id=res.parent_id,
                state=self.make_state(res),
                secondary_parent_id=res.secondary_parent_id,
                metrics=res.metrics,
                signature=res.signature,
                epoch=epoch,
                root_id=root,
                origin_id=origin,
                operator=res.operator,
            )

        def _outranks(a: SearchNode[TState], b: SearchNode[TState]) -> bool:
            """Whether *a* beats *b* under the strategy's own relation."""
            if not b.valid:
                return a.valid
            if not a.valid:
                return False
            return a.id in self.strategy.top_tier_ids([a, b])

        def _note_accepted(new_node: SearchNode[TState], res: Result) -> None:
            """Record an accepted candidate. Nothing here decides it is best:
            that is the evaluator's call and it happens at epoch boundaries."""
            if collector is not None:
                collector.on_accepted(
                    new_node,
                    is_new_best=False,
                    elapsed=time.monotonic() - start_time,
                    llm_type=res.llm_type,
                )
            if res.llm_type:
                log.info(f"[{res.llm_type.upper()} ACCEPTED] node={new_node.id}")
            else:
                log.debug(f"[ACCEPTED] node={new_node.id}")

        def _archive_seed(node: SearchNode[TState]) -> None:
            """Keep an LLM seed available to the fronts of later epochs.

            One entry per lineage, and only the best of it, so a lineage the
            model revisits cannot claim more of the front than a lineage it got
            right first time. Over the cap the worst entry goes, which leaves
            the archive holding the seeds most likely to still be worth editing.
            """
            current = seed_archive.get(node.root_id)
            if current is not None and not _outranks(node, current):
                return
            seed_archive[node.root_id] = node
            if len(seed_archive) > seed_archive_cap:
                # The entry the rest of the archive beats most often. Dominance
                # rather than a score, so no measure is privileged here either.
                entries = list(seed_archive.values())
                losses = {
                    n.root_id: sum(1 for m in entries if m is not n and _outranks(m, n))
                    for n in entries
                }
                worst = max(entries, key=lambda n: losses[n.root_id])
                del seed_archive[worst.root_id]

        def _process_seed_result(res: Result) -> None:
            new_node = _make_node(res, new_lineage=True)
            seed_children.append(new_node)
            _archive_seed(new_node)
            node_states[new_node.id] = new_node.state
            node_metrics[new_node.id] = dict(new_node.metrics)
            _note_accepted(new_node, res)
            self.storage.save_node(new_node, tasks_completed)

        def _close_generation() -> None:
            """Merge the finished batch of children into the pool.

            Survival is an NSGA-II truncation of parents+children by
            non-dominated rank then crowding distance, the same comparison
            parent selection uses. Doing it per arriving child would mean a
            full sort per result, which at pool size 20 costs about as much as
            producing the candidate did.
            """
            nonlocal active_pool, epoch_no_improve

            if not pending_children:
                return

            combined = active_pool + pending_children
            survivors = self.strategy.select_survivors(combined, active_pool_size)
            kept = {n.id for n in survivors}

            # Progress is a new candidate reaching the best-ranked tier. Read
            # off the dominance relation, so it needs no blended score and no
            # threshold on a magnitude -- an epoch goes stale when nothing new
            # can get to the front any more, whatever the numbers happen to be
            # denominated in.
            new_ids = {n.id for n in pending_children}
            top_tier = self.strategy.top_tier_ids(combined)
            if new_ids & top_tier:
                epoch_no_improve = 0
                if collector is not None:
                    collector.on_no_improve_reset()

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
                    # Content only for candidates that reached the best-ranked
                    # tier. A run admits most of what it produces -- one wrote
                    # 106,640 files -- and a node that never outranked anything
                    # is not worth reading back. The lineage row is written
                    # either way, so the record of what happened is complete.
                    self.storage.save_node(
                        child, tasks_completed, keep_content=child.id in top_tier
                    )
                    continue
                # A child can be the run's best and still lose its generation on
                # another objective. Save it anyway: save_best is about to write
                # it out, and lineage.csv should not omit the winner.
                if child is best_node:
                    self.storage.save_node(child, tasks_completed)
                log.debug(f"[REJECTED] node={child.id} (dominated by the pool)")
                if collector is not None:
                    collector.on_pool_rejected(is_llm=False)

            for node in active_pool:
                if node.id not in kept:
                    node_states.pop(node.id, None)
                    node_metrics.pop(node.id, None)
                    self.storage.record_eviction(node.id, tasks_completed)

            # Keep arrival order rather than the selector's rank order: the
            # pool is an unordered set to every reader, and reshuffling it each
            # generation would churn the dashboard for nothing.
            active_pool = [n for n in combined if n.id in kept]
            pending_children.clear()

        def _is_no_op(res: Result) -> bool:
            """Whether this candidate measures exactly as its parent does.

            Not the same test as rejecting an identical file, which is all the
            worker can see. An operator may rewrite the markup and leave the
            render untouched -- reordering elements that do not overlap is the
            clearest case -- and the result is a candidate that differs in bytes
            and not in anything the search can perceive.

            Those are worse than wasted. Identical objectives cannot be ranked
            against the parent, so the candidate survives selection wherever the
            parent does and the operator policy is told it succeeded. Measured
            on one run, 58% of all candidates were of this kind, none of them
            catchable by the byte comparison, and the operator that produces
            them most reliably had taken 74% of the policy's weight.
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
                if collector is not None:
                    collector.on_unchanged(res)
                log.debug(
                    f"Task {res.task_id} measured identically to its parent "
                    f"({res.operator})"
                )
                return

            new_node = _make_node(res)
            pending_children.append(new_node)
            _note_accepted(new_node, res)

            # Progress is decided when the generation closes, where the pool is
            # ranked -- see _close_generation. A candidate cannot be known to
            # have reached the top tier before it has been ranked against one.
            if len(pending_children) >= lambda_size:
                _close_generation()

        def _do_epoch_transition(reason: str) -> None:
            nonlocal epoch, epoch_started_at, checks_without_gain
            nonlocal panel_at_epoch_open, epochs_without_gain, epochs_exhausted

            epoch_started_at = tasks_completed
            # The evaluator's best carries across epochs -- it is an absolute
            # score, and a later epoch has to beat what the run already has --
            # but the patience counting restarts with the epoch.
            checks_without_gain = 0

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

            # Ask the evaluator what the epoch just ended actually bought,
            # before deciding to pay for another batch of seeds. The pool it
            # sees is the one the epoch finished with, since _close_generation
            # has already run. A second call here is close to free: the score
            # is absolute and cached per node, so re-ranking the same field in
            # _begin_seed_phase re-prices only what is new.
            _run_panel_check()
            if epoch_improvement_patience > 0 and best_panel is not None:
                if panel_at_epoch_open is not None:
                    if panel_at_epoch_open - best_panel > epoch_improvement:
                        epochs_without_gain = 0
                    else:
                        epochs_without_gain += 1
                panel_at_epoch_open = best_panel
                if epochs_without_gain >= epoch_improvement_patience:
                    log.info(
                        f"Epochs stopped paying: {epochs_without_gain} in a row "
                        f"improved the evaluator's best by no more than "
                        f"{epoch_improvement:g} (best {best_panel:.6f})."
                    )
                    epochs_exhausted = True
                    return
            _begin_seed_phase()

        def _run_panel_check() -> None:
            """Put the current front to the evaluator and record its verdict.

            The field is the best-ranked distinct candidates, capped: the top
            tier can be most of the pool, and evaluating near-clones spends the
            expensive part of the run learning nothing. Whatever the evaluator
            has already scored costs nothing to include, so the cap is about
            new work, not about the size of the field.
            """
            nonlocal best_panel, last_eval_at, checks_without_gain, best_node

            # Before anything else, including ranking a field: with no
            # evaluator there is no check to run, and building the field costs
            # the strategy a pass over the pool for a verdict nobody can give.
            if self.rank_front is None:
                return

            last_eval_at = tasks_completed
            checks_without_gain += 1
            if collector is not None:
                collector.on_evaluator_check(
                    checks_without_gain=checks_without_gain,
                    patience=epoch_eval_patience or 0,
                )
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
                checks_without_gain = 0
                best_node = top
                log.info(f"Evaluator: node={top.id} score={value:.6f}")
                if collector is not None:
                    collector.on_evaluator_best(
                        value, elapsed=time.monotonic() - start_time
                    )

        def _check_epoch_end():
            nonlocal pool_refilling

            if pool_refilling:
                if len(active_pool) < active_pool_size:
                    return
                pool_refilling = False

            staleness = (
                epoch_patience is not None and epoch_no_improve >= epoch_patience
            )
            # Still reported, no longer a criterion: read against the epoch's
            # opening value it was a ratio to a moment, and across real runs that
            # moment was a trough as often as a peak -- epoch 0 opened on the
            # "too little data" sentinel of 1.0 and later epochs ended above
            # their own baseline, so the rule fired at once or never.
            pool_div = pool_diversity(active_pool)

            if collector is not None:
                collector.on_pool_state(diversity=pool_div)
                collector.on_epoch_progress(tasks_completed - epoch_started_at)

            # A ceiling on how long one epoch may run. Staleness measures
            # whether the pool has stopped producing; this measures how long the
            # proxy has been left unsupervised, which is a different thing.
            # Measured on one run, the gap between top-tier entries has a median
            # of 68 tasks and a 99th percentile of 285, so staleness at 500 only
            # arrives at the far tail -- 150,800 tasks into the epoch, all of it
            # without the evaluator seeing anything. The epoch boundary is where
            # the evaluator ranks the front and the model re-seeds from its
            # choice, so capping the epoch caps the drift.
            # Ask the evaluator what it makes of the current front, now and
            # then rather than only at the boundary. The cheap measures can be
            # improved without the drawing getting better -- one run drove them
            # 64% down while the evaluator saw no difference at all -- and the
            # only way to notice is to ask the evaluator while it is happening.
            if (
                self.rank_front is not None
                and epoch_eval_interval
                and tasks_completed - last_eval_at >= epoch_eval_interval
            ):
                _run_panel_check()

            panel_stale = (
                epoch_eval_patience is not None
                and epoch_eval_patience > 0
                and best_panel is not None
                and checks_without_gain >= epoch_eval_patience
            )

            over_budget = (
                epoch_max_tasks is not None
                and epoch_max_tasks > 0
                and tasks_completed - epoch_started_at >= epoch_max_tasks
            )

            # Any one of these ending the epoch, rather than all of them
            # agreeing. Each is meant to be set tight enough that its firing is
            # sufficient on its own -- a pool that has collapsed into clones is
            # done whatever the score is still doing -- and the two rules fail
            # very differently when one is set wrongly. Under this rule a
            # criterion that seldom reaches its threshold simply sits idle;
            # requiring all of them would let that same criterion block every
            # transition, so the LLM would never re-seed and the epochs would
            # be spent as one long local search.
            #
            # Diversity stays opt-in. A pool collapses into agreement long
            # before it stops improving, so a threshold that looks safe ends
            # search while it is still working; staleness on the best score is
            # the criterion that measures the thing we mean by converged.
            if staleness:
                reason = (
                    f"staleness ({epoch_no_improve} >="
                    f" {epoch_patience} tasks without improvement)"
                )
            elif panel_stale:
                reason = (
                    "the evaluator has not seen a better candidate in "
                    f"{checks_without_gain} checks"
                )
            elif over_budget:
                reason = (
                    f"epoch budget ({tasks_completed - epoch_started_at} >="
                    f" {epoch_max_tasks} tasks)"
                )
            else:
                return

            _do_epoch_transition(reason)

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

            best_node is whatever the evaluator chose at the last epoch
            boundary. It is included in the comparison rather than replaced, so
            this cannot come out worse by the evaluator's own judgement than its
            previous pick.

            The whole pool is evaluated, not the capped front an epoch boundary
            gets: FRONT_EVAL_CAP exists because a boundary pays that cost every
            epoch, and this happens once. Measured on the bench, restricting it
            to the front would have left most of the gap unclaimed -- on one
            case the pool held a candidate the evaluator scored 8x better than
            the one the measures alone would have picked.
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
                if epochs is not None and epoch >= epochs:
                    log.info(f"Max epochs ({epochs}) reached.")
                    break
                if epochs_exhausted:
                    break

                _dispatch_tasks()

                continue_loop, res = _fetch_result()
                if not continue_loop:
                    break
                if res is None:
                    continue

                # A derived candidate rode along in a reply that was already
                # paid for: it never occupied a worker slot and was never
                # dispatched, so it frees nothing and completes no task.
                if not res.derived:
                    in_flight -= 1
                    tasks_completed += 1

                # Belonging to the open batch and being one of its deliveries
                # are different questions once a reply can carry several
                # candidates. The extras belong -- they must not be dropped as
                # having outlived the epoch -- but the batch is only satisfied
                # by the calls it asked for.
                in_batch = res.task_id in seed_task_ids
                # Outlived its epoch: the pool it was measured against is gone.
                stale = phase == SEED_PHASE and not in_batch
                if in_batch and not res.derived:
                    seeds_completed += 1
                    # A batch that came back short used to just run short. Ask
                    # for a replacement instead: the epoch is the only thing
                    # that puts new structure into the pool, and one that opens
                    # on two candidates cannot do it.
                    if not res.valid and seed_retries_left > 0:
                        seeds_target += 1
                        seed_retries_left -= 1
                        log.info(
                            f"Seed edit failed; asking for a replacement "
                            f"({seed_retries_left} left this epoch)"
                        )
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
                    # A failed result names its operator only when that operator
                    # produced nothing to score. Charge the draw: it consumed a
                    # slot and returned no candidate, which is what a zero
                    # reward means. Leaving it unreported instead would park the
                    # operator at its prior weight and let it keep drawing.
                    if res.operator is not None and operator_policy is not None:
                        operator_policy.update(res.operator, 0.0)
                    if res.llm_type:
                        log.info(
                            f"[{res.llm_type.upper()} INVALID] "
                            f"task={res.task_id} msg={res.invalid_msg}"
                        )
                    else:
                        log.debug(f"Task {res.task_id} rejected: {res.invalid_msg}")
                    if collector is not None:
                        collector.on_invalid(res)
                elif in_batch and phase == SEED_PHASE:
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
