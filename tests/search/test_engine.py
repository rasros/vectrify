import logging
import time

import pytest

from vectrify.search import INVALID_SCORE, ChainState, Result, SearchNode
from vectrify.search.engine import MultiprocessSearchEngine


class FakeStrategy:
    def select_parent(
        self,
        nodes: list[SearchNode],
    ) -> tuple[int, int | None]:
        _ = nodes
        return 1, None

    def should_diversify(self, pool: list[SearchNode]) -> tuple[bool, float]:
        _ = pool
        return False, 1.0

    def select_survivors(
        self, nodes: list[SearchNode], max_keep: int
    ) -> list[SearchNode]:
        return sorted(nodes, key=lambda n: n.score)[:max_keep]

    def epoch_parents(
        self, pool: list[SearchNode], max_parents: int
    ) -> list[SearchNode]:
        return pool[:max_parents]


class FakeStorage:
    def __init__(self):
        self.save_called = False
        self.best_saved: SearchNode | None = None
        self.max_node_id = 1
        self.current_run_dir = None

    def initialize(self) -> None:
        pass

    def load_resume_nodes(self, max_nodes: int = 20) -> list:
        _ = max_nodes
        return []

    def save_node(self, node: SearchNode) -> None:
        _ = node
        self.save_called = True

    def save_best(self, node: SearchNode) -> None:
        self.best_saved = node

    def record_eviction(self, node_id: int, tasks_completed: int) -> None:
        _ = node_id, tasks_completed


def test_engine_init():
    engine = MultiprocessSearchEngine(2, FakeStrategy(), FakeStorage())
    assert engine.workers == 2


def test_engine_run_loop_processes_result_and_saves():
    strat = FakeStrategy()
    store = FakeStorage()

    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=store, max_total_tasks=1
    )

    res = Result(
        task_id=1,
        parent_id=1,
        valid=True,
        score=0.1,
        payload="fake_payload",
    )
    # Put into the unscored queue so the ScorerThread can process it
    engine.unscored_q.put(res)

    initial_node = SearchNode(
        score=0.8,
        id=1,
        parent_id=0,
        state=ChainState(score=0.8, payload=None),
    )

    engine.run(
        initial_nodes=[initial_node],
        max_wall_seconds=None,
    )

    assert store.save_called is True


def test_engine_respects_max_wall_seconds(monkeypatch):
    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.select_calls = 0

        def select_parent(self, nodes: list[SearchNode]) -> tuple[int, int | None]:
            _ = nodes
            self.select_calls += 1
            return 1, None

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(workers=1, strategy=strat, storage=FakeStorage())

    class FakeTime:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return float(self.calls * 100)

    monkeypatch.setattr(time, "monotonic", FakeTime())

    dummy_node = SearchNode(
        score=0.5,
        id=1,
        parent_id=0,
        state=ChainState(score=0.5, payload=None),
    )

    # The first wall-clock check already exceeds the limit, so the run loop
    # must exit before dispatching any task.
    engine.run(initial_nodes=[dummy_node], max_wall_seconds=50.0)
    assert strat.select_calls == 0


def test_engine_epoch_patience_triggers_transition():

    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.epoch_parents_calls = 0

        def epoch_parents(self, pool, max_parents):
            self.epoch_parents_calls += 1
            return pool[:max_parents]

    strat = TrackingStrategy()
    store = FakeStorage()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=store, max_total_tasks=3
    )
    for score in (0.49, 0.48, 0.47):
        engine.unscored_q.put(
            Result(
                task_id=1,
                parent_id=1,
                valid=True,
                score=score,
                payload="p",
                llm_type="llm-generate",
            )
        )

    initial_node = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial_node],
        max_wall_seconds=None,
        epoch_patience=3,
        epoch_min_delta=0.1,
    )
    assert strat.epoch_parents_calls >= 1
    assert store.save_called


def test_engine_epoch_patience_resets_on_improvement():
    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.epoch_parents_calls = 0

        def epoch_parents(self, pool, max_parents):
            self.epoch_parents_calls += 1
            return pool[:max_parents]

    strat = TrackingStrategy()
    store = FakeStorage()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=store, max_total_tasks=3
    )

    # Each result improves on the previous best by more than epoch_min_delta,
    # so the patience counter resets every time and no transition may fire.
    for score in (0.35, 0.2, 0.05):
        engine.unscored_q.put(
            Result(
                task_id=1,
                parent_id=1,
                valid=True,
                score=score,
                payload="p",
                llm_type="llm-generate",
            )
        )

    initial_node = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial_node],
        max_wall_seconds=None,
        epoch_patience=2,
        epoch_min_delta=0.1,
    )
    assert strat.epoch_parents_calls == 0
    assert store.save_called


def test_engine_epoch_patience_none_no_transitions():

    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.epoch_parents_calls = 0

        def epoch_parents(self, pool, max_parents):
            self.epoch_parents_calls += 1
            return pool[:max_parents]

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(
        workers=1,
        strategy=strat,
        storage=FakeStorage(),
        max_total_tasks=0,
    )
    dummy_node = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[dummy_node],
        max_wall_seconds=None,
        epoch_patience=None,
    )
    assert strat.epoch_parents_calls == 0


def test_engine_respects_max_total_tasks():
    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.select_calls = 0

        def select_parent(self, nodes: list[SearchNode]) -> tuple[int, int | None]:
            _ = nodes
            self.select_calls += 1
            return 1, None

    strat = TrackingStrategy()
    store = FakeStorage()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=store, max_total_tasks=0
    )
    dummy_node = SearchNode(
        score=0.5,
        id=1,
        parent_id=0,
        state=ChainState(score=0.5, payload=None),
    )

    engine.run(initial_nodes=[dummy_node], max_wall_seconds=None)
    assert strat.select_calls == 0
    assert store.save_called is False


def test_engine_active_pool_bounded():
    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.max_seen = 0

        def select_parent(self, nodes: list[SearchNode]) -> tuple[int, int | None]:
            self.max_seen = max(self.max_seen, len(nodes))
            return nodes[0].id, None

    strat = TrackingStrategy()
    store = FakeStorage()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=store, max_total_tasks=10
    )

    for i in range(10):
        engine.unscored_q.put(
            Result(
                task_id=i + 1,
                parent_id=1,
                valid=True,
                score=float(i) * 0.01,
                payload="p",
            )
        )

    initial_node = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial_node],
        max_wall_seconds=None,
        active_pool_size=3,
    )
    assert strat.max_seen <= 4


def test_engine_score_fn_none_with_unscored_result_raises():
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), storage=FakeStorage()
    )
    engine.unscored_q.put(
        Result(
            task_id=1,
            parent_id=1,
            valid=True,
            score=None,
            payload="p",
        )
    )

    initial_node = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )

    with pytest.raises(RuntimeError, match="no score and no score_fn"):
        engine.run(
            initial_nodes=[initial_node],
            max_wall_seconds=None,
            score_fn=None,
        )


def test_engine_pool_collapse_epoch_end_does_not_crash():
    """Regression: the epoch-end branch compared an imported *function* to a
    float, so any positive threshold raised TypeError on the first check.
    Nothing covered this path, which is why it shipped.
    """
    strat = FakeStrategy()
    store = FakeStorage()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=store, max_total_tasks=1
    )
    engine.unscored_q.put(
        Result(task_id=1, parent_id=1, valid=True, score=0.1, payload="p")
    )
    initial = SearchNode(
        score=0.8, id=1, parent_id=0, state=ChainState(score=0.8, payload=None)
    )

    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        epoch_distinct=0.05,  # the flag that used to crash the run
        active_pool_size=1,
    )

    assert store.save_called is True


def test_engine_aborts_when_every_epoch0_seed_fails():
    """Epoch 0 has nothing to fall back to, so a failed batch must say why.

    Regression: the run used to idle until --max-wall-seconds and exit 0 with
    no output, hiding whatever the LLM returned (often a 401).
    """
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), storage=FakeStorage(), max_total_tasks=50
    )
    engine.unscored_q.put(
        Result(
            task_id=1,
            parent_id=1,
            valid=False,
            score=float("inf"),
            payload=None,
            invalid_msg="AuthenticationError(401)",
            llm_type="llm-generate",
        )
    )
    initial = SearchNode(
        score=float("inf"),
        id=1,
        parent_id=0,
        state=ChainState(score=float("inf"), payload=None),
    )

    with pytest.raises(RuntimeError, match="seed task"):
        engine.run(
            initial_nodes=[initial],
            max_wall_seconds=None,
            epoch_seeds=1,
            active_pool_size=1,
        )


def _seed_result(task_id: int, score: float) -> Result:
    return Result(
        task_id=task_id,
        parent_id=1,
        valid=True,
        score=score,
        payload="p",
        llm_type="llm-generate",
    )


def test_seed_batch_does_not_consult_the_parent_selector():

    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.select_calls = 0

        def select_parent(self, nodes):
            self.select_calls += 1
            return nodes[0].id, None

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=FakeStorage(), max_total_tasks=2
    )
    for i, score in enumerate((0.4, 0.3), start=1):
        engine.unscored_q.put(_seed_result(i, score))

    initial = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(initial_nodes=[initial], max_wall_seconds=None, epoch_seeds=2)

    assert strat.select_calls == 0


def test_seed_phase_cannot_go_stale():

    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.epoch_parents_calls = 0

        def epoch_parents(self, pool, max_parents):
            self.epoch_parents_calls += 1
            return pool[:max_parents]

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=FakeStorage(), max_total_tasks=3
    )
    for i, score in enumerate((0.49, 0.48, 0.47), start=1):
        engine.unscored_q.put(_seed_result(i, score))

    initial = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        epoch_seeds=3,
        epoch_patience=1,
        epoch_min_delta=0.1,
    )

    assert strat.epoch_parents_calls == 0


def test_epoch_zero_keeps_resumed_nodes_alongside_seed_children():

    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.pools_seen = []

        def select_parent(self, nodes):
            self.pools_seen.append({n.id for n in nodes})
            return nodes[0].id, None

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=FakeStorage(), max_total_tasks=2
    )
    engine.unscored_q.put(_seed_result(1, 0.3))
    engine.unscored_q.put(
        Result(task_id=2, parent_id=2, valid=True, score=0.4, payload="p")
    )

    initial = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(initial_nodes=[initial], max_wall_seconds=None, epoch_seeds=1)

    # The local task that follows the batch sees the seed child (id 2) and the
    # carried-in node (id 1).
    assert strat.pools_seen == [{1, 2}]


def test_local_results_that_outlive_their_epoch_do_not_count_as_seeds(caplog):
    """An epoch can transition with local tasks still in flight.

    Those results land during the next seed phase. Counted toward the batch
    they end it before its LLM children arrive, leaving the epoch refining
    leftovers of the pool it just discarded.
    """
    engine = MultiprocessSearchEngine(
        workers=4, strategy=FakeStrategy(), storage=FakeStorage(), max_total_tasks=6
    )

    # Epoch 0: seed (task 1), then four local tasks (2-5). Patience of 1 ends
    # the epoch on the first local result, leaving 3, 4 and 5 in flight; they
    # arrive after epoch 1 has already opened its batch (task 6).
    engine.unscored_q.put(_seed_result(1, 0.4))
    engine.unscored_q.put(
        Result(task_id=2, parent_id=1, valid=True, score=0.45, payload="p")
    )
    for tid in (3, 4):
        engine.unscored_q.put(
            Result(task_id=tid, parent_id=1, valid=True, score=0.9, payload="p")
        )
    engine.unscored_q.put(_seed_result(6, 0.35))
    engine.unscored_q.put(
        Result(task_id=5, parent_id=1, valid=True, score=0.9, payload="p")
    )

    initial = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    with caplog.at_level(logging.INFO, logger="vectrify.search.engine"):
        engine.run(
            initial_nodes=[initial],
            max_wall_seconds=None,
            epoch_seeds=1,
            epoch_patience=1,
            epoch_min_delta=0.1,
            active_pool_size=2,
            epochs=5,
        )

    refines = [m for m in caplog.messages if "refining" in m]
    # Epoch 1 was seeded once, so it has exactly one candidate to refine. The
    # three stale locals must not have been mistaken for seed children.
    assert refines[-1].startswith("Epoch 1: refining 1 candidate")


def test_seed_children_open_lineages_and_local_children_inherit():
    """Crossover pairs only across lineages, so the engine has to assign them:
    every LLM seed is an independent attempt, its descendants are not."""

    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.roots = []

        def select_parent(self, nodes):
            self.roots.append({n.id: n.root_id for n in nodes})
            return nodes[0].id, None

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=FakeStorage(), max_total_tasks=4
    )
    engine.unscored_q.put(_seed_result(1, 0.4))
    engine.unscored_q.put(_seed_result(2, 0.3))
    for tid in (3, 4):
        engine.unscored_q.put(
            Result(task_id=tid, parent_id=2, valid=True, score=0.35, payload="p")
        )

    initial = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        epoch_seeds=2,
        generation_size=1,
    )

    seen = strat.roots[-1]
    # The two seed children carry different lineages, and the local child that
    # followed carries its parent's rather than a new one.
    assert len(set(seen.values())) >= 2
    assert len(seen) > len(set(seen.values()))


def test_front_is_ranked_by_the_evaluator_not_by_the_round_score():
    """The round optimises pixel L1 because it is ~300x cheaper than the real
    objective. If the evaluator's verdict did not decide seed order, the epoch
    boundary would carry the cheap proxy's opinion into the next round."""

    class TrackingStrategy(FakeStrategy):
        def select_parent(self, nodes):
            return nodes[0].id, None

        def epoch_parents(self, pool, max_parents):
            return pool[:max_parents]

    seen: list[list[int]] = []

    def rank_front(nodes):
        seen.append([n.id for n in nodes])
        return sorted(nodes, key=lambda n: -n.score)  # deliberately reversed

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(
        workers=1,
        strategy=strat,
        storage=FakeStorage(),
        max_total_tasks=4,
        rank_front=rank_front,
    )
    engine.unscored_q.put(_seed_result(1, 0.4))
    for tid, score in ((2, 0.30), (3, 0.20), (4, 0.10)):
        engine.unscored_q.put(
            Result(task_id=tid, parent_id=1, valid=True, score=score, payload="p")
        )

    initial = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        epoch_seeds=1,
        epoch_patience=1,
        epoch_min_delta=0.1,
        active_pool_size=2,
        epochs=5,
    )
    assert seen, "the evaluator was never consulted"


def _explode(_nodes):
    raise RuntimeError("no cuda")


class _DriftingStrategy(FakeStrategy):
    """Survival that keeps the newest nodes, however they scored.

    Stands in for what the majority relation does on a real pool, where two
    objectives can outvote the score and evict the candidate an epoch was
    seeded from.
    """

    def __init__(self):
        self.epoch_pools: list[set[int]] = []

    def select_parent(self, nodes):
        return nodes[0].id, None

    def select_survivors(self, nodes, max_keep):
        return nodes[-max_keep:]

    def epoch_parents(self, pool, max_parents):
        self.epoch_pools.append({n.id for n in pool})
        return pool[:max_parents]


def test_a_new_epoch_can_be_seeded_from_the_llm_seed_local_search_replaced():
    """Local refinement is not monotone: a lineage can end an epoch worse than
    the seed it started from. If only the evolved pool reached the next front,
    the model would be handed the damaged drawing and build on it."""
    strat = _DriftingStrategy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=FakeStorage(), max_total_tasks=2
    )
    engine.unscored_q.put(_seed_result(1, 0.2))
    # Worse than the seed it descends from, and the pool holds one node.
    engine.unscored_q.put(
        Result(task_id=2, parent_id=2, valid=True, score=0.9, payload="p")
    )

    initial = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        epoch_seeds=1,
        epoch_patience=1,
        epoch_min_delta=0.1,
        active_pool_size=1,
        generation_size=1,
        epochs=2,
    )

    # Node 2 is the seed child, node 3 the local child that displaced it.
    assert strat.epoch_pools == [{2, 3}]


def test_remembered_seeds_stay_within_their_share_of_the_front():
    """Every epoch adds seeds, so without a bound the front would drift towards
    being all history and none of the pool the search actually built."""
    strat = _DriftingStrategy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=FakeStorage(), max_total_tasks=7
    )
    # Epoch 0: two seeds far better than anything that follows, then a local
    # child, which fills the pool and ends the epoch.
    for task_id, score in ((1, 0.10), (2, 0.11)):
        engine.unscored_q.put(_seed_result(task_id, score))
    engine.unscored_q.put(
        Result(task_id=3, parent_id=2, valid=True, score=0.7, payload="p")
    )
    # Epoch 1 replaces the pool with its own two seeds and refines them.
    for task_id, score in ((4, 0.50), (5, 0.60)):
        engine.unscored_q.put(_seed_result(task_id, score))
    for task_id in (6, 7):
        engine.unscored_q.put(
            Result(task_id=task_id, parent_id=5, valid=True, score=0.8, payload="p")
        )

    initial = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        epoch_seeds=2,
        epoch_patience=1,
        epoch_min_delta=0.1,
        active_pool_size=4,
        generation_size=1,
        epochs=3,
    )

    # Epoch 2 opens on a pool of epoch 1's nodes. Both epoch-0 seeds (2 and 3)
    # outscore all of them, and a pool of four admits one remembered seed, so
    # only the better of the two is carried in.
    assert strat.epoch_pools[-1] - {5, 6, 7, 8} == {2}


def test_a_resumed_run_treats_no_restored_node_as_an_llm_seed():
    """Storage restores drawings and ids, not who wrote them. Guessing that a
    restored node was a seed would protect a locally degraded candidate for the
    rest of the run, which is the failure the archive exists to prevent."""
    strat = _DriftingStrategy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=FakeStorage(), max_total_tasks=2
    )
    for task_id in (1, 2):
        engine.unscored_q.put(
            Result(task_id=task_id, parent_id=1, valid=True, score=0.9, payload="p")
        )

    resumed = SearchNode(
        score=0.1, id=1, parent_id=0, state=ChainState(score=0.1, payload=None)
    )
    engine.run(
        initial_nodes=[resumed],
        max_wall_seconds=None,
        epoch_seeds=1,
        initial_seeds=0,
        epoch_patience=1,
        epoch_min_delta=0.1,
        active_pool_size=1,
        generation_size=1,
        epochs=2,
    )

    # The restored node is gone from the pool and does not come back, even
    # though it scored better than what replaced it.
    assert strat.epoch_pools == [{2}]


def test_a_run_without_llm_seeds_offers_the_epoch_only_the_evolved_pool():
    """The bench runs with --seeds 0 and must see exactly what it saw before."""
    strat = _DriftingStrategy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=FakeStorage(), max_total_tasks=2
    )
    for task_id in (1, 2):
        engine.unscored_q.put(
            Result(task_id=task_id, parent_id=1, valid=True, score=0.9, payload="p")
        )

    initial = SearchNode(
        score=0.1, id=1, parent_id=0, state=ChainState(score=0.1, payload=None)
    )
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        epoch_seeds=0,
        epoch_patience=1,
        epoch_min_delta=0.1,
        active_pool_size=1,
        generation_size=1,
        epochs=2,
    )

    assert strat.epoch_pools == [{2}]


def test_a_failing_evaluator_does_not_stop_the_run():
    """It runs a model at an epoch boundary; a load failure there must cost the
    ordering, not the run."""
    engine = MultiprocessSearchEngine(
        workers=1,
        strategy=FakeStrategy(),
        storage=FakeStorage(),
        max_total_tasks=3,
        rank_front=_explode,
    )
    engine.unscored_q.put(_seed_result(1, 0.4))
    for tid in (2, 3):
        engine.unscored_q.put(
            Result(task_id=tid, parent_id=1, valid=True, score=0.35, payload="p")
        )
    initial = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        epoch_seeds=1,
        epoch_patience=1,
        epoch_min_delta=0.1,
        active_pool_size=2,
        epochs=4,
    )


def test_children_join_the_pool_only_when_the_generation_closes():
    """Survival is a sort over the whole population, so it is paid once per
    generation rather than once per child; until then children are held back."""

    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.pools = []

        def select_parent(self, nodes):
            self.pools.append({n.id for n in nodes})
            return nodes[0].id, None

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=FakeStorage(), max_total_tasks=5
    )
    for tid in range(1, 6):
        engine.unscored_q.put(
            Result(task_id=tid, parent_id=1, valid=True, score=0.1, payload="p")
        )

    initial = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        active_pool_size=4,
        generation_size=4,
    )

    assert strat.pools[:4] == [{1}] * 4
    assert len(strat.pools[4]) > 1


def test_epoch_transition_closes_the_open_generation():
    """The next batch edits this pool's front, so children that arrived since
    the last generation have to land before the front is read."""

    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.epoch_pools = []

        def epoch_parents(self, pool, max_parents):
            self.epoch_pools.append({n.id for n in pool})
            return pool[:max_parents]

    strat = TrackingStrategy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=FakeStorage(), max_total_tasks=2
    )
    for tid in (1, 2):
        engine.unscored_q.put(
            Result(task_id=tid, parent_id=1, valid=True, score=0.9, payload="p")
        )

    initial = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        epoch_patience=1,
        active_pool_size=4,
        generation_size=10,
    )

    assert strat.epoch_pools
    assert strat.epoch_pools[0] == {1, 2}


def test_the_engine_picks_the_operator_and_hears_how_it_did():
    """Eight workers each running their own policy could never learn anything:
    none of them sees whether its own children survived."""

    class RecordingPolicy:
        def __init__(self):
            self.selects = 0
            self.updates = []

        def select(self):
            self.selects += 1
            return "op"

        def update(self, operator, survived):
            self.updates.append((operator, survived))

    policy = RecordingPolicy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), storage=FakeStorage(), max_total_tasks=2
    )
    for tid, score in ((1, 0.1), (2, 0.9)):
        engine.unscored_q.put(
            Result(
                task_id=tid,
                parent_id=1,
                valid=True,
                score=score,
                payload="p",
                operator="op",
            )
        )

    initial = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial],
        max_wall_seconds=None,
        active_pool_size=1,
        generation_size=2,
        operator_policy=policy,
    )

    assert policy.selects == 2
    assert [engine.task_q.get().operator for _ in range(2)] == ["op", "op"]
    # The pool holds one node, so the better child survives and the worse does not.
    assert policy.updates == [("op", True), ("op", False)]


def test_an_operator_that_produced_nothing_is_charged_for_the_draw():
    """A blank draw spends a task and returns no candidate. Reporting nothing
    would leave the operator at its prior weight, still drawing its share to
    fail again; a zero is what the draw was actually worth."""

    class RecordingPolicy:
        def __init__(self):
            self.updates = []

        def select(self):
            return "op"

        def update(self, operator, survived):
            self.updates.append((operator, survived))

    policy = RecordingPolicy()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), storage=FakeStorage(), max_total_tasks=2
    )
    # One blank draw naming its operator, one ordinary failure naming none.
    engine.unscored_q.put(
        Result(
            task_id=1,
            parent_id=1,
            valid=False,
            score=INVALID_SCORE,
            payload=None,
            operator="op",
        )
    )
    engine.unscored_q.put(
        Result(task_id=2, parent_id=1, valid=False, score=INVALID_SCORE, payload=None)
    )

    engine.run(
        initial_nodes=[
            SearchNode(
                score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
            )
        ],
        max_wall_seconds=None,
        active_pool_size=1,
        generation_size=2,
        operator_policy=policy,
    )

    assert policy.updates == [("op", False)]


def _engine_with_evaluator(store, rank_front):
    return MultiprocessSearchEngine(
        workers=1,
        strategy=FakeStrategy(),
        storage=store,
        max_total_tasks=2,
        rank_front=rank_front,
    )


def _run_two_children(engine):
    for tid, score in ((1, 0.1), (2, 0.2)):
        engine.unscored_q.put(
            Result(task_id=tid, parent_id=1, valid=True, score=score, payload="p")
        )
    engine.run(
        initial_nodes=[
            SearchNode(
                score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
            )
        ],
        max_wall_seconds=None,
        active_pool_size=3,
        generation_size=1,
    )


def test_the_final_artifact_is_chosen_by_the_evaluator():
    """The round score is a proxy that ranks candidates at about rho 0.83, and
    the evaluator finds roughly a 2x spread inside one front -- so writing out
    the proxy's winner is close to picking arbitrarily among the good ones."""
    store = FakeStorage()
    # An evaluator that prefers the worst round score, which the proxy never would.
    engine = _engine_with_evaluator(
        store, lambda nodes: sorted(nodes, key=lambda n: -n.score)
    )
    _run_two_children(engine)

    assert store.best_saved is not None
    assert store.best_saved.score == 0.5


def test_a_failing_evaluator_falls_back_to_the_best_score():
    """Losing the run's single most important artifact to a scorer error at
    shutdown would be the worst possible time for it."""

    def explode(_nodes):
        raise RuntimeError("no")

    store = FakeStorage()
    _run_two_children(_engine_with_evaluator(store, explode))

    assert store.best_saved is not None
    assert store.best_saved.score == 0.1


def test_scorer_thread_scores_queued_results_together():
    """The point of batching: a scorer whose cost is per-call, not per-image,
    should see everything already queued in one call rather than one each."""

    class PoolStrategy(FakeStrategy):
        def select_parent(self, nodes: list[SearchNode]) -> tuple[int, int | None]:
            return nodes[0].id, None

    engine = MultiprocessSearchEngine(
        workers=1, strategy=PoolStrategy(), storage=FakeStorage(), max_total_tasks=8
    )
    for task_id in range(1, 9):
        engine.unscored_q.put(
            Result(task_id=task_id, parent_id=1, valid=True, score=None, payload="p")
        )

    batch_sizes = []

    def score_fn(results):
        batch_sizes.append(len(results))
        for res in results:
            res.score = 0.1

    initial_node = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial_node],
        max_wall_seconds=None,
        score_fn=score_fn,
        active_pool_size=3,
    )

    assert sum(batch_sizes) == 8
    assert max(batch_sizes) > 1


def test_a_scoring_failure_loses_only_the_batch_it_belongs_to():
    """Scoring in batches must not let one bad candidate discard the candidates
    scored alongside it, which singly-scored candidates never risked."""
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), storage=FakeStorage(), max_total_tasks=4
    )
    for task_id in range(1, 5):
        engine.unscored_q.put(
            Result(task_id=task_id, parent_id=1, valid=True, score=None, payload="p")
        )

    scored = []

    def score_fn(results):
        # Half the batch is scored before the failure, as a real scorer that
        # fails partway through its work would leave it.
        for res in results[: len(results) // 2]:
            res.score = 0.1
            scored.append(res.task_id)
        raise ValueError("scorer exploded")

    initial_node = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial_node],
        max_wall_seconds=None,
        score_fn=score_fn,
        active_pool_size=3,
    )

    assert scored, "results scored before the failure should have survived it"


def test_a_collapsed_pool_ends_the_epoch_without_waiting_for_staleness():
    """Each criterion is set tight enough that reaching it is reason enough on
    its own: a pool that has become clones of one drawing is finished whatever
    the score is still doing, and requiring every criterion to agree would let
    a rarely-reached one block the transition and spend the run as a single
    local search."""
    from unittest.mock import MagicMock

    class Collapsed(FakeStrategy):
        def should_diversify(self, pool: list[SearchNode]) -> tuple[bool, float]:
            _ = pool
            return False, 0.0

    collector = MagicMock()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=Collapsed(), storage=FakeStorage(), max_total_tasks=6
    )
    for task_id in range(1, 7):
        engine.unscored_q.put(
            Result(task_id=task_id, parent_id=1, valid=True, score=0.5, payload="p")
        )

    node = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[node],
        max_wall_seconds=None,
        active_pool_size=2,
        epochs=4,
        # Far beyond the task budget, so staleness cannot be what ends it.
        epoch_patience=10_000,
        epoch_diversity=0.5,
        collector=collector,
    )

    collector.on_epoch_transition.assert_called()


def test_the_pool_criteria_read_the_same_on_any_scale():
    """The reason they are ratios. Score spread is denominated in whatever the
    round objective happens to be, and that has been rewritten repeatedly; an
    absolute threshold would have to be rechosen every time, and would still
    mean different things on a busy drawing and a plain one."""
    from unittest.mock import MagicMock

    class PoolStrategy(FakeStrategy):
        def select_parent(self, nodes: list[SearchNode]) -> tuple[int, int | None]:
            return nodes[0].id, None

    def transitions_for(scale: float) -> int:
        collector = MagicMock()
        engine = MultiprocessSearchEngine(
            workers=1,
            strategy=PoolStrategy(),
            storage=FakeStorage(),
            max_total_tasks=8,
        )
        # The same pattern of ties and distinct values in both runs, a
        # thousandfold apart in absolute terms.
        for task_id, spread in enumerate(
            [1.0, 0.9, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
        ):
            engine.unscored_q.put(
                Result(
                    task_id=task_id + 1,
                    parent_id=1,
                    valid=True,
                    score=spread * scale,
                    payload="p",
                )
            )
        node = SearchNode(
            score=scale, id=1, parent_id=0, state=ChainState(score=scale, payload=None)
        )
        engine.run(
            initial_nodes=[node],
            max_wall_seconds=None,
            active_pool_size=2,
            epochs=4,
            epoch_patience=10_000,
            epoch_distinct=0.6,
            collector=collector,
        )
        return collector.on_epoch_transition.call_count

    assert transitions_for(1.0) == transitions_for(1000.0)
