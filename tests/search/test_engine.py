import logging
import time

import pytest

from vectrify.search import ChainState, Result, SearchNode
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
        self.best_saved = None
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


def test_engine_low_variance_epoch_end_does_not_crash():
    """Regression: the low-variance branch compared the imported score_std
    *function* to a float, so any positive --epoch-variance raised TypeError on
    the first epoch-end check. Nothing covered this path, which is why it
    shipped.
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
        epoch_variance=0.05,  # the flag that used to crash the run
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
