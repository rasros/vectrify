import time

import pytest

from vectrify.search import ChainState, Result, SearchNode
from vectrify.search.engine import MultiprocessSearchEngine


class FakeStrategy:
    @property
    def top_k_count(self) -> int:
        return 1

    def select_parent(
        self,
        nodes: list[SearchNode],
        progress: float,
    ) -> tuple[int, int | None]:
        _ = (nodes, progress)
        return 1, None

    def create_new_state(
        self,
        result: Result,
    ) -> ChainState:
        return ChainState(
            score=result.score,
            payload="new_fake_payload",
        )

    def should_diversify(self, pool: list[SearchNode]) -> tuple[bool, float]:
        _ = pool
        return False, 1.0

    def epoch_seeds(self, pool: list[SearchNode], max_seeds: int) -> list[SearchNode]:
        return pool[:max_seeds]


class FakeStorage:
    def __init__(self):
        self.save_called = False
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

    def record_eviction(self, node_id: int, tasks_completed: int) -> None:
        _ = node_id, tasks_completed


def test_engine_init():
    engine = MultiprocessSearchEngine(2, FakeStrategy(), FakeStorage())
    assert engine.workers == 2


def test_engine_run_loop_terminates_on_max_accepts():
    strat = FakeStrategy()
    store = FakeStorage()

    engine = MultiprocessSearchEngine(workers=1, strategy=strat, storage=store)

    res = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
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
        max_accepts=1,
        max_wall_seconds=None,
    )

    assert store.save_called is True


def test_engine_respects_max_wall_seconds(monkeypatch):
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), storage=FakeStorage()
    )

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

    engine.run(initial_nodes=[dummy_node], max_accepts=10, max_wall_seconds=50.0)
    assert True


def test_engine_epoch_patience_triggers_transition():

    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.epoch_seeds_calls = 0

        def epoch_seeds(self, pool, max_seeds):
            self.epoch_seeds_calls += 1
            return pool[:max_seeds]

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
                worker_slot=0,
                valid=True,
                score=score,
                payload="p",
            )
        )

    initial_node = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial_node],
        max_accepts=100,
        max_wall_seconds=None,
        epoch_patience=3,
        epoch_min_delta=0.1,
    )
    assert strat.epoch_seeds_calls >= 1
    assert store.save_called


def test_engine_epoch_patience_resets_on_improvement():
    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.epoch_seeds_calls = 0

        def epoch_seeds(self, pool, max_seeds):
            self.epoch_seeds_calls += 1
            return pool[:max_seeds]

    strat = TrackingStrategy()
    store = FakeStorage()
    engine = MultiprocessSearchEngine(
        workers=1, strategy=strat, storage=store, max_total_tasks=3
    )

    for score in (0.1, 0.09, 0.08):
        engine.unscored_q.put(
            Result(
                task_id=1,
                parent_id=1,
                worker_slot=0,
                valid=True,
                score=score,
                payload="p",
            )
        )

    initial_node = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    engine.run(
        initial_nodes=[initial_node],
        max_accepts=100,
        max_wall_seconds=None,
        epoch_patience=2,
        epoch_min_delta=0.1,
    )
    assert strat.epoch_seeds_calls >= 1
    assert store.save_called


def test_engine_epoch_patience_none_no_transitions():

    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.epoch_seeds_calls = 0

        def epoch_seeds(self, pool, max_seeds):
            self.epoch_seeds_calls += 1
            return pool[:max_seeds]

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
        max_accepts=10,
        max_wall_seconds=None,
        epoch_patience=None,
    )
    assert strat.epoch_seeds_calls == 0


def test_engine_respects_max_total_tasks():
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), storage=FakeStorage(), max_total_tasks=0
    )
    dummy_node = SearchNode(
        score=0.5,
        id=1,
        parent_id=0,
        state=ChainState(score=0.5, payload=None),
    )

    engine.run(initial_nodes=[dummy_node], max_accepts=10, max_wall_seconds=None)
    assert True


def test_engine_active_pool_bounded():
    class TrackingStrategy(FakeStrategy):
        def __init__(self):
            self.max_seen = 0

        def select_parent(
            self, nodes: list[SearchNode], progress: float
        ) -> tuple[int, int | None]:
            _ = progress
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
                worker_slot=0,
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
        max_accepts=100,
        max_wall_seconds=None,
        active_pool_size=3,
    )
    assert strat.max_seen <= 4


def test_engine_init_error_raises():
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), storage=FakeStorage()
    )
    engine.result_q.put({"init_error": "missing API key"})

    initial_node = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )

    with pytest.raises(RuntimeError, match="Worker initialization failed"):
        engine.run(initial_nodes=[initial_node], max_accepts=10, max_wall_seconds=None)


def test_engine_score_fn_none_with_unscored_result_raises():
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), storage=FakeStorage()
    )
    engine.unscored_q.put(
        Result(
            task_id=1,
            parent_id=1,
            worker_slot=0,
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
            max_accepts=10,
            max_wall_seconds=None,
            score_fn=None,
        )
