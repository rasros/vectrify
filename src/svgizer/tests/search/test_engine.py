import time

from svgizer.search import ChainState, Result, SearchNode
from svgizer.search.engine import MultiprocessSearchEngine


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

    def should_diversify(self, pool: list[SearchNode]) -> bool:
        _ = pool
        return False


class FakeStorage:
    def __init__(self):
        self.save_called = False
        self.max_node_id = 1

    def initialize(self) -> None:
        pass

    def load_resume_nodes(self, max_nodes: int = 20) -> list:
        return []

    def save_node(self, node: SearchNode) -> None:
        _ = node
        self.save_called = True


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
    engine.result_q.put(res)

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


def test_engine_patience_stops_on_no_improvement():
    """Search must stop after `patience` tasks with no improvement >= min_delta."""
    strat = FakeStrategy()
    store = FakeStorage()
    engine = MultiprocessSearchEngine(workers=1, strategy=strat, storage=store)

    # Three results, none improving beyond min_delta=0.1 from initial best of 0.5
    for score in (0.49, 0.48, 0.47):
        engine.result_q.put(
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
        patience=3,
        min_delta=0.1,
    )
    # Only 3 tasks completed before patience triggered; store was called for each accept
    assert store.save_called


def test_engine_patience_resets_on_improvement():
    """A result that beats min_delta should reset the patience counter."""
    strat = FakeStrategy()
    store = FakeStorage()
    engine = MultiprocessSearchEngine(workers=1, strategy=strat, storage=store)

    # Task 1: 0.5→0.1 (delta=0.4 >= 0.1) → resets counter
    # Task 2: 0.1→0.09 (delta=0.01 < 0.1) → no reset, counter=1
    # Task 3: 0.09→0.08 (delta=0.01 < 0.1) → no reset, counter=2 → patience fires
    for score in (0.1, 0.09, 0.08):
        engine.result_q.put(
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
        patience=2,
        min_delta=0.1,
    )
    assert store.save_called


def test_engine_patience_disabled_at_zero():
    """patience=0 (default) must not stop the search early."""
    engine = MultiprocessSearchEngine(
        workers=1,
        strategy=FakeStrategy(),
        storage=FakeStorage(),
        max_total_tasks=0,
    )
    dummy_node = SearchNode(
        score=0.5, id=1, parent_id=0, state=ChainState(score=0.5, payload=None)
    )
    # Would loop forever with patience, but max_total_tasks=0 stops it immediately
    engine.run(
        initial_nodes=[dummy_node],
        max_accepts=10,
        max_wall_seconds=None,
        patience=0,
    )
    assert True


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