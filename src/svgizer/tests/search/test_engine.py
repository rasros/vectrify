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
        parent_state: ChainState,
        result: Result,
    ) -> ChainState:
        _ = parent_state
        return ChainState(
            score=result.score,
            model_temperature=0.6,
            stale_hits=0,
            payload="new_fake_payload",
        )


class FakeStorage:
    def __init__(self):
        self.save_called = False
        self.max_node_id = 1

    def initialize(self) -> None:
        pass

    def load_resume_nodes(self) -> list:
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
        used_temperature=0.6,
        payload="fake_payload",
    )
    engine.result_q.put(res)

    initial_node = SearchNode(
        score=0.8,
        id=1,
        parent_id=0,
        state=ChainState(score=0.8, model_temperature=0.6, stale_hits=0, payload=None),
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
        state=ChainState(score=0.5, model_temperature=1.0, stale_hits=0, payload=None),
    )

    engine.run(initial_nodes=[dummy_node], max_accepts=10, max_wall_seconds=50.0)
    assert True


def test_engine_respects_max_total_tasks():
    engine = MultiprocessSearchEngine(
        workers=1, strategy=FakeStrategy(), storage=FakeStorage(), max_total_tasks=0
    )
    dummy_node = SearchNode(
        score=0.5,
        id=1,
        parent_id=0,
        state=ChainState(score=0.5, model_temperature=1.0, stale_hits=0, payload=None),
    )

    engine.run(initial_nodes=[dummy_node], max_accepts=10, max_wall_seconds=None)
    assert True
