from svgizer.search import ChainState, Result, SearchNode
from svgizer.search.engine import MultiprocessSearchEngine


class FakeStrategy:
    @property
    def top_k_count(self) -> int:
        return 1

    def select_parent(self, _nodes, _progress):
        return 1, None

    def create_new_state(self, _parent_state, result):
        return ChainState(
            score=result.score,
            model_temperature=0.6,
            stale_hits=0,
            payload="new_fake_payload",
        )


class FakeStorage:
    def __init__(self):
        self.save_called = False

    @property
    def max_node_id(self) -> int:
        return 0

    def initialize(self) -> None:
        pass

    def load_resume_nodes(self) -> list:
        return []

    def save_final_svg(self, _content: str) -> None:
        pass

    def load_seed_svg(self, _path: str) -> str:
        return ""

    def save_node(self, _node):
        self.save_called = True


def test_engine_init():
    engine = MultiprocessSearchEngine(2, FakeStrategy(), FakeStorage())
    assert engine.workers == 2


def test_engine_run_loop_terminates_on_max_accepts(monkeypatch):
    strat = FakeStrategy()
    store = FakeStorage()

    res = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
        valid=True,
        score=0.1,
        used_temperature=0.6,
        payload="fake_payload",
    )

    engine = MultiprocessSearchEngine(1, strat, store)

    # Use **_kwargs to handle the 'timeout' parameter used in engine.py
    monkeypatch.setattr(engine.result_q, "get", lambda **_kwargs: res)
    monkeypatch.setattr(engine.task_q, "put", lambda _obj: None)

    initial_node = SearchNode(
        score=0.8,
        id=1,
        parent_id=0,
        state=ChainState(score=0.8, model_temperature=0.6, stale_hits=0, payload=None),
    )

    best = engine.run(
        initial_nodes=[initial_node],
        max_accepts=1,
        max_wall_seconds=None,
    )

    assert best is not None
    assert best.score == 0.1
    assert store.save_called is True
