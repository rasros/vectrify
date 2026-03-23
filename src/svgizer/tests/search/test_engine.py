from svgizer.search import ChainState, Result, SearchNode
from svgizer.search.engine import MultiprocessSearchEngine


class FakeStrategy:
    def select_parent(self, nodes, progress):
        return 1, None

    def create_new_state(self, parent_state, result):
        return ChainState(
            score=result.score,
            model_temperature=0.6,
            stale_hits=0,
            payload="new_fake_payload"
        )


class FakeStorage:
    write_lineage_enabled = False

    def __init__(self):
        self.save_called = False

    def save_node(self, node):
        self.save_called = True
        return "fake_path.txt"


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

    monkeypatch.setattr(engine.result_q, "get", lambda timeout=None: res)
    monkeypatch.setattr(engine.task_q, "put", lambda obj: None)

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

    assert best.score == 0.1
    assert store.save_called is True