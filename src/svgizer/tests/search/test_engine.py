from svgizer.models import ChainState, Result, SearchNode
from svgizer.search.engine import MultiprocessSearchEngine


class FakeStrategy:
    def select_parent(self, nodes, progress):
        return 1, None

    def create_new_state(self, parent_state, result):
        return ChainState(None, None, None, 0.1, 0.6, 0, None)


class FakeStorage:
    write_lineage_enabled = False

    def __init__(self):
        self.save_called = False

    def save_node(self, node):
        self.save_called = True
        return "fake_path.svg"


def test_engine_init():
    engine = MultiprocessSearchEngine(2, FakeStrategy(), FakeStorage(), "simple")
    assert engine.workers == 2


def test_engine_run_loop_terminates_on_max_accepts(monkeypatch):
    strat = FakeStrategy()
    store = FakeStorage()

    res = Result(
        task_id=1,
        parent_id=1,
        worker_slot=0,
        svg="<svg/>",
        valid=True,
        invalid_msg=None,
        raster_png=b"fake",
        score=0.1,
        used_temperature=0.6,
        change_summary="better",
    )

    engine = MultiprocessSearchEngine(1, strat, store, "simple")

    # 1. Bypass multiprocessing queues
    monkeypatch.setattr(engine.result_q, "get", lambda timeout=None: res)
    monkeypatch.setattr(engine.task_q, "put", lambda obj: None)

    # 2. Bypass PIL trying to parse the b"fake" bytes
    monkeypatch.setattr(
        "svgizer.search.engine.make_preview_data_url",
        lambda png, side: "data:image/png;base64,fake",
    )
    monkeypatch.setattr(
        "svgizer.search.engine.png_bytes_to_data_url",
        lambda png: "data:image/png;base64,fake",
    )

    initial_node = SearchNode(
        score=0.8,
        id=1,
        parent_id=0,
        state=ChainState(None, None, None, 0.8, 0.6, 0, None),
    )

    best = engine.run(
        initial_nodes=[initial_node],
        max_accepts=1,
        max_wall_seconds=None,
        openai_image_long_side=512,
        original_dims=(100, 100),
    )

    assert best.score == 0.1
    assert store.save_called is True
