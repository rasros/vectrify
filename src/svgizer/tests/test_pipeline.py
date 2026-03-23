import pytest

from svgizer.diff import ScorerType
from svgizer.pipeline import run_svg_search
from svgizer.search import StrategyType


class FakeStorage:
    def __init__(self):
        self.initialized = False
        self._max_id = 0

    @property
    def max_node_id(self) -> int:
        return self._max_id

    def initialize(self):
        self.initialized = True

    def load_resume_nodes(self):
        return []

    def save_node(self, node):
        self._max_id = max(self._max_id, node.id)

    def save_final_svg(self, _content):
        pass


class FakeEngine:
    started_workers = False
    ran = False

    def __init__(self, *args, **kwargs):
        pass

    def start_workers(self, _target, _params):
        FakeEngine.started_workers = True

    def run(self, *_args, **_kwargs):
        FakeEngine.ran = True
        return


def test_run_svg_search_flow(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    # Mock image opening and scorer factory
    monkeypatch.setattr(
        "svgizer.pipeline.Image.open",
        lambda _path: pytest.importorskip("PIL.Image").new("RGB", (10, 10)),
    )
    monkeypatch.setattr(
        "svgizer.pipeline.get_scorer",
        lambda _scorer_type: type(
            "Scorer", (), {"prepare_reference": lambda _s, _i: None}
        )(),
    )
    monkeypatch.setattr("os.path.isfile", lambda _path: True)
    monkeypatch.setattr("svgizer.pipeline.MultiprocessSearchEngine", FakeEngine)

    store = FakeStorage()

    run_svg_search(
        image_path="test.png",
        storage=store,
        seed_svg_path=None,
        max_accepts=1,
        workers=1,
        base_model_temperature=0.6,
        openai_image_long_side=512,
        max_wall_seconds=None,
        log_level="INFO",
        scorer_type=ScorerType.SIMPLE,
        strategy_type=StrategyType.GREEDY,
        goal=None,
    )

    assert store.initialized is True
    assert FakeEngine.started_workers is True
    assert FakeEngine.ran is True
