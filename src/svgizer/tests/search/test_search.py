import pytest

from svgizer.diff import ScorerType
from svgizer.search import StrategyType, run_search


class FakeStorage:
    write_lineage_enabled = False

    def __init__(self):
        self.initialized = False

    def initialize(self):
        self.initialized = True

    def load_resume_nodes(self, *args, **kwargs):
        return ([], None, 0)


class FakeEngine:
    started_workers = False
    ran = False

    def __init__(self, *args, **kwargs):
        pass

    def start_workers(self, params):
        FakeEngine.started_workers = True

    def run(self, *args, **kwargs):
        FakeEngine.ran = True
        return


class FakeImage:
    @property
    def size(self):
        return (100, 100)

    def convert(self, mode):
        return self

    def save(self, buf, format):
        buf.write(b"fake_png_bytes")


class FakeScorer:
    def prepare_reference(self, img):
        return None

    def score(self, ref, png):
        return 0.5


def test_run_search_fails_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        run_search(
            "any.png",
            FakeStorage(),
            None,
            1,
            1,
            0.6,
            512,
            None,
            "INFO",
            ScorerType.SIMPLE,
            StrategyType.GREEDY,
        )

    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_run_search_flow(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    # Swap out dependencies with our explicit fake classes
    monkeypatch.setattr("svgizer.search.search.Image.open", lambda path: FakeImage())
    monkeypatch.setattr(
        "svgizer.search.search.get_scorer", lambda scorer_type: FakeScorer()
    )
    monkeypatch.setattr("os.path.isfile", lambda path: True)

    # Swap out the Engine class
    monkeypatch.setattr("svgizer.search.search.MultiprocessSearchEngine", FakeEngine)

    store = FakeStorage()

    run_search(
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
        strategy=StrategyType.GREEDY,
    )

    assert store.initialized is True
    assert FakeEngine.started_workers is True
    assert FakeEngine.ran is True
