import base64
import io

from PIL import Image

from vectrify.image_utils import png_bytes_to_data_url, resize_long_side
from vectrify.tests.helpers import make_png as _make_png
from vectrify.vector.worker import _should_use_llm, _use_llm


def test_use_llm_no_svg_uses_llm_when_rate_nonzero():
    for _ in range(20):
        assert _use_llm(has_content=False, llm_rate=1.0, llm_pressure=1.0) is True


def test_use_llm_rate_zero_never_calls():
    for _ in range(20):
        assert _use_llm(has_content=False, llm_rate=0.0, llm_pressure=0.0) is False
        assert _use_llm(has_content=True, llm_rate=0.0, llm_pressure=1.0) is False


def test_use_llm_rate_one_always_calls():
    for _ in range(20):
        assert _use_llm(has_content=True, llm_rate=1.0, llm_pressure=1.0) is True


def test_use_llm_intermediate_rate_is_probabilistic():
    results = [
        _use_llm(has_content=True, llm_rate=0.5, llm_pressure=1.0) for _ in range(200)
    ]
    assert any(results), "Expected some True values at rate=0.5"
    assert not all(results), "Expected some False values at rate=0.5"


def test_use_llm_zero_pressure_never_calls_when_has_content():
    for _ in range(20):
        assert _use_llm(has_content=True, llm_rate=1.0, llm_pressure=0.0) is False


def test_use_llm_takes_priority_over_crossover(monkeypatch):
    """LLM must be checked before crossover so pressure actually triggers LLM calls.

    Regression: previously crossover was checked first, so once the pool had
    ≥2 nodes every task went to crossover and llm_pressure was never effective.
    """
    import queue

    from vectrify.formats.models import VectorStatePayload
    from vectrify.search import ChainState, Task
    from vectrify.vector import worker as worker_module
    from vectrify.vector.worker import WorkerContext, worker_loop

    png = _make_png()

    class FakeClient:
        def __init__(self):
            self.generate_calls = 0

        def generate(self, blocks, config):
            _ = (blocks, config)
            self.generate_calls += 1
            return "<svg/>"

    class FakePlugin:
        def __init__(self):
            self.crossover_calls = 0

        def build_generate_prompt(self, *args, **kwargs):
            _ = (args, kwargs)
            return []

        def apply_edit(self, parent, raw):
            _ = parent
            return raw

        def extract_from_llm(self, raw):
            return raw

        def crossover(self, a, b, orig_img_fast):
            _ = (b, orig_img_fast)
            self.crossover_calls += 1
            return a, "crossover"

        def mutate(self, content, orig_img_fast):
            _ = orig_img_fast
            return content, "mutation"

        def validate(self, content):
            _ = content
            return True, None

        def rasterize(self, content, out_w, out_h):
            _ = (content, out_w, out_h)
            return png

    client = FakeClient()
    plugin = FakePlugin()
    monkeypatch.setattr(worker_module, "get_provider", lambda *_a, **_kw: client)
    # Make _use_llm deterministic: random() < llm_rate * llm_pressure always true.
    monkeypatch.setattr(worker_module.random, "random", lambda: 0.0)

    parent_state = ChainState(
        score=0.5,
        payload=VectorStatePayload(
            content="<svg><rect/></svg>",
            raster_data_url=None,
            raster_preview_data_url=None,
            origin=None,
        ),
    )
    # Both dispatch branches are eligible: LLM pressure is maxed AND a
    # secondary parent with content makes crossover possible.
    task = Task(
        task_id=1,
        parent_id=1,
        parent_state=parent_state,
        secondary_parent_id=2,
        secondary_parent_state=parent_state,
        llm_pressure=1.0,
    )

    task_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()
    task_q.put(task)
    task_q.put(None)

    ctx = WorkerContext(
        format_plugin=plugin,
        image_data_url=png_bytes_to_data_url(png),
        original_png_bytes=png,
        original_w=32,
        original_h=32,
        image_long_side=16,
        log_level="ERROR",
        log_file=None,
        goal=None,
        llm_provider="openai",
        llm_model="test-model",
        reasoning="",
        api_key=None,
        llm_rate=1.0,
    )
    worker_loop(task_q, result_q, ctx)

    result = result_q.get_nowait()
    assert result.llm_type == "llm-generate"
    assert client.generate_calls == 1
    assert plugin.crossover_calls == 0


def _compute_preview(png: bytes, long_side: int) -> str:
    full_img = Image.open(io.BytesIO(png)).convert("RGB")
    preview_img = resize_long_side(full_img, long_side)
    buf = io.BytesIO()
    preview_img.save(buf, format="PNG")
    return png_bytes_to_data_url(buf.getvalue())


def test_worker_preview_downscales_image():
    png = _make_png(size=256)
    preview = _compute_preview(png, long_side=64)

    _, b64 = preview.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert max(img.size) <= 64


def test_worker_preview_preserves_small_image():
    png = _make_png(size=32)
    preview = _compute_preview(png, long_side=128)

    _, b64 = preview.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert img.size == (32, 32)


def test_should_use_llm_rate_zero_overrides_force_llm():
    """Regression: epoch-0 seed tasks set force_llm, which bypassed the rate
    entirely, so --llm-rate 0 still issued LLM calls and offline runs were
    impossible.
    """
    assert (
        _should_use_llm(
            force_llm=True, has_content=False, llm_rate=0.0, llm_pressure=1.0
        )
        is False
    )
    assert (
        _should_use_llm(
            force_llm=True, has_content=True, llm_rate=0.0, llm_pressure=1.0
        )
        is False
    )


def test_should_use_llm_honours_force_llm_when_enabled():
    assert (
        _should_use_llm(
            force_llm=True, has_content=True, llm_rate=0.01, llm_pressure=0.0
        )
        is True
    )


def test_should_use_llm_without_content_needs_a_nonzero_rate():
    assert (
        _should_use_llm(
            force_llm=False, has_content=False, llm_rate=1.0, llm_pressure=1.0
        )
        is True
    )
    assert (
        _should_use_llm(
            force_llm=False, has_content=False, llm_rate=0.0, llm_pressure=1.0
        )
        is False
    )
