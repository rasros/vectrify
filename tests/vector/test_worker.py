import base64
import io
import queue

import pytest
from PIL import Image

from tests.helpers import make_png as _make_png
from vectrify.formats.models import VectorStatePayload
from vectrify.image_utils import png_bytes_to_data_url, resize_long_side
from vectrify.search import ChainState, Result, Task
from vectrify.vector import worker as worker_module
from vectrify.vector.worker import WorkerContext, worker_loop


class FakeClient:
    def __init__(self):
        self.generate_calls = 0

    def generate(self, blocks, config):
        _ = (blocks, config)
        self.generate_calls += 1
        return "<svg/>"


class FakePlugin:
    def __init__(self, png: bytes, *, no_change: bool = False):
        self.png = png
        self.crossover_calls = 0
        self.mutate_ops = []
        self.mutate_calls = 0
        # When set, the local operators hand back exactly what they were given,
        # standing in for an operator that found nothing it could change.
        self.no_change = no_change

    def _edited(self, content: str) -> str:
        return content if self.no_change else f"{content}<!--edited-->"

    def build_generate_prompt(self, *args, **kwargs):
        _ = (args, kwargs)
        return []

    def apply_edit(self, parent, raw):
        _ = parent
        return raw

    def extract_from_llm(self, raw):
        return raw

    def crossover(self, a, b):
        _ = b
        self.crossover_calls += 1
        return self._edited(a), "crossover"

    def element_targets(self, content, reference_png):
        _ = content, reference_png
        return {}

    def mutate(self, content, operator=None, targets=None):
        _ = targets
        self.mutate_ops.append(operator)
        self.mutate_calls += 1
        return self._edited(content), "mutation"

    def validate(self, content):
        _ = content
        return True, None

    def rasterize(self, content, out_w, out_h):
        _ = (content, out_w, out_h)
        return self.png


def _run_one(
    task: Task, monkeypatch, *, no_change: bool = False
) -> tuple[Result, FakeClient, FakePlugin]:
    png = _make_png()
    client, plugin = FakeClient(), FakePlugin(png, no_change=no_change)
    monkeypatch.setattr(worker_module, "get_provider", lambda *_a, **_kw: client)

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
        resolution_llm=16,
        log_level="ERROR",
        log_file=None,
        goal=None,
        llm_provider="openai",
        llm_model="test-model",
        reasoning="",
        api_key=None,
    )
    worker_loop(task_q, result_q, ctx)
    return result_q.get_nowait(), client, plugin


@pytest.fixture
def parent_state():
    return ChainState(
        score=0.5,
        payload=VectorStatePayload(
            content="<svg><rect/></svg>",
            raster_data_url=None,
            raster_preview_data_url=None,
            origin=None,
        ),
    )


def test_force_llm_calls_the_model_even_when_crossover_is_possible(
    parent_state, monkeypatch
):
    task = Task(
        task_id=1,
        parent_id=1,
        parent_state=parent_state,
        secondary_parent_id=2,
        secondary_parent_state=parent_state,
        force_llm=True,
    )
    result, client, plugin = _run_one(task, monkeypatch)

    assert result.llm_type == "llm-generate"
    assert client.generate_calls == 1
    assert plugin.crossover_calls == 0


def test_local_task_never_calls_the_model(parent_state, monkeypatch):
    task = Task(
        task_id=1,
        parent_id=1,
        parent_state=parent_state,
        secondary_parent_id=2,
        secondary_parent_state=parent_state,
        force_llm=False,
    )
    result, client, plugin = _run_one(task, monkeypatch)

    assert result.llm_type is None
    assert client.generate_calls == 0
    assert plugin.crossover_calls == 1


def test_local_task_without_secondary_parent_mutates(parent_state, monkeypatch):
    task = Task(task_id=1, parent_id=1, parent_state=parent_state, force_llm=False)
    result, client, plugin = _run_one(task, monkeypatch)

    assert result.llm_type is None
    assert client.generate_calls == 0
    assert plugin.mutate_calls == 1


def test_unchanged_candidate_is_rejected_and_charged_to_its_operator(
    parent_state, monkeypatch
):
    """An operator that finds nothing to change hands the parent straight back.
    Scoring that clone costs a full task and admits it wherever the parent
    already sits, so the policy reads a failed draw as a success."""
    task = Task(task_id=1, parent_id=1, parent_state=parent_state, force_llm=False)
    result, _client, _plugin = _run_one(task, monkeypatch, no_change=True)

    assert result.valid is False
    assert result.payload.content is None
    # The name that actually ran, so the policy charges the right arm.
    assert result.operator == "mutation"


def test_ordinary_failure_does_not_name_an_operator(parent_state, monkeypatch):
    """Only a blank draw is charged. A candidate that fails to validate is a
    different event and the operator that ran is not reliably known there."""
    png = _make_png()
    client, plugin = FakeClient(), FakePlugin(png)
    plugin.validate = lambda _content: (False, "broken")  # type: ignore[method-assign]
    monkeypatch.setattr(worker_module, "get_provider", lambda *_a, **_kw: client)

    task_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()
    task_q.put(Task(task_id=1, parent_id=1, parent_state=parent_state, force_llm=False))
    task_q.put(None)
    ctx = WorkerContext(
        format_plugin=plugin,
        image_data_url=png_bytes_to_data_url(png),
        original_png_bytes=png,
        original_w=32,
        original_h=32,
        resolution_llm=16,
        log_level="ERROR",
        log_file=None,
        goal=None,
        llm_provider="openai",
        llm_model="test-model",
        reasoning="",
        api_key=None,
    )
    worker_loop(task_q, result_q, ctx)
    result = result_q.get_nowait()

    assert result.valid is False
    assert result.operator is None


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
