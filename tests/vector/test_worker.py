import queue

import pytest

from tests.helpers import make_png as _make_png
from vectrify.search import ChainState, Result, Task
from vectrify.vector import worker as worker_module
from vectrify.vector.payloads import VectorStatePayload
from vectrify.vector.worker import WorkerContext, worker_loop


class FakeOperators:
    """Stands in for the SVG functions the worker calls, and records calls."""

    def __init__(self, png: bytes, *, no_change: bool = False):
        self.png = png
        self.crossover_calls = 0
        self.mutate_ops: list[str | None] = []
        # When set, the operators hand back exactly what they were given,
        # standing in for an operator that found nothing it could change.
        self.no_change = no_change
        self.valid = True

    def _edited(self, content: str) -> str:
        return content if self.no_change else f"{content}<!--edited-->"

    def install(self, monkeypatch) -> None:
        def crossover(a, _b, _scope=None):
            self.crossover_calls += 1
            return self._edited(a), "crossover"

        def mutate(content, op=None, _targets=None, _scope=None):
            self.mutate_ops.append(op)
            return self._edited(content), "mutation"

        patches = {
            "apply_crossover": crossover,
            "apply_mutation": mutate,
            "element_targets": lambda _content, _png: {},
            "is_valid_svg": lambda _content: (self.valid, None if self.valid else "x"),
            "rasterize_svg_to_png_bytes": lambda _content, **_size: self.png,
        }
        for name, value in patches.items():
            monkeypatch.setattr(worker_module, name, value)


def _run_one(task: Task, operators: FakeOperators, monkeypatch) -> Result:
    operators.install(monkeypatch)
    task_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()
    task_q.put(task)
    task_q.put(None)
    ctx = WorkerContext(
        original_png_bytes=operators.png,
        original_w=32,
        original_h=32,
        log_level="ERROR",
    )
    worker_loop(task_q, result_q, ctx)
    return result_q.get_nowait()


@pytest.fixture
def parent_state():
    return ChainState(
        payload=VectorStatePayload(content="<svg><rect/></svg>", origin=None)
    )


def test_a_second_parent_means_crossover(parent_state, monkeypatch):
    operators = FakeOperators(_make_png())
    task = Task(
        task_id=1,
        parent_id=1,
        parent_state=parent_state,
        secondary_parent_id=2,
        secondary_parent_state=parent_state,
    )
    result = _run_one(task, operators, monkeypatch)

    assert result.valid
    assert operators.crossover_calls == 1
    assert result.payload.raster_png == operators.png
    assert result.operator == "crossover"


def test_one_parent_means_the_chosen_mutation(parent_state, monkeypatch):
    operators = FakeOperators(_make_png())
    task = Task(
        task_id=1, parent_id=1, parent_state=parent_state, operator="Mutation: x"
    )
    result = _run_one(task, operators, monkeypatch)

    assert result.valid
    assert operators.mutate_ops == ["Mutation: x"]
    assert result.payload.content.endswith("<!--edited-->")


def test_unchanged_candidate_is_rejected_and_charged_to_its_operator(
    parent_state, monkeypatch
):
    """An operator that finds nothing to change hands the parent straight back.
    Scoring that clone costs a full task and admits it wherever the parent
    already sits, so the policy reads a failed draw as a success."""
    task = Task(task_id=1, parent_id=1, parent_state=parent_state)
    result = _run_one(task, FakeOperators(_make_png(), no_change=True), monkeypatch)

    assert result.valid is False
    assert result.payload.content is None
    # The name that actually ran, so the policy charges the right arm.
    assert result.operator == "mutation"


def test_ordinary_failure_does_not_name_an_operator(parent_state, monkeypatch):
    """Only a blank draw is charged. A candidate that fails to validate is a
    different event and the operator that ran is not reliably known there."""
    operators = FakeOperators(_make_png())
    operators.valid = False
    result = _run_one(
        Task(task_id=1, parent_id=1, parent_state=parent_state), operators, monkeypatch
    )

    assert result.valid is False
    assert result.operator is None
