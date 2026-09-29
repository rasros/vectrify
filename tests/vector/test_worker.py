import pytest

from tests.helpers import make_png as _make_png
from vectrify.vector import worker as worker_module
from vectrify.vector.worker import WorkerContext, mutate

PARENT = "<svg><rect/></svg>"


class FakeOperators:
    """Stands in for the SVG functions the worker calls, and records calls."""

    def __init__(self, png: bytes, *, no_change: bool = False):
        self.png = png
        self.mutate_ops: list[str | None] = []
        # When set, the operators hand back exactly what they were given,
        # standing in for an operator that found nothing it could change.
        self.no_change = no_change
        self.valid = True

    def install(self, monkeypatch) -> None:
        def apply(content, op=None, _targets=None, _scope=None):
            self.mutate_ops.append(op)
            return (content if self.no_change else f"{content}<!--edited-->"), "applied"

        patches = {
            "apply_mutation": apply,
            "element_targets": lambda _content, _png: {},
            "is_valid_svg": lambda _content: (self.valid, None if self.valid else "x"),
            "rasterize_svg_to_png_bytes": lambda _content, **_size: self.png,
        }
        for name, value in patches.items():
            monkeypatch.setattr(worker_module, name, value)
        # Set directly: init_worker also makes the process ignore Ctrl-C.
        context = WorkerContext(
            original_png_bytes=self.png, original_w=32, original_h=32
        )
        monkeypatch.setattr(worker_module, "_context", context)


@pytest.fixture
def operators(monkeypatch):
    def make(**options):
        fake = FakeOperators(_make_png(), **options)
        fake.install(monkeypatch)
        return fake

    return make


def test_a_task_applies_the_chosen_operator_and_renders_the_child(operators):
    fake = operators()
    mutant = mutate(PARENT, "numeric", seed=1)
    assert fake.mutate_ops == ["numeric"]
    assert mutant.content == f"{PARENT}<!--edited-->"
    assert mutant.png == fake.png
    assert mutant.operator == "applied"


def test_an_unchanged_child_is_no_candidate_but_names_its_operator(operators):
    operators(no_change=True)
    mutant = mutate(PARENT, "numeric", seed=1)
    assert mutant.content is None
    assert mutant.operator == "applied"


def test_an_invalid_child_is_no_candidate(operators):
    fake = operators()
    fake.valid = False
    mutant = mutate(PARENT, "numeric", seed=1)
    assert mutant.content is None
    assert mutant.operator == "numeric"
