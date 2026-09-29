from tests.vector.test_search import SVG
from vectrify.document import import_svg
from vectrify.vector import worker as worker_module
from vectrify.vector.nodes import Paths, frozen
from vectrify.vector.worker import Renderer, WorkerContext, mutate


def context_and_paths():
    document = import_svg(SVG)
    paths = Paths({"p": document.geometry_for("p")}, {})
    return WorkerContext(SVG, (64, 64), frozen(document, paths)), paths


def install(monkeypatch, context):
    # Set directly: init_worker also makes the process ignore Ctrl-C.
    monkeypatch.setattr(worker_module, "_context", context)
    monkeypatch.setattr(worker_module, "_render", Renderer(context))


def test_a_task_applies_its_move_and_renders_the_child(monkeypatch):
    context, paths = context_and_paths()
    install(monkeypatch, context)
    mutant = mutate(paths, "simplify", seed=1)
    assert mutant.move == "simplify"
    assert mutant.state is not None
    assert mutant.state.nodes() == paths.nodes() - 1
    assert mutant.png is not None


def test_a_move_with_nothing_to_change_is_no_candidate(monkeypatch):
    context, paths = context_and_paths()
    install(monkeypatch, context)
    mutant = mutate(paths, "strokes", seed=1)
    assert mutant.state is None
    assert mutant.move == "strokes"
