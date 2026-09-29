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
    assert mutant.image is not None


def test_a_move_with_nothing_to_change_is_no_candidate(monkeypatch):
    context, paths = context_and_paths()
    install(monkeypatch, context)
    mutant = mutate(paths, "strokes", seed=1)
    assert mutant.state is None
    assert mutant.move == "strokes"


def test_rendering_in_slices_matches_rendering_the_whole_drawing():
    import random

    import numpy as np

    from vectrify.vector import nodes

    squares = [
        f'<rect x="{2 + 6 * (i % 10)}" y="{2 + 6 * (i // 10)}" width="5" height="5" '
        f'fill="#{(i * 37) % 256:02x}40{(255 - i * 7) % 256:02x}"/>'
        for i in range(40)
    ]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" '
        f'viewBox="0 0 64 64">{"".join(squares[:20])}'
        '<path id="p" fill="#000000" d="M14 14 L50 14 L50 50 L14 50 Z"/>'
        f"{''.join(squares[20:])}</svg>"
    )
    document = import_svg(svg)
    paths = Paths({"p": document.geometry_for("p")}, {})
    context = WorkerContext(svg, (64, 64), frozen(document, paths))
    sliced = Renderer(context)
    assert sliced.layers is not None
    whole = Renderer(context)
    whole.layers = None
    moved = nodes.nudge(paths, random.Random(1), context.fixed)
    assert moved is not None
    difference = np.abs(sliced(moved).astype(int) - whole(moved).astype(int))
    assert difference.max() <= 2
