"""The node search climbs, spends its removal budget, stops, and repeats."""

import threading

from PIL import Image, ImageDraw

from vectrify.document import import_svg
from vectrify.score.simple import SimpleFallbackScorer
from vectrify.vector.nodes import Paths, frozen
from vectrify.vector.search import SearchSettings, run_search
from vectrify.vector.worker import WorkerContext

# A square drawn with far more points than it needs, a little off target.
SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" '
    'viewBox="0 0 64 64"><rect width="64" height="64" fill="#ffffff"/>'
    '<path id="p" fill="#000000" d="M14 14 L24 14 L34 14 L44 14 L50 14 '
    'L50 32 L50 50 L32 50 L14 50 L14 32 Z"/></svg>'
)


def setup():
    target = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(target).rectangle((16, 16, 47, 47), fill="black")
    scorer = SimpleFallbackScorer()
    reference = scorer.prepare_reference(target)
    document = import_svg(SVG)
    paths = Paths({"p": document.geometry_for("p")}, {})
    context = WorkerContext(SVG, (64, 64), frozen(document, paths))
    return paths, (lambda png: scorer.score(reference, png)), context


def test_the_climb_never_ends_worse_than_it_started():
    paths, score, context = setup()
    seen = []
    outcome = run_search(
        paths,
        score,
        context,
        SearchSettings(max_total_tasks=40, random_seed=7),
        progress=seen.append,
    )
    assert outcome.tasks_completed == 40
    assert outcome.best.score <= outcome.start.score
    assert seen[-1].tasks_completed == outcome.tasks_completed


def test_simplify_removes_points_within_the_tolerance_of_the_start():
    paths, score, context = setup()
    tolerance = 0.02
    outcome = run_search(
        paths,
        score,
        context,
        SearchSettings(
            moves=(),
            simplify=True,
            tolerance=tolerance,
            max_total_tasks=60,
            random_seed=3,
        ),
    )
    assert outcome.best.state.nodes() < paths.nodes()
    assert outcome.best.score <= outcome.start.score + tolerance


def test_one_worker_with_a_seed_repeats_exactly():
    paths, score, context = setup()
    settings = SearchSettings(
        moves=("shape", "detail"), simplify=True, max_total_tasks=30, random_seed=5
    )
    first = run_search(paths, score, context, settings)
    second = run_search(paths, score, context, settings)
    assert first.best.state.key() == second.best.state.key()
    assert first.accepted == second.accepted


def test_stop_event_ends_the_search_early():
    paths, score, context = setup()
    stop = threading.Event()

    def progress(state):
        if state.tasks_completed >= 5:
            stop.set()

    outcome = run_search(
        paths,
        score,
        context,
        SearchSettings(max_total_tasks=10_000),
        stop=stop,
        progress=progress,
    )
    assert 5 <= outcome.tasks_completed < 100
