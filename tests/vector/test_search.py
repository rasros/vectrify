"""The search climbs from its best seed, with a stop event and progress."""

import threading

from PIL import Image

from vectrify.image_utils import png_bytes, rasterize_svg
from vectrify.score.simple import SimpleFallbackScorer
from vectrify.vector.search import SearchSettings, run_search
from vectrify.vector.worker import WorkerContext

SEED = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48">'
    '<rect width="48" height="48" fill="#ffffff"/>'
    '<rect x="14" y="10" width="20" height="24" fill="#3355aa"/>'
    '<circle cx="30" cy="30" r="8" fill="#aa3322"/></svg>'
)
WORSE = SEED.replace("#3355aa", "#00ff00")


def target():
    image = Image.new("RGB", (48, 48), "white")
    image.paste((40, 80, 180), (10, 10, 30, 30))
    image.paste((180, 40, 30), (24, 24, 40, 40))
    return image


def setup():
    scorer = SimpleFallbackScorer()
    reference = scorer.prepare_reference(target())

    def score(png: bytes) -> float:
        return scorer.score(reference, png)

    context = WorkerContext(
        original_png_bytes=png_bytes(target()), original_w=48, original_h=48
    )
    return score, context


def test_search_starts_from_the_best_seed_and_never_ends_worse():
    score, context = setup()
    seen = []
    outcome = run_search(
        [WORSE, SEED],
        score,
        context,
        SearchSettings(max_total_tasks=30, random_seed=7),
        progress=seen.append,
    )
    assert outcome.start.content == SEED
    assert outcome.start.score == score(rasterize_svg(SEED, 48, 48))
    assert outcome.tasks_completed == 30
    assert outcome.best.score <= outcome.start.score
    assert outcome.ranked[0] == outcome.best
    scores = [c.score for c in outcome.ranked]
    assert scores == sorted(scores)
    assert len({c.content for c in outcome.ranked}) == len(outcome.ranked)
    assert seen[-1].tasks_completed == outcome.tasks_completed


def test_one_worker_with_a_seed_repeats_exactly():
    score, context = setup()
    settings = SearchSettings(max_total_tasks=20, random_seed=3)
    first = run_search([SEED], score, context, settings)
    second = run_search([SEED], score, context, settings)
    assert first.best == second.best
    assert first.accepted == second.accepted


def test_stop_event_ends_the_search_early():
    score, context = setup()
    stop = threading.Event()

    def progress(state):
        if state.tasks_completed >= 5:
            stop.set()

    outcome = run_search(
        [SEED],
        score,
        context,
        SearchSettings(max_total_tasks=10_000),
        stop=stop,
        progress=progress,
    )
    assert 5 <= outcome.tasks_completed < 100
