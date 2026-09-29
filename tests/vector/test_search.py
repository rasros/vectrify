"""The search runs in-process from a Reference, with a stop event and progress."""

import threading

from PIL import Image

from tests.helpers import rasterize
from vectrify.vector.reference import Reference
from vectrify.vector.search import SearchSettings, run_search, seed_node
from vectrify.vector.worker import WorkerContext

SEED = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48">'
    '<rect width="48" height="48" fill="#ffffff"/>'
    '<rect x="14" y="10" width="20" height="24" fill="#3355aa"/>'
    '<circle cx="30" cy="30" r="8" fill="#aa3322"/></svg>'
)


def target():
    image = Image.new("RGB", (48, 48), "white")
    image.paste((40, 80, 180), (10, 10, 30, 30))
    image.paste((180, 40, 30), (24, 24, 40, 40))
    return image


def setup():
    reference = Reference.build(target(), score_resolution=48, segment_count=2)
    context = WorkerContext(
        original_png_bytes=reference.png,
        original_w=reference.width,
        original_h=reference.height,
        log_level="ERROR",
        random_seed=7,
    )
    seed = seed_node(
        reference,
        SEED,
        rasterize(SEED, 48, 48),
        node_id=1,
        origin="test seed",
    )
    return reference, context, seed


def test_seed_node_is_measured_on_every_objective():
    reference, _context, seed = setup()
    assert seed.valid
    assert {"edge", "colour", "shape", "detail"} <= set(seed.metrics)
    assert len(seed.metrics) == 4 + len(reference.segments)


def test_search_runs_local_operators_and_returns_its_pool():
    reference, context, seed = setup()
    seen = []
    outcome = run_search(
        reference,
        [seed],
        context,
        SearchSettings(pool_size=4, max_total_tasks=30, epochs=1),
        progress=seen.append,
    )
    assert outcome.tasks_completed >= 30
    assert outcome.pool
    assert all(node.valid for node in outcome.pool)
    assert outcome.best is not None
    assert seen[-1].tasks_completed == outcome.tasks_completed


def test_stop_event_ends_the_search_early():
    reference, context, seed = setup()
    stop = threading.Event()

    def progress(state):
        if state.tasks_completed >= 5:
            stop.set()

    outcome = run_search(
        reference,
        [seed],
        context,
        SearchSettings(pool_size=4, max_total_tasks=10_000, epochs=1),
        stop=stop,
        progress=progress,
    )
    assert 5 <= outcome.tasks_completed < 100
