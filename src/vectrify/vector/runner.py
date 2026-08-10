import io
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vectrify.dashboard import Dashboard
    from vectrify.formats.base import FormatPlugin
    from vectrify.search.stats import SearchStats

from PIL import Image, UnidentifiedImageError

from vectrify.cli import (
    DEFAULT_EPOCH_DIVERSITY,
    DEFAULT_MAX_TOTAL_TASKS,
    DEFAULT_POOL_SIZE,
    DEFAULT_RESOLUTION_LLM,
    DEFAULT_TOURNAMENT_SIZE,
)
from vectrify.formats.models import VectorStatePayload
from vectrify.image_utils import (
    downscale_png_bytes,
    png_bytes_to_data_url,
    resize_long_side,
)
from vectrify.llm.models import api_key_env
from vectrify.score import ScorerType, get_scorer
from vectrify.score.base import DEFAULT_CONFIG
from vectrify.score.complexity import WORST_REGION
from vectrify.score.regions import DEFAULT_TILE_SIZE, snap_raster, worst_region_score
from vectrify.score.vision import DEFAULT_VISION_MODEL
from vectrify.search import (
    INVALID_SCORE,
    ChainState,
    MultiprocessSearchEngine,
    NsgaStrategy,
    SearchNode,
    StorageAdapter,
)
from vectrify.search.collector import StatCollector
from vectrify.utils import setup_logger, start_log_listener
from vectrify.vector.resume import filter_to_pool_size, resume_nodes
from vectrify.vector.state import VectorStateBuilder
from vectrify.vector.worker import WorkerContext, worker_loop

log = logging.getLogger("main")


def _load_image(image_path: str, long_side: int) -> tuple[Image.Image, bytes, int, int]:
    """Open the reference image and return (img, png_bytes, width, height).

    Downscaled to *long_side*, which makes the raster the single resolution in
    the run: candidates are rendered at this size, scored at this size, and the
    scorer's crop count follows from it. A source image's own dimensions would
    otherwise silently set the cost -- a 2000px input is 100 crops per
    candidate against 9 for a 700px one.

    Raises FileNotFoundError if the path does not exist and ValueError if the
    file exists but is not a decodable image.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"input image could not be read as an image: {image_path} ({exc})"
        ) from exc
    img = resize_long_side(
        img, snap_raster(long_side, DEFAULT_CONFIG.tile_size or DEFAULT_TILE_SIZE)
    )
    w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return img, buf.getvalue(), w, h


def resolve_seeds(seeds: int | None, pool_size: int) -> int:
    """LLM calls that open each epoch; None means derive from the pool size."""
    return pool_size // 10 if seeds is None else max(0, seeds)


def initial_seed_tasks(epoch_seeds: int, initial_nodes: list[SearchNode]) -> int:
    """Epoch 0's batch size, discounted by candidates already carried in.

    A resumed node is a seed that has already been paid for, so a resume that
    restores a full batch should spend nothing on generating another.
    """
    seeded = sum(1 for n in initial_nodes if n.state.payload.content)
    return max(0, epoch_seeds - seeded)


def run_vector_search(
    image_path: str,
    storage: StorageAdapter,
    workers: int,
    resolution: int,
    max_wall_seconds: float | None,
    log_level: str,
    scorer_type: ScorerType,
    goal: str | None,
    llm_provider: str,
    llm_model: str,
    reasoning: str,
    format_plugin: "FormatPlugin",
    resolution_llm: int = DEFAULT_RESOLUTION_LLM,
    write_lineage: bool = True,
    save_raster: bool = False,
    epoch_patience: int | None = None,
    epoch_min_delta: float = 1e-4,
    pool_size: int = DEFAULT_POOL_SIZE,
    seeds: int | None = None,
    epoch_diversity: float = DEFAULT_EPOCH_DIVERSITY,
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
    epoch_variance: float | None = None,
    epochs: int | None = None,
    max_total_tasks: int = DEFAULT_MAX_TOTAL_TASKS,
    vision_model: str = DEFAULT_VISION_MODEL,
    stats: "SearchStats | None" = None,
    dashboard: "Dashboard | None" = None,
) -> None:
    epoch_seeds = resolve_seeds(seeds, pool_size)

    # Validate the reference image up front so a missing or corrupt input fails
    # before storage.initialize() creates the output directory tree.
    original_img, original_png_bytes, original_w, original_h = _load_image(
        image_path, resolution
    )

    storage.initialize()
    assert storage.current_run_dir is not None
    run_log_file = storage.current_run_dir / "search.log"
    # The dashboard owns the terminal, so console logging is suppressed only
    # then; otherwise stderr keeps receiving records alongside the log file.
    setup_logger(log_level, log_file=run_log_file, console=dashboard is None)
    log_queue, log_listener = start_log_listener()

    # Suppress tqdm / HF noise before any library imports or workers spawn.
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    api_key = os.getenv(api_key_env(llm_provider))

    # Load the scoring model in the background so epoch-0 LLM seeding can run
    # concurrently with (potentially slow) HuggingFace model downloads/init.
    _scorer: list[Any] = []
    _scoring_ref: list[Any] = []
    _scorer_error: list[Exception] = []
    scorer_ready = threading.Event()

    def _init_scorer() -> None:
        try:
            s = get_scorer(
                scorer_type,
                vision_model=vision_model,
            )
            ref = s.prepare_reference(original_img)
            _scorer.append(s)
            _scoring_ref.append(ref)
            log.info("Scoring model ready.")
        except Exception as exc:
            _scorer_error.append(exc)
            log.error(f"Scorer initialisation failed: {exc}")
        finally:
            scorer_ready.set()

    def _start_scorer_thread() -> None:
        threading.Thread(target=_init_scorer, daemon=True, name="ScorerInit").start()

    resumed_items = storage.load_resume_nodes()
    if resumed_items:
        # Resume path: scorer needed before engine starts — kick it off now
        # and wait for it just before scoring the resumed nodes.
        _start_scorer_thread()

    initial_nodes: list[SearchNode] = []

    if resumed_items:
        scorer_ready.wait()
        if _scorer_error:
            raise RuntimeError(
                f"Scorer failed to initialise: {_scorer_error[0]}"
            ) from _scorer_error[0]

        initial_nodes = resume_nodes(
            resumed_items=resumed_items,
            format_plugin=format_plugin,
            original_img=original_img,
            original_w=original_w,
            original_h=original_h,
            resolution_llm=resolution_llm,
            pool_size=pool_size,
            workers=workers,
            scorer=_scorer[0],
            scoring_ref=_scoring_ref[0],
            storage=storage,
        )
        initial_nodes = filter_to_pool_size(initial_nodes, pool_size)

    # With the LLM disabled the search can only mutate existing candidates, so
    # without at least one it would dispatch nothing and idle until the wall
    # clock. Fail immediately with the reason instead.
    if epoch_seeds <= 0 and not any(
        n.state.payload.content for n in initial_nodes if n.state.payload
    ):
        raise ValueError(
            "--seeds 0 disables all LLM calls, but there are no existing "
            "candidates to mutate. Resume a previous run with --resume, or "
            "allow LLM calls so the first candidate can be generated."
        )

    if not initial_nodes:
        initial_nodes.append(
            SearchNode(
                score=INVALID_SCORE,
                id=0,
                parent_id=0,
                state=ChainState(
                    INVALID_SCORE,
                    VectorStatePayload(None, None, None, None, None),
                ),
            )
        )

    collector = (
        StatCollector(stats, run_dir=storage.current_run_dir)
        if stats is not None
        else None
    )
    if collector is not None:
        collector.configure_run(
            epoch_diversity=epoch_diversity,
            epoch_variance=epoch_variance or 0.0,
        )
        valid = [n for n in initial_nodes if n.score < INVALID_SCORE]
        if valid:
            collector.seed_initial_score(min(valid, key=lambda n: n.score).score)

    first_batch = initial_seed_tasks(epoch_seeds, initial_nodes)
    if first_batch < epoch_seeds:
        log.info(
            f"Epoch 0: {first_batch} LLM seed task(s) "
            f"(batch={epoch_seeds}, already seeded={epoch_seeds - first_batch})"
        )

    engine = MultiprocessSearchEngine(
        workers=workers,
        strategy=NsgaStrategy[VectorStatePayload](
            pool_size=pool_size,
            epoch_diversity=epoch_diversity,
            tournament_size=tournament_size,
        ),
        storage=storage,
        max_total_tasks=max_total_tasks,
        make_state=VectorStateBuilder(
            resolution_llm=resolution_llm,
            write_lineage=write_lineage,
            save_raster=save_raster,
        ),
    )

    # What the LLM sees, deliberately not the raster: vision billing tiles at
    # 512px, so a 700px prompt image costs 3x a 512px one for detail the model
    # does not need — scoring reads the full-resolution raster, not this.
    model_png = downscale_png_bytes(original_png_bytes, resolution_llm)
    worker_ctx = WorkerContext(
        format_plugin=format_plugin,
        image_data_url=png_bytes_to_data_url(model_png),
        original_png_bytes=original_png_bytes,
        original_w=original_w,
        original_h=original_h,
        resolution_llm=resolution_llm,
        log_level=log_level,
        log_file=str(run_log_file),
        goal=goal,
        llm_provider=llm_provider,
        llm_model=llm_model,
        reasoning=reasoning,
        api_key=api_key,
        log_queue=log_queue,
    )

    def score_fn(res):
        scorer_ready.wait()
        if _scorer_error:
            raise RuntimeError(
                f"Scorer failed to initialise: {_scorer_error[0]}"
            ) from _scorer_error[0]
        scorer = _scorer[0]
        ref = _scoring_ref[0]
        result = scorer.score(ref, res.payload.raster_png)
        if res.payload.raster_png:
            # One grid serves both consumers: the worst_region objective and the
            # heatmap drawn from the same distances. Computing it here rather
            # than inside diff_heatmap is what keeps this to a single extra
            # vision pass instead of two.
            grid = scorer.region_distance_grid(ref, res.payload.raster_png)
            if grid is not None:
                res.metrics[WORST_REGION] = worst_region_score(grid)
            # Only for the --save-heatmap sidecar now that no prompt carries
            # a difference map; skipped entirely otherwise.
            res.payload.heatmap_png = (
                scorer.diff_heatmap(
                    ref,
                    res.payload.raster_png,
                    long_side=resolution_llm,
                    grid=grid,
                )
                if getattr(storage, "save_heatmap", False)
                else None
            )
        return result

    if dashboard is not None:
        logging.getLogger().addHandler(dashboard.log_handler)

    dashboard_entered = False
    try:
        if dashboard is not None:
            dashboard.__enter__()
            dashboard_entered = True

        if not resumed_items:
            # Fresh start: start scorer after dashboard so HF output doesn't
            # appear above the Live display.
            _start_scorer_thread()

        engine.start_workers(worker_loop, worker_ctx)

        engine.run(
            initial_nodes,
            max_wall_seconds=max_wall_seconds,
            epoch_patience=epoch_patience,
            epoch_min_delta=epoch_min_delta,
            active_pool_size=pool_size,
            score_fn=score_fn,
            epoch_seeds=epoch_seeds,
            initial_seeds=first_batch,
            epochs=epochs,
            epoch_variance=epoch_variance,
            collector=collector,
        )
    finally:
        log_listener.stop()
        if dashboard is not None and dashboard_entered:
            dashboard.__exit__(None, None, None)
        if dashboard is not None:
            logging.getLogger().removeHandler(dashboard.log_handler)
