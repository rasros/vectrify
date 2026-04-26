import dataclasses
import io
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vectrify.dashboard import Dashboard
    from vectrify.formats.base import FormatPlugin
    from vectrify.search.stats import SearchStats

from PIL import Image

from vectrify.formats.models import VectorStatePayload
from vectrify.image_utils import (
    downscale_png_bytes,
    png_bytes_to_data_url,
)
from vectrify.score import ScorerType, get_scorer
from vectrify.search import (
    INVALID_SCORE,
    BeamSearchStrategy,
    ChainState,
    MultiprocessSearchEngine,
    NsgaStrategy,
    SearchNode,
    StorageAdapter,
    StrategyType,
)
from vectrify.search.collector import StatCollector
from vectrify.utils import setup_logger, start_log_listener
from vectrify.vector.adapter import VectorStrategyAdapter
from vectrify.vector.resume import filter_to_pool_size, resume_nodes
from vectrify.vector.worker import WorkerContext, worker_loop

log = logging.getLogger("main")


@dataclasses.dataclass
class _EngineParams:
    pool_size: int
    seed_tasks: int
    epoch_patience: int | None
    epoch_min_delta: float
    max_epochs: int | None
    epoch_pool_size: int | None
    epoch_steps: int | None


def _load_image(image_path: str) -> tuple[Image.Image, bytes, int, int]:
    """Open the reference image and return (img, png_bytes, width, height)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return img, buf.getvalue(), w, h


def _build_engine_params(
    strategy_type: StrategyType,
    pool_size: int,
    seeds: int,
    beams: int,
    initial_nodes: list[SearchNode],
    epoch_patience: int | None,
    epoch_min_delta: float,
    max_epochs: int | None,
    epoch_pool_size: int | None,
    epoch_steps: int | None,
) -> _EngineParams:
    """Compute engine configuration from search parameters."""
    is_beam = strategy_type == StrategyType.BEAM

    if is_beam:
        return _EngineParams(
            pool_size=beams,
            seed_tasks=beams,
            epoch_patience=epoch_patience,
            epoch_min_delta=epoch_min_delta,
            max_epochs=max_epochs,
            epoch_pool_size=None,
            epoch_steps=epoch_steps,
        )

    seed_target = pool_size // 10 if seeds == 0 else seeds
    seeded = sum(1 for n in initial_nodes if n.state.payload.content)
    seed_tasks = max(0, seed_target - seeded)
    if seed_tasks > 0:
        log.info(
            f"Epoch 0: {seed_tasks} LLM seed tasks "
            f"(target={seed_target}, already seeded={seeded})"
        )

    return _EngineParams(
        pool_size=pool_size,
        seed_tasks=seed_tasks,
        epoch_patience=epoch_patience,
        epoch_min_delta=epoch_min_delta,
        max_epochs=max_epochs,
        epoch_pool_size=epoch_pool_size,
        epoch_steps=epoch_steps,
    )


def run_vector_search(
    image_path: str,
    storage: StorageAdapter,
    workers: int,
    image_long_side: int,
    max_wall_seconds: float | None,
    log_level: str,
    scorer_type: ScorerType,
    strategy_type: StrategyType,
    goal: str | None,
    llm_provider: str,
    llm_model: str,
    reasoning: str,
    format_plugin: "FormatPlugin",
    write_lineage: bool = True,
    save_raster: bool = False,
    epoch_patience: int | None = None,
    epoch_min_delta: float = 1e-4,
    llm_rate: float = 0.2,
    pool_size: int = 20,
    seeds: int = 0,
    beams: int = 10,
    cull_keep: float = 0.5,
    epoch_diversity: float = 0.10,
    epoch_variance: float | None = None,
    max_epochs: int | None = None,
    epoch_pool_size: int | None = None,
    epoch_steps: int | None = None,
    vision_model: str = "ensemble",
    stats: "SearchStats | None" = None,
    dashboard: "Dashboard | None" = None,
) -> None:
    storage.initialize()
    assert storage.current_run_dir is not None
    run_log_file = storage.current_run_dir / "search.log"
    setup_logger(log_level, log_file=run_log_file)
    log_queue, log_listener = start_log_listener()

    # Suppress tqdm / HF noise before any library imports or workers spawn.
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    original_img, original_png_bytes, original_w, original_h = _load_image(image_path)
    api_key = os.getenv(f"{llm_provider.upper()}_API_KEY")

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
                provider_name=llm_provider,
                api_key=api_key,
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
            image_long_side=image_long_side,
            pool_size=pool_size,
            workers=workers,
            scorer=_scorer[0],
            scoring_ref=_scoring_ref[0],
            storage=storage,
        )
        initial_nodes = filter_to_pool_size(initial_nodes, pool_size, strategy_type)

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
            llm_rate=llm_rate,
            epoch_diversity=epoch_diversity,
            epoch_variance=epoch_variance or 0.0,
            epoch_steps=epoch_steps or 0,
        )
        valid = [n for n in initial_nodes if n.score < INVALID_SCORE]
        if valid:
            collector.seed_initial_score(min(valid, key=lambda n: n.score).score)

    is_beam = strategy_type == StrategyType.BEAM
    base_strategy = (
        BeamSearchStrategy[VectorStatePayload](beams=beams, cull_keep=cull_keep)
        if is_beam
        else NsgaStrategy[VectorStatePayload](
            pool_size=pool_size,
            crossover_distance_threshold=10,
            epoch_diversity=epoch_diversity,
        )
    )

    ep = _build_engine_params(
        strategy_type=strategy_type,
        pool_size=pool_size,
        seeds=seeds,
        beams=beams,
        initial_nodes=initial_nodes,
        epoch_patience=epoch_patience,
        epoch_min_delta=epoch_min_delta,
        max_epochs=max_epochs,
        epoch_pool_size=epoch_pool_size,
        epoch_steps=epoch_steps,
    )

    strategy = VectorStrategyAdapter(
        base_strategy, image_long_side, write_lineage, save_raster
    )
    engine = MultiprocessSearchEngine(
        workers=workers, strategy=strategy, storage=storage
    )

    model_png = downscale_png_bytes(original_png_bytes, image_long_side)
    worker_ctx = WorkerContext(
        format_plugin=format_plugin,
        image_data_url=png_bytes_to_data_url(model_png),
        original_png_bytes=original_png_bytes,
        original_w=original_w,
        original_h=original_h,
        image_long_side=image_long_side,
        log_level=log_level,
        log_file=str(run_log_file),
        goal=goal,
        llm_provider=llm_provider,
        llm_model=llm_model,
        reasoning=reasoning,
        api_key=api_key,
        total_workers=workers,
        llm_rate=llm_rate,
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
            res.payload.heatmap_png = scorer.diff_heatmap(
                ref, res.payload.raster_png, long_side=image_long_side
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
            epoch_patience=ep.epoch_patience,
            epoch_min_delta=ep.epoch_min_delta,
            active_pool_size=ep.pool_size,
            score_fn=score_fn,
            seed_tasks=ep.seed_tasks,
            max_epochs=ep.max_epochs,
            epoch_pool_size=ep.epoch_pool_size,
            epoch_variance=epoch_variance,
            epoch_steps=ep.epoch_steps,
            collector=collector,
        )
    finally:
        log_listener.stop()
        if dashboard is not None and dashboard_entered:
            dashboard.__exit__(None, None, None)
        if dashboard is not None:
            logging.getLogger().removeHandler(dashboard.log_handler)
