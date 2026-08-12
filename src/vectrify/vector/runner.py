import io
import logging
import os
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
from vectrify.score.complexity import (
    FRONT_SCORE,
    NODE_RATIO,
    WORST_REGION_4,
    WORST_REGION_16,
    ZIP_RATIO,
)
from vectrify.score.regions import complexity_ratio, region_worst_scores
from vectrify.score.simple import SimpleFallbackScorer
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
from vectrify.search.operators import FixedWeightPolicy
from vectrify.utils import setup_logger, start_log_listener
from vectrify.vector.resume import filter_to_pool_size, resume_nodes
from vectrify.vector.state import VectorStateBuilder
from vectrify.vector.worker import WorkerContext, worker_loop

log = logging.getLogger("main")


def _load_image(image_path: str, long_side: int) -> tuple[Image.Image, bytes, int, int]:
    """Open the reference image and return (img, png_bytes, width, height).

    Downscaled to *long_side*, which makes the raster the single resolution in
    the run: candidates are rendered at this size and written in its coordinate
    space. A source image's own dimensions would otherwise silently set the
    cost of every rasterization in the run.

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
    img = resize_long_side(img, long_side)
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
    # Selects the evaluator that ranks the converged Pareto front, not the
    # round's scorer -- the round is always pixel L1.
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
    random_seed: int | None = None,
    vision_model: str = DEFAULT_VISION_MODEL,  # for the front evaluator
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

    # The round scores on pixels, so there is no model to load and nothing to
    # overlap it with. The configured --scorer selects the evaluator that ranks
    # the converged Pareto front instead, which runs once per epoch.
    loop_scorer = SimpleFallbackScorer()
    loop_ref = loop_scorer.prepare_reference(original_img)

    # The zero-complexity candidate's error. Every complexity ratio is charged
    # against how much of this a candidate removes, so it is what stops a blank
    # canvas -- which removes none of it -- from owning a Pareto front slot.
    blank = Image.new("RGB", original_img.size, "white")
    blank_buf = io.BytesIO()
    blank.save(blank_buf, format="PNG")
    blank_error = loop_scorer.score(loop_ref, blank_buf.getvalue())
    log.info(f"Blank-canvas error: {blank_error:.6f}")
    log.info(
        f"Round scoring: pixel L1. Front evaluator: {ScorerType(scorer_type).value}"
        f" ({vision_model})."
    )

    resumed_items = storage.load_resume_nodes()

    initial_nodes: list[SearchNode] = []

    if resumed_items:
        initial_nodes = resume_nodes(
            resumed_items=resumed_items,
            format_plugin=format_plugin,
            original_img=original_img,
            original_w=original_w,
            original_h=original_h,
            resolution_llm=resolution_llm,
            pool_size=pool_size,
            workers=workers,
            scorer=loop_scorer,
            scoring_ref=loop_ref,
            blank_error=blank_error,
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

    # Built on first use so a run that never reaches an epoch boundary -- and a
    # machine without CUDA using --scorer simple -- never pays for the model.
    _front: list[Any] = []

    def _front_scorer() -> tuple[Any, Any]:
        if not _front:
            scorer = get_scorer(scorer_type, vision_model=vision_model)
            _front.extend([scorer, scorer.prepare_reference(original_img)])
        return _front[0], _front[1]

    def rank_front(nodes: list[SearchNode]) -> list[SearchNode]:
        """Order a converged front by the run's real objective.

        Re-rasterises rather than reading the node's stored render, which is
        only kept when --write-lineage or --save-raster is on.
        """
        scorer, ref = _front_scorer()
        scored: list[tuple[float, SearchNode]] = []
        for node in nodes:
            content = getattr(node.state.payload, "content", None)
            if not content:
                continue
            try:
                png = format_plugin.rasterize(
                    content, out_w=original_w, out_h=original_h
                )
                value = scorer.score(ref, png)
            except Exception as exc:
                log.debug(f"Front evaluation skipped node {node.id}: {exc}")
                continue
            node.metrics[FRONT_SCORE] = value
            scored.append((value, node))

        if not scored:
            return nodes
        scored.sort(key=lambda pair: pair[0])
        log.info(
            f"Front evaluated: {len(scored)} candidate(s), "
            f"best {scored[0][0]:.6f}, worst {scored[-1][0]:.6f}"
        )
        return [node for _value, node in scored]

    engine = MultiprocessSearchEngine(
        workers=workers,
        strategy=NsgaStrategy[VectorStatePayload](
            pool_size=pool_size,
            epoch_diversity=epoch_diversity,
            tournament_size=tournament_size,
        ),
        storage=storage,
        max_total_tasks=max_total_tasks,
        rank_front=rank_front,
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
        random_seed=random_seed,
        log_queue=log_queue,
    )

    def score_fn(res):
        result = loop_scorer.score(loop_ref, res.payload.raster_png)
        png = res.payload.raster_png
        if png:
            worst = region_worst_scores(loop_ref.image, png)
            res.metrics[WORST_REGION_4] = worst[2]
            res.metrics[WORST_REGION_16] = worst[4]
            res.metrics[ZIP_RATIO] = complexity_ratio(
                res.metrics.get("zip_complexity", 0.0), result, blank_error
            )
            res.metrics[NODE_RATIO] = complexity_ratio(
                res.metrics.get("node_complexity", 0.0), result, blank_error
            )
            res.payload.heatmap_png = (
                loop_scorer.diff_heatmap(loop_ref, png, long_side=resolution_llm)
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
            operator_policy=FixedWeightPolicy(format_plugin.mutation_weights()),
            collector=collector,
        )
    finally:
        log_listener.stop()
        if dashboard is not None and dashboard_entered:
            dashboard.__exit__(None, None, None)
        if dashboard is not None:
            logging.getLogger().removeHandler(dashboard.log_handler)
