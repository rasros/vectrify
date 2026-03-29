import concurrent.futures
import dataclasses
import io
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from svgizer.dashboard import Dashboard
    from svgizer.formats.base import FormatPlugin
    from svgizer.search.stats import SearchStats

from PIL import Image

from svgizer.formats.models import VectorStatePayload
from svgizer.image_utils import (
    downscale_png_bytes,
    make_preview_data_url,
    png_bytes_to_data_url,
)
from svgizer.score import ScorerType, get_scorer
from svgizer.score.complexity import visual_complexity
from svgizer.score.simple import SimpleFallbackScorer
from svgizer.search import (
    INVALID_SCORE,
    ChainState,
    GreedyHillClimbingStrategy,
    MultiprocessSearchEngine,
    NsgaStrategy,
    SearchNode,
    StorageAdapter,
    StrategyType,
)
from svgizer.search.collector import StatCollector
from svgizer.search.diversity import simhash
from svgizer.search.nsga import crowding_distance, non_dominated_sort
from svgizer.utils import setup_logger
from svgizer.vector.adapter import VectorStrategyAdapter
from svgizer.vector.worker import WorkerContext, worker_loop

log = logging.getLogger("main")


@dataclasses.dataclass
class _EngineParams:
    pool_size: int
    seed_tasks: int
    epoch_patience: int | None
    epoch_min_delta: float
    max_epochs: int | None
    epoch_pool_size: int | None


def _load_image(image_path: str) -> tuple[Image.Image, bytes, int, int]:
    """Open the reference image and return (img, png_bytes, width, height)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return img, buf.getvalue(), w, h


def _prefilter_nodes(
    prepped_nodes: list,
    original_img: Image.Image,
    max_keep: int,
) -> list:
    """Use SimpleFallbackScorer + complexity Pareto front to reduce candidates."""
    simple_scorer = SimpleFallbackScorer()
    simple_ref = simple_scorer.prepare_reference(original_img)

    simple_scores = []
    for _, _, png, _, _, _ in prepped_nodes:
        try:
            simple_scores.append(simple_scorer.score(simple_ref, png))
        except Exception:
            simple_scores.append(1.0)

    complexities = [item[4] for item in prepped_nodes]
    max_s = max(simple_scores, default=1.0) or 1.0
    max_c = max(complexities, default=1.0) or 1.0

    temp_nodes = [
        SearchNode(
            score=simple_scores[i],
            id=i,
            parent_id=0,
            state=ChainState(score=simple_scores[i], payload=None),
            complexity=complexities[i],
        )
        for i in range(len(prepped_nodes))
    ]
    objectives = {
        i: (simple_scores[i] / max_s, complexities[i] / max_c)
        for i in range(len(prepped_nodes))
    }

    fronts = non_dominated_sort(temp_nodes, objectives)
    kept: list[int] = []
    for front in fronts:
        if len(kept) >= max_keep:
            break
        distances = crowding_distance(front, objectives)
        for node in sorted(front, key=lambda n: -distances[n.id]):
            if len(kept) >= max_keep:
                break
            kept.append(node.id)

    return [prepped_nodes[i] for i in kept]


def _resume_nodes(
    resumed_items: list[tuple[int, str]],
    format_plugin: "FormatPlugin",
    original_img: Image.Image,
    original_w: int,
    original_h: int,
    image_long_side: int,
    pool_size: int,
    workers: int,
    scorer: Any,
    scoring_ref: Any,
    storage: StorageAdapter,
) -> list[SearchNode]:
    """Deduplicate, rasterize, pre-filter, and re-score resumed nodes."""
    log.info(f"Resuming {len(resumed_items)} nodes. Deduplicating and re-scoring...")

    unique_items = []
    seen_sigs: set[int] = set()
    for old_id, content_text in resumed_items:
        sig = simhash(content_text)
        if sig is not None:
            if sig in seen_sigs:
                log.debug(f"Skipping duplicate Node {old_id} during resume.")
                continue
            seen_sigs.add(sig)
        unique_items.append((old_id, content_text, sig))

    log.info(f"Filtered to {len(unique_items)} unique nodes.")

    def _prep(item: tuple) -> tuple:
        old_id, content_text, sig = item
        png = format_plugin.rasterize(content_text, out_w=original_w, out_h=original_h)
        preview = make_preview_data_url(png, image_long_side)
        complexity = visual_complexity(png)
        return old_id, content_text, png, preview, complexity, sig

    prepped: list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_prep, item) for item in unique_items]
        for future in concurrent.futures.as_completed(futures):
            try:
                prepped.append(future.result())
            except Exception as e:
                log.error(f"Failed to prep resume node: {e}")

    if len(prepped) > 2 * pool_size:
        log.info(
            f"Pre-filtering {len(prepped)} resume nodes "
            f"to {2 * pool_size} using simple scorer + complexity Pareto front..."
        )
        prepped = _prefilter_nodes(prepped, original_img, 2 * pool_size)
        log.info(f"Pre-filter done: {len(prepped)} nodes selected.")

    initial_nodes: list[SearchNode] = []
    current_new_id = 1
    for old_id, content_text, png, preview, complexity, sig in prepped:
        try:
            new_score = scorer.score(scoring_ref, png)
            node = SearchNode(
                score=new_score,
                id=current_new_id,
                parent_id=0,
                complexity=complexity,
                signature=sig,
                state=ChainState(
                    score=new_score,
                    payload=VectorStatePayload(
                        content=content_text,
                        raster_data_url=None,
                        raster_preview_data_url=preview,
                        change_summary=f"Imported from Node {old_id}",
                        invalid_msg=None,
                    ),
                ),
            )
            storage.save_node(node)
            initial_nodes.append(node)
            current_new_id += 1
        except Exception as e:
            log.error(f"Failed to import Node {old_id}: {e}")

    return initial_nodes


def _filter_resumed_nodes(
    initial_nodes: list[SearchNode],
    pool_size: int,
    strategy_type: StrategyType,
) -> list[SearchNode]:
    """Trim resumed nodes down to pool_size using NSGA or score sorting."""
    if len(initial_nodes) <= pool_size:
        return initial_nodes

    log.info(f"Filtering {len(initial_nodes)} rescored nodes down to {pool_size}...")

    if strategy_type == StrategyType.NSGA:
        max_score = (
            max(
                (n.score for n in initial_nodes if n.score < INVALID_SCORE),
                default=1.0,
            )
            or 1.0
        )
        max_comp = max((n.complexity for n in initial_nodes), default=1.0) or 1.0
        objectives = {
            n.id: (n.score / max_score, n.complexity / max_comp) for n in initial_nodes
        }
        fronts = non_dominated_sort(initial_nodes, objectives)
        filtered: list[SearchNode] = []
        for front in fronts:
            if len(filtered) >= pool_size:
                break
            distances = crowding_distance(front, objectives)
            for node in sorted(front, key=lambda n: -distances[n.id]):
                if len(filtered) >= pool_size:
                    break
                filtered.append(node)
        return filtered

    return sorted(initial_nodes, key=lambda n: n.score)[:pool_size]


def _build_engine_params(
    strategy_type: StrategyType,
    pool_size: int,
    seeds: int,
    initial_nodes: list[SearchNode],
    epoch_patience: int | None,
    epoch_min_delta: float,
    max_epochs: int | None,
    epoch_pool_size: int | None,
) -> _EngineParams:
    """Compute engine configuration from search parameters."""
    is_greedy = strategy_type == StrategyType.GREEDY

    if is_greedy:
        return _EngineParams(
            pool_size=1,
            seed_tasks=0,
            epoch_patience=epoch_patience,
            epoch_min_delta=epoch_min_delta,
            max_epochs=max_epochs,
            epoch_pool_size=None,
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
    epoch_diversity: float = 0.10,
    epoch_variance: float | None = None,
    max_epochs: int | None = None,
    epoch_pool_size: int | None = None,
    vision_model: str = "ensemble",
    stats: "SearchStats | None" = None,
    dashboard: "Dashboard | None" = None,
) -> None:
    storage.initialize()
    assert storage.current_run_dir is not None
    run_log_file = storage.current_run_dir / "search.log"
    setup_logger(log_level, log_file=run_log_file)

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

        initial_nodes = _resume_nodes(
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
        initial_nodes = _filter_resumed_nodes(initial_nodes, pool_size, strategy_type)

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
        )
        valid = [n for n in initial_nodes if n.score < INVALID_SCORE]
        if valid:
            collector.seed_initial_score(min(valid, key=lambda n: n.score).score)

    is_greedy = strategy_type == StrategyType.GREEDY
    base_strategy = (
        GreedyHillClimbingStrategy[VectorStatePayload]()
        if is_greedy
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
        initial_nodes=initial_nodes,
        epoch_patience=epoch_patience,
        epoch_min_delta=epoch_min_delta,
        max_epochs=max_epochs,
        epoch_pool_size=epoch_pool_size,
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
    )

    def score_fn(res):
        scorer_ready.wait()
        if _scorer_error:
            raise RuntimeError(
                f"Scorer failed to initialise: {_scorer_error[0]}"
            ) from _scorer_error[0]
        return _scorer[0].score(_scoring_ref[0], res.payload.raster_png)

    # Start workers before the dashboard enters so that subprocess spawn output
    # (which goes to the inherited stderr fd) doesn't disrupt Rich Live's cursor
    # positioning.
    engine.start_workers(worker_loop, worker_ctx)

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
            collector=collector,
        )
    finally:
        if dashboard is not None and dashboard_entered:
            dashboard.__exit__(None, None, None)
        if dashboard is not None:
            logging.getLogger().removeHandler(dashboard.log_handler)
