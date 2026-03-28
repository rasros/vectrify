import concurrent.futures
import io
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from svgizer.dashboard import Dashboard
    from svgizer.search.stats import SearchStats

from PIL import Image

from svgizer.image_utils import (
    downscale_png_bytes,
    make_preview_data_url,
    png_bytes_to_data_url,
    rasterize_svg_to_png_bytes,
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
from svgizer.svg.adapter import SvgStatePayload, SvgStrategyAdapter
from svgizer.svg.worker import worker_loop
from svgizer.utils import setup_logger

log = logging.getLogger("main")


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


def run_svg_search(
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
    write_lineage: bool = True,
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
    if dashboard is not None:
        logging.getLogger().addHandler(dashboard.log_handler)

    dashboard_entered = False
    try:
        if dashboard is not None:
            dashboard.__enter__()
            dashboard_entered = True
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        original_img = Image.open(image_path).convert("RGB")
        original_w, original_h = original_img.size

        buf = io.BytesIO()
        original_img.save(buf, format="PNG")
        original_png_bytes = buf.getvalue()

        api_key = os.getenv(f"{llm_provider.upper()}_API_KEY")
        scorer = get_scorer(
            scorer_type,
            provider_name=llm_provider,
            api_key=api_key,
            vision_model=vision_model,
        )
        scoring_ref = scorer.prepare_reference(original_img)

        initial_nodes = []
        current_new_id = 1

        resumed_items = storage.load_resume_nodes()
        if resumed_items:
            log.info(
                f"Resuming {len(resumed_items)} nodes. Deduplicating and re-scoring..."
            )

            unique_items = []
            seen_sigs = set()
            for old_id, svg_text in resumed_items:
                sig = simhash(svg_text)
                if sig is not None:
                    if sig in seen_sigs:
                        log.debug(f"Skipping duplicate Node {old_id} during resume.")
                        continue
                    seen_sigs.add(sig)
                unique_items.append((old_id, svg_text, sig))

            log.info(f"Filtered to {len(unique_items)} unique nodes.")

            # Helper for threaded rasterization
            def _prep_resume_node(item):
                old_id, svg_text, sig = item
                png = rasterize_svg_to_png_bytes(
                    svg_text, out_w=original_w, out_h=original_h
                )
                preview = make_preview_data_url(png, image_long_side)
                complexity = visual_complexity(png)
                return old_id, svg_text, png, preview, complexity, sig

            prepped_nodes = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_prep_resume_node, item) for item in unique_items
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        prepped_nodes.append(future.result())
                    except Exception as e:
                        log.error(f"Failed to prep resume node: {e}")

            if len(prepped_nodes) > 2 * pool_size:
                log.info(
                    f"Pre-filtering {len(prepped_nodes)} resume nodes "
                    f"to {2 * pool_size} using simple scorer + complexity Pareto front..."
                )
                prepped_nodes = _prefilter_nodes(
                    prepped_nodes, original_img, 2 * pool_size
                )
                log.info(f"Pre-filter done: {len(prepped_nodes)} nodes selected.")

            for old_id, svg_text, png, preview, complexity, sig in prepped_nodes:
                try:
                    new_score = scorer.score(scoring_ref, png)

                    imported_node = SearchNode(
                        score=new_score,
                        id=current_new_id,
                        parent_id=0,
                        complexity=complexity,
                        signature=sig,
                        state=ChainState(
                            score=new_score,
                            payload=SvgStatePayload(
                                svg=svg_text,
                                raster_data_url=None,
                                raster_preview_data_url=preview,
                                change_summary=f"Imported from Node {old_id}",
                                invalid_msg=None,
                            ),
                        ),
                    )
                    storage.save_node(imported_node)
                    initial_nodes.append(imported_node)
                    current_new_id += 1
                except Exception as e:
                    log.error(f"Failed to import Node {old_id}: {e}")

        if len(initial_nodes) > pool_size:
            log.info(
                f"Filtering {len(initial_nodes)} rescored nodes down to {pool_size}..."
            )

            if strategy_type == StrategyType.NSGA:
                max_score = (
                    max(
                        (n.score for n in initial_nodes if n.score < float("inf")),
                        default=1.0,
                    )
                    or 1.0
                )
                max_comp = (
                    max((n.complexity for n in initial_nodes), default=1.0) or 1.0
                )
                objectives = {
                    n.id: (n.score / max_score, n.complexity / max_comp)
                    for n in initial_nodes
                }

                fronts = non_dominated_sort(initial_nodes, objectives)
                filtered_nodes = []

                for front in fronts:
                    if len(filtered_nodes) >= pool_size:
                        break

                    distances = crowding_distance(front, objectives)
                    front_sorted = sorted(front, key=lambda n: -distances[n.id])

                    for node in front_sorted:
                        if len(filtered_nodes) >= pool_size:
                            break
                        filtered_nodes.append(node)

                initial_nodes = filtered_nodes
            else:
                initial_nodes = sorted(initial_nodes, key=lambda n: n.score)[:pool_size]

        if not initial_nodes:
            initial_nodes.append(
                SearchNode(
                    score=INVALID_SCORE,
                    id=0,
                    parent_id=0,
                    state=ChainState(
                        INVALID_SCORE,
                        SvgStatePayload(None, None, None, None, None),
                    ),
                )
            )

        # Seed stats so the dashboard isn't blank on start.
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
            valid = [n for n in initial_nodes if n.score < float("inf")]
            if valid:
                best_initial = min(valid, key=lambda n: n.score)
                collector.seed_initial_score(best_initial.score)

        is_greedy = strategy_type == StrategyType.GREEDY

        if is_greedy:
            base_strategy = GreedyHillClimbingStrategy[SvgStatePayload]()
        else:
            base_strategy = NsgaStrategy[SvgStatePayload](
                pool_size=pool_size,
                crossover_distance_threshold=10,
                epoch_diversity=epoch_diversity,
            )

        if is_greedy:
            engine_pool_size = 1
            engine_seed_tasks = 0
            engine_epoch_patience = epoch_patience
            engine_epoch_min_delta = epoch_min_delta
            engine_max_epochs = max_epochs
            engine_epoch_pool_size = None
        else:
            engine_pool_size = pool_size
            seed_target = pool_size // 10 if seeds == 0 else seeds
            seeded = sum(1 for n in initial_nodes if n.state.payload.svg)
            engine_seed_tasks = max(0, seed_target - seeded)
            engine_epoch_patience = epoch_patience
            engine_epoch_min_delta = epoch_min_delta
            engine_max_epochs = max_epochs
            engine_epoch_pool_size = epoch_pool_size
            if engine_seed_tasks > 0:
                log.info(
                    f"Epoch 0: {engine_seed_tasks} LLM seed tasks "
                    f"(target={seed_target}, already seeded={seeded})"
                )

        strategy = SvgStrategyAdapter(base_strategy, image_long_side, write_lineage)
        engine = MultiprocessSearchEngine(
            workers=workers, strategy=strategy, storage=storage
        )

        model_png = downscale_png_bytes(original_png_bytes, image_long_side)

        worker_params = {
            "image_data_url": png_bytes_to_data_url(model_png),
            "original_png_bytes": original_png_bytes,
            "original_w": original_w,
            "original_h": original_h,
            "image_long_side": image_long_side,
            "log_level": log_level,
            "log_file": str(run_log_file),
            "goal": goal,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "reasoning": reasoning,
            "api_key": api_key,
            "total_workers": workers,
            "llm_rate": llm_rate,
        }

        def score_fn(res):
            return scorer.score(scoring_ref, res.payload.raster_png)

        engine.start_workers(worker_loop, worker_params)
        engine.run(
            initial_nodes,
            max_wall_seconds=max_wall_seconds,
            epoch_patience=engine_epoch_patience,
            epoch_min_delta=engine_epoch_min_delta,
            active_pool_size=engine_pool_size,
            score_fn=score_fn,
            seed_tasks=engine_seed_tasks,
            max_epochs=engine_max_epochs,
            epoch_pool_size=engine_epoch_pool_size,
            epoch_variance=epoch_variance,
            collector=collector,
        )
    finally:
        if dashboard is not None and dashboard_entered:
            dashboard.__exit__(None, None, None)
            logging.getLogger().removeHandler(dashboard.log_handler)
