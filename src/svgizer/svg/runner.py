import io
import logging
import os

from PIL import Image

from svgizer.diff import ScorerType, get_scorer
from svgizer.image_utils import (
    downscale_png_bytes,
    make_preview_data_url,
    png_bytes_to_data_url,
    rasterize_svg_to_png_bytes,
)
from svgizer.search import (
    INVALID_SCORE,
    ChainState,
    GeneticPoolStrategy,
    GreedyHillClimbingStrategy,
    MultiprocessSearchEngine,
    SearchNode,
    SearchStrategy,
    StorageAdapter,
    StrategyType,
)
from svgizer.svg.adapter import SvgStatePayload, SvgStrategyAdapter, make_is_svg_stale
from svgizer.svg.worker import worker_loop
from svgizer.utils import setup_logger

log = logging.getLogger("main")


def run_svg_search(
    image_path: str,
    storage: StorageAdapter,
    seed_svg_path: str | None,
    max_accepts: int,
    workers: int,
    base_model_temperature: float,
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
) -> None:
    setup_logger(log_level)

    # 1. Image & Scorer Setup
    original_img = Image.open(image_path).convert("RGB")
    original_w, original_h = original_img.size

    buf = io.BytesIO()
    original_img.save(buf, format="PNG")
    original_png_bytes = buf.getvalue()

    scorer = get_scorer(scorer_type)
    scoring_ref = scorer.prepare_reference(original_img)

    # 2. Storage & Resume Logic
    storage.initialize()
    initial_nodes = storage.load_resume_nodes()

    if initial_nodes:
        log.info(f"Resumed {len(initial_nodes)} nodes from storage.")

    # 3. Seed Handling
    if seed_svg_path:
        try:
            seed_svg = storage.load_seed_svg(seed_svg_path)
            png = rasterize_svg_to_png_bytes(
                seed_svg, out_w=original_w, out_h=original_h
            )
            score = scorer.score(scoring_ref, png)

            seed_node = SearchNode(
                score=score,
                id=storage.max_node_id + 1,
                parent_id=0,
                state=ChainState(
                    score=score,
                    model_temperature=base_model_temperature,
                    stale_hits=0,
                    payload=SvgStatePayload(
                        svg=seed_svg,
                        raster_data_url=None,
                        raster_preview_data_url=make_preview_data_url(
                            png, image_long_side
                        ),
                        change_summary=None,
                        invalid_msg=None,
                    ),
                ),
            )
            initial_nodes.append(seed_node)
            storage.save_node(seed_node)
        except Exception as e:
            log.error(f"Failed to load seed SVG: {e}")

    if not initial_nodes:
        initial_nodes.append(
            SearchNode(
                score=INVALID_SCORE,
                id=0,
                parent_id=0,
                state=ChainState(
                    INVALID_SCORE,
                    base_model_temperature,
                    0,
                    SvgStatePayload(None, None, None, None, None),
                ),
            )
        )

    # 4. Search Execution
    base_strategy: SearchStrategy[SvgStatePayload]
    if strategy_type == StrategyType.GREEDY:
        base_strategy = GreedyHillClimbingStrategy[SvgStatePayload]()
    else:
        base_strategy = GeneticPoolStrategy[SvgStatePayload](
            top_k=3, is_stale_fn=make_is_svg_stale(0.995)
        )

    strategy = SvgStrategyAdapter(base_strategy, image_long_side, write_lineage)

    engine = MultiprocessSearchEngine(
        workers=workers, strategy=strategy, storage=storage
    )

    # Prepare worker parameters
    model_png = downscale_png_bytes(original_png_bytes, image_long_side)
    api_key_env_var = f"{llm_provider.upper()}_API_KEY"

    worker_params = {
        "image_data_url": png_bytes_to_data_url(model_png),
        "original_png_bytes": original_png_bytes,
        "original_w": original_w,
        "original_h": original_h,
        "log_level": log_level,
        "scorer_type": scorer_type,
        "goal": goal,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "reasoning": reasoning,
        "api_key": os.getenv(api_key_env_var),
        "worker_max_temp": 1.6,
        "worker_temp_step": 0.07,
    }

    engine.start_workers(worker_loop, worker_params)
    best_node = engine.run(initial_nodes, max_accepts, max_wall_seconds)

    if best_node and best_node.state.payload.svg:
        storage.save_final_svg(best_node.state.payload.svg)
        log.info(f"Done. Best score: {best_node.score:.6f}")
