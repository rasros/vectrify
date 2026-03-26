import io
import logging
import os

from PIL import Image

from svgizer.image_utils import (
    downscale_png_bytes,
    make_preview_data_url,
    png_bytes_to_data_url,
    rasterize_svg_to_png_bytes,
)
from svgizer.score import ScorerType, get_scorer
from svgizer.score.complexity import svg_complexity
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
from svgizer.svg.adapter import SvgStatePayload, SvgStrategyAdapter
from svgizer.svg.worker import worker_loop
from svgizer.utils import setup_logger

log = logging.getLogger("main")


def run_svg_search(
    image_path: str,
    storage: StorageAdapter,
    max_accepts: int,
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
    patience: int | None = None,
    min_delta: float = 1e-4,
    llm_rate: float = 0.2,
    pool_size: int = 20,
    warmup_llm: int = -1,
    diversity_threshold: float = 0.97,
    diversity_boost_threshold: float = 0.10,
) -> None:
    setup_logger(log_level)

    # 1. Image & Scorer Setup
    original_img = Image.open(image_path).convert("RGB")
    original_w, original_h = original_img.size

    buf = io.BytesIO()
    original_img.save(buf, format="PNG")
    original_png_bytes = buf.getvalue()

    api_key = os.getenv(f"{llm_provider.upper()}_API_KEY")
    scorer = get_scorer(scorer_type, provider_name=llm_provider, api_key=api_key)
    scoring_ref = scorer.prepare_reference(original_img)

    storage.initialize()

    initial_nodes = []
    current_new_id = 1

    # 3. Resume Phase
    resumed_items = storage.load_resume_nodes()
    if resumed_items:
        log.info(f"Resuming {len(resumed_items)} nodes. Re-scoring...")
        for old_id, svg_text in resumed_items:
            try:
                png = rasterize_svg_to_png_bytes(
                    svg_text, out_w=original_w, out_h=original_h
                )
                new_score = scorer.score(scoring_ref, png)

                imported_node = SearchNode(
                    score=new_score,
                    id=current_new_id,
                    parent_id=0,
                    complexity=svg_complexity(svg_text),
                    content=svg_text,  # Crucial for NCD calculations
                    state=ChainState(
                        score=new_score,
                        payload=SvgStatePayload(
                            svg=svg_text,
                            raster_data_url=None,
                            raster_preview_data_url=make_preview_data_url(
                                png, image_long_side
                            ),
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

    # 5. Empty Start Fallback
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

    # 6. Search Execution Setup
    if strategy_type == StrategyType.GREEDY:
        base_strategy = GreedyHillClimbingStrategy[SvgStatePayload]()
    else:
        base_strategy = NsgaStrategy[SvgStatePayload](
            pool_size=pool_size,
            crossover_prob=0.25,
            diversity_threshold=diversity_threshold,
            diversity_boost_threshold=diversity_boost_threshold,
        )

    # Warmup: force LLM for the first N tasks so the pool is seeded with
    # LLM-generated SVGs before local mutations dominate.
    warmup_target = pool_size // 10 if warmup_llm < 0 else warmup_llm
    seeded = sum(1 for n in initial_nodes if n.state.payload.svg)
    warmup_tasks = max(0, warmup_target - seeded)

    # Resume Shock: Check if the loaded pool is completely homogenous
    warmup_diverse = 0
    if seeded > 0 and base_strategy.should_diversify(initial_nodes):
        log.warning("Initial pool lacks diversity. Triggering resume shock.")
        warmup_diverse = max(1, pool_size // 4)

    if warmup_tasks > 0 or warmup_diverse > 0:
        log.info(
            f"Warmup: {warmup_tasks} regular LLM tasks, "
            f"{warmup_diverse} diverse LLM tasks "
            f"(target={warmup_target}, already seeded={seeded})"
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
        max_accepts,
        max_wall_seconds,
        patience,
        min_delta,
        pool_size,
        score_fn,
        warmup_tasks,
        warmup_diverse=warmup_diverse,
    )
