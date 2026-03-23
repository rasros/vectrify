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
    MultiprocessSearchEngine,
    SearchNode,
    StrategyType,
)
from svgizer.storage import StorageAdapter
from svgizer.svg_adapter import SvgStatePayload, SvgStrategyAdapter, is_svg_stale
from svgizer.utils import setup_logger
from svgizer.worker import worker_loop

log = logging.getLogger("main")


def run_svg_search(
    image_path: str,
    storage: StorageAdapter,
    seed_svg_path: str | None,
    max_accepts: int,
    workers: int,
    base_model_temperature: float,
    openai_image_long_side: int,
    max_wall_seconds: float | None,
    log_level: str,
    scorer_type: ScorerType,
    strategy_type: StrategyType,
    goal: str | None,
) -> None:
    setup_logger(log_level)

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")
    if not os.path.isfile(image_path):
        raise SystemExit(f"Input image '{image_path}' does not exist.")

    storage.initialize()

    original_img = Image.open(image_path).convert("RGB")
    original_w, original_h = original_img.size

    buf = io.BytesIO()
    original_img.save(buf, format="PNG")
    original_png_bytes = buf.getvalue()

    model_png_bytes = (
        downscale_png_bytes(original_png_bytes, openai_image_long_side)
        if openai_image_long_side > 0
        else original_png_bytes
    )
    openai_original_data_url = png_bytes_to_data_url(model_png_bytes)

    scorer = get_scorer(scorer_type)
    scoring_ref = scorer.prepare_reference(original_img)

    base_strategy = GeneticPoolStrategy(
        top_k=3, temp_step=0.3, max_temp=1.6, is_stale_fn=is_svg_stale
    )

    strategy = SvgStrategyAdapter(
        base_strategy=base_strategy,
        openai_image_long_side=openai_image_long_side,
        write_lineage=storage.write_lineage_enabled,
    )

    initial_nodes = []

    prior_nodes, _, max_id = storage.load_resume_nodes(
        log, base_model_temperature, original_w, original_h, openai_image_long_side
    )

    if prior_nodes:
        log.info(f"Resuming: Recalculating scores for {len(prior_nodes)} nodes...")
        for n in prior_nodes:
            if n.state.payload.svg:
                full_png = rasterize_svg_to_png_bytes(
                    n.state.payload.svg, out_w=original_w, out_h=original_h
                )
                n.score = n.state.score = scorer.score(scoring_ref, full_png)
        initial_nodes.extend(prior_nodes)

    if seed_svg_path:
        try:
            seed_svg = storage.load_seed_svg(seed_svg_path)
            full_png = rasterize_svg_to_png_bytes(
                seed_svg, out_w=original_w, out_h=original_h
            )
            seed_score = scorer.score(scoring_ref, full_png)

            payload = SvgStatePayload(
                svg=seed_svg,
                raster_data_url=None,
                raster_preview_data_url=make_preview_data_url(
                    full_png, openai_image_long_side
                ),
                change_summary=None,
                invalid_msg=None,
            )

            seed_node = SearchNode(
                score=seed_score,
                id=max_id + 1,
                parent_id=0,
                state=ChainState(
                    score=seed_score,
                    model_temperature=base_model_temperature,
                    stale_hits=0,
                    payload=payload,
                ),
            )
            initial_nodes.append(seed_node)
            storage.save_node(seed_node)
        except Exception as e:
            raise SystemExit(f"--seed-svg error: {e}")

    if not initial_nodes:
        empty_payload = SvgStatePayload(None, None, None, None, None)
        initial_nodes.append(
            SearchNode(
                score=INVALID_SCORE,
                id=0,
                parent_id=0,
                state=ChainState(
                    score=INVALID_SCORE,
                    model_temperature=base_model_temperature,
                    stale_hits=0,
                    payload=empty_payload,
                ),
            )
        )

    engine = MultiprocessSearchEngine(
        workers=workers, strategy=strategy, storage=storage, max_total_tasks=10000
    )

    worker_params = {
        "openai_original_data_url": openai_original_data_url,
        "original_png_bytes": original_png_bytes,
        "original_w": original_w,
        "original_h": original_h,
        "openai_image_long_side": openai_image_long_side,
        "log_level": log_level,
        "scorer_type": scorer_type,
        "goal": goal,
    }

    engine.start_workers(worker_loop, worker_params)

    log.info("Starting Search Engine...")
    best_node = engine.run(
        initial_nodes=initial_nodes,
        max_accepts=max_accepts,
        max_wall_seconds=max_wall_seconds,
    )

    if best_node and best_node.state.payload.svg:
        storage.save_final_svg(best_node.state.payload.svg)
        log.info(f"Search complete. Best score: {best_node.score:.6f}")
    else:
        log.error("Search failed to produce a valid SVG.")
