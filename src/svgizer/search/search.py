import io
import logging
import os
import time
from typing import Optional

from PIL import Image

from svgizer.diff import get_scorer
from svgizer.image_utils import (
    downscale_png_bytes,
    png_bytes_to_data_url,
    rasterize_svg_to_png_bytes,
    make_preview_data_url,
)
from svgizer.models import ChainState, SearchNode, INVALID_SCORE
from svgizer.openai_iface import is_valid_svg
from svgizer.storage import StorageAdapter
from svgizer.utils import setup_logger

# New imports from the decoupled search package
from svgizer.search.genetic import GeneticPoolStrategy
from svgizer.search.engine import MultiprocessSearchEngine

log = logging.getLogger("main")

def run_search(
        image_path: str,
        storage: StorageAdapter,
        seed_svg_path: Optional[str],
        max_accepts: int,
        workers: int,
        base_model_temperature: float,
        openai_image_long_side: int,
        max_wall_seconds: Optional[float],
        log_level: str,
        scorer_type: str,
) -> None:
    """
    Main entry point for the SVG optimization search.
    Orchestrates initialization and hands off execution to the SearchEngine.
    """
    setup_logger(log_level)

    # 1. Environment & Input Validation
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")
    if not os.path.isfile(image_path):
        raise SystemExit(f"Input image '{image_path}' does not exist.")

    storage.initialize()

    # 2. Image Preparation
    original_img = Image.open(image_path).convert("RGB")
    original_w, original_h = original_img.size

    buf = io.BytesIO()
    original_img.save(buf, format="PNG")
    original_png_bytes = buf.getvalue()

    # Prepare the specific version of the original image the LLM will see
    model_png_bytes = (
        downscale_png_bytes(original_png_bytes, openai_image_long_side)
        if openai_image_long_side > 0
        else original_png_bytes
    )
    openai_original_data_url = png_bytes_to_data_url(model_png_bytes)

    # 3. Scorer & Strategy Initialization
    scorer = get_scorer(scorer_type)
    scoring_ref = scorer.prepare_reference(original_img)

    # Instantiate the 'Brains' of the search
    strategy = GeneticPoolStrategy(
        top_k=3,
        temp_step=0.3,
        max_temp=1.6
    )

    # 4. Resume or Seed Logic (Building the Initial Node Set)
    initial_nodes = []

    # Attempt Resume
    prior_nodes, _, max_id = storage.load_resume_nodes(
        log, base_model_temperature, original_w, original_h, openai_image_long_side
    )
    if prior_nodes:
        log.info(f"Resuming: Recalculating scores for {len(prior_nodes)} nodes...")
        for n in prior_nodes:
            if n.state.svg:
                full_png = rasterize_svg_to_png_bytes(n.state.svg, out_w=original_w, out_h=original_h)
                n.score = n.state.score = scorer.score(scoring_ref, full_png)
        initial_nodes.extend(prior_nodes)

    # Handle Seed SVG if no resume nodes exist or as an addition
    if seed_svg_path:
        try:
            seed_svg = storage.load_seed_svg(seed_svg_path)
            valid, err = is_valid_svg(seed_svg)
            if not valid: raise ValueError(err)

            full_png = rasterize_svg_to_png_bytes(seed_svg, out_w=original_w, out_h=original_h)
            seed_score = scorer.score(scoring_ref, full_png)

            seed_node = SearchNode(
                score=seed_score,
                id=max_id + 1,
                parent_id=0,
                state=ChainState(
                    svg=seed_svg,
                    raster_data_url=None,
                    raster_preview_data_url=make_preview_data_url(full_png, openai_image_long_side),
                    score=seed_score,
                    model_temperature=base_model_temperature,
                    stale_hits=0,
                    invalid_msg=None
                )
            )
            initial_nodes.append(seed_node)
            storage.save_node(seed_node)
        except Exception as e:
            raise SystemExit(f"--seed-svg error: {e}")

    # Fallback to an empty root if nothing else exists
    if not initial_nodes:
        initial_nodes.append(SearchNode(
            score=INVALID_SCORE, id=0, parent_id=0,
            state=ChainState(None, None, None, INVALID_SCORE, base_model_temperature, 0, None)
        ))

    # 5. Execution Engine Setup
    engine = MultiprocessSearchEngine(
        workers=workers,
        strategy=strategy,
        storage=storage,
        scorer_type=scorer_type
    )

    # Pass constant parameters to workers
    worker_params = {
        "openai_original_data_url": openai_original_data_url,
        "original_png_bytes": original_png_bytes,
        "original_w": original_w,
        "original_h": original_h,
        "openai_image_long_side": openai_image_long_side,
        "log_level": log_level,
        "scorer_type": scorer_type,
    }

    # 6. Run Search
    engine.start_workers(worker_params)

    log.info("Starting Search Engine...")
    best_node = engine.run(
        initial_nodes=initial_nodes,
        max_accepts=max_accepts,
        max_wall_seconds=max_wall_seconds,
        openai_image_long_side=openai_image_long_side,
        original_dims=(original_w, original_h)
    )

    # 7. Cleanup & Finalize
    if best_node and best_node.state.svg:
        storage.save_final_svg(best_node.state.svg)
        log.info(f"Search complete. Best score: {best_node.score:.6f}")
    else:
        log.error("Search failed to produce a valid SVG.")