import concurrent.futures
import logging
from typing import TYPE_CHECKING, Any

from PIL import Image

from vectrify.formats.models import VectorStatePayload
from vectrify.image_utils import make_preview_data_url
from vectrify.score.complexity import structural_complexity, visual_complexity
from vectrify.score.simple import SimpleFallbackScorer
from vectrify.search import (
    INVALID_SCORE,
    ChainState,
    SearchNode,
    StorageAdapter,
    StrategyType,
)
from vectrify.search.diversity import simhash
from vectrify.search.nsga import build_objectives, pareto_select

if TYPE_CHECKING:
    from vectrify.formats.base import FormatPlugin

log = logging.getLogger(__name__)


def prefilter_nodes(
    prepped_nodes: list,
    original_img: Image.Image,
    max_keep: int,
) -> list:
    """Reduce candidates using SimpleFallbackScorer + complexity Pareto front.

    Each entry in prepped_nodes is a tuple of (old_id, content_text, png_bytes,
    preview_data_url, visual_complexity, structural_complexity, signature).
    Returns at most max_keep entries from the Pareto-optimal front.

    This selects on the same three objectives the search itself uses, so a
    cheap prefilter cannot favour candidates the real selection would discard.
    """
    simple_scorer = SimpleFallbackScorer()
    simple_ref = simple_scorer.prepare_reference(original_img)

    simple_scores = []
    for _, _, png, _, _, _, _ in prepped_nodes:
        try:
            simple_scores.append(simple_scorer.score(simple_ref, png))
        except Exception:
            simple_scores.append(1.0)

    temp_nodes = [
        SearchNode(
            score=simple_scores[i],
            id=i,
            parent_id=0,
            state=ChainState(score=simple_scores[i], payload=None),
            visual_complexity=prepped_nodes[i][4],
            structural_complexity=prepped_nodes[i][5],
        )
        for i in range(len(prepped_nodes))
    ]
    objectives = build_objectives(temp_nodes)
    kept = pareto_select(temp_nodes, objectives, max_keep)
    return [prepped_nodes[node.id] for node in kept]


def resume_nodes(
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
    """Deduplicate, rasterize, pre-filter, and re-score a set of resumed nodes.

    Saves each accepted node to storage and returns the resulting SearchNode list.
    """
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
        return (
            old_id,
            content_text,
            png,
            preview,
            visual_complexity(png),
            structural_complexity(content_text),
            sig,
        )

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
        prepped = prefilter_nodes(prepped, original_img, 2 * pool_size)
        log.info(f"Pre-filter done: {len(prepped)} nodes selected.")

    initial_nodes: list[SearchNode] = []
    current_new_id = 1
    for old_id, content_text, png, preview, vis_comp, str_comp, sig in prepped:
        try:
            new_score = scorer.score(scoring_ref, png)
            node = SearchNode(
                score=new_score,
                id=current_new_id,
                parent_id=0,
                visual_complexity=vis_comp,
                structural_complexity=str_comp,
                signature=sig,
                state=ChainState(
                    score=new_score,
                    payload=VectorStatePayload(
                        content=content_text,
                        raster_data_url=None,
                        raster_preview_data_url=preview,
                        origin=f"Imported from Node {old_id}",
                    ),
                ),
            )
            storage.save_node(node)
            initial_nodes.append(node)
            current_new_id += 1
        except Exception as e:
            log.error(f"Failed to import Node {old_id}: {e}")

    return initial_nodes


def filter_to_pool_size(
    nodes: list[SearchNode],
    pool_size: int,
    strategy_type: StrategyType,
) -> list[SearchNode]:
    """Trim nodes down to pool_size using NSGA Pareto selection or score sorting."""
    if len(nodes) <= pool_size:
        return nodes

    log.info(f"Filtering {len(nodes)} rescored nodes down to {pool_size}...")

    if strategy_type == StrategyType.NSGA:
        # Pareto-select among valid nodes only (an infinite score would
        # corrupt the normalization); top up with invalid ones if short,
        # matching the old behavior where they sorted into the last fronts.
        valid = [n for n in nodes if n.score < INVALID_SCORE]
        filtered = pareto_select(valid, build_objectives(valid), pool_size)
        if len(filtered) < pool_size:
            kept_ids = {n.id for n in filtered}
            invalid = [n for n in nodes if n.id not in kept_ids]
            filtered.extend(invalid[: pool_size - len(filtered)])
        return filtered

    return sorted(nodes, key=lambda n: n.score)[:pool_size]
