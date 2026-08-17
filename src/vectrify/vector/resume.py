import concurrent.futures
import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from PIL import Image

from vectrify.formats.models import VectorStatePayload
from vectrify.image_utils import make_preview_data_url
from vectrify.score.compare import compare
from vectrify.score.complexity import detail_excess
from vectrify.score.edges import overlap_distance
from vectrify.score.metrics import COLOUR, DETAIL, EDGE, SHAPE
from vectrify.score.simple import SimpleFallbackScorer
from vectrify.search import (
    INVALID_SCORE,
    VALID_SCORE,
    ChainState,
    SearchNode,
    StorageAdapter,
)
from vectrify.search.diversity import simhash
from vectrify.search.nsga import build_objectives, pareto_select

if TYPE_CHECKING:
    from vectrify.formats.base import FormatPlugin

log = logging.getLogger(__name__)


@dataclasses.dataclass
class PreppedNode:
    """A resumed candidate, rasterized and measured, ready to be re-scored.

    A dataclass rather than a tuple: the metric values used to be read
    positionally, so registering a metric silently shifted every later index.
    """

    old_id: int
    content: str
    png: bytes
    preview_data_url: str
    metrics: dict[str, float]
    signature: int | None


def prefilter_nodes(
    prepped_nodes: list[PreppedNode],
    original_img: Image.Image,
    max_keep: int,
) -> list[PreppedNode]:
    """Reduce candidates using SimpleFallbackScorer + a front over the metrics.

    Returns at most max_keep entries from the Pareto-optimal front. This selects
    on the same objectives the search itself uses, so a cheap prefilter cannot
    favour candidates the real selection would discard.
    """
    simple_scorer = SimpleFallbackScorer()
    simple_ref = simple_scorer.prepare_reference(original_img)

    simple_scores = []
    for item in prepped_nodes:
        try:
            simple_scores.append(simple_scorer.score(simple_ref, item.png))
        except Exception:
            simple_scores.append(1.0)

    temp_nodes = [
        SearchNode(
            score=simple_scores[i],
            id=i,
            parent_id=0,
            state=ChainState(score=simple_scores[i], payload=None),
            metrics=item.metrics,
        )
        for i, item in enumerate(prepped_nodes)
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
    resolution_llm: int,
    pool_size: int,
    workers: int,
    scoring_ref: Any,
    reference_detail: float,
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

    def _prep(item: tuple) -> PreppedNode:
        old_id, content_text, sig = item
        png = format_plugin.rasterize(content_text, out_w=original_w, out_h=original_h)
        return PreppedNode(
            old_id=old_id,
            content=content_text,
            png=png,
            preview_data_url=make_preview_data_url(png, resolution_llm),
            metrics={},
            signature=sig,
        )

    prepped: list[PreppedNode] = []
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
            f"to {2 * pool_size} using simple scorer + metric front..."
        )
        prepped = prefilter_nodes(prepped, original_img, 2 * pool_size)
        log.info(f"Pre-filter done: {len(prepped)} nodes selected.")

    initial_nodes: list[SearchNode] = []
    current_new_id = 1
    for item in prepped:
        try:
            # Resumed nodes compete directly against freshly scored ones, so
            # they are measured the same way. Leaving the metrics absent would
            # read as 0.0 -- best possible for a minimised objective -- and let
            # every import dominate the candidates actually being measured.
            comparison = compare(scoring_ref, item.png)
            metrics = dict(item.metrics)
            metrics[EDGE] = overlap_distance(
                comparison.reference_edges, comparison.candidate_edges
            )
            metrics[COLOUR] = float(comparison.colour.mean())
            metrics[SHAPE] = comparison.shape
            metrics[DETAIL] = detail_excess(reference_detail, item.png)
            new_score = VALID_SCORE
            node = SearchNode(
                score=new_score,
                id=current_new_id,
                parent_id=0,
                metrics=metrics,
                signature=item.signature,
                state=ChainState(
                    score=new_score,
                    payload=VectorStatePayload(
                        content=item.content,
                        raster_data_url=None,
                        raster_preview_data_url=item.preview_data_url,
                        origin=f"Imported from Node {item.old_id}",
                    ),
                ),
            )
            storage.save_node(node)
            initial_nodes.append(node)
            current_new_id += 1
        except Exception as e:
            log.error(f"Failed to import Node {item.old_id}: {e}")

    return initial_nodes


def filter_to_pool_size(
    nodes: list[SearchNode],
    pool_size: int,
) -> list[SearchNode]:
    """Trim nodes down to pool_size using NSGA Pareto selection."""
    if len(nodes) <= pool_size:
        return nodes

    log.info(f"Filtering {len(nodes)} rescored nodes down to {pool_size}...")

    # Pareto-select among valid nodes only (an infinite score would corrupt the
    # normalization); top up with invalid ones if short, matching the old
    # behavior where they sorted into the last fronts.
    valid = [n for n in nodes if n.score < INVALID_SCORE]
    filtered = pareto_select(valid, build_objectives(valid), pool_size)
    if len(filtered) < pool_size:
        kept_ids = {n.id for n in filtered}
        invalid = [n for n in nodes if n.id not in kept_ids]
        filtered.extend(invalid[: pool_size - len(filtered)])
    return filtered
