import io
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vectrify.dashboard import Dashboard
    from vectrify.formats.base import FormatPlugin
    from vectrify.search.stats import SearchStats

from PIL import Image, UnidentifiedImageError

from vectrify.cli import (
    DEFAULT_EPOCH_EVAL_INTERVAL,
    DEFAULT_EPOCH_EVAL_PATIENCE,
    DEFAULT_EPOCH_IMPROVEMENT,
    DEFAULT_EPOCH_IMPROVEMENT_PATIENCE,
    DEFAULT_EPOCH_MAX_TASKS,
    DEFAULT_MAX_TOTAL_TASKS,
    DEFAULT_POOL_SIZE,
    DEFAULT_RESOLUTION_LLM,
    DEFAULT_SEEDS,
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
from vectrify.score.base import DEFAULT_CONFIG
from vectrify.score.compare import compare, prepare
from vectrify.score.complexity import detail, detail_excess
from vectrify.score.edges import overlap_distance
from vectrify.score.metrics import (
    COLOUR,
    DETAIL,
    EDGE,
    FRONT_SCORE,
    SHAPE,
)
from vectrify.score.utils import MAX_SCORE
from vectrify.score.vision import DEFAULT_VISION_MODEL
from vectrify.search import (
    ChainState,
    MultiprocessSearchEngine,
    NsgaStrategy,
    SearchNode,
    StorageAdapter,
)
from vectrify.search.collector import StatCollector
from vectrify.search.operators import Exp3Policy, FixedWeightPolicy
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


def resolve_seeds(seeds: int | None) -> int:
    """LLM calls that open each epoch; None takes the default.

    It used to derive from the pool size, which tied the LLM budget to a number
    chosen for entirely separate reasons -- widening the pool silently bought
    more LLM calls.
    """
    return DEFAULT_SEEDS if seeds is None else max(0, seeds)


def initial_seed_tasks(epoch_seeds: int, initial_nodes: list[SearchNode]) -> int:
    """Epoch 0's batch size, discounted by candidates already carried in.

    A resumed node is a seed that has already been paid for, so a resume that
    restores a full batch should spend nothing on generating another.
    """
    seeded = sum(1 for n in initial_nodes if n.state.payload.content)
    return max(0, epoch_seeds - seeded)


def evaluate_front(
    nodes: list[SearchNode],
    *,
    front_scorer: Callable[[], tuple[Any, Any]],
    format_plugin: Any,
    out_w: int,
    out_h: int,
) -> list[SearchNode]:
    """Order *nodes* by the evaluator, best first, scoring only what is new.

    *front_scorer* is called for (scorer, reference) and only when there is
    something to score, so a call the cache answers in full never builds a
    model.

    Re-rasterises rather than reading a node's stored render, which is only
    kept when --write-lineage or --save-raster is on.
    """
    renders: list[tuple[bytes, SearchNode]] = []
    for node in nodes:
        # Already judged, and the judgement travels: the panel's score is a
        # calibrated distance to the target, so it means the same thing in
        # every call. Re-rasterising and re-embedding a node the evaluator has
        # already seen would buy an identical number at full price -- and a run
        # asks about the same pool members repeatedly.
        if FRONT_SCORE in node.metrics:
            continue
        content = getattr(node.state.payload, "content", None)
        if not content:
            continue
        try:
            renders.append(
                (format_plugin.rasterize(content, out_w=out_w, out_h=out_h), node)
            )
        except Exception as exc:
            log.debug(f"Front evaluation skipped node {node.id}: {exc}")

    if renders:
        scorer, ref = front_scorer()
        pngs = [png for png, _ in renders]
        try:
            values = scorer.rank(ref, pngs)
        except AttributeError:
            values = [scorer.score(ref, png) for png in pngs]
        except Exception as exc:
            log.warning(f"Front evaluation failed, keeping rank order: {exc}")
            return nodes

        for value, (_png, node) in zip(values, renders, strict=True):
            node.metrics[FRONT_SCORE] = value

    # Every node the panel has ever scored, freshly measured or recalled.
    scored = [
        (node.metrics[FRONT_SCORE], node)
        for node in nodes
        if FRONT_SCORE in node.metrics
    ]
    if not scored:
        return nodes
    scored.sort(key=lambda pair: pair[0])
    log.info(
        f"Front evaluated: {len(scored)} candidate(s) "
        f"({len(renders)} newly scored), "
        f"best {scored[0][0]:.6f}, worst {scored[-1][0]:.6f}"
    )
    return [node for _value, node in scored]


def run_vector_search(
    image_path: str,
    storage: StorageAdapter,
    workers: int,
    resolution: int,
    max_wall_seconds: float | None,
    log_level: str,
    # Selects the evaluator that ranks the converged Pareto front, not the
    # evaluator's scorer -- the per-candidate measures are always pixel work.
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
    pool_size: int = DEFAULT_POOL_SIZE,
    seeds: int | None = None,
    epoch_max_tasks: int | None = DEFAULT_EPOCH_MAX_TASKS,
    epoch_eval_interval: int | None = DEFAULT_EPOCH_EVAL_INTERVAL,
    epoch_eval_patience: int | None = DEFAULT_EPOCH_EVAL_PATIENCE,
    epoch_improvement: float = DEFAULT_EPOCH_IMPROVEMENT,
    epoch_improvement_patience: int = DEFAULT_EPOCH_IMPROVEMENT_PATIENCE,
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
    adaptive_operators: bool = True,
    epochs: int | None = None,
    max_total_tasks: int | None = DEFAULT_MAX_TOTAL_TASKS,
    random_seed: int | None = None,
    vision_model: str = DEFAULT_VISION_MODEL,  # for the front evaluator
    stats: "SearchStats | None" = None,
    dashboard: "Dashboard | None" = None,
) -> None:
    epoch_seeds = resolve_seeds(seeds)

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

    # A small encoder rather than a pixel measure: where selection decides, one
    # mutation from the parent, it is right about its accepted mutations far
    # more often, and that ratio is what the search compounds. The configured
    # --scorer still selects the evaluator that ranks a converged front.
    # Sized against the scorer thread's own work rather than the worker count:
    # it is one batch of candidates at a time, and oversubscribing here would
    # only take cores from the workers producing them.
    pixel_pool = ThreadPoolExecutor(
        max_workers=min(8, (os.cpu_count() or 4)), thread_name_prefix="pixel"
    )

    scoring_img = resize_long_side(original_img, DEFAULT_CONFIG.target_long_side)
    pixel_ref = prepare(scoring_img)
    # The target's own detail, measured once, at the size candidates are
    # rasterized at -- original_png_bytes, not the smaller image the pixel
    # comparison resizes to. Compressed size grows with pixel count, so reading
    # the reference at scoring resolution and candidates at render resolution
    # charges every candidate for the difference between the two: measured on
    # the duck, the same image reads 7,607 at 256 and 41,602 at 700, and every
    # candidate in a run came out 58-142% "busier" than a target it matched.
    reference_detail = detail(original_png_bytes)
    log.info(
        "Measures: edge overlap, colour distance, shape moments and a detail "
        "budget, traded off by dominance, no model. "
        f"Front evaluator: {ScorerType(scorer_type).value} ({vision_model})."
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
            scoring_ref=pixel_ref,
            reference_detail=reference_detail,
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
                valid=False,
                id=0,
                parent_id=0,
                state=ChainState(
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
            epoch_max_tasks=epoch_max_tasks,
            epoch_patience=epoch_patience,
            eval_patience=epoch_eval_patience,
            epochs=epochs,
        )

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
        return evaluate_front(
            nodes,
            front_scorer=_front_scorer,
            format_plugin=format_plugin,
            out_w=original_w,
            out_h=original_h,
        )

    engine = MultiprocessSearchEngine(
        workers=workers,
        strategy=NsgaStrategy[VectorStatePayload](
            pool_size=pool_size,
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
        source_name=Path(image_path).name,
        llm_provider=llm_provider,
        llm_model=llm_model,
        reasoning=reasoning,
        api_key=api_key,
        random_seed=random_seed,
        log_queue=log_queue,
    )

    def _pixel_objectives(res) -> None:
        png = res.payload.raster_png
        if not png:
            return
        try:
            comparison = compare(pixel_ref, png)
            res.metrics[EDGE] = overlap_distance(
                comparison.reference_edges, comparison.candidate_edges
            )
            res.metrics[COLOUR] = float(comparison.colour.mean())
            res.metrics[SHAPE] = comparison.shape
            res.metrics[DETAIL] = detail_excess(reference_detail, png)
            # Measured, so valid. `score` carries no magnitude any more: the
            # measures are ranked by dominance and the only score in the run is
            # the evaluator's, recorded as FRONT_SCORE on the nodes it sees.
            res.measured = True
        except Exception as exc:
            log.debug(f"Pixel objectives skipped: {exc}")

    def score_fn(results):
        # Two measures of different kinds, chromatic and structural, and no
        # model: against damage of a known severity the pair orders candidates
        # better than any embedding configuration tried, at no forward pass.
        #
        # Spread across threads because they run on the one scorer thread, in a
        # decode-resize-convolve pass per candidate that is where a run's
        # throughput was going: profiled, workers sat idle above four of them
        # while this serialised. The work is numpy and Pillow, both of which
        # drop the GIL, so threads genuinely overlap here.
        if len(results) > 1:
            list(pixel_pool.map(_pixel_objectives, results))
        else:
            for res in results:
                _pixel_objectives(res)

        for res in results:
            if not res.measured:
                # Nothing rendered, so nothing can be measured.
                res.score = MAX_SCORE

    if dashboard is not None:
        logging.getLogger().addHandler(dashboard.log_handler)

    dashboard_entered = False
    try:
        if dashboard is not None:
            dashboard.__enter__()
            dashboard_entered = True

        weights = format_plugin.mutation_weights()
        operator_policy = (
            Exp3Policy(
                weights, reward_scale=format_plugin.operator_reward_scale()
            )
            if adaptive_operators
            else FixedWeightPolicy(weights)
        )

        engine.start_workers(worker_loop, worker_ctx)

        engine.run(
            initial_nodes,
            max_wall_seconds=max_wall_seconds,
            epoch_patience=epoch_patience,
            active_pool_size=pool_size,
            score_fn=score_fn,
            epoch_seeds=epoch_seeds,
            initial_seeds=first_batch,
            epochs=epochs,
            epoch_max_tasks=epoch_max_tasks,
            epoch_eval_interval=epoch_eval_interval,
            epoch_eval_patience=epoch_eval_patience,
            epoch_improvement=epoch_improvement,
            epoch_improvement_patience=epoch_improvement_patience,
            operator_policy=operator_policy,
            collector=collector,
        )

        if isinstance(operator_policy, Exp3Policy):
            probs = operator_policy.probabilities()
            log.info(
                "Final operator mix: "
                + ", ".join(
                    f"{name}={p:.2f}"
                    for name, p in sorted(probs.items(), key=lambda kv: -kv[1])
                )
            )
    finally:
        log_listener.stop()
        if dashboard is not None and dashboard_entered:
            dashboard.__exit__(None, None, None)
        if dashboard is not None:
            logging.getLogger().removeHandler(dashboard.log_handler)
