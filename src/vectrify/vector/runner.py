import contextlib
import dataclasses
import io
import logging
import multiprocessing as mp
import os
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vectrify.dashboard import Dashboard
    from vectrify.formats.base import SvgBackend
    from vectrify.search.stats import SearchStats

from PIL import Image, UnidentifiedImageError

from vectrify.cli import (
    DEFAULT_CROSSOVER_DISTANCE,
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
    crop_single_color_background,
    downscale_png_bytes,
    png_bytes_to_data_url,
    resize_long_side,
)
from vectrify.llm.models import api_key_env
from vectrify.refine.samvg import (
    SAMVG_MAX_SIDE,
    SAMVG_MODEL,
    SAMVG_POINTS_PER_BATCH,
    generate_svg,
)
from vectrify.score import ScorerType, choose_scorer
from vectrify.score.metrics import (
    FRONT_SCORE,
)
from vectrify.score.segments import (
    save_segments as save_segment_map,
)
from vectrify.score.vision import DEFAULT_VISION_MODEL
from vectrify.search import (
    ChainState,
    SearchNode,
    StorageAdapter,
)
from vectrify.search.collector import StatCollector
from vectrify.search.operators import Exp3Policy
from vectrify.utils import setup_logger, start_log_listener
from vectrify.vector.reference import Reference
from vectrify.vector.resume import filter_to_pool_size, resume_nodes
from vectrify.vector.search import (
    SearchSettings,
    operator_policy,
    run_search,
    seed_node,
)
from vectrify.vector.worker import WorkerContext

log = logging.getLogger("main")


@dataclasses.dataclass(frozen=True)
class VectorSearchConfig:
    """Options for a vector-search run, independent of its input/output wiring.

    ``run_vector_search`` remains the public, CLI-friendly entry point.  This
    object gives programmatic callers and the runner internals a single value
    to pass around as the option set grows.
    """

    resolution_llm: int = DEFAULT_RESOLUTION_LLM
    score_resolution: int | None = None
    edge_tolerance: float | None = None
    write_lineage: bool = True
    save_raster: bool = False
    save_segments: bool = True
    epoch_patience: int | None = None
    pool_size: int = DEFAULT_POOL_SIZE
    seeds: int | None = None
    epoch_max_tasks: int | None = DEFAULT_EPOCH_MAX_TASKS
    epoch_eval_interval: int | None = DEFAULT_EPOCH_EVAL_INTERVAL
    epoch_eval_patience: int | None = DEFAULT_EPOCH_EVAL_PATIENCE
    epoch_improvement: float = DEFAULT_EPOCH_IMPROVEMENT
    epoch_improvement_patience: int = DEFAULT_EPOCH_IMPROVEMENT_PATIENCE
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE
    crossover_distance: int = DEFAULT_CROSSOVER_DISTANCE
    adaptive_operators: bool = True
    epochs: int | None = None
    max_total_tasks: int | None = DEFAULT_MAX_TOTAL_TASKS
    random_seed: int | None = None
    vision_model: str = DEFAULT_VISION_MODEL
    auto_crop: bool = True
    segment_count: int = 8
    samvg_seed: bool = False
    samvg_model: str = SAMVG_MODEL
    samvg_max_side: int = SAMVG_MAX_SIDE
    samvg_points_per_batch: int = SAMVG_POINTS_PER_BATCH
    samvg_min_pixels: int = 32
    samvg_min_impact: float = 3e-6
    samvg_max_layers: int = 512
    samvg_segments: int = 16
    samvg_fill_holes: bool = True
    samvg_hybrid_strokes: bool = True
    samvg_ocr: bool = True
    dry_run: bool = False


def _load_image(
    image_path: str, long_side: int, *, auto_crop: bool = True
) -> tuple[Image.Image, bytes, int, int]:
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
    if auto_crop:
        img = crop_single_color_background(img)
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


def preflight_report(
    *,
    image_size: tuple[int, int],
    segment_count: int,
    scorer_type: ScorerType | str,
    parameters: Mapping[str, Any] | None = None,
) -> str:
    """Report the local runtime features a real run would depend on."""
    lines = [
        "dry_run=true",
        f"image={image_size[0]}x{image_size[1]}",
        f"segments={segment_count}",
        f"scorer_requested={ScorerType(scorer_type).value}",
    ]
    try:
        import torch

        lines.extend(
            [
                f"torch={torch.__version__}",
                f"cuda_available={torch.cuda.is_available()}",
            ]
        )
        if torch.cuda.is_available():
            try:
                torch.empty(1, device="cuda")
                torch.cuda.synchronize()
                lines.extend(
                    [
                        "cuda_test=passed",
                        f"cuda_device={torch.cuda.get_device_name(0)}",
                    ]
                )
            except Exception as exc:
                lines.append(f"cuda_test=failed: {exc}")
        elif hasattr(torch.backends, "mps"):
            lines.append(f"mps_available={torch.backends.mps.is_available()}")
    except ImportError:
        lines.append("torch=unavailable")
    if parameters is not None:
        lines.append("parameters:")
        lines.extend(f"{name}={value!r}" for name, value in sorted(parameters.items()))
    return "\n".join(lines) + "\n"


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
    format_plugin: "SvgBackend",
    resolution_llm: int = DEFAULT_RESOLUTION_LLM,
    score_resolution: int | None = None,
    edge_tolerance: float | None = None,
    write_lineage: bool = True,
    save_raster: bool = False,
    save_segments: bool = True,
    epoch_patience: int | None = None,
    pool_size: int = DEFAULT_POOL_SIZE,
    seeds: int | None = None,
    epoch_max_tasks: int | None = DEFAULT_EPOCH_MAX_TASKS,
    epoch_eval_interval: int | None = DEFAULT_EPOCH_EVAL_INTERVAL,
    epoch_eval_patience: int | None = DEFAULT_EPOCH_EVAL_PATIENCE,
    epoch_improvement: float = DEFAULT_EPOCH_IMPROVEMENT,
    epoch_improvement_patience: int = DEFAULT_EPOCH_IMPROVEMENT_PATIENCE,
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
    crossover_distance: int = DEFAULT_CROSSOVER_DISTANCE,
    adaptive_operators: bool = True,
    epochs: int | None = None,
    max_total_tasks: int | None = DEFAULT_MAX_TOTAL_TASKS,
    random_seed: int | None = None,
    vision_model: str = DEFAULT_VISION_MODEL,  # for the front evaluator
    auto_crop: bool = True,
    segment_count: int = 8,
    samvg_seed: bool = False,
    samvg_model: str = SAMVG_MODEL,
    samvg_max_side: int = SAMVG_MAX_SIDE,
    samvg_points_per_batch: int = SAMVG_POINTS_PER_BATCH,
    samvg_min_pixels: int = 32,
    samvg_min_impact: float = 3e-6,
    samvg_max_layers: int = 512,
    samvg_segments: int = 16,
    samvg_fill_holes: bool = True,
    samvg_hybrid_strokes: bool = True,
    samvg_ocr: bool = True,
    dry_run: bool = False,
    dry_run_parameters: Mapping[str, Any] | None = None,
    stats: "SearchStats | None" = None,
    dashboard: "Dashboard | None" = None,
    config: VectorSearchConfig | None = None,
) -> None:
    # Keep the established keyword-heavy API for the CLI and integrations,
    # while allowing internal/programmatic callers to pass one cohesive option
    # object.  A supplied config deliberately owns every search option.
    config = config or VectorSearchConfig(
        resolution_llm=resolution_llm,
        score_resolution=score_resolution,
        edge_tolerance=edge_tolerance,
        write_lineage=write_lineage,
        save_raster=save_raster,
        save_segments=save_segments,
        epoch_patience=epoch_patience,
        pool_size=pool_size,
        seeds=seeds,
        epoch_max_tasks=epoch_max_tasks,
        epoch_eval_interval=epoch_eval_interval,
        epoch_eval_patience=epoch_eval_patience,
        epoch_improvement=epoch_improvement,
        epoch_improvement_patience=epoch_improvement_patience,
        tournament_size=tournament_size,
        crossover_distance=crossover_distance,
        adaptive_operators=adaptive_operators,
        epochs=epochs,
        max_total_tasks=max_total_tasks,
        random_seed=random_seed,
        vision_model=vision_model,
        auto_crop=auto_crop,
        segment_count=segment_count,
        samvg_seed=samvg_seed,
        samvg_model=samvg_model,
        samvg_max_side=samvg_max_side,
        samvg_points_per_batch=samvg_points_per_batch,
        samvg_min_pixels=samvg_min_pixels,
        samvg_min_impact=samvg_min_impact,
        samvg_max_layers=samvg_max_layers,
        samvg_segments=samvg_segments,
        samvg_fill_holes=samvg_fill_holes,
        samvg_hybrid_strokes=samvg_hybrid_strokes,
        samvg_ocr=samvg_ocr,
        dry_run=dry_run,
    )
    resolution_llm = config.resolution_llm
    score_resolution = config.score_resolution
    edge_tolerance = config.edge_tolerance
    write_lineage = config.write_lineage
    save_raster = config.save_raster
    save_segments = config.save_segments
    epoch_patience = config.epoch_patience
    pool_size = config.pool_size
    seeds = config.seeds
    epoch_max_tasks = config.epoch_max_tasks
    epoch_eval_interval = config.epoch_eval_interval
    epoch_eval_patience = config.epoch_eval_patience
    epoch_improvement = config.epoch_improvement
    epoch_improvement_patience = config.epoch_improvement_patience
    tournament_size = config.tournament_size
    crossover_distance = config.crossover_distance
    adaptive_operators = config.adaptive_operators
    epochs = config.epochs
    max_total_tasks = config.max_total_tasks
    random_seed = config.random_seed
    vision_model = config.vision_model
    auto_crop = config.auto_crop
    segment_count = config.segment_count
    samvg_seed = config.samvg_seed
    samvg_model = config.samvg_model
    samvg_max_side = config.samvg_max_side
    samvg_points_per_batch = config.samvg_points_per_batch
    samvg_min_pixels = config.samvg_min_pixels
    samvg_min_impact = config.samvg_min_impact
    samvg_max_layers = config.samvg_max_layers
    samvg_segments = config.samvg_segments
    samvg_fill_holes = config.samvg_fill_holes
    samvg_hybrid_strokes = config.samvg_hybrid_strokes
    samvg_ocr = config.samvg_ocr
    dry_run = config.dry_run
    epoch_seeds = resolve_seeds(seeds)

    # Validate the reference image up front so a missing or corrupt input fails
    # before storage.initialize() creates the output directory tree.
    original_img, original_png_bytes, original_w, original_h = _load_image(
        image_path, resolution, auto_crop=auto_crop
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

    reference = Reference.build(
        original_img,
        score_resolution=score_resolution,
        edge_tolerance=edge_tolerance,
        segment_count=segment_count,
    )
    if save_segments:
        save_segment_map(
            list(reference.segments), storage.current_run_dir, reference.scoring_image
        )
    log.info(
        "Target attention: %d edge-aware Voronoi mask(s).", len(reference.segments)
    )
    if dry_run:
        report = preflight_report(
            image_size=(original_w, original_h),
            segment_count=len(reference.segments),
            scorer_type=scorer_type,
            parameters=dry_run_parameters,
        )
        (storage.current_run_dir / "dry-run.txt").write_text(report)
        log.info("Dry run complete; no workers or LLM calls were started.")
        log_listener.stop()
        return

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
            reference=reference,
            storage=storage,
        )
        initial_nodes = filter_to_pool_size(initial_nodes, pool_size)

    # This is intentionally an *additional* seed, rather than one of the LLM
    # batch: it gives the search a segmentation-derived structural hypothesis
    # without reducing the configured LLM exploration budget.  It is created
    # in the main process so SAM is loaded once, not once per worker.
    resumed_seed_nodes = list(initial_nodes)
    if samvg_seed:
        try:
            content = format_plugin.extract_from_llm(
                generate_svg(
                    original_img,
                    min_pixels=samvg_min_pixels,
                    min_impact=samvg_min_impact,
                    max_layers=samvg_max_layers,
                    segments=samvg_segments,
                    fill_holes=samvg_fill_holes,
                    hybrid_strokes=samvg_hybrid_strokes,
                    ocr=samvg_ocr,
                    max_side=samvg_max_side,
                    model=samvg_model,
                    points_per_batch=samvg_points_per_batch,
                    rasterize=lambda svg, width, height: format_plugin.rasterize(
                        svg, out_w=width, out_h=height
                    ),
                )
            )
            valid, error = format_plugin.validate(content)
            if not valid:
                raise ValueError(error or "generated SVG failed validation")
            seed = seed_node(
                reference,
                content,
                format_plugin.rasterize(content, out_w=original_w, out_h=original_h),
                node_id=max((node.id for node in initial_nodes), default=0) + 1,
                origin="SAMVG-inspired seed",
                resolution_llm=resolution_llm,
            )
            storage.save_node(seed)
            initial_nodes.append(seed)
            initial_nodes = filter_to_pool_size(initial_nodes, pool_size)
            root = ET.fromstring(content)
            layer_count = sum(
                element.tag.split("}")[-1] == "path" for element in root.iter()
            )
            log.info("Added SAMVG-inspired seed with %d traced layer(s).", layer_count)
        except Exception as exc:
            raise RuntimeError(f"SAMVG-inspired seed generation failed: {exc}") from exc

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

    first_batch = initial_seed_tasks(epoch_seeds, resumed_seed_nodes)
    if first_batch < epoch_seeds:
        log.info(
            f"Epoch 0: {first_batch} LLM seed task(s) "
            f"(batch={epoch_seeds}, already seeded={epoch_seeds - first_batch})"
        )

    # Chosen now rather than at the first epoch boundary. The choice decides
    # what every score in this run means, so a run that cannot have the scorer
    # it asked for should fail before it spends anything, and one that silently
    # degrades has to leave a record something downstream can read.
    choice = choose_scorer(scorer_type, vision_model=vision_model)
    if storage.current_run_dir is not None:
        with contextlib.suppress(OSError):
            (storage.current_run_dir / "scorer.txt").write_text(choice.as_record())
    if choice.degraded:
        log.error(f"This run is NOT comparable with a {choice.requested} run.")

    # The reference is still built on first use: a run that never reaches an
    # epoch boundary never pays for the embedding pass.
    _front: list[Any] = []
    # GPU-bound path fitting happens in workers while the panel/vision
    # evaluator runs in the main process. One shared gate makes the device a
    # bounded resource instead of multiplying its memory footprint by the CPU
    # worker count.
    gpu_gate = mp.get_context("spawn").Semaphore(1)
    format_plugin.gpu_gate = gpu_gate

    def _front_scorer() -> tuple[Any, Any]:
        if not _front:
            _front.extend(
                [choice.scorer, choice.scorer.prepare_reference(original_img)]
            )
        return _front[0], _front[1]

    def rank_front(nodes: list[SearchNode]) -> list[SearchNode]:
        gpu_gate.acquire()
        try:
            return evaluate_front(
                nodes,
                front_scorer=_front_scorer,
                format_plugin=format_plugin,
                out_w=original_w,
                out_h=original_h,
            )
        finally:
            gpu_gate.release()

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

    if dashboard is not None:
        logging.getLogger().addHandler(dashboard.log_handler)

    dashboard_entered = False
    try:
        if dashboard is not None:
            dashboard.__enter__()
            dashboard_entered = True

        policy = operator_policy(format_plugin, adaptive_operators)
        run_search(
            reference,
            initial_nodes,
            worker_ctx,
            SearchSettings(
                workers=workers,
                pool_size=pool_size,
                tournament_size=tournament_size,
                crossover_distance=crossover_distance,
                adaptive_operators=adaptive_operators,
                epoch_seeds=epoch_seeds,
                initial_seeds=first_batch,
                epochs=epochs,
                epoch_patience=epoch_patience,
                epoch_max_tasks=epoch_max_tasks,
                epoch_eval_interval=epoch_eval_interval,
                epoch_eval_patience=epoch_eval_patience,
                epoch_improvement=epoch_improvement,
                epoch_improvement_patience=epoch_improvement_patience,
                max_total_tasks=max_total_tasks,
                max_wall_seconds=max_wall_seconds,
                resolution_llm=resolution_llm,
                write_lineage=write_lineage,
                save_raster=save_raster,
            ),
            storage=storage,
            rank_front=rank_front,
            policy=policy,
            collector=collector,
        )

        # Repeated here because the warning at startup is thousands of lines
        # above the number it invalidates.
        log.info(f"Run scored with: {choice.summary()}")

        if isinstance(policy, Exp3Policy):
            probs = policy.probabilities()
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
