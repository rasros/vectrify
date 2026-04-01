import argparse
import os

from svgizer.score import ScorerType
from svgizer.search import StrategyType

DEFAULT_OUTPUT = "output.svg"
DEFAULT_PROVIDER = "auto"
DEFAULT_SCORER = "auto"
DEFAULT_VISION_MODEL = "google/siglip-so400m-patch14-384"
DEFAULT_STRATEGY = "nsga"
BEAM_ONLY_PARAMS = {"beams", "cull_keep"}
NSGA_ONLY_PARAMS = {"epoch_diversity", "epoch_variance", "epoch_seeds"}
DEFAULT_MAX_EPOCHS = 1
DEFAULT_WORKERS = os.cpu_count() or 4
DEFAULT_MAX_WALL_SECONDS = 60 * 60
DEFAULT_RESUME = False
DEFAULT_WRITE_LINEAGE = True
DEFAULT_SAVE_RASTER = True
DEFAULT_SAVE_HEATMAP = False
DEFAULT_IMAGE_LONG_SIDE = 512
DEFAULT_REASONING = "medium"
DEFAULT_LLM_RATE = 1 / DEFAULT_WORKERS
DEFAULT_POOL_SIZE = 100
DEFAULT_SEEDS = 0
DEFAULT_BEAMS = 10
DEFAULT_CULL_KEEP = 0.5
DEFAULT_EPOCH_DIVERSITY = 0.0
DEFAULT_EPOCH_VARIANCE = 0.0
DEFAULT_EPOCH_SEEDS = 0
DEFAULT_EPOCH_PATIENCE = 0
DEFAULT_EPOCH_MIN_DELTA = 1e-4
DEFAULT_EPOCH_STEPS = 0
DEFAULT_FORMAT = "svg"
DEFAULT_LOG_LEVEL = "INFO"


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SVGizer: Evolutionary SVG approximation using Vision LLMs "
            "and pool-based refinement."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("image", help="Path to input raster image (PNG/JPEG/WEBP/GIF).")

    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT, help="Final SVG path."
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "gemini", "auto"],
        default=DEFAULT_PROVIDER,
        help="LLM provider to use. 'auto' checks environment variables.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name. Defaults depend on the active provider.",
    )

    parser.add_argument(
        "--scorer",
        type=str,
        choices=[e.value for e in ScorerType],
        default=DEFAULT_SCORER,
        help="Difference scoring backend (vision, simple, llm, or auto).",
    )

    parser.add_argument(
        "--vision-model",
        type=str,
        default=DEFAULT_VISION_MODEL,
        dest="vision_model",
        help="HuggingFace vision model for perceptual scoring.",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=[e.value for e in StrategyType],
        default=DEFAULT_STRATEGY,
        help="Search strategy/evolution algorithm to use.",
    )

    parser.add_argument(
        "--goal",
        default=None,
        help=(
            "Custom prompt/goal to guide the SVG generation "
            "(e.g., 'Make lines thicker')."
        ),
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        dest="max_epochs",
        help="Maximum number of epochs to run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel worker processes (cpu count).",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=DEFAULT_MAX_WALL_SECONDS,
        help="Maximum runtime in seconds (0 to disable).",
    )

    parser.add_argument(
        "--reasoning",
        type=str,
        default=DEFAULT_REASONING,
        help="Reasoning effort level across supported LLMs.",
    )
    parser.add_argument(
        "--image-long-side",
        type=int,
        default=DEFAULT_IMAGE_LONG_SIDE,
        help="Downscale reference/preview images to this long-side dimension.",
    )

    parser.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RESUME,
        help="Resume search from existing nodes in the output directory.",
    )
    parser.add_argument(
        "--resume-top",
        type=int,
        default=None,
        dest="resume_top",
        help="When resuming, keep only the N best-scoring nodes. Default: load all.",
    )
    parser.add_argument(
        "--write-lineage",
        dest="write_lineage",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WRITE_LINEAGE,
        help="Write a CSV and directory of all accepted SVG nodes.",
    )
    parser.add_argument(
        "--save-raster",
        dest="save_raster",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_RASTER,
        help="Save a .png alongside each accepted node file.",
    )
    parser.add_argument(
        "--save-heatmap",
        dest="save_heatmap",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_HEATMAP,
        help="Save a .heatmap.png perceptual diff alongside each accepted node file.",
    )

    parser.add_argument(
        "--llm-rate",
        type=float,
        default=DEFAULT_LLM_RATE,
        help="Fraction of tasks (0.0-1.0) that call the LLM;"
        " the rest use local operations (crossover, mutations).",
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help="Number of top nodes kept in the active pool for parent selection.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        dest="seeds",
        help=(
            "Target number of LLM-seeded nodes for epoch 0. "
            "Resumed nodes count toward this. "
            "Defaults to pool-size // 10 when 0."
        ),
    )

    parser.add_argument(
        "--beams",
        type=int,
        default=DEFAULT_BEAMS,
        help=(
            "Number of beams (parallel hill-climbers) for the beam strategy. "
            "Each epoch starts with this many fresh LLM seeds. "
            "Ignored by the nsga strategy."
        ),
    )

    parser.add_argument(
        "--cull-keep",
        type=float,
        default=DEFAULT_CULL_KEEP,
        dest="cull_keep",
        help=(
            "Fraction of beams eligible for expansion in the beam strategy. "
            "Only the top scoring fraction are mutated/LLM-edited; the rest "
            "starve and are evicted by better candidates. "
            "1.0 disables culling. Ignored by nsga."
        ),
    )

    parser.add_argument(
        "--epoch-diversity",
        type=float,
        default=DEFAULT_EPOCH_DIVERSITY,
        dest="epoch_diversity",
        help=(
            "Trigger an epoch transition when mean pairwise genome diversity "
            "drops below this threshold. 0 disables diversity-based epoch transitions."
        ),
    )

    parser.add_argument(
        "--epoch-variance",
        type=float,
        default=DEFAULT_EPOCH_VARIANCE,
        dest="epoch_variance",
        help=(
            "Trigger an epoch transition when the std dev of scores in the active pool "
            "drops below this threshold (pool converged). "
            "0 disables variance-based epoch transitions."
        ),
    )

    parser.add_argument(
        "--epoch-seeds",
        type=int,
        default=DEFAULT_EPOCH_SEEDS,
        dest="epoch_seeds",
        help=(
            "Number of Pareto-front nodes carried into each new epoch. "
            "0 defaults to pool-size // 4."
        ),
    )

    parser.add_argument(
        "--epoch-patience",
        type=int,
        default=DEFAULT_EPOCH_PATIENCE,
        dest="epoch_patience",
        help=(
            "End the current epoch and re-seed from the Pareto front if the best score "
            "does not improve by --epoch-min-delta over this many consecutive tasks."
            "0 disables staleness-based epoch transitions."
        ),
    )
    parser.add_argument(
        "--epoch-min-delta",
        type=float,
        default=DEFAULT_EPOCH_MIN_DELTA,
        help=(
            "Minimum absolute score improvement required to reset the patience counter."
        ),
    )
    parser.add_argument(
        "--epoch-steps",
        type=int,
        default=DEFAULT_EPOCH_STEPS,
        dest="epoch_steps",
        help=(
            "Maximum number of completed tasks per epoch before forcing an epoch "
            "transition. 0 means unlimited."
        ),
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["svg", "graphviz", "typst"],
        default=DEFAULT_FORMAT,
        help="Output vector format to generate.",
    )

    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    ns = parser.parse_args(args)

    if ns.max_wall_seconds is not None and ns.max_wall_seconds <= 0:
        ns.max_wall_seconds = None

    if ns.max_epochs < 1:
        raise SystemExit("Error: --max-epochs must be at least 1")
    if ns.workers <= 0 or ns.pool_size <= 0:
        raise SystemExit("Error: --workers and --pool-size must be > 0")
    if ns.image_long_side < 0:
        raise SystemExit("Error: Configuration values cannot be negative")

    if ns.resume_top is not None:
        ns.resume = True

    def _flags(params):
        return ", ".join("--" + p.replace("_", "-") for p in sorted(params))

    is_beam = ns.strategy == StrategyType.BEAM.value
    if is_beam:
        nsga_set = {
            p for p in NSGA_ONLY_PARAMS if getattr(ns, p, None) not in (None, 0, 0.0)
        }
        if nsga_set:
            raise SystemExit(
                f"Error: {_flags(nsga_set)} are nsga-only parameters"
                " and cannot be used with --strategy beam."
            )
    else:
        beam_defaults = {"beams": DEFAULT_BEAMS, "cull_keep": DEFAULT_CULL_KEEP}
        beam_set = {p for p in BEAM_ONLY_PARAMS if getattr(ns, p) != beam_defaults[p]}
        if beam_set:
            raise SystemExit(
                f"Error: {_flags(beam_set)} are beam-only parameters"
                " and cannot be used with --strategy nsga."
            )

    return ns
