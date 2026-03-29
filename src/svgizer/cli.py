import argparse
import os

from svgizer.score import ScorerType
from svgizer.search import StrategyType

DEFAULT_MAX_EPOCHS = -1
DEFAULT_WORKERS = os.cpu_count() or 4
DEFAULT_MAX_WALL_SECONDS = 60 * 15
DEFAULT_RESUME = False
DEFAULT_WRITE_LINEAGE = True
DEFAULT_IMAGE_LONG_SIDE = 512
DEFAULT_REASONING = "medium"
DEFAULT_LLM_RATE = 1 / (DEFAULT_WORKERS * 5)
DEFAULT_POOL_SIZE = 100
DEFAULT_EPOCH_DIVERSITY = 0.0
DEFAULT_EPOCH_VARIANCE = 0.0


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SVGizer: Evolutionary SVG approximation using Vision LLMs "
            "and pool-based refinement."
        )
    )

    parser.add_argument("image", help="Path to input raster image (PNG/JPEG/WEBP/GIF).")

    parser.add_argument("--output", "-o", default="output.svg", help="Final SVG path.")

    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "gemini", "auto"],
        default="auto",
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
        default=ScorerType.AUTO.value,
        help="Difference scoring backend (vision, simple, llm, or auto).",
    )

    parser.add_argument(
        "--vision-model",
        type=str,
        default="google/siglip-so400m-patch14-384",
        dest="vision_model",
        help="HuggingFace vision model for perceptual scoring.",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=[e.value for e in StrategyType],
        default=StrategyType.NSGA.value,
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
        help="Maximum number of epochs to run (0 = one epoch only, -1 for unlimited).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel worker processes (default: {DEFAULT_WORKERS}, cpu count).",
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
        default=False,
        help="Save a .png alongside each accepted node file (useful for non-SVG formats).",
    )

    parser.add_argument(
        "--llm-rate",
        type=float,
        default=DEFAULT_LLM_RATE,
        help=(
            f"Fraction of tasks (0.0-1.0) that call the LLM; the rest use local "
            f"operations (crossover, mutations). Default: {DEFAULT_LLM_RATE}."
        ),
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help=(
            f"Number of top nodes kept in the active pool for parent selection. "
            f"Default: {DEFAULT_POOL_SIZE}."
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=0,
        dest="seeds",
        help=(
            "Target number of LLM-seeded nodes for epoch 0. "
            "Resumed nodes count toward this. "
            "Defaults to pool-size // 10 when 0."
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
        default=0,
        dest="epoch_seeds",
        help=(
            "Number of Pareto-front nodes carried into each new epoch. "
            "0 defaults to pool-size // 4."
        ),
    )

    parser.add_argument(
        "--epoch-patience",
        type=int,
        default=0,
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
        default=1e-4,
        help=(
            "Minimum absolute score improvement required to reset the patience counter."
        ),
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["svg", "graphviz"],
        default="svg",
        help="Output vector format to generate (default: svg).",
    )

    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    ns = parser.parse_args(args)

    if ns.max_wall_seconds is not None and ns.max_wall_seconds <= 0:
        ns.max_wall_seconds = None

    if ns.max_epochs < -1:
        raise SystemExit("Error: --max-epochs cannot be less than -1")
    if ns.workers <= 0 or ns.pool_size <= 0:
        raise SystemExit("Error: --workers and --pool-size must be > 0")
    if ns.image_long_side < 0:
        raise SystemExit("Error: Configuration values cannot be negative")

    return ns
