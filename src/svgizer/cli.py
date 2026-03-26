import argparse
import os

from svgizer.score import ScorerType
from svgizer.search import StrategyType

DEFAULT_MAX_ACCEPTS = 1000
DEFAULT_WORKERS = os.cpu_count() or 4
DEFAULT_MAX_WALL_SECONDS = 0
DEFAULT_RESUME = False
DEFAULT_WRITE_LINEAGE = True
DEFAULT_IMAGE_LONG_SIDE = 1024
DEFAULT_REASONING = "medium"
DEFAULT_LLM_RATE = 1 / DEFAULT_WORKERS
DEFAULT_POOL_SIZE = 200
DEFAULT_DIVERSITY_THRESHOLD = 0.97
DEFAULT_DIVERSITY_BOOST_THRESHOLD = 0.10


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
        help="Difference scoring backend (dreamsim, simple, llm, or auto).",
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
        "--max-accepts",
        type=int,
        default=DEFAULT_MAX_ACCEPTS,
        help="Number of successful refinements to reach.",
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
        "--warmup-llm",
        type=int,
        default=-1,
        dest="warmup_llm",
        help=(
            "Target number of LLM-seeded nodes before switching to hybrid mode. "
            "Resumed nodes count toward this. "
            "Defaults to pool-size // 10 when -1."
        ),
    )

    parser.add_argument(
        "--diversity-threshold",
        type=float,
        default=DEFAULT_DIVERSITY_THRESHOLD,
        help=(
            f"NCD/edit-distance ratio above which nodes are considered near-duplicates "
            f"during selection filtering. Default: {DEFAULT_DIVERSITY_THRESHOLD}."
        ),
    )

    parser.add_argument(
        "--diversity-boost-threshold",
        type=float,
        default=DEFAULT_DIVERSITY_BOOST_THRESHOLD,
        help=(
            f"Mean NCD threshold below which the pool is considered converged, "
            f"triggering a fresh LLM diversity seed. "
            f"Default: {DEFAULT_DIVERSITY_BOOST_THRESHOLD}."
        ),
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "Stop early if best score does not improve by --min-delta over this many "
            "consecutive tasks. 0 disables early stopping."
        ),
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help=(
            "Minimum absolute score improvement required to reset the patience counter."
        ),
    )

    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    ns = parser.parse_args(args)

    if ns.max_wall_seconds is not None and ns.max_wall_seconds <= 0:
        ns.max_wall_seconds = None

    if ns.max_accepts <= 0 or ns.workers <= 0 or ns.pool_size <= 0:
        raise SystemExit("Error: --max-accepts, --workers, and --pool-size must be > 0")
    if ns.image_long_side < 0:
        raise SystemExit("Error: Configuration values cannot be negative")

    return ns
