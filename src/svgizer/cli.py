import argparse

from svgizer.diff import ScorerType
from svgizer.search import StrategyType

DEFAULT_MAX_ACCEPTS = 32
DEFAULT_WORKERS = 4
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_WALL_SECONDS = 0
DEFAULT_RESUME = True
DEFAULT_WRITE_LINEAGE = True
DEFAULT_IMAGE_LONG_SIDE = 512
DEFAULT_REASONING = "medium"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "SVGizer: Evolutionary SVG approximation using Vision LLMs "
            "and pool-based refinement."
        )
    )

    parser.add_argument("image", help="Path to input raster image (PNG/JPEG/WEBP/GIF).")

    parser.add_argument("--output", "-o", default="output.svg", help="Final SVG path.")
    parser.add_argument(
        "--seed-svg", default=None, help="Path to an SVG file to seed the search pool."
    )

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
        default=StrategyType.GENETIC.value,
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
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--max_wall_seconds",
        type=float,
        default=DEFAULT_MAX_WALL_SECONDS,
        help="Maximum runtime in seconds (0 to disable).",
    )

    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Base LLM temperature for generation.",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        default=DEFAULT_REASONING,
        choices=["none", "low", "medium", "high"],
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
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()

    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        args.max_wall_seconds = None

    if args.max_accepts <= 0 or args.workers <= 0:
        raise SystemExit("Error: --max-accepts and --workers must be > 0")
    if args.temperature < 0 or args.image_long_side < 0:
        raise SystemExit("Error: Configuration values cannot be negative")

    return args