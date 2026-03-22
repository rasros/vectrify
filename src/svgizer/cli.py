import argparse

from svgizer.diff import ScorerType

DEFAULT_MAX_ACCEPTS = 32
DEFAULT_WORKERS = 4
DEFAULT_MODEL_TEMP = 0.6
DEFAULT_MAX_WALL_SECONDS = 0  # 0 disables
DEFAULT_RESUME = True
DEFAULT_WRITE_LINEAGE = True
DEFAULT_OPENAI_IMAGE_LONG_SIDE = 512


def parse_args():
    parser = argparse.ArgumentParser(
        description="SVGizer: Evolutionary SVG approximation using Vision LLMs and pool-based refinement."
    )

    # Positional
    parser.add_argument("image", help="Path to input raster image (PNG/JPEG/WEBP/GIF).")

    # Output & Input
    parser.add_argument("--output", "-o", default="output.svg", help="Final SVG path.")
    parser.add_argument(
        "--seed-svg", default=None, help="Path to an SVG file to seed the search pool."
    )

    # Scorer Configuration
    parser.add_argument(
        "--scorer",
        type=str,
        choices=[e.value for e in ScorerType],
        default=ScorerType.AUTO.value,
        help="Difference scoring backend (dreamsim, simple, llm, or auto).",
    )

    # Search Constraints
    parser.add_argument("--max-accepts", type=int, default=DEFAULT_MAX_ACCEPTS,
                        help="Number of successful refinements to reach.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Number of parallel worker processes.")
    parser.add_argument("--max-wall-seconds", type=float, default=DEFAULT_MAX_WALL_SECONDS,
                        help="Maximum runtime in seconds (0 to disable).")

    # Model Parameters
    parser.add_argument("--model-temp", type=float, default=DEFAULT_MODEL_TEMP,
                        help="Base LLM temperature.")
    parser.add_argument(
        "--openai-image-long-side", type=int, default=DEFAULT_OPENAI_IMAGE_LONG_SIDE,
        help="Downscale reference/preview images to this long-side dimension."
    )

    # State Management
    parser.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RESUME,
        help="Resume search from existing nodes in the output directory."
    )
    parser.add_argument(
        "--write-lineage",
        dest="write_lineage",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WRITE_LINEAGE,
        help="Write a CSV and directory of all accepted SVG nodes."
    )

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Logic conversion for wall time
    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        args.max_wall_seconds = None

    # Bounds validation
    if args.max_accepts <= 0 or args.workers <= 0:
        raise SystemExit("Error: --max-accepts and --workers must be > 0")
    if args.model_temp < 0 or args.openai_image_long_side < 0:
        raise SystemExit("Error: Configuration values cannot be negative")

    return args