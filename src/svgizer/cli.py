import argparse

DEFAULT_MAX_ACCEPTS = 32
DEFAULT_WORKERS = 4
DEFAULT_MODEL_TEMP = 0.2

DEFAULT_MAX_TOTAL_TASKS = 10_000
DEFAULT_MAX_WALL_SECONDS = 0  # 0 disables
DEFAULT_RESUME = True

DEFAULT_TOP_K = 3
DEFAULT_WRITE_LINEAGE = True

# Parent selection:
# With probability p(progress), pick a parent from the current top-K pool (elite selection).
# p decays from START -> END as accepted_valid approaches max_accepts.
DEFAULT_ELITE_PROB_START = 0.70
DEFAULT_ELITE_PROB_END = 0.10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipelined SVG approximation of an image using OpenAI (pool-based refinement)."
    )
    parser.add_argument("image", help="Path to input raster image (PNG/JPEG/WEBP/GIF).")

    parser.add_argument(
        "--output",
        "-o",
        default="output.svg",
        help="Final SVG path (default: output.svg). Intermediate files are written alongside it.",
    )

    parser.add_argument(
        "--seed-svg",
        default=None,
        help=(
            "Path to an SVG file to seed the pool / resume from. "
            "If provided, it is rasterized and scored against the input image and becomes the initial best node."
        ),
    )

    parser.add_argument(
        "--max-accepts",
        type=int,
        default=DEFAULT_MAX_ACCEPTS,
        help=f"Stop after this many accepted valid SVGs (default: {DEFAULT_MAX_ACCEPTS}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of multiprocessing workers (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--model-temp",
        type=float,
        default=DEFAULT_MODEL_TEMP,
        help=f"Base sampling temperature for SVG generation (default: {DEFAULT_MODEL_TEMP}).",
    )

    parser.add_argument(
        "--max-total-tasks",
        type=int,
        default=DEFAULT_MAX_TOTAL_TASKS,
        help=f"Hard cap on total OpenAI calls enqueued (default: {DEFAULT_MAX_TOTAL_TASKS}).",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=DEFAULT_MAX_WALL_SECONDS,
        help=f"Wall-clock time limit in seconds (default: {DEFAULT_MAX_WALL_SECONDS}; 0 disables).",
    )

    parser.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RESUME,
        help=f"Resume from previously written node files (default: {DEFAULT_RESUME}).",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Maintain this many best nodes in-memory for parent selection (default: {DEFAULT_TOP_K}).",
    )

    parser.add_argument(
        "--elite-prob-start",
        type=float,
        default=DEFAULT_ELITE_PROB_START,
        help=f"Probability of choosing a parent from top-K at the start (default: {DEFAULT_ELITE_PROB_START}).",
    )
    parser.add_argument(
        "--elite-prob-end",
        type=float,
        default=DEFAULT_ELITE_PROB_END,
        help=f"Probability of choosing a parent from top-K near the end (default: {DEFAULT_ELITE_PROB_END}).",
    )

    parser.add_argument(
        "--write-lineage",
        dest="write_lineage",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WRITE_LINEAGE,
        help=f"Write lineage TSV/DOT files (default: {DEFAULT_WRITE_LINEAGE}).",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )

    args = parser.parse_args()

    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        args.max_wall_seconds = None

    if args.max_accepts <= 0:
        raise SystemExit("--max-accepts must be > 0")
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")
    if args.model_temp < 0:
        raise SystemExit("--model-temp must be >= 0")
    if args.max_total_tasks <= 0:
        raise SystemExit("--max-total-tasks must be > 0")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")

    if not (0.0 <= args.elite_prob_start <= 1.0):
        raise SystemExit("--elite-prob-start must be in [0, 1]")
    if not (0.0 <= args.elite_prob_end <= 1.0):
        raise SystemExit("--elite-prob-end must be in [0, 1]")

    return args
