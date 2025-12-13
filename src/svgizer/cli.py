import argparse

# High-level run controls
DEFAULT_MAX_ACCEPTS = 32  # previous default was max_iter(8) * num_beams(4) = 32
DEFAULT_WORKERS = 4

# Proposal controls (how many candidate SVGs we try per accepted step, pipelined in parallel)
DEFAULT_PROPOSALS_PER_STEP = 4
DEFAULT_MODEL_TEMP = 1.0

# Limits / safety
DEFAULT_MAX_TOTAL_TASKS = 10_000
DEFAULT_MAX_WALL_SECONDS = 0  # 0 disables
DEFAULT_RESUME = True

# Outputs
DEFAULT_TOP_K = 4
DEFAULT_WRITE_TOP_K_EACH = 10

# Simulated annealing defaults (score-space)
DEFAULT_ANNEAL_T0 = 0.03
DEFAULT_ANNEAL_ALPHA = 0.995
DEFAULT_ANNEAL_MIN_T = 0.002

# Exploration helper: occasionally propose from best-ever rather than current
DEFAULT_PROPOSE_FROM_BEST_PROB = 0.2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipelined SVG approximation of an image using OpenAI (simulated annealing search)."
    )
    parser.add_argument("image", help="Path to input raster image (PNG/JPEG/WEBP/GIF).")

    parser.add_argument(
        "--output",
        "-o",
        default="output.svg",
        help="Final SVG path (default: output.svg). Intermediate files are written alongside it.",
    )

    # Search controls (simplified)
    parser.add_argument(
        "--max-accepts",
        type=int,
        default=DEFAULT_MAX_ACCEPTS,
        help=f"Stop after this many accepted valid SVGs (default: {DEFAULT_MAX_ACCEPTS}).",
    )
    parser.add_argument(
        "--proposals-per-step",
        type=int,
        default=DEFAULT_PROPOSALS_PER_STEP,
        help=f"How many proposals to try per accepted step (default: {DEFAULT_PROPOSALS_PER_STEP}).",
    )
    parser.add_argument(
        "--model-temp",
        type=float,
        default=DEFAULT_MODEL_TEMP,
        help=f"Base sampling temperature for SVG generation (default: {DEFAULT_MODEL_TEMP}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of multiprocessing workers (default: {DEFAULT_WORKERS}).",
    )

    # Simulated annealing knobs (score-space)
    parser.add_argument(
        "--anneal-t0",
        type=float,
        default=DEFAULT_ANNEAL_T0,
        help=f"Initial simulated annealing temperature in score units (default: {DEFAULT_ANNEAL_T0}).",
    )
    parser.add_argument(
        "--anneal-alpha",
        type=float,
        default=DEFAULT_ANNEAL_ALPHA,
        help=f"Cooling multiplier applied on each accepted move (default: {DEFAULT_ANNEAL_ALPHA}).",
    )
    parser.add_argument(
        "--anneal-min-t",
        type=float,
        default=DEFAULT_ANNEAL_MIN_T,
        help=f"Minimum annealing temperature floor (default: {DEFAULT_ANNEAL_MIN_T}).",
    )
    parser.add_argument(
        "--propose-from-best-prob",
        type=float,
        default=DEFAULT_PROPOSE_FROM_BEST_PROB,
        help=(
            "Probability to propose from best-ever instead of current (default: "
            f"{DEFAULT_PROPOSE_FROM_BEST_PROB})."
        ),
    )

    # Limits / safety
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

    # Resume / outputs
    parser.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RESUME,
        help=f"Resume from previously written *_nodeXXXXX_score*.svg files (default: {DEFAULT_RESUME}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Keep this many best snapshots for periodic TOP-K writing (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--write-top-k-each",
        type=int,
        default=DEFAULT_WRITE_TOP_K_EACH,
        help=f"Write TOP snapshot every N accepted nodes (0 disables). Default: {DEFAULT_WRITE_TOP_K_EACH}.",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )

    args = parser.parse_args()

    # Normalize: 0 means disabled for wall limit
    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        args.max_wall_seconds = None

    # Validation
    if args.max_accepts <= 0:
        raise SystemExit("--max-accepts must be > 0")
    if args.proposals_per_step <= 0:
        raise SystemExit("--proposals-per-step must be > 0")
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")

    if args.max_total_tasks <= 0:
        raise SystemExit("--max-total-tasks must be > 0")

    if args.anneal_t0 < 0:
        raise SystemExit("--anneal-t0 must be >= 0")
    if not (0 < args.anneal_alpha <= 1.0):
        raise SystemExit("--anneal-alpha must be in (0, 1]")
    if args.anneal_min_t < 0:
        raise SystemExit("--anneal-min-t must be >= 0")
    if not (0.0 <= args.propose_from_best_prob <= 1.0):
        raise SystemExit("--propose-from-best-prob must be in [0, 1]")

    if args.model_temp < 0:
        raise SystemExit("--model-temp must be >= 0")

    return args
