import argparse
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from vectrify.formats import FORMAT_NAMES
from vectrify.score import ScorerType
from vectrify.score.vision import DEFAULT_VISION_MODEL

DEFAULT_OUTPUT = "output.svg"
DEFAULT_PROVIDER = "auto"
DEFAULT_SCORER = "auto"
DEFAULT_EPOCHS = 2
DEFAULT_WORKERS = os.cpu_count() or 4
DEFAULT_MAX_WALL_SECONDS = 60 * 60
DEFAULT_RESUME = False
DEFAULT_WRITE_LINEAGE = True
DEFAULT_SAVE_RASTER = True
DEFAULT_SAVE_HEATMAP = False
DEFAULT_DASHBOARD = True
DEFAULT_RESOLUTION = 768
DEFAULT_RESOLUTION_LLM = 512
DEFAULT_REASONING = "medium"

DEFAULT_POOL_SIZE = 100
DEFAULT_EPOCH_DIVERSITY = 0.0
DEFAULT_EPOCH_VARIANCE = 0.0
DEFAULT_EPOCH_PATIENCE = 200
DEFAULT_EPOCH_MIN_DELTA = 1e-4
DEFAULT_TOURNAMENT_SIZE = 2
DEFAULT_MAX_TOTAL_TASKS = 10000
DEFAULT_FORMAT = "svg"
DEFAULT_LOG_LEVEL = "INFO"

DESCRIPTION = (
    "Vectorize raster images into SVG, Graphviz, or Typst by combining vision "
    "LLMs with NSGA-II multi-objective evolutionary search. Each epoch opens "
    "with a batch of LLM candidates and then refines them with local search."
)

EPILOG = """\
Examples
--------
  Quickstart (auto-detects provider from $OPENAI_API_KEY /
              $ANTHROPIC_API_KEY / $GEMINI_API_KEY):
      vectrify input.png -o output.svg

  Bigger LLM batch per epoch, longer wall-clock cap:
      vectrify photo.jpg -o sketch.svg --seeds 20 --max-wall-seconds 1800

  Steer the search with a custom goal:
      vectrify logo.png --goal "Use thick strokes only and avoid gradients"

  Output a Graphviz DOT diagram instead of SVG:
      vectrify diagram.png -o out.dot --format graphviz

  Resume an earlier run and keep only the 20 best nodes:
      vectrify input.png --resume --resume-top 20

Environment
-----------
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY  one is required

Docs: https://github.com/rasros/vectrify
"""


def _get_version() -> str:
    try:
        return _pkg_version("vectrify")
    except PackageNotFoundError:
        return "0.0.0+local"


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vectrify",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "image",
        help="Input raster image (PNG, JPEG, WEBP, or GIF).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help="Output file path. Extension should match --format. "
        f"Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=list(FORMAT_NAMES),
        default=DEFAULT_FORMAT,
        help=f"Output vector format. Default: {DEFAULT_FORMAT}",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    g_llm = parser.add_argument_group("LLM provider")
    g_llm.add_argument(
        "--provider",
        type=str,
        choices=["auto", "openai", "anthropic", "gemini"],
        default=DEFAULT_PROVIDER,
        help="LLM provider. 'auto' picks whichever *_API_KEY is set "
        f"(openai > anthropic > gemini). Default: {DEFAULT_PROVIDER}",
    )
    g_llm.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="NAME",
        help="Model name. Defaults to a recent flagship model for the active provider.",
    )
    g_llm.add_argument(
        "--reasoning",
        type=str,
        choices=["low", "medium", "high"],
        default=DEFAULT_REASONING,
        help="Reasoning effort for thinking-capable models. "
        f"Default: {DEFAULT_REASONING}",
    )

    g_score = parser.add_argument_group("Scoring")
    g_score.add_argument(
        "--scorer",
        type=str,
        choices=[e.value for e in ScorerType],
        default=DEFAULT_SCORER,
        help="Perceptual scorer. 'auto' uses 'vision' if torch+transformers "
        f"are installed, else 'simple'. Default: {DEFAULT_SCORER}",
    )
    g_score.add_argument(
        "--vision-model",
        type=str,
        default=DEFAULT_VISION_MODEL,
        dest="vision_model",
        metavar="HF_REPO",
        help="HuggingFace model id for the vision scorer (CLIP/SigLIP-style). "
        f"Default: {DEFAULT_VISION_MODEL}",
    )

    g_search = parser.add_argument_group("Search strategy")
    g_search.add_argument(
        "--goal",
        default=None,
        metavar="TEXT",
        help="Custom prompt steering the LLM "
        "(e.g. 'Make lines thicker and avoid gradients').",
    )
    g_search.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        metavar="N",
        help="Active pool size used for parent selection. "
        f"Default: {DEFAULT_POOL_SIZE}",
    )
    g_search.add_argument(
        "--seeds",
        type=int,
        default=None,
        dest="seeds",
        metavar="N",
        help="LLM calls that open every epoch. Their children become that "
        "epoch's entire pool, which local mutation and crossover then refine; "
        "no other task calls the LLM, so total calls are at most "
        "epochs x seeds. Resumed candidates count toward epoch 0's batch. "
        "0 disables the LLM entirely (requires --resume). "
        "Defaults to pool-size // 10.",
    )

    g_epoch = parser.add_argument_group("Epoch control")
    g_epoch.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        dest="epochs",
        metavar="N",
        help="Epochs to run. Each one opens with a batch of seeds LLM calls, "
        "so this and --seeds together fix the run's entire LLM spend. "
        f"Default: {DEFAULT_EPOCHS}",
    )
    g_epoch.add_argument(
        "--epoch-patience",
        type=int,
        default=DEFAULT_EPOCH_PATIENCE,
        dest="epoch_patience",
        metavar="N",
        help="End the epoch and re-seed if the best score does not improve by "
        "--epoch-min-delta over this many consecutive local tasks. 0 disables. "
        f"Default: {DEFAULT_EPOCH_PATIENCE}",
    )
    g_epoch.add_argument(
        "--epoch-min-delta",
        type=float,
        default=DEFAULT_EPOCH_MIN_DELTA,
        metavar="DELTA",
        help="Minimum score improvement that resets --epoch-patience. "
        f"Default: {DEFAULT_EPOCH_MIN_DELTA}",
    )
    g_epoch.add_argument(
        "--epoch-diversity",
        type=float,
        default=DEFAULT_EPOCH_DIVERSITY,
        dest="epoch_diversity",
        metavar="THR",
        help="End epoch when mean pairwise genome diversity "
        "drops below this threshold. 0 disables.",
    )
    g_epoch.add_argument(
        "--epoch-variance",
        type=float,
        default=DEFAULT_EPOCH_VARIANCE,
        dest="epoch_variance",
        metavar="THR",
        help="End epoch when score std dev in the active pool "
        "drops below this threshold. 0 disables.",
    )
    g_search.add_argument(
        "--tournament-size",
        type=int,
        default=DEFAULT_TOURNAMENT_SIZE,
        dest="tournament_size",
        metavar="N",
        help="Candidates compared per parent-selection tournament. "
        "Higher means stronger bias toward visual quality and faster "
        "convergence, at the cost of pool diversity. "
        f"Default: {DEFAULT_TOURNAMENT_SIZE}",
    )

    g_resume = parser.add_argument_group("Resume")
    g_resume.add_argument(
        "--resume",
        dest="resume",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RESUME,
        help="Resume search from existing nodes in the output directory.",
    )
    g_resume.add_argument(
        "--resume-top",
        type=int,
        default=None,
        dest="resume_top",
        metavar="N",
        help="When resuming, keep only the N best-scoring nodes (implies --resume).",
    )

    g_artifacts = parser.add_argument_group("Output artifacts")
    g_artifacts.add_argument(
        "--write-lineage",
        dest="write_lineage",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WRITE_LINEAGE,
        help="Write lineage.csv and per-node files for every accepted node.",
    )
    g_artifacts.add_argument(
        "--save-raster",
        dest="save_raster",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_RASTER,
        help="Save a rendered .png alongside each accepted node.",
    )
    g_artifacts.add_argument(
        "--save-heatmap",
        dest="save_heatmap",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_HEATMAP,
        help="Save a perceptual diff .heatmap.png alongside each accepted node.",
    )

    g_runtime = parser.add_argument_group("Runtime")
    g_runtime.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"Parallel worker processes. Default: cpu count ({DEFAULT_WORKERS})",
    )
    g_runtime.add_argument(
        "--max-wall-seconds",
        type=float,
        default=DEFAULT_MAX_WALL_SECONDS,
        metavar="SECS",
        help="Wall-clock budget. 0 (or negative) disables. "
        f"Default: {DEFAULT_MAX_WALL_SECONDS}s",
    )
    g_runtime.add_argument(
        "--max-total-tasks",
        type=int,
        default=DEFAULT_MAX_TOTAL_TASKS,
        dest="max_total_tasks",
        metavar="N",
        help="Hard cap on total tasks (mutations, crossovers, and LLM calls) "
        f"across the entire run. Default: {DEFAULT_MAX_TOTAL_TASKS}",
    )
    g_runtime.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        metavar="PX",
        help="Working resolution for the whole run, and the parameter that "
        "most affects output quality and cost. The reference and every "
        "candidate are rendered and scored at this long-side; it is rounded up "
        "to a whole number of scorer crops, and it fixes the coordinate space "
        "candidates are written in (SVG viewBox, Typst page). Higher resolves "
        f"finer detail and costs proportionally more. Default: "
        f"{DEFAULT_RESOLUTION}",
    )
    g_runtime.add_argument(
        "--resolution-llm",
        type=int,
        default=DEFAULT_RESOLUTION_LLM,
        metavar="PX",
        help="Long-side of the images sent to the LLM (target, current render, "
        "difference map). Separate from resolution because it does not affect "
        "scoring at all, and vision pricing tiles at 512px, so going above that "
        f"triples the cost of every image. Default: {DEFAULT_RESOLUTION_LLM}",
    )
    g_runtime.add_argument(
        "--dashboard",
        dest="dashboard",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DASHBOARD,
        help="Show the live progress dashboard. Automatically disabled when "
        "stdout is not a terminal (e.g. piped or redirected).",
    )
    g_runtime.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Console log verbosity. Default: {DEFAULT_LOG_LEVEL}",
    )
    g_runtime.add_argument(
        "--debug",
        action="store_true",
        help="Print the full traceback on failure instead of a one-line error.",
    )

    ns = parser.parse_args(args)

    if ns.max_wall_seconds is not None and ns.max_wall_seconds <= 0:
        ns.max_wall_seconds = None

    if ns.epochs < 1:
        raise SystemExit("Error: --epochs must be at least 1")
    if ns.workers <= 0 or ns.pool_size <= 0:
        raise SystemExit("Error: --workers and --pool-size must be > 0")
    if ns.resolution <= 0:
        raise SystemExit("Error: --resolution must be > 0")
    if ns.resolution_llm <= 0:
        raise SystemExit("Error: --resolution-llm must be > 0")
    if ns.max_total_tasks <= 0:
        raise SystemExit("Error: --max-total-tasks must be > 0")

    if ns.seeds is not None and ns.seeds < 0:
        raise SystemExit("Error: --seeds must be 0 or greater")
    if ns.seeds == 0 and not ns.resume and ns.resume_top is None:
        raise SystemExit(
            "Error: --seeds 0 disables every LLM call, so the search has "
            "nothing to mutate unless it starts from existing candidates. "
            "Add --resume, or raise --seeds."
        )

    if ns.tournament_size < 2:
        raise SystemExit("Error: --tournament-size must be at least 2")

    if ns.resume_top is not None:
        ns.resume = True

    return ns
