import argparse
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from vectrify.score import ScorerType
from vectrify.search import StrategyType

DEFAULT_OUTPUT = "output.svg"
DEFAULT_PROVIDER = "auto"
DEFAULT_SCORER = "auto"
DEFAULT_VISION_MODEL = "google/siglip-so400m-patch14-384"
DEFAULT_STRATEGY = "nsga"
BEAM_ONLY_PARAMS = {"beams", "cull_keep"}
NSGA_ONLY_PARAMS = {"epoch_diversity", "epoch_variance", "epoch_seeds"}
DEFAULT_MAX_EPOCHS = 4
DEFAULT_WORKERS = os.cpu_count() or 4
DEFAULT_MAX_WALL_SECONDS = 60 * 60
DEFAULT_RESUME = False
DEFAULT_WRITE_LINEAGE = True
DEFAULT_SAVE_RASTER = True
DEFAULT_SAVE_HEATMAP = False
DEFAULT_IMAGE_LONG_SIDE = 512
DEFAULT_REASONING = "medium"
DEFAULT_LLM_RATE = min(2 / DEFAULT_WORKERS, 0.2)
DEFAULT_POOL_SIZE = 100
DEFAULT_SEEDS = 0
DEFAULT_BEAMS = 10
DEFAULT_CULL_KEEP = 0.5
DEFAULT_EPOCH_DIVERSITY = 0.0
DEFAULT_EPOCH_VARIANCE = 0.0
DEFAULT_EPOCH_SEEDS = 0
DEFAULT_EPOCH_PATIENCE = 20
DEFAULT_EPOCH_MIN_DELTA = 1e-4
DEFAULT_EPOCH_STEPS = 50
DEFAULT_MAX_LLM_CALLS = 0  # 0 = unlimited / off
DEFAULT_FORMAT = "svg"
DEFAULT_LOG_LEVEL = "INFO"

DESCRIPTION = (
    "Vectorize raster images into SVG, Graphviz, or Typst by combining vision "
    "LLMs with NSGA-II multi-objective evolutionary search."
)

EPILOG = """\
Examples
--------
  Quickstart (auto-detects provider from $OPENAI_API_KEY /
              $ANTHROPIC_API_KEY / $GEMINI_API_KEY):
      vectrify input.png -o output.svg

  Bigger LLM budget per epoch, longer wall-clock cap:
      vectrify photo.jpg -o sketch.svg --epoch-patience 60 --max-wall-seconds 1800

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
        choices=["svg", "graphviz", "typst"],
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
        "--strategy",
        type=str,
        choices=[e.value for e in StrategyType],
        default=DEFAULT_STRATEGY,
        help=f"Search algorithm. Default: {DEFAULT_STRATEGY}",
    )
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
        default=DEFAULT_SEEDS,
        dest="seeds",
        metavar="N",
        help="Target LLM-seeded nodes for epoch 0. Resumed nodes count "
        "toward this. 0 uses pool-size // 10.",
    )
    g_search.add_argument(
        "--llm-rate",
        type=float,
        default=DEFAULT_LLM_RATE,
        metavar="RATE",
        help="Fraction of tasks (0.0-1.0) that call the LLM; the rest run "
        f"local mutations and crossover. Default: {DEFAULT_LLM_RATE:.2f}",
    )
    g_search.add_argument(
        "--beams",
        type=int,
        default=DEFAULT_BEAMS,
        metavar="N",
        help="[beam-only] Parallel hill-climbers; each epoch starts with "
        f"this many fresh LLM seeds. Default: {DEFAULT_BEAMS}",
    )
    g_search.add_argument(
        "--cull-keep",
        type=float,
        default=DEFAULT_CULL_KEEP,
        dest="cull_keep",
        metavar="FRAC",
        help="[beam-only] Fraction of beams eligible for expansion. Lower "
        f"values prune harder; 1.0 disables culling. Default: {DEFAULT_CULL_KEEP}",
    )

    g_epoch = parser.add_argument_group("Epoch control")
    g_epoch.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        dest="max_epochs",
        metavar="N",
        help=f"Maximum epochs to run. Default: {DEFAULT_MAX_EPOCHS}",
    )
    g_epoch.add_argument(
        "--epoch-patience",
        type=int,
        default=DEFAULT_EPOCH_PATIENCE,
        dest="epoch_patience",
        metavar="N",
        help="End the epoch and re-seed if best score does not improve by "
        "--epoch-min-delta over this many consecutive LLM calls "
        "(local mutations are not counted). 0 disables. "
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
        "--epoch-steps",
        type=int,
        default=DEFAULT_EPOCH_STEPS,
        dest="epoch_steps",
        metavar="N",
        help="Hard cap on LLM calls per epoch before forcing transition "
        "(local mutations are not counted). 0 means unlimited. "
        f"Default: {DEFAULT_EPOCH_STEPS}",
    )
    g_epoch.add_argument(
        "--epoch-diversity",
        type=float,
        default=DEFAULT_EPOCH_DIVERSITY,
        dest="epoch_diversity",
        metavar="THR",
        help="[nsga-only] End epoch when mean pairwise genome diversity "
        "drops below this threshold. 0 disables.",
    )
    g_epoch.add_argument(
        "--epoch-variance",
        type=float,
        default=DEFAULT_EPOCH_VARIANCE,
        dest="epoch_variance",
        metavar="THR",
        help="[nsga-only] End epoch when score std dev in the active pool "
        "drops below this threshold. 0 disables.",
    )
    g_epoch.add_argument(
        "--epoch-seeds",
        type=int,
        default=DEFAULT_EPOCH_SEEDS,
        dest="epoch_seeds",
        metavar="N",
        help="[nsga-only] Pareto-front nodes carried into each new epoch. "
        "0 uses pool-size // 4.",
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
        "--max-llm-calls",
        type=int,
        default=DEFAULT_MAX_LLM_CALLS,
        dest="max_llm_calls",
        metavar="N",
        help="Hard cap on total LLM calls across the entire run; ends the "
        "run as soon as it is reached. 0 disables. Useful as a strict "
        f"cost bound. Default: {DEFAULT_MAX_LLM_CALLS} (unlimited)",
    )
    g_runtime.add_argument(
        "--image-long-side",
        type=int,
        default=DEFAULT_IMAGE_LONG_SIDE,
        metavar="PX",
        help="Downscale reference and preview images to this long-side. "
        f"Default: {DEFAULT_IMAGE_LONG_SIDE}",
    )
    g_runtime.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Console log verbosity. Default: {DEFAULT_LOG_LEVEL}",
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
