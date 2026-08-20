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
# An epoch is where the LLM sees the front and rewrites what local search
# cannot reach. Two spends most of a run refining whatever the first batch
# happened to produce, and a run that draws a poor first batch has one chance
# to recover from it.
#
# A ceiling now rather than a budget: --epoch-improvement-patience is what
# ends a run, when an epoch stops buying anything the evaluator can see. At 4
# this was the binding limit and it bound early -- measured on two runs of the
# same image, both ended on the epoch count inside 13 minutes of a one-hour
# wall, and one of them found its best candidate on the very last node it
# produced. Fifty is high enough to be reached only when the wall or the
# improvement test does not fire first.
DEFAULT_EPOCHS = 50
# How much better an epoch has to leave the evaluator's best for the epoch to
# have been worth its seed batch. Zero: any improvement at all counts, which
# is the weakest form of the test and the one that assumes least. Raise it to
# stop paying for epochs that only buy noise.
DEFAULT_EPOCH_IMPROVEMENT = 0.0
# Consecutive epochs improving by no more than that margin before the run
# stops. One, so a single epoch that buys nothing ends the run: an epoch costs
# a batch of LLM calls, and the evidence for waiting longer would have to come
# from a measurement nobody has taken. Raise it if an epoch's verdict turns
# out to be noisy enough that one refusal is not a settled opinion -- which is
# the reason --epoch-eval-patience inside an epoch sits at five, not one.
DEFAULT_EPOCH_IMPROVEMENT_PATIENCE = 1
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
# LLM calls opening each epoch. Five rather than ten, and a fixed number rather
# than pool-size // 10: the divisor tied the LLM budget to a pool size chosen for
# entirely separate reasons, so widening the pool silently bought more LLM calls.
#
# Five because the LLM calls are what produce candidates the evaluator rewards
# and local search is what drifts away from them. Measured on a 45-minute run,
# ten calls opened the only epoch that fit and the evaluator's best arrived in
# the first 2000 of 146,806 tasks; nothing in the remainder beat it. Half the
# batch is half the wait before an epoch can end and re-seed, at the same cost
# per epoch, so the same LLM spend buys twice as many chances to re-seed.
DEFAULT_SEEDS = 5
# Off, after trying it on. A pool collapses into agreement long before it stops
# improving: on a measured run the score spread had fallen to a fiftieth of its
# peak by task 500 of an epoch whose best went on to improve a further 77%
# before going stale at task 5200. Given floors of 0.10 and 0.05, all four
# epochs of a real run ended on a pool measure rather than on staleness and the
# run stopped after 3821 of its 12000 tasks.
# A ceiling on how long one epoch may run before the evaluator gets to steer
# again. Off by default: the right value is a judgement about how much drift on
# the cheap measures is acceptable between evaluations, and nothing measured so
# far pins it. Measured on one run, staleness at 500 first fired 150,800 tasks
# into an epoch -- all of it with no evaluator in the loop, which is where the
# proxy runs away from what a viewer would call better.
DEFAULT_EPOCH_MAX_TASKS = None
# How often the evaluator is asked about the front mid-epoch, in tasks. The
# cheap measures can be driven a long way without the drawing improving -- one
# run took them 64% down while the evaluator saw no difference at all -- and
# asking it only at the boundary means noticing that after the fact.
#
# A check costs one panel call over the front, about 13s measured, against a
# throughput near 60 tasks/s. Nothing is asked twice: the evaluator's score is
# absolute and cached per node, so a check re-prices only what is new.
DEFAULT_EPOCH_EVAL_INTERVAL = 2000
# Evaluator checks without a better candidate before the epoch ends and the
# model re-seeds. Counted in checks because that is the only unit that does not
# depend on something else: a generation is 100 accepted candidates, so its size
# in tasks moves with the acceptance rate and with --pool-size, and a threshold
# in generations below one interval's worth would fire before a check could ever
# intervene.
#
# Five, which is five checks and so five intervals of drift -- 10,000 tasks at
# the default interval. Measured on one 45-minute run the evaluator's best came
# at the first check and 73 further checks over 145,000 tasks never beat it,
# while the front it was shown degraded 40%, so the tolerance wanted is far
# below what an unset value gave. Five rather than two because a single
# evaluator verdict is noisy and an epoch ending on one costs a seed batch to
# reopen; five consecutive refusals is a settled opinion.
#
# It interacts with two other limits. Each ending costs an LLM batch, so
# --epochs is what caps the spend, and at this patience an epoch is roughly
# 10,000 tasks -- so --epochs 4 ends a run around 40,000 tasks, well short of a
# one-hour wall budget. Raise --epochs to spend the rest.
DEFAULT_EPOCH_EVAL_PATIENCE = 5
# Tasks without improvement before an epoch is called converged. Measured over
# eleven runs and 145 improvements, the gap between one improvement and the
# next is 25 tasks at the median, 142 at the 95th percentile and 497 at the
# longest observed -- so the previous 200 stopped epochs while improvements
# were still arriving. That measurement is also censored, since those runs were
# themselves cut off at 200 and cannot show what waiting longer would have
# found, which argues for clearing the observed maximum rather than sitting
# just above the percentile.
DEFAULT_EPOCH_PATIENCE = 500
DEFAULT_TOURNAMENT_SIZE = 2
DEFAULT_ADAPTIVE_OPERATORS = True
# Unset: the run is bounded by --epochs and --max-wall-seconds, which are the
# limits that describe what the search is meant to do. A task cap was binding
# before either of them -- at a measured 18 tasks/sec, 10000 tasks is nine
# minutes against a one-hour wall budget, and an epoch that runs to staleness
# takes 3600-5200 tasks, so four epochs could not fit inside it.
DEFAULT_MAX_TOTAL_TASKS = None
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
        help="Front evaluator, which orders a converged front. 'panel' puts "
        "every pair to five image encoders and takes the majority; 'vision' "
        "uses one encoder; 'auto' uses 'panel' if torch+transformers are "
        f"installed, else 'simple'. Default: {DEFAULT_SCORER}",
    )
    g_score.add_argument(
        "--vision-model",
        type=str,
        default=DEFAULT_VISION_MODEL,
        dest="vision_model",
        metavar="HF_REPO",
        help="HuggingFace model id for the vision scorer, any image encoder "
        "transformers can load. "
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
        default=DEFAULT_SEEDS,
        dest="seeds",
        metavar="N",
        help="LLM calls that open every epoch. Their children become that "
        "epoch's entire pool, which local mutation and crossover then refine; "
        "no other task calls the LLM, so total calls are at most "
        "epochs x seeds. Resumed candidates count toward epoch 0's batch. "
        f"0 disables the LLM entirely (requires --resume). Default: "
        f"{DEFAULT_SEEDS}",
    )

    g_epoch = parser.add_argument_group(
        "Epoch control. Any convergence criterion that is set can end an epoch "
        "on its own, so each wants a threshold tight enough that reaching it "
        "means the search is genuinely done. The two pool criteria are read as "
        "fractions of where the epoch started, so one setting means the same "
        "thing on every image."
    )
    g_epoch.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        dest="epochs",
        metavar="N",
        help="Ceiling on epochs. Each one opens with a batch of seeds LLM "
        "calls, so this and --seeds together bound the run's LLM spend. What "
        "normally ends a run is --epoch-improvement-patience. "
        f"Default: {DEFAULT_EPOCHS}",
    )
    g_epoch.add_argument(
        "--epoch-improvement",
        type=float,
        default=DEFAULT_EPOCH_IMPROVEMENT,
        dest="epoch_improvement",
        metavar="D",
        help="How much an epoch must improve the evaluator's best score to "
        "count as having paid for itself. 0 counts any improvement. "
        f"Default: {DEFAULT_EPOCH_IMPROVEMENT}",
    )
    g_epoch.add_argument(
        "--epoch-improvement-patience",
        type=int,
        default=DEFAULT_EPOCH_IMPROVEMENT_PATIENCE,
        dest="epoch_improvement_patience",
        metavar="N",
        help="Stop after N consecutive epochs that improve the evaluator's "
        "best by no more than --epoch-improvement. 0 disables, leaving "
        f"--epochs and --max-wall-seconds. Default: "
        f"{DEFAULT_EPOCH_IMPROVEMENT_PATIENCE}",
    )
    g_epoch.add_argument(
        "--epoch-eval-interval",
        type=int,
        default=DEFAULT_EPOCH_EVAL_INTERVAL,
        dest="epoch_eval_interval",
        metavar="N",
        help="Ask the evaluator about the front every N local tasks, not only "
        "at the epoch boundary. Scores are cached per candidate, so a check "
        f"re-prices only what is new. 0 disables. Default: "
        f"{DEFAULT_EPOCH_EVAL_INTERVAL}",
    )
    g_epoch.add_argument(
        "--epoch-eval-patience",
        type=int,
        default=DEFAULT_EPOCH_EVAL_PATIENCE,
        dest="epoch_eval_patience",
        metavar="N",
        help="End the epoch once this many consecutive evaluator checks pass "
        "without a better candidate. Unset by default.",
    )
    g_epoch.add_argument(
        "--epoch-max-tasks",
        type=int,
        default=DEFAULT_EPOCH_MAX_TASKS,
        dest="epoch_max_tasks",
        metavar="N",
        help="End the epoch after this many local tasks whether or not it has "
        "gone stale, so the evaluator ranks the front and the model re-seeds "
        "from its choice at least this often. Unset by default.",
    )
    g_epoch.add_argument(
        "--epoch-patience",
        type=int,
        default=DEFAULT_EPOCH_PATIENCE,
        dest="epoch_patience",
        metavar="N",
        help="End the epoch and re-seed if no candidate reaches the "
        "best-ranked tier over this many consecutive local tasks. 0 disables. "
        f"Default: {DEFAULT_EPOCH_PATIENCE}",
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

    g_search.add_argument(
        "--adaptive-operators",
        dest="adaptive_operators",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ADAPTIVE_OPERATORS,
        help="Learn which mutation operators are working on this image and "
        "shift towards them, instead of drawing from one fixed weight table. "
        f"Default: {'on' if DEFAULT_ADAPTIVE_OPERATORS else 'off'}",
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
        "across the entire run. Unset by default; --epochs and "
        "--max-wall-seconds bound the run.",
    )
    g_runtime.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        metavar="PX",
        help="Working resolution for the whole run. The reference and every "
        "candidate are rendered at this long-side, and it fixes the coordinate "
        "space candidates are written in (SVG viewBox, Typst page). Higher "
        "resolves finer geometry and costs proportionally more. Default: "
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
        "--random-seed",
        type=int,
        default=None,
        dest="random_seed",
        metavar="N",
        help="Seed the mutation/crossover RNG. With --workers 1 this makes a "
        "run reproducible; above that, scheduling still varies. Default: unseeded",
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
    if ns.max_total_tasks is not None and ns.max_total_tasks <= 0:
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
