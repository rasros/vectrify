"""Benchmark the local (non-LLM) search over the bench/cases corpus.

Every run is LLM-free: each case's seeds open the pool and the in-process
NSGA-II search runs from them, so only mutation, crossover and Pareto
selection do any work. Two invocations with the same --reps are paired case for
case, which is what makes a change to the search measurable.

    uv run python scripts/bench_search.py run --out before.json
    # ... change the search ...
    uv run python scripts/bench_search.py run --out after.json
    uv run python scripts/bench_search.py compare before.json after.json
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

from PIL import Image

from vectrify.image_utils import rasterize_svg, resize_long_side
from vectrify.score import ScorerType

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CASES = REPO / "bench" / "cases"
DEFAULT_POOL_SIZE = 100


def case_seeds(case: Path) -> list[Path]:
    return sorted((case / "seeds").glob("*.svg"))


def discover_cases(cases_dir: Path) -> list[Path]:
    found = sorted(
        d for d in cases_dir.iterdir() if (d / "target.png").is_file() and case_seeds(d)
    )
    if not found:
        raise SystemExit(f"no cases with target.png + seeds/*.svg under {cases_dir}")
    return found


_VISION: dict[str, object] = {}


def vision_score(target_png: Path, content: str, resolution: int) -> float:
    """Score the run's final artifact the way a real run's evaluator would.

    The pixel curve measures what the round optimises, which makes two searches
    comparable but says nothing about whether a gain is visible. This is the
    objective that actually matters, so it is what a change has to move.

    The model is loaded once per bench process and the reference embedded once
    per case; both are far too expensive to redo per run.
    """
    from PIL import Image

    from vectrify.image_utils import resize_long_side
    from vectrify.score.ensemble import EnsembleScorer

    scorer = _VISION.get("scorer")
    if scorer is None:
        # The same panel the run is judged by. A finished artifact is a single
        # candidate, so there is no vote to take and this is the panel's mean.
        scorer = EnsembleScorer()
        _VISION["scorer"] = scorer

    key = str(target_png)
    if _VISION.get("reference_key") != key:
        target = resize_long_side(Image.open(target_png).convert("RGB"), resolution)
        _VISION["reference"] = scorer.prepare_reference(target)
        _VISION["reference_key"] = key
        _VISION["size"] = target.size

    width, height = _VISION["size"]
    png = rasterize_svg(content, width, height)
    return scorer.score(_VISION["reference"], png)


def run_case(case: Path, seed: int, args) -> dict:
    """One search over a case's seed pool; the evaluator judges start and end."""
    from vectrify.score import choose_scorer
    from vectrify.vector.reference import Reference
    from vectrify.vector.search import (
        SearchSettings,
        evaluate_front,
        run_search,
        seed_node,
    )
    from vectrify.vector.worker import WorkerContext

    target = resize_long_side(
        Image.open(case / "target.png").convert("RGB"), args.resolution
    )
    width, height = target.size
    reference = Reference.build(target)
    seeds = [
        seed_node(
            reference,
            content,
            rasterize_svg(content, width, height),
            node_id=index,
            origin=f"Seed {index}",
        )
        for index, content in enumerate(
            (s.read_text(encoding="utf-8") for s in case_seeds(case)), start=1
        )
    ]
    choice = choose_scorer(ScorerType(args.scorer))

    def rank_front(nodes):
        return evaluate_front(
            nodes,
            front_scorer=lambda: (
                choice.scorer,
                choice.scorer.prepare_reference(target),
            ),
            out_w=width,
            out_h=height,
        )

    context = WorkerContext(
        original_png_bytes=reference.png,
        original_w=width,
        original_h=height,
        log_level="ERROR",
        random_seed=seed,
    )
    outcome = run_search(
        reference,
        seeds,
        context,
        SearchSettings(
            workers=args.workers,
            pool_size=args.pool_size,
            adaptive_operators=args.adaptive_operators,
            epochs=args.epochs,
            epoch_eval_interval=args.eval_interval,
            max_total_tasks=args.tasks,
        ),
        rank_front=rank_front,
    )
    if outcome.best is None:
        raise SystemExit(f"{case.name} seed={seed} produced no candidate")
    start = min(
        vision_score(case / "target.png", n.state.payload.content, args.resolution)
        for n in seeds
    )
    final = vision_score(
        case / "target.png", outcome.best.state.payload.content, args.resolution
    )
    return {
        "case": case.name,
        "seed": seed,
        "tasks": outcome.tasks_completed,
        "start": start,
        "vision": final,
        "gain": (start - final) / start if start > 0 else 0.0,
    }


def cmd_run(args) -> None:
    cases = discover_cases(Path(args.cases))
    runs = []
    for case in cases:
        for rep in range(args.reps):
            result = run_case(case, args.seed_base + rep, args)
            runs.append(result)
            print(
                f"{result['case']:<14} seed={result['seed']:<3} "
                f"tasks={result['tasks']:<6} {result['start']:.6f} -> "
                f"{result['vision']:.6f}  gain={result['gain']:.1%}",
                flush=True,
            )

    payload = {
        "config": {
            "tasks": args.tasks,
            "reps": args.reps,
            "eval_interval": args.eval_interval,
            "workers": args.workers,
            "resolution": args.resolution,
            "scorer": args.scorer,
            "epochs": args.epochs,
            "seed_base": args.seed_base,
        },
        "runs": runs,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    _summarise(runs)


def _summarise(runs: list[dict]) -> None:
    print(f"\n{'case':<14} {'start':>10} {'vision':>10} {'gain':>8}")
    by_case: dict[str, list[dict]] = {}
    for r in runs:
        by_case.setdefault(r["case"], []).append(r)
    for case, rows in by_case.items():
        print(
            f"{case:<14} {statistics.fmean(r['start'] for r in rows):>10.6f} "
            f"{statistics.fmean(r['vision'] for r in rows):>10.6f} "
            f"{statistics.fmean(r['gain'] for r in rows):>7.1%}"
        )
    print(
        f"{'OVERALL':<14} {statistics.fmean(r['start'] for r in runs):>10.6f} "
        f"{statistics.fmean(r['vision'] for r in runs):>10.6f} "
        f"{statistics.fmean(r['gain'] for r in runs):>7.1%}"
    )


def _bootstrap_ci(deltas: list[float], rounds: int = 20000) -> tuple[float, float]:
    rng = random.Random(0)
    n = len(deltas)
    means = sorted(
        statistics.fmean(rng.choice(deltas) for _ in range(n)) for _ in range(rounds)
    )
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def cmd_compare(args) -> None:
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())

    if before["config"] != after["config"]:
        print(
            "WARNING: configs differ; the comparison is not paired.\n",
            file=sys.stderr,
        )

    keyed = {(r["case"], r["seed"]): r for r in before["runs"]}
    pairs = [
        (keyed[(r["case"], r["seed"])], r)
        for r in after["runs"]
        if (r["case"], r["seed"]) in keyed
    ]
    if not pairs:
        raise SystemExit("no (case, seed) pairs in common")

    print(f"{len(pairs)} paired runs\n")
    print(f"{'metric':<8} {'before':>10} {'after':>10} {'delta':>11}  95% CI")
    for metric in ("vision",):
        deltas = [a[metric] - b[metric] for b, a in pairs]
        lo, hi = _bootstrap_ci(deltas)
        mean = statistics.fmean(deltas)
        better = sum(1 for d in deltas if d < 0)
        print(
            f"{metric:<8} {statistics.fmean(b[metric] for b, _ in pairs):>10.6f} "
            f"{statistics.fmean(a[metric] for _, a in pairs):>10.6f} "
            f"{mean:>+11.6f}  [{lo:+.6f}, {hi:+.6f}]  better in {better}/{len(deltas)}"
        )

    print("\nlower is better; a CI entirely below 0 is an improvement")
    print(f"\n{'case':<14} {'vision delta':>13}")
    by_case: dict[str, list[tuple[dict, dict]]] = {}
    for b, a in pairs:
        by_case.setdefault(a["case"], []).append((b, a))
    for case, rows in by_case.items():
        print(
            f"{case:<14} "
            f"{statistics.fmean(a['vision'] - b['vision'] for b, a in rows):>+13.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the corpus and write results JSON")
    run.add_argument("--out", required=True, metavar="PATH")
    run.add_argument("--cases", default=str(DEFAULT_CASES), metavar="DIR")
    # The round scores on pixels now, so a task costs ~1 ms rather than ~300 ms
    # and a meaningful budget is thousands of tasks rather than hundreds.
    run.add_argument("--tasks", type=int, default=4000, metavar="N")
    run.add_argument("--reps", type=int, default=3, metavar="N")
    run.add_argument("--workers", type=int, default=1, metavar="N")
    run.add_argument("--resolution", type=int, default=384, metavar="PX")
    run.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        dest="pool_size",
        metavar="N",
    )
    # Selects the evaluator that ranks a converged front, not the round's
    # scorer: the round is always pixel L1.
    #
    # The reported `vision` column is the evaluator panel's mean distance
    # across its members, which is not on the same scale as the single-model
    # number older result files carry: compare within a set of runs, not across
    # the change.
    run.add_argument(
        "--scorer", default="simple", choices=[e.value for e in ScorerType]
    )
    run.add_argument("--epochs", type=int, default=1, metavar="N")
    run.add_argument("--seed-base", type=int, default=1000, dest="seed_base")
    run.add_argument(
        "--eval-interval",
        type=int,
        default=2000,
        dest="eval_interval",
        metavar="N",
        help="Candidates between front evaluations within an epoch",
    )
    run.add_argument(
        "--adaptive-operators",
        dest="adaptive_operators",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="paired comparison of two results files")
    cmp_.add_argument("before")
    cmp_.add_argument("after")
    cmp_.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
