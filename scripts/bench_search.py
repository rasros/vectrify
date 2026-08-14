"""Benchmark the local (non-LLM) search over the bench/cases corpus.

Every run is LLM-free: each case's seed.svg is planted as a previous run and the
search resumes from it with --seeds 0, so only mutation, crossover and Pareto
selection do any work. Two invocations with the same --reps are paired case for
case, which is what makes a change to the search measurable.

    uv run python scripts/bench_search.py run --out before.json
    # ... change the search ...
    uv run python scripts/bench_search.py run --out after.json
    uv run python scripts/bench_search.py compare before.json after.json
"""

import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from vectrify.cli import DEFAULT_POOL_SIZE
from vectrify.formats.svg.plugin import SvgPlugin

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CASES = REPO / "bench" / "cases"
SEED_RUN = "1970-01-01_00-00-00"


def case_seeds(case: Path) -> list[Path]:
    return sorted((case / "seeds").glob("*.svg"))


def discover_cases(cases_dir: Path) -> list[Path]:
    found = sorted(
        d for d in cases_dir.iterdir() if (d / "target.png").is_file() and case_seeds(d)
    )
    if not found:
        raise SystemExit(f"no cases with target.png + seeds/*.svg under {cases_dir}")
    return found


def plant_seed(work: Path, case: Path) -> Path:
    """Lay out a fake previous run so --resume picks every seed up.

    All of them, not one: a pool descended from a single ancestor gives
    crossover nothing to recombine, which is not the regime a real epoch runs
    in -- an epoch opens with several independent LLM candidates.
    """
    project = work / "out"
    nodes = project / "runs" / SEED_RUN / "nodes"
    nodes.mkdir(parents=True)
    for index, seed in enumerate(case_seeds(case), start=1):
        (nodes / f"1.000000_{index}.svg").write_text(seed.read_text(encoding="utf-8"))
    return work / "out.svg"


def read_curve(project: Path) -> list[float]:
    """Running-best score after each accepted node, oldest run last."""
    runs = sorted(p for p in (project / "runs").iterdir() if p.name != SEED_RUN)
    lineage = runs[-1] / "lineage.csv"
    scores: list[float] = []
    with lineage.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("score") or ""
            try:
                value = float(raw)
            except ValueError:
                continue  # eviction rows carry no score
            if value == float("inf"):
                continue
            scores.append(value)

    best, curve = float("inf"), []
    for value in scores:
        best = min(best, value)
        curve.append(best)
    return curve


_VISION: dict[str, object] = {}


def vision_score(target_png: Path, artifact: Path, resolution: int) -> float:
    """Score the run's final artifact the way a real run's evaluator would.

    The pixel curve measures what the round optimises, which makes two searches
    comparable but says nothing about whether a gain is visible. This is the
    objective that actually matters, so it is what a change has to move.

    The model is loaded once per bench process and the reference embedded once
    per case; both are far too expensive to redo per run.
    """
    from PIL import Image

    from vectrify.image_utils import resize_long_side
    from vectrify.score.vision import VisionScorer

    scorer = _VISION.get("scorer")
    if scorer is None:
        scorer = VisionScorer()
        _VISION["scorer"] = scorer

    key = str(target_png)
    if _VISION.get("reference_key") != key:
        target = resize_long_side(Image.open(target_png).convert("RGB"), resolution)
        _VISION["reference"] = scorer.prepare_reference(target)
        _VISION["reference_key"] = key
        _VISION["size"] = target.size

    width, height = _VISION["size"]
    png = SvgPlugin().rasterize(
        artifact.read_text(encoding="utf-8"), out_w=width, out_h=height
    )
    return scorer.score(_VISION["reference"], png)


def run_case(case: Path, seed: int, args) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        output = plant_seed(work, case)
        cmd = [
            "vectrify",
            str(case / "target.png"),
            "-o",
            str(output),
            "--seeds",
            "0",
            "--resume",
            "--random-seed",
            str(seed),
            "--workers",
            str(args.workers),
            "--max-total-tasks",
            str(args.tasks),
            "--resolution",
            str(args.resolution),
            "--scorer",
            args.scorer,
            "--epochs",
            str(args.epochs),
            "--epoch-patience",
            "0",
            "--max-wall-seconds",
            "0",
            "--no-dashboard",
            "--no-save-raster",
            "--log-level",
            "ERROR",
            "--pool-size",
            str(args.pool_size),
        ]
        if not args.adaptive_operators:
            cmd.append("--no-adaptive-operators")
        done = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        if done.returncode != 0:
            raise SystemExit(
                f"{case.name} seed={seed} failed ({done.returncode}):\n"
                f"{done.stderr[-2000:]}"
            )
        curve = read_curve(work / "out")
        vision = vision_score(case / "target.png", output, args.resolution)

    if not curve:
        raise SystemExit(f"{case.name} seed={seed} produced no scored nodes")
    start, final = curve[0], curve[-1]
    return {
        "case": case.name,
        "seed": seed,
        "nodes": len(curve),
        "start": start,
        "final": final,
        "vision": vision,
        # Mean of the running best: rewards reaching a score early, not just
        # ending there, so a faster search scores better at equal final value.
        "auc": statistics.fmean(curve),
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
                f"nodes={result['nodes']:<4} {result['start']:.6f} -> "
                f"{result['final']:.6f}  auc={result['auc']:.6f}  "
                f"vision={result['vision']:.6f}",
                flush=True,
            )

    payload = {
        "config": {
            "tasks": args.tasks,
            "reps": args.reps,
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
    print(f"\n{'case':<14} {'vision':>10} {'final':>10} {'auc':>10} {'gain':>8}")
    by_case: dict[str, list[dict]] = {}
    for r in runs:
        by_case.setdefault(r["case"], []).append(r)
    for case, rows in by_case.items():
        print(
            f"{case:<14} {statistics.fmean(r['vision'] for r in rows):>10.6f} "
            f"{statistics.fmean(r['final'] for r in rows):>10.6f} "
            f"{statistics.fmean(r['auc'] for r in rows):>10.6f} "
            f"{statistics.fmean(r['gain'] for r in rows):>7.1%}"
        )
    print(
        f"{'OVERALL':<14} {statistics.fmean(r['vision'] for r in runs):>10.6f} "
        f"{statistics.fmean(r['final'] for r in runs):>10.6f} "
        f"{statistics.fmean(r['auc'] for r in runs):>10.6f} "
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
    for metric in ("vision", "final", "auc"):
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
    print(f"\n{'case':<14} {'vision delta':>13} {'final delta':>13} {'auc delta':>12}")
    by_case: dict[str, list[tuple[dict, dict]]] = {}
    for b, a in pairs:
        by_case.setdefault(a["case"], []).append((b, a))
    for case, rows in by_case.items():
        print(
            f"{case:<14} "
            f"{statistics.fmean(a['vision'] - b['vision'] for b, a in rows):>+13.6f} "
            f"{statistics.fmean(a['final'] - b['final'] for b, a in rows):>+13.6f} "
            f"{statistics.fmean(a['auc'] - b['auc'] for b, a in rows):>+12.6f}"
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
    # scorer: the round is always pixel L1. At --epochs 1 the run ends before
    # any front is handed over, so the default avoids loading torch for a model
    # that never runs.
    run.add_argument("--scorer", default="simple", choices=["simple", "vision"])
    run.add_argument("--epochs", type=int, default=1, metavar="N")
    run.add_argument("--seed-base", type=int, default=1000, dest="seed_base")
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
