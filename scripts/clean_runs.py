#!/usr/bin/env python3
"""
Clean up run directories, keeping only the front (score vs the other objectives)
and the top 20 nodes by score. All other node files are deleted.

Usage:
    uv run scripts/clean_runs.py <project_dir_or_svg_output>
    uv run scripts/clean_runs.py output.svg           # resolves to output/runs/
    uv run scripts/clean_runs.py my_project/runs/     # direct runs dir
    uv run scripts/clean_runs.py my_project/runs/2024-01-01_12-00-00/  # single run

Options:
    --dry-run    Print what would be deleted without deleting.
    --top N      Number of top-score nodes to keep (default: 20).
"""

import argparse
import csv
import re
import sys
from pathlib import Path

from vectrify.run_dirs import OUTPUT_EXTENSIONS, project_runs_dir, run_dirs_in
from vectrify.score.metrics import (
    FRONT_SCORE,
    METRIC_NAMES,
    SCORER_METRICS,
    read_metrics,
    row_has_metrics,
)
from vectrify.search.nsga import pareto_front


def collect_node_files(nodes_dir: Path) -> list[dict]:
    """
    Read node files (any known output extension) from a nodes directory.
    Supports the shapes storage has written:
      Current: {id}.{ext}                     e.g. 2.svg
      Current: eval{score}_{id}.{ext}         e.g. eval0.004392_2.svg
      Legacy:  {round_score}_{id}.{ext}       e.g. 0.069113_2.svg
      Legacy:  score{score}_node{id}_...      e.g. score00000.069113_node00002_...

    Only the eval prefix carries a score that means anything: it is the
    evaluator's, the run's only score. The legacy leading number was a blended
    proxy that nothing ranked on, so it is parsed for the id and ignored.
    """
    ext_pattern = "|".join(re.escape(e) for e in OUTPUT_EXTENSIONS)
    # New format: plain score_id.ext
    # `inf` must be its own alternative: storage writes f"{score:.6f}", which
    # yields a bare "inf" for INVALID_SCORE, so requiring digits first made the
    # optional (?:inf)? branch dead and left inf_*.svg files unmatched entirely.
    _plain = re.compile(rf"^(\d+)(?:{ext_pattern})$")
    _eval = re.compile(rf"^eval(-?[0-9.]+)_(\d+)(?:{ext_pattern})$")
    _legacy = re.compile(rf"^(inf|[0-9.]+)_(\d+)(?:{ext_pattern})$")
    # Old format: score00000.069113_node00002_parent00000.svg
    _old = re.compile(rf"^score([0-9.]+)_node(\d+)_parent\d+(?:{ext_pattern})$")

    nodes = []
    for node_path in sorted(nodes_dir.iterdir()):
        bare = _plain.match(node_path.name)
        if bare:
            score, node_id = float("inf"), int(bare.group(1))
        else:
            m = (
                _eval.match(node_path.name)
                or _legacy.match(node_path.name)
                or _old.match(node_path.name)
            )
            if not m:
                continue
            try:
                score = float(m.group(1))
            except ValueError:
                score = float("inf")
            node_id = int(m.group(2))
        nodes.append(
            {
                "id": node_id,
                "score": score,
                "path": node_path,
                **dict.fromkeys(METRIC_NAMES),
            }
        )
    return nodes


def load_metrics_from_lineage(lineage_csv: Path, nodes: list[dict]) -> None:
    """Fill in every registered metric from lineage.csv where available.

    Columns come from the metric registry, so a newly registered metric is
    picked up without editing this function.
    """
    if not lineage_csv.exists():
        return
    id_to_node = {n["id"]: n for n in nodes}
    try:
        with lineage_csv.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    node = id_to_node.get(int(row["id"]))
                    if node is None:
                        continue
                    # Skip sparse eviction rows, which would zero the metrics
                    # the node's real row already supplied.
                    if row_has_metrics(row):
                        node.update(read_metrics(row))
                except (KeyError, ValueError):
                    pass
    except Exception as e:
        print(f"  Warning: could not read {lineage_csv}: {e}", file=sys.stderr)


def clean_run_dir(run_dir: Path, top_n: int, dry_run: bool) -> tuple[int, int]:
    """
    Clean a single run directory. Returns (kept, deleted) counts.
    """
    nodes_dir = run_dir / "nodes"
    if not nodes_dir.exists():
        return 0, 0

    nodes = collect_node_files(nodes_dir)
    if not nodes:
        return 0, 0

    load_metrics_from_lineage(run_dir / "lineage.csv", nodes)

    # Nodes the lineage could actually describe. A file whose metrics never
    # arrived says nothing about whether it is worth keeping, and treating a
    # missing measure as 0.0 would make it unbeatable on that axis.
    measured = [n for n in nodes if all(n.get(m) is not None for m in SCORER_METRICS)]

    keep_ids: set[int] = set()

    # The best-ranked tier under the same relation the search selects by. Not a
    # score: the measures are traded off by dominance and there is no blend of
    # them to sort on.
    if measured:
        for node in pareto_front(
            measured, key=lambda n: tuple(n[m] for m in SCORER_METRICS)
        ):
            keep_ids.add(node["id"])

    # Then the best the evaluator saw, which is the only score in a run and
    # exists on the handful of nodes it was shown.
    evaluated = [n for n in nodes if n.get(FRONT_SCORE) is not None]
    for node in sorted(evaluated, key=lambda n: n[FRONT_SCORE])[:top_n]:
        keep_ids.add(node["id"])

    kept = 0
    deleted = 0
    for node in nodes:
        if node["id"] in keep_ids:
            kept += 1
        else:
            if dry_run:
                print(f"  [dry-run] would delete {node['path'].name}")
            else:
                node["path"].unlink(missing_ok=True)
                # Remove paired .png if present (old storage format)
                png_path = node["path"].with_suffix(".png")
                png_path.unlink(missing_ok=True)
            deleted += 1

    return kept, deleted


def resolve_run_dirs(path: Path) -> list[tuple[Path, list[Path]]]:
    """
    Return a list of (runs_dir, [run_dir, ...]) groups to clean.

    Accepts:
      - an output file (.svg/.dot/.typ) → output/runs/
      - project_dir/            → project_dir/runs/
      - project_dir/runs/       → all run dirs inside
      - single run dir          → that dir only
      - arbitrary directory     → recursively find all */runs/ beneath it
    """
    runs_dir = project_runs_dir(path)
    if runs_dir is not None:
        if not runs_dir.exists():
            print(f"Runs directory not found: {runs_dir}", file=sys.stderr)
            sys.exit(1)
        return [(runs_dir, run_dirs_in(runs_dir))]

    # Check if it looks like a single timestamped run dir (has a nodes/ subdir)
    if (path / "nodes").is_dir():
        return [(path.parent, [path])]

    # Recurse: find all runs/ directories anywhere beneath path
    all_runs_dirs = sorted(path.rglob("runs"), key=lambda p: str(p))
    groups = [(rd, run_dirs_in(rd)) for rd in all_runs_dirs if rd.is_dir()]
    if not groups:
        print(f"No runs/ directories found under {path}", file=sys.stderr)
        sys.exit(1)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path",
        help=(
            "SVG output path, project dir, runs dir, single run dir,"
            " or root dir to recurse."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without deleting.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Number of top-score nodes to keep (default: 20).",
    )
    args = parser.parse_args()

    groups = resolve_run_dirs(Path(args.path))

    total_kept = 0
    total_deleted = 0

    for runs_dir, run_dirs in groups:
        if len(groups) > 1:
            print(f"\n{runs_dir}:")
        for run_dir in run_dirs:
            kept, deleted = clean_run_dir(run_dir, top_n=args.top, dry_run=args.dry_run)
            if kept + deleted > 0:
                action = "would keep" if args.dry_run else "kept"
                del_word = "would delete" if args.dry_run else "deleted"
                print(f"  {run_dir.name}: {action} {kept}, {del_word} {deleted}")
            total_kept += kept
            total_deleted += deleted

    del_word = "would delete" if args.dry_run else "deleted"
    print(f"\nTotal: kept {total_kept}, {del_word} {total_deleted}")


if __name__ == "__main__":
    main()
