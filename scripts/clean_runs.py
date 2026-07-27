#!/usr/bin/env python3
"""
Clean up run directories, keeping only the Pareto front (score vs complexity)
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
from vectrify.search.nsga import pareto_front


def collect_node_files(nodes_dir: Path) -> list[dict]:
    """
    Read node files (any known output extension) from a nodes directory.
    Supports two filename formats:
      New: {score}_{id}.{ext}           e.g. 0.069113_2.svg
      Old: score{score}_node{id}_...    e.g. score00000.069113_node00002_parent00000.svg
    """
    ext_pattern = "|".join(re.escape(e) for e in OUTPUT_EXTENSIONS)
    # New format: plain score_id.ext
    _new = re.compile(rf"^([0-9.]+(?:inf)?)_(\d+)(?:{ext_pattern})$")
    # Old format: score00000.069113_node00002_parent00000.svg
    _old = re.compile(rf"^score([0-9.]+)_node(\d+)_parent\d+(?:{ext_pattern})$")

    nodes = []
    for node_path in sorted(nodes_dir.iterdir()):
        m = _new.match(node_path.name) or _old.match(node_path.name)
        if not m:
            continue
        try:
            score = float(m.group(1))
        except ValueError:
            score = float("inf")
        node_id = int(m.group(2))
        nodes.append(
            {"id": node_id, "score": score, "path": node_path, "complexity": None}
        )
    return nodes


def load_complexities_from_lineage(lineage_csv: Path, nodes: list[dict]) -> None:
    """Fill in complexity from lineage.csv where available."""
    if not lineage_csv.exists():
        return
    id_to_node = {n["id"]: n for n in nodes}
    try:
        with lineage_csv.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    node_id = int(row["id"])
                    complexity = float(row["complexity"])
                    if node_id in id_to_node:
                        id_to_node[node_id]["complexity"] = complexity
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

    load_complexities_from_lineage(run_dir / "lineage.csv", nodes)
    for node in nodes:
        if node["complexity"] is None:
            node["complexity"] = 0.0

    valid = [n for n in nodes if n["score"] < float("inf")]

    keep_ids: set[int] = set()

    # Pareto front (score vs complexity)
    if valid:
        for node in pareto_front(valid, key=lambda n: (n["score"], n["complexity"])):
            keep_ids.add(node["id"])

    # Top N by score
    for node in sorted(valid, key=lambda n: n["score"])[:top_n]:
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
