#!/usr/bin/env python3
"""
Clean up run directories, keeping only the Pareto front (score vs complexity)
and the top 20 nodes by score. All other .svg files are deleted.

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
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path


_PATH_COMMANDS = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")


def svg_complexity(svg: str) -> float:
    compressed_size = len(zlib.compress(svg.encode("utf-8"), level=9))
    try:
        root = ET.fromstring(svg)
        element_count = sum(1 for _ in root.iter())
        path_vertices = sum(
            len(_PATH_COMMANDS.findall(el.get("d", "")))
            for el in root.iter()
            if el.get("d")
        )
    except ET.ParseError:
        return float(compressed_size)
    return float(compressed_size + element_count * 50 + path_vertices * 5)


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if a Pareto-dominates b (lower is better for both objectives)."""
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])


def pareto_front(nodes: list[dict]) -> list[dict]:
    """Return nodes on the Pareto front (minimising score and complexity)."""
    front = []
    for candidate in nodes:
        dominated = False
        for other in nodes:
            if other is candidate:
                continue
            if _dominates(
                (other["score"], other["complexity"]),
                (candidate["score"], candidate["complexity"]),
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def collect_svg_files(nodes_dir: Path) -> list[dict]:
    """
    Read SVG files from a nodes directory. Supports two filename formats:
      New: {score}_{id}.svg                          e.g. 0.069113_2.svg
      Old: score{score}_node{id}_parent{pid}.svg     e.g. score00000.069113_node00002_parent00000.svg
    """
    # New format: plain score_id.svg
    _new = re.compile(r"^([0-9.]+(?:inf)?)_(\d+)\.svg$")
    # Old format: score00000.069113_node00002_parent00000.svg
    _old = re.compile(r"^score([0-9.]+)_node(\d+)_parent\d+\.svg$")

    nodes = []
    for svg_path in nodes_dir.glob("*.svg"):
        m = _new.match(svg_path.name) or _old.match(svg_path.name)
        if not m:
            continue
        try:
            score = float(m.group(1))
        except ValueError:
            score = float("inf")
        node_id = int(m.group(2))
        nodes.append({"id": node_id, "score": score, "path": svg_path, "complexity": None})
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


def fill_missing_complexities(nodes: list[dict]) -> None:
    """Compute complexity from SVG text for nodes where it's still missing."""
    for node in nodes:
        if node["complexity"] is None:
            try:
                text = node["path"].read_text(encoding="utf-8")
                node["complexity"] = svg_complexity(text)
            except Exception:
                node["complexity"] = 0.0


def clean_run_dir(run_dir: Path, top_n: int, dry_run: bool) -> tuple[int, int]:
    """
    Clean a single run directory. Returns (kept, deleted) counts.
    """
    nodes_dir = run_dir / "nodes"
    if not nodes_dir.exists():
        return 0, 0

    nodes = collect_svg_files(nodes_dir)
    if not nodes:
        return 0, 0

    load_complexities_from_lineage(run_dir / "lineage.csv", nodes)
    fill_missing_complexities(nodes)

    valid = [n for n in nodes if n["score"] < float("inf")]

    keep_ids: set[int] = set()

    # Pareto front (score vs complexity)
    if valid:
        for node in pareto_front(valid):
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


def _runs_dirs_from_runs(runs_dir: Path) -> list[Path]:
    return sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )


def resolve_run_dirs(path: Path) -> list[tuple[Path, list[Path]]]:
    """
    Return a list of (runs_dir, [run_dir, ...]) groups to clean.

    Accepts:
      - output.svg              → output/runs/
      - project_dir/            → project_dir/runs/
      - project_dir/runs/       → all run dirs inside
      - single run dir          → that dir only
      - arbitrary directory     → recursively find all */runs/ beneath it
    """
    if path.suffix == ".svg":
        runs_dir = path.parent / path.stem / "runs"
        if not runs_dir.exists():
            print(f"Runs directory not found: {runs_dir}", file=sys.stderr)
            sys.exit(1)
        return [(runs_dir, _runs_dirs_from_runs(runs_dir))]

    if path.name == "runs" and path.is_dir():
        return [(path, _runs_dirs_from_runs(path))]

    if (path / "runs").is_dir():
        runs_dir = path / "runs"
        return [(runs_dir, _runs_dirs_from_runs(runs_dir))]

    # Check if it looks like a single timestamped run dir (has a nodes/ subdir)
    if (path / "nodes").is_dir():
        return [(path.parent, [path])]

    # Recurse: find all runs/ directories anywhere beneath path
    all_runs_dirs = sorted(path.rglob("runs"), key=lambda p: str(p))
    groups = [
        (rd, _runs_dirs_from_runs(rd))
        for rd in all_runs_dirs
        if rd.is_dir()
    ]
    if not groups:
        print(f"No runs/ directories found under {path}", file=sys.stderr)
        sys.exit(1)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="SVG output path, project dir, runs dir, single run dir, or root dir to recurse.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting.")
    parser.add_argument("--top", type=int, default=20, metavar="N", help="Number of top-score nodes to keep (default: 20).")
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
                print(f"  {run_dir.name}: {action} {kept}, {'would delete' if args.dry_run else 'deleted'} {deleted}")
            total_kept += kept
            total_deleted += deleted

    print(f"\nTotal: kept {total_kept}, {'would delete' if args.dry_run else 'deleted'} {total_deleted}")


if __name__ == "__main__":
    main()
