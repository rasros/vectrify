#!/usr/bin/env python3
"""
Plot stats for one or more runs.

Usage:
    uv run scripts/plot_run.py <run_dir>
    uv run scripts/plot_run.py output/runs/2024-01-01_12-00-00
    uv run scripts/plot_run.py output/runs/           # overlay all runs
    uv run scripts/plot_run.py output.svg             # all runs for this output

Options:
    --output FILE   Save to FILE instead of showing interactively.
    --top N         Only plot the N most recent runs when given a runs/ dir (default: all).
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.rcParams["figure.dpi"] = 192  # crisp on HiDPI / 4K displays
matplotlib.use("Agg")  # safe default; switch_backend below upgrades to GUI if available

import matplotlib.pyplot as plt

for _backend in ("TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
    try:
        plt.switch_backend(_backend)
        break
    except Exception:
        pass
import matplotlib.ticker as mticker


# ── Data loading ─────────────────────────────────────────────────────────────

def load_stats(run_dir: Path) -> dict:
    """Load wide-format stats.csv."""
    path = run_dir / "stats.csv"
    if not path.exists():
        return {}

    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {}

    last = rows[-1]

    def _float(key, default=0.0):
        try:
            v = last.get(key, "")
            return float(v) if v != "" else default
        except (ValueError, TypeError):
            return default

    tasks   = _float("tasks_completed") or 1
    llm     = _float("llm_call_count")
    mut     = _float("mutation_call_count")
    acc     = _float("accepted_count")
    llm_acc = _float("llm_accepted_count")
    llm_inv = _float("llm_invalid_count")
    mut_acc = _float("mutation_accepted_count")

    stats: dict = {
        "elapsed_seconds":        _float("elapsed"),
        "best_score":             _float("best_score", float("inf")),
        "tasks_completed":        tasks,
        "accepted_count":         acc,
        "pool_rejected_count":    _float("pool_rejected_count"),
        "invalid_count":          _float("invalid_count"),
        "accept_rate":            acc / tasks,
        "pool_rejected_rate":     _float("pool_rejected_count") / tasks,
        "invalid_rate":           _float("invalid_count") / tasks,
        "llm_call_count":         llm,
        "llm_accepted_count":     llm_acc,
        "llm_invalid_count":      llm_inv,
        "llm_valid_rate":         (llm - llm_inv) / llm if llm else 0.0,
        "llm_accept_rate":        llm_acc / llm if llm else 0.0,
        "mutation_call_count":    mut,
        "mutation_accepted_count": mut_acc,
        "mutation_accept_rate":   mut_acc / mut if mut else 0.0,
        "llm_rate":               _float("llm_rate"),
        "llm_pressure_final":     _float("llm_pressure"),
        "epochs_completed":       _float("epoch"),
        "epoch_patience_config":  _float("epoch_patience"),
        "epoch_diversity_config": _float("epoch_diversity"),
        "epoch_variance_config":  _float("epoch_variance"),
        "pool_diversity_final":   _float("pool_diversity"),
        "pool_score_std_final":   _float("pool_score_std"),
    }

    # Reconstruct score_history from rows where best_score decreased.
    history = []
    prev_best = float("inf")
    for row in rows:
        try:
            elapsed = float(row.get("elapsed", 0) or 0)
            bs_raw = row.get("best_score", "")
            if bs_raw == "":
                continue
            bs = float(bs_raw)
            if bs < prev_best:
                history.append((elapsed, bs))
                prev_best = bs
        except (ValueError, TypeError):
            pass
    stats["score_history"] = history

    # Convergence + rates time series:
    # (elapsed, pool_diversity, pool_score_std, epoch, llm_pressure, accept_rate)
    convergence = []
    for row in rows:
        try:
            t_comp = float(row.get("tasks_completed", 0) or 0)
            a_comp = float(row.get("accepted_count", 0) or 0)
            convergence.append((
                float(row.get("elapsed", 0) or 0),
                float(row.get("pool_diversity", 0) or 0),
                float(row.get("pool_score_std", 0) or 0),
                int(float(row.get("epoch", 0) or 0)),
                float(row.get("llm_pressure", 0) or 0),
                a_comp / t_comp if t_comp else 0.0,
            ))
        except (ValueError, TypeError):
            pass
    stats["convergence_history"] = convergence

    return stats


def load_lineage(run_dir: Path) -> list[dict]:
    path = run_dir / "lineage.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "id": int(row["id"]),
                    "parent": int(row["parent"]),
                    "epoch": int(row.get("epoch", 0) or 0),
                    "score": float(row["score"]),
                    "complexity": float(row.get("complexity", 0) or 0),
                })
            except (KeyError, ValueError):
                pass
    return rows


def resolve_run_dirs(path: Path, top: int | None) -> list[Path]:
    if path.suffix == ".svg":
        runs_dir = path.parent / path.stem / "runs"
    elif path.name == "runs" and path.is_dir():
        runs_dir = path
    elif (path / "runs").is_dir():
        runs_dir = path / "runs"
    elif (path / "nodes").is_dir() or (path / "lineage.csv").exists():
        return [path]
    else:
        # recurse
        found = sorted(path.rglob("runs"), key=lambda p: str(p))
        dirs = []
        for rd in found:
            dirs.extend(sorted([d for d in rd.iterdir() if d.is_dir()], key=lambda d: d.name))
        return dirs[-top:] if top else dirs

    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
    return run_dirs[-top:] if top else run_dirs


# ── Pareto helper ──────────────────────────────────────────────────────────────

def _pareto_top10(lin: list[dict]) -> list[dict]:
    """Return up to 10 Pareto-front nodes (minimise score and complexity), sorted by score."""
    valid = [r for r in lin if r["score"] < float("inf")]
    if not valid:
        return []
    front = []
    for candidate in valid:
        dominated = False
        for other in valid:
            if other is candidate:
                continue
            if (other["score"] <= candidate["score"]
                    and other["complexity"] <= candidate["complexity"]
                    and (other["score"] < candidate["score"]
                         or other["complexity"] < candidate["complexity"])):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda r: r["score"])[:10]


# ── Plot helpers ──────────────────────────────────────────────────────────────

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# Distinct highlight colors for top-10 Pareto nodes (cycled if >10)
PARETO_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
]


def _label(run_dir: Path) -> str:
    return run_dir.name


def plot_score_history(ax, runs: list[tuple[Path, dict]], lineages: list[list[dict]]):
    ax.set_title("Best score over time")
    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel("Score (lower = better)")

    for i, ((run_dir, stats), lin) in enumerate(zip(runs, lineages)):
        history = stats.get("score_history", [])
        if not history:
            continue
        xs = [0.0] + [t for t, _ in history]
        ys = [history[0][1]] + [s for _, s in history]
        color = COLORS[i % len(COLORS)]
        ax.step(xs, ys, where="post", color=color, label=_label(run_dir), linewidth=1.5)
        ax.scatter([t for t, _ in history], [s for _, s in history],
                   color=color, s=20, zorder=3)

        # Epoch transition lines
        ch = stats.get("convergence_history", [])
        prev_epoch = ch[0][3] if ch else 0
        for elapsed, _div, _std, ep, _pr, _ar in ch:
            if ep != prev_epoch:
                ax.axvline(elapsed, color="grey", linewidth=0.8, linestyle=":", alpha=0.8)
                ax.text(elapsed, ys[-1], f" e{ep}", fontsize=8, color="grey", va="bottom")
                prev_epoch = ep


    if len(runs) > 1:
        ax.legend(fontsize=7, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))


def plot_pareto(ax, runs: list[tuple[Path, dict]], lineages: list[list[dict]]):
    ax.set_title("Score vs complexity")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Score")

    for i, ((run_dir, _), lin) in enumerate(zip(runs, lineages)):
        valid = [r for r in lin if r["score"] < float("inf")]
        if not valid:
            continue
        color = COLORS[i % len(COLORS)]
        ax.scatter([r["complexity"] for r in valid],
                   [r["score"] for r in valid],
                   s=6, alpha=0.3, color=color, label=_label(run_dir) if len(runs) > 1 else None)

        top10 = _pareto_top10(lin)
        for j, node in enumerate(top10):
            pc = PARETO_COLORS[j % len(PARETO_COLORS)]
            ax.scatter([node["complexity"]], [node["score"]],
                       marker="*", s=160, color=pc, zorder=5,
                       edgecolors="black", linewidths=0.4)
            ax.annotate(f"#{node['id']}", (node["complexity"], node["score"]),
                        fontsize=8, color=pc,
                        xytext=(4, 4), textcoords="offset points")

    if len(runs) > 1:
        ax.legend(fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))


def plot_convergence(ax, runs: list[tuple[Path, dict]], lineages: list[list[dict]]):
    ax.set_title("Pool convergence over time")
    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel("Pool diversity", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax.twinx()
    ax2.set_ylabel("Score std dev", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    for i, (run_dir, stats) in enumerate(runs):
        ch = stats.get("convergence_history", [])
        if not ch:
            continue
        color_div = COLORS[i % len(COLORS)]
        color_std = COLORS[(i + 1) % len(COLORS)]
        xs          = [r[0] for r in ch]
        diversities = [r[1] for r in ch]
        stds        = [r[2] for r in ch]
        ax.plot(xs, diversities, color=color_div, linewidth=1.2,
                label=f"{_label(run_dir)} diversity")
        ax2.plot(xs, stds, color=color_std, linewidth=1.2, linestyle="--",
                 label=f"{_label(run_dir)} score std")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper right")



def plot_summary_text(ax, runs: list[tuple[Path, dict]], lineages: list[list[dict]]):
    ax.axis("off")
    ax.set_title("Run summary", loc="left")
    lines = []
    for (run_dir, stats), lin in zip(runs, lineages):
        if not stats:
            continue
        best = stats.get("best_score")
        best_str = f"{best:.6f}" if best is not None and best < float("inf") else "—"
        lines.append(f"[{_label(run_dir)}]")
        lines.append(f"  best score      {best_str}")
        lines.append(f"  elapsed         {stats.get('elapsed_seconds', 0):.0f}s")
        lines.append(f"  tasks           {int(stats.get('tasks_completed') or 0):,}")
        lines.append(f"  accepted        {int(stats.get('accepted_count') or 0):,}  ({stats.get('accept_rate', 0)*100:.1f}%)")
        lines.append(f"  pool-rejected   {int(stats.get('pool_rejected_count') or 0):,}  ({stats.get('pool_rejected_rate', 0)*100:.1f}%)")
        lines.append(f"  invalid         {int(stats.get('invalid_count') or 0):,}  ({stats.get('invalid_rate', 0)*100:.1f}%)")
        lines.append(f"  llm calls       {int(stats.get('llm_call_count') or 0):,}")
        lines.append(f"  llm valid       {stats.get('llm_valid_rate', 0)*100:.1f}%")
        lines.append(f"  llm accept      {stats.get('llm_accept_rate', 0)*100:.1f}%")
        lines.append(f"  llm pressure    {stats.get('llm_pressure_final', 0):.3f}")
        lines.append(f"  mut calls       {int(stats.get('mutation_call_count') or 0):,}")
        lines.append(f"  mut accept      {stats.get('mutation_accept_rate', 0)*100:.1f}%")
        lines.append(f"  epochs          {int(stats.get('epochs_completed') or 0)}")
        lines.append(f"  diversity       {stats.get('pool_diversity_final', 0):.4f}")
        lines.append(f"  score std       {stats.get('pool_score_std_final', 0):.6f}")

        top10 = _pareto_top10(lin)
        if top10:
            lines.append("")
            lines.append(f"  pareto top 10:")
            lines.append(f"  {'#':>2}  {'id':>6}  {'score':>10}  {'complexity':>10}  ep")
            for rank, node in enumerate(top10, 1):
                lines.append(
                    f"  {rank:>2}  {node['id']:>6}  {node['score']:>10.6f}"
                    f"  {node['complexity']:>10.0f}  {node['epoch']}"
                )
        lines.append("")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=7, verticalalignment="top", fontfamily="monospace")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Run dir, runs/ dir, output.svg, or project dir.")
    parser.add_argument("--output", "-o", default=None,
                        help="Save plot to this file (png/pdf/svg). Default: show interactively.")
    parser.add_argument("--top", type=int, default=None, metavar="N",
                        help="Only plot the N most recent runs.")
    args = parser.parse_args()

    run_dirs = resolve_run_dirs(Path(args.path), args.top)
    if not run_dirs:
        print(f"No run directories found at {args.path}", file=sys.stderr)
        sys.exit(1)

    runs = [(d, load_stats(d)) for d in run_dirs]
    lineages = [load_lineage(d) for d in run_dirs]

    # Filter to runs that have any data at all
    combined = [(r, s, l) for r, s, l in zip(run_dirs, [s for _, s in runs], lineages)
                if s or l]
    if not combined:
        print("No stats.csv or lineage.csv found in the given run directories.",
              file=sys.stderr)
        sys.exit(1)
    run_dirs_f = [r for r, _, _ in combined]
    runs_f = [(r, s) for r, s, _ in combined]
    lineages_f = [l for _, _, l in combined]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"SVGizer run stats — {run_dirs_f[0].parent.name if len(run_dirs_f) == 1 else str(Path(args.path))}",
        fontsize=11, fontweight="bold",
    )
    plt.subplots_adjust(hspace=0.35, wspace=0.32, left=0.07, right=0.98, top=0.93, bottom=0.06)

    plot_score_history(axes[0, 0], runs_f, lineages_f)
    plot_pareto(axes[0, 1], runs_f, lineages_f)
    plot_convergence(axes[1, 0], runs_f, lineages_f)
    plot_summary_text(axes[1, 1], runs_f, lineages_f)

    if args.output:
        out = Path(args.output)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        if matplotlib.get_backend().lower() == "agg":
            out = Path("/tmp/svgizer_plot.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"No GUI backend available. Saved to {out}", file=sys.stderr)
            print(f"Use --output FILE to save to a specific path.")
        else:
            plt.show()


if __name__ == "__main__":
    main()
