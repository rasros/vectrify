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
    --top N         Only plot the N most recent runs when given a runs/ dir
                    (default: all).
"""

import argparse
import contextlib
import csv
import functools
import sys
from pathlib import Path

import matplotlib

from vectrify.run_dirs import project_runs_dir, run_dirs_in
from vectrify.score.metrics import (
    METRIC_NAMES,
    read_metrics,
)
from vectrify.search.nsga import pareto_front
from vectrify.search.stats import derived_rates

matplotlib.rcParams["figure.dpi"] = 192  # crisp on HiDPI / 4K displays
matplotlib.use("Agg")  # safe default; switch_backend below upgrades to GUI if available

import matplotlib.pyplot as plt  # noqa: E402

for _backend in ("TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
    try:
        plt.switch_backend(_backend)
        break
    except Exception:
        pass

import matplotlib.ticker as mticker  # noqa: E402

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

    counts = {
        name: _float(name)
        for name in (
            "tasks_completed",
            "accepted_count",
            "pool_rejected_count",
            "invalid_count",
            "llm_call_count",
            "llm_accepted_count",
            "llm_invalid_count",
            "mutation_call_count",
            "mutation_accepted_count",
        )
    }

    stats: dict = {
        "elapsed_seconds": _float("elapsed"),
        "best_score": _float("best_score", float("inf")),
        **counts,
        # Rates come from the same definitions the live dashboard uses.
        **derived_rates(counts),
        "seeds_target": _float("seeds_target"),
        "epochs_completed": _float("epoch"),
        "epoch_patience_config": _float("epoch_patience"),
        "epoch_max_tasks_config": _float("epoch_max_tasks"),
        "pool_diversity_final": _float("pool_diversity"),
        "pool_score_std_final": _float("pool_score_std"),
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
    # (elapsed, pool_diversity, pool_score_std, epoch, accept_rate)
    convergence = []
    for row in rows:
        try:
            t_comp = float(row.get("tasks_completed", 0) or 0)
            a_comp = float(row.get("accepted_count", 0) or 0)
            convergence.append(
                (
                    float(row.get("elapsed", 0) or 0),
                    float(row.get("pool_diversity", 0) or 0),
                    float(row.get("pool_score_std", 0) or 0),
                    int(float(row.get("epoch", 0) or 0)),
                    a_comp / t_comp if t_comp else 0.0,
                )
            )
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
            with contextlib.suppress(KeyError, ValueError):
                rows.append(
                    {
                        "id": int(row["id"]),
                        "parent": int(row["parent"]),
                        "epoch": int(row.get("epoch", 0) or 0),
                        "score": float(row["score"]),
                        # Metric columns come from the registry.
                        **read_metrics(row),
                    }
                )
    return rows


def load_final_pool_ids(run_dir: Path) -> set[int] | None:
    """Return node IDs in the final active pool, derived from eviction records."""
    path = run_dir / "lineage.csv"
    if not path.exists():
        return None
    all_ids: set[int] = set()
    evicted_ids: set[int] = set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            with contextlib.suppress(KeyError, ValueError):
                nid = int(row["id"])
                all_ids.add(nid)
                if row.get("evicted", ""):
                    evicted_ids.add(nid)
    pool = all_ids - evicted_ids
    return pool or None


def resolve_run_dirs(path: Path, top: int | None) -> list[Path]:
    runs_dir = project_runs_dir(path)
    if runs_dir is None:
        if (path / "nodes").is_dir() or (path / "lineage.csv").exists():
            return [path]
        # recurse
        found = sorted(path.rglob("runs"), key=lambda p: str(p))
        dirs = []
        for rd in found:
            dirs.extend(run_dirs_in(rd))
        return dirs[-top:] if top else dirs

    run_dirs = run_dirs_in(runs_dir)
    return run_dirs[-top:] if top else run_dirs


# ── Pareto helper ──────────────────────────────────────────────────────────────


def _pareto_top10(lin: list[dict], pool_ids: set[int] | None = None) -> list[dict]:
    """Return up to 10 Pareto-front nodes, sorted by score.

    Minimises the same three objectives the search selects on: score, visual
    the structural and chromatic measures. Note the front plot draws only two
    of them, so a front node can look dominated in that 2-D projection while
    being genuinely non-dominated in three.

    If *pool_ids* is provided only nodes in the final active pool are
    considered; otherwise all lineage nodes are used as a fallback.
    """
    candidates = lin if pool_ids is None else [r for r in lin if r["id"] in pool_ids]
    valid = [r for r in candidates if r["score"] < float("inf")]
    if not valid:
        return []
    front = pareto_front(
        valid, key=lambda r: (r["score"], *(r[m] for m in METRIC_NAMES))
    )
    return sorted(front, key=lambda r: r["score"])[:10]


# ── Plot helpers ──────────────────────────────────────────────────────────────

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# Distinct highlight colors for top-10 Pareto nodes (cycled if >10)
PARETO_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
    "#66c2a5",
    "#fc8d62",
]


@functools.cache
def plot_score_history(ax, runs: list[tuple[Path, dict]], lineages: list[list[dict]]):
    ax.set_title("Best score over time")
    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel("Score (lower = better)")
    ax.grid(True, color="grey", alpha=0.15, linewidth=0.5)

    for i, ((run_dir, stats), _lin) in enumerate(zip(runs, lineages, strict=False)):
        history = stats.get("score_history", [])
        if not history:
            continue
        xs = [0.0] + [t for t, _ in history]
        ys = [history[0][1]] + [s for _, s in history]
        color = COLORS[i % len(COLORS)]
        ax.step(xs, ys, where="post", color=color, label=run_dir.name, linewidth=1.5)
        ax.scatter(
            [t for t, _ in history],
            [s for _, s in history],
            color=color,
            s=20,
            zorder=3,
        )

        # Epoch transition lines
        ch = stats.get("convergence_history", [])
        prev_epoch = ch[0][3] if ch else 0
        for elapsed, _div, _std, ep, _ar in ch:
            if ep != prev_epoch:
                ax.axvline(
                    elapsed, color="grey", linewidth=0.8, linestyle=":", alpha=0.8
                )
                ax.text(
                    elapsed, ys[-1], f" e{ep}", fontsize=8, color="grey", va="bottom"
                )
                prev_epoch = ep

    if len(runs) > 1:
        ax.legend(fontsize=7, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))


def plot_pareto(
    ax,
    runs: list[tuple[Path, dict]],
    lineages: list[list[dict]],
    pool_ids_list: list[set[int] | None],
):
    axis_metric = METRIC_NAMES[0]
    axis_name = axis_metric.replace("_", " ")
    ax.set_title(f"Score vs {axis_name}")
    ax.set_xlabel(
        f"{axis_name.capitalize()}  "
        f"(stars: {len(METRIC_NAMES) + 1}-objective Pareto front)"
    )
    ax.set_ylabel("Score")
    ax.grid(True, color="grey", alpha=0.15, linewidth=0.5)

    for i, ((run_dir, _), lin, pool_ids) in enumerate(
        zip(runs, lineages, pool_ids_list, strict=False)
    ):
        candidates = (
            lin if pool_ids is None else [r for r in lin if r["id"] in pool_ids]
        )
        valid = [r for r in candidates if r["score"] < float("inf")]
        if not valid:
            continue
        color = COLORS[i % len(COLORS)]
        ax.scatter(
            [r[axis_metric] for r in valid],
            [r["score"] for r in valid],
            s=6,
            alpha=0.3,
            color=color,
            label=run_dir.name if len(runs) > 1 else None,
        )

        top10 = _pareto_top10(lin, pool_ids)
        for j, node in enumerate(top10):
            pc = PARETO_COLORS[j % len(PARETO_COLORS)]
            ax.scatter(
                [node[axis_metric]],
                [node["score"]],
                marker="*",
                s=160,
                color=pc,
                zorder=5,
                edgecolors="black",
                linewidths=0.4,
            )
            ax.annotate(
                f"#{node['id']}",
                (node[axis_metric], node["score"]),
                fontsize=8,
                color=pc,
                xytext=(4, 4),
                textcoords="offset points",
            )

    if len(runs) > 1:
        ax.legend(fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))


def plot_convergence(ax, runs: list[tuple[Path, dict]], lineages: list[list[dict]]):  # noqa: ARG001
    ax.set_title("Pool convergence over time")
    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel("Pool diversity", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_yscale("log")

    ax2 = ax.twinx()
    ax2.set_ylabel("Score variance", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_yscale("log")
    ax2.grid(True, color="grey", alpha=0.15, linewidth=0.5)

    for i, (run_dir, stats) in enumerate(runs):
        ch = stats.get("convergence_history", [])
        if not ch:
            continue
        color_div = COLORS[i % len(COLORS)]
        color_std = COLORS[(i + 1) % len(COLORS)]
        xs = [r[0] for r in ch]
        diversities = [r[1] for r in ch]
        variances = [r[2] ** 2 for r in ch]
        ax.plot(
            xs,
            diversities,
            color=color_div,
            linewidth=1.2,
            label=f"{run_dir.name} diversity",
        )
        ax2.plot(
            xs,
            variances,
            color=color_std,
            linewidth=1.2,
            linestyle="--",
            label=f"{run_dir.name} score variance",
        )

        prev_epoch = ch[0][3] if ch else 0
        for elapsed, _div, _std, ep, _ar in ch:
            if ep != prev_epoch:
                ax.axvline(
                    elapsed, color="grey", linewidth=0.8, linestyle=":", alpha=0.8
                )
                ax.text(
                    elapsed,
                    0.02,
                    f" e{ep}",
                    fontsize=8,
                    color="grey",
                    va="bottom",
                    transform=ax.get_xaxis_transform(),
                )
                prev_epoch = ep

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper right")


def plot_summary_text(
    ax,
    runs: list[tuple[Path, dict]],
    lineages: list[list[dict]],
    pool_ids_list: list[set[int] | None],
):
    ax.axis("off")
    ax.set_title("Run summary", loc="left")
    lines = []
    for (run_dir, stats), lin, pool_ids in zip(
        runs, lineages, pool_ids_list, strict=False
    ):
        if not stats:
            continue
        best = stats.get("best_score")
        best_str = f"{best:.6f}" if best is not None and best < float("inf") else "—"
        lines.append(f"[{run_dir.name}]")
        lines.append(f"  best score      {best_str}")
        lines.append(f"  elapsed         {stats.get('elapsed_seconds', 0):.0f}s")
        lines.append(f"  tasks           {int(stats.get('tasks_completed') or 0):,}")
        acc_pct = stats.get("accept_rate", 0) * 100
        lines.append(
            f"  accepted        {int(stats.get('accepted_count') or 0):,}"
            f"  ({acc_pct:.1f}%)"
        )
        rej_pct = stats.get("pool_rejected_rate", 0) * 100
        lines.append(
            f"  pool-rejected   {int(stats.get('pool_rejected_count') or 0):,}"
            f"  ({rej_pct:.1f}%)"
        )
        inv_pct = stats.get("invalid_rate", 0) * 100
        lines.append(
            f"  invalid         {int(stats.get('invalid_count') or 0):,}"
            f"  ({inv_pct:.1f}%)"
        )
        lines.append(f"  llm calls       {int(stats.get('llm_call_count') or 0):,}")
        lines.append(f"  llm valid       {stats.get('llm_valid_rate', 0) * 100:.1f}%")
        lines.append(f"  llm accept      {stats.get('llm_accept_rate', 0) * 100:.1f}%")
        lines.append(f"  seeds/epoch     {int(stats.get('seeds_target') or 0)}")
        lines.append(
            f"  mut calls       {int(stats.get('mutation_call_count') or 0):,}"
        )
        lines.append(
            f"  mut accept      {stats.get('mutation_accept_rate', 0) * 100:.1f}%"
        )
        lines.append(f"  epochs          {int(stats.get('epochs_completed') or 0)}")
        lines.append(f"  diversity       {stats.get('pool_diversity_final', 0):.4f}")
        std = stats.get("pool_score_std_final", 0)
        lines.append(f"  score variance  {std**2:.6f}")

        pool_note = "" if pool_ids is None else " (final pool)"
        top10 = _pareto_top10(lin, pool_ids)
        if top10:
            lines.append("")
            lines.append(f"  pareto top 10{pool_note}:")
            metric_head = "".join(f"  {m:>9}" for m in METRIC_NAMES)
            lines.append(f"  {'#':>2}  {'id':>6}  {'score':>10}{metric_head}  ep")
            for rank, node in enumerate(top10, 1):
                # `.4g` keeps the byte-count metrics readable while still
                # showing the sub-1.0 region distances as something but zero.
                metric_cells = "".join(f"  {node[m]:>9.4g}" for m in METRIC_NAMES)
                lines.append(
                    f"  {rank:>2}  {node['id']:>6}  {node['score']:>10.6f}"
                    f"{metric_cells}  {node['epoch']}"
                )
        lines.append("")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        fontfamily="monospace",
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help="Run dir, runs/ dir, output.svg, or project dir.")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Save plot to this file (png/pdf/svg). Default: show interactively.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        metavar="N",
        help="Only plot the N most recent runs.",
    )
    args = parser.parse_args()

    run_dirs = resolve_run_dirs(Path(args.path), args.top)
    if not run_dirs:
        print(f"No run directories found at {args.path}", file=sys.stderr)
        sys.exit(1)

    runs = [(d, load_stats(d)) for d in run_dirs]
    lineages = [load_lineage(d) for d in run_dirs]
    pool_ids_all = [load_final_pool_ids(d) for d in run_dirs]

    # Filter to runs that have any data at all
    combined = [
        (r, s, lin, pids)
        for r, s, lin, pids in zip(
            run_dirs, [s for _, s in runs], lineages, pool_ids_all, strict=False
        )
        if s or lin
    ]
    if not combined:
        print(
            "No stats.csv or lineage.csv found in the given run directories.",
            file=sys.stderr,
        )
        sys.exit(1)
    run_dirs_f = [r for r, _, _, _ in combined]
    runs_f = [(r, s) for r, s, _, _ in combined]
    lineages_f = [lin for _, _, lin, _ in combined]
    pool_ids_f = [pids for _, _, _, pids in combined]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    title = run_dirs_f[0].parent.name if len(run_dirs_f) == 1 else str(Path(args.path))
    fig.suptitle(
        f"Vectrify run stats — {title}",
        fontsize=11,
        fontweight="bold",
    )
    plt.subplots_adjust(
        hspace=0.35, wspace=0.32, left=0.07, right=0.98, top=0.93, bottom=0.06
    )

    plot_score_history(axes[0, 0], runs_f, lineages_f)
    plot_pareto(axes[0, 1], runs_f, lineages_f, pool_ids_f)
    plot_convergence(axes[1, 0], runs_f, lineages_f)
    plot_summary_text(axes[1, 1], runs_f, lineages_f, pool_ids_f)

    if args.output:
        out = Path(args.output)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        if matplotlib.get_backend().lower() == "agg":
            out = Path("/tmp/vectrify_plot.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"No GUI backend available. Saved to {out}", file=sys.stderr)
            print("Use --output FILE to save to a specific path.")
        else:
            plt.show()


if __name__ == "__main__":
    main()
