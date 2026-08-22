"""Small, deterministic corpus for the layout-managed vector backends.

The SVG corpus measures freeform drawing.  These cases deliberately measure
the jobs Typst and DOT can actually perform: a fixed-page infographic and a
directed pipeline.  The target is one known-good source and the five seeds are
different, valid starting decompositions.  They do not need an LLM to evolve.
"""

from __future__ import annotations

from pathlib import Path


def _typst(primary: str, accent: str, progress: str, stroke: str) -> str:
    """A compact status-card layout, rich enough to exercise scene operators."""
    return (
        "#set page(width: 384pt, height: 384pt, margin: 0pt)\n"
        "#place(dx: 40pt, dy: 36pt)[#text(size: 24pt, weight: "
        '"bold", fill: rgb("172554"))[Project pulse]]\n'
        "#place(dx: 40pt, dy: 90pt)[#rect(width: 304pt, height: 112pt, "
        f'radius: 16pt, fill: rgb("{primary}"))]\n'
        "#place(dx: 66pt, dy: 117pt)[#text(size: 15pt, fill: white)"
        "[This week]]\n"
        "#place(dx: 66pt, dy: 145pt)[#text(size: 32pt, weight: "
        '"bold", fill: white)[24 tasks]]\n'
        "#place(dx: 40pt, dy: 234pt)[#rect(width: 304pt, height: 20pt, "
        'radius: 10pt, fill: rgb("e2e8f0"))]\n'
        "#place(dx: 40pt, dy: 234pt)[#rect(width: 208pt, height: 20pt, "
        f'radius: 10pt, fill: rgb("{progress}"))]\n'
        "#place(dx: 40pt, dy: 279pt)[#circle(radius: 18pt, "
        f'fill: rgb("{accent}"))]\n'
        "#place(dx: 74pt, dy: 273pt)[#text(size: 15pt, fill: "
        'rgb("334155"))[On track for Friday]]\n'
        "#place(dx: 40pt, dy: 327pt)[#line(start: (0pt, 0pt), "
        f'end: (304pt, 0pt), stroke: {stroke}pt + rgb("cbd5e1"))]\n'
    )


TYPST_TARGET = _typst("4f7ecb", "f59e0b", "84cc16", "2")
TYPST_SEEDS = [
    _typst("588bcf", "f97316", "65a30d", "1"),
    _typst("3d6fb5", "eab308", "a3e635", "4"),
    _typst("6f9bd7", "fb923c", "4d7c0f", "3"),
    _typst("4878c1", "d97706", "bef264", "5"),
    _typst("557fbb", "fbbf24", "22c55e", "2.5"),
]


def _dot(fill: str, ranksep: str, nodesep: str, penwidth: str) -> str:
    return f'''digraph Pipeline {{
    graph [rankdir=LR, ranksep={ranksep}, nodesep={nodesep}, bgcolor="white", pad=0.15];
    node [shape=box, style="rounded,filled", fontname="Arial", fontsize=16,
          color="#334155", penwidth={penwidth}, fillcolor="{fill}", margin="0.20,0.12"];
    edge [color="#475569", penwidth={penwidth}, arrowsize=0.8];
    ingest [label="Ingest"];
    clean [label="Clean"];
    publish [label="Publish", fillcolor="#bef264"];
    ingest -> clean -> publish;
}}\n'''


DOT_TARGET = _dot("#bfdbfe", "0.65", "0.40", "1.6")
DOT_SEEDS = [
    _dot("#c7ddf8", "0.45", "0.28", "1.0"),
    _dot("#a8c9ef", "0.92", "0.65", "2.5"),
    _dot("#d1e4fb", "0.72", "0.32", "2.0"),
    _dot("#b2d3f4", "0.55", "0.52", "1.2"),
    _dot("#c5daf4", "0.83", "0.46", "2.2"),
]


CASES = {
    "typst-process": ("typst", TYPST_TARGET, TYPST_SEEDS),
    "graphviz-pipeline": ("graphviz", DOT_TARGET, DOT_SEEDS),
}


def generate(root: Path) -> None:
    """Write non-SVG seeds and renderer-produced targets.

    This is intentionally optional at regeneration time: committed targets
    let a checkout run the corpus tests even when a renderer extra is absent.
    """
    from vectrify.formats.graphviz.plugin import GraphvizPlugin
    from vectrify.formats.typst.plugin import TypstPlugin

    plugins = {"typst": TypstPlugin(), "graphviz": GraphvizPlugin()}
    suffixes = {"typst": ".typ", "graphviz": ".dot"}
    for name, (format_name, target, seeds) in CASES.items():
        case_dir = root / name
        seeds_dir = case_dir / "seeds"
        seeds_dir.mkdir(parents=True, exist_ok=True)
        suffix = suffixes[format_name]
        for stale in seeds_dir.glob(f"*{suffix}"):
            stale.unlink()
        for index, seed in enumerate(seeds, start=1):
            (seeds_dir / f"{index}{suffix}").write_text(seed, encoding="utf-8")
        plugin = plugins[format_name]
        (case_dir / "target.png").write_bytes(plugin.rasterize(target, 384, 384))
