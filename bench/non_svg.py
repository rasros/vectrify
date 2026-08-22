"""Small, deterministic corpus for the layout-managed vector backends.

The SVG corpus measures freeform drawing.  These cases deliberately measure
the jobs Typst and DOT can actually perform: a fixed-page infographic and a
directed pipeline.  The target is one known-good source and the five seeds are
different, valid starting decompositions.  They do not need an LLM to evolve.
"""

from __future__ import annotations

from pathlib import Path


def _typst(page: str, circle: str, card: str, line: str) -> str:
    return (
        "#set page(width: 384pt, height: 384pt, margin: 0pt)\n"
        f"#place(dx: 80pt, dy: 80pt)[#circle(radius: {circle}pt, "
        f'fill: rgb("{page}"))]\n'
        "#place(dx: 230pt, dy: 80pt)[#rect(width: 74pt, height: 74pt, "
        f'radius: 10pt, fill: rgb("{card}"))]\n'
        "#place(dx: 128pt, dy: 208pt)[#rect(width: 130pt, height: 48pt, "
        'radius: 8pt, fill: rgb("65a30d"))]\n'
        "#place(dx: 155pt, dy: 123pt)[#line(start: (0pt, 0pt), "
        f'end: (78pt, 85pt), stroke: {line}pt + rgb("334155"))]\n'
    )


TYPST_TARGET = _typst("4f7ecb", "36", "e6a23c", "4")
TYPST_SEEDS = [
    _typst("588bcf", "30", "e0a04a", "2"),
    _typst("3d6fb5", "43", "f0ad38", "6"),
    _typst("6f9bd7", "34", "d58f2c", "3"),
    _typst("4878c1", "39", "e8b14d", "5"),
    _typst("557fbb", "32", "d99a40", "7"),
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
