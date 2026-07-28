import logging
import random
import re
from collections.abc import Callable

from PIL import Image

from vectrify.formats.micro_search import with_micro_search

log = logging.getLogger(__name__)

_NODE_SHAPES = [
    "box",
    "ellipse",
    "circle",
    "diamond",
    "parallelogram",
    "trapezium",
    "hexagon",
    "octagon",
    "doublecircle",
    "Mrecord",
]
_EDGE_STYLES = ["solid", "dashed", "dotted", "bold"]
_COLORS = [
    "black",
    "gray",
    "lightblue",
    "lightgreen",
    "lightyellow",
    "orange",
    "pink",
    "red",
    "blue",
    "green",
    "purple",
    "white",
]
_FILL_COLORS = [
    "lightblue",
    "lightgreen",
    "lightyellow",
    "lightcoral",
    "lightsalmon",
    "white",
    "lightgray",
    "bisque",
    "lavender",
]
_FONT_SIZES = ["8", "10", "12", "14", "16", "18"]
_RANK_DIRS = ["TB", "LR", "BT", "RL"]
_ARROW_HEADS = ["normal", "vee", "dot", "odot", "none", "box", "open"]
_PORTS = ["n", "ne", "e", "se", "s", "sw", "w", "nw", "c"]
_NODE_KEYWORDS = frozenset({"node", "edge", "graph", "strict", "digraph", "subgraph"})


def _parse_node_names(dot: str) -> list[str]:
    """Extract node identifiers from a DOT string."""
    names = []
    for m in re.finditer(
        r'^\s*"?([A-Za-z0-9_]+)"?\s*(?:\[|;|->|--|\n)', dot, re.MULTILINE
    ):
        names.append(m.group(1))
    return list(dict.fromkeys(names))


def _set_graph_attr(dot: str, key: str, value: str) -> str:
    """Set or replace a top-level graph attribute."""
    pattern = rf"({key}\s*=\s*)[^\s;,\]]+"
    if re.search(pattern, dot):
        return re.sub(pattern + r"([;\s])", rf"\g<1>{value}\2", dot, count=1)
    return re.sub(r"(\{)", rf"\1\n    {key}={value};", dot, count=1)


def _random_node_attr_tweak(dot: str) -> str:
    op = random.choice(["shape", "color", "fillcolor", "fontsize", "style", "penwidth"])
    if op == "shape":
        val = random.choice(_NODE_SHAPES)
        if re.search(r"shape\s*=", dot):
            dot = re.sub(r"shape\s*=\s*\w+", f"shape={val}", dot, count=1)
        elif "node [" in dot:
            dot = re.sub(r"node\s*\[", f"node [shape={val},", dot, count=1)
    elif op == "color":
        val = random.choice(_COLORS)
        if re.search(r"color\s*=", dot):
            dot = re.sub(r"color\s*=\s*\w+", f"color={val}", dot, count=1)
    elif op == "fillcolor":
        val = random.choice(_FILL_COLORS)
        if re.search(r"fillcolor\s*=", dot):
            dot = re.sub(r"fillcolor\s*=\s*\w+", f"fillcolor={val}", dot, count=1)
        else:
            dot = dot + f"\n    node [style=filled, fillcolor={val}];"
    elif op == "fontsize":
        val = random.choice(_FONT_SIZES)
        if re.search(r"fontsize\s*=", dot):
            dot = re.sub(r"fontsize\s*=\s*\d+", f"fontsize={val}", dot, count=1)
    elif op == "style":
        val = random.choice(["filled", "dashed", "bold", "dotted"])
        if re.search(r"style\s*=", dot):
            dot = re.sub(r"style\s*=\s*\w+", f"style={val}", dot, count=1)
    elif op == "penwidth":
        val = str(round(random.uniform(0.5, 4.0), 1))
        if re.search(r"penwidth\s*=", dot):
            dot = re.sub(r"penwidth\s*=\s*[\d.]+", f"penwidth={val}", dot, count=1)
        elif "node [" in dot:
            dot = re.sub(r"node\s*\[", f"node [penwidth={val},", dot, count=1)
        else:
            dot = dot + f"\n    node [penwidth={val}];"
    return dot


def _random_edge_attr_tweak(dot: str) -> str:
    op = random.choice(["style", "color", "arrowhead", "penwidth", "headport"])
    if op == "style":
        val = random.choice(_EDGE_STYLES)
        if re.search(r"edge\s*\[", dot):
            dot = re.sub(r"edge\s*\[", f"edge [style={val},", dot, count=1)
        else:
            dot = dot + f"\n    edge [style={val}];"
    elif op == "color":
        val = random.choice(_COLORS)
        if "edge [" in dot:
            dot = re.sub(r"edge\s*\[", f"edge [color={val},", dot, count=1)
        else:
            dot = dot + f"\n    edge [color={val}];"
    elif op == "arrowhead":
        val = random.choice(_ARROW_HEADS)
        if re.search(r"arrowhead\s*=", dot):
            dot = re.sub(r"arrowhead\s*=\s*\w+", f"arrowhead={val}", dot, count=1)
        else:
            dot = dot + f"\n    edge [arrowhead={val}];"
    elif op == "penwidth":
        val = str(round(random.uniform(0.5, 4.0), 1))
        if re.search(r"edge\s*\[", dot):
            dot = re.sub(r"edge\s*\[", f"edge [penwidth={val},", dot, count=1)
        else:
            dot = dot + f"\n    edge [penwidth={val}];"
    elif op == "headport":
        val = random.choice(_PORTS)
        if re.search(r"headport\s*=", dot):
            dot = re.sub(r"headport\s*=\s*\w+", f"headport={val}", dot, count=1)
        else:
            dot = dot + f"\n    edge [headport={val}];"
    return dot


# Graph attribute -> random value generator, so the dispatch is exhaustive
# by construction rather than by an if-chain with an unreachable fallthrough.
_LAYOUT_ATTRS: list[tuple[str, Callable[[], str]]] = [
    ("rankdir", lambda: random.choice(_RANK_DIRS)),
    ("splines", lambda: random.choice(["true", "false", "ortho", "curved", "line"])),
    ("nodesep", lambda: str(round(random.uniform(0.25, 1.5), 2))),
    ("ranksep", lambda: str(round(random.uniform(0.3, 2.0), 2))),
]


def _random_layout_tweak(dot: str) -> str:
    attr, make_value = random.choice(_LAYOUT_ATTRS)
    return _set_graph_attr(dot, attr, make_value())


def _remove_node(dot: str) -> str:
    """Remove a random node and all lines referencing it."""
    names = [n for n in _parse_node_names(dot) if n.lower() not in _NODE_KEYWORDS]
    if len(names) <= 1:
        return dot
    target = random.choice(names)
    esc = re.escape(target)
    lines = dot.splitlines(keepends=True)
    return "".join(line for line in lines if not re.search(rf'\b"?{esc}"?\b', line))


_MUTATIONS = [
    (_random_node_attr_tweak, "Mutation: node attributes", 3),
    (_random_edge_attr_tweak, "Mutation: edge attributes", 3),
    (_random_layout_tweak, "Mutation: layout tweak", 3),
    (_remove_node, "Mutation: removed node", 1),
]


def _apply_one_mutation(dot: str) -> tuple[str, str]:
    fns, labels, weights = zip(*_MUTATIONS, strict=True)
    fn, label = random.choices(
        list(zip(fns, labels, strict=True)), weights=list(weights), k=1
    )[0]
    return fn(dot), label


def render_dot_png(dot: str) -> bytes:
    """Render DOT source to PNG bytes at graphviz's natural size."""
    import graphviz

    return graphviz.Source(dot).pipe(format="png", quiet=True)


def _rasterize_dot(dot: str) -> bytes | None:
    try:
        return render_dot_png(dot)
    except Exception:
        return None


_ATTR_BLOCK_RE = re.compile(r"^\s*(?:node|edge|graph)\s*\[[^\]]*\];?\s*$", re.MULTILINE)


def mutate_with_micro_search(
    parent_dot: str,
    orig_img_fast: Image.Image,
    num_trials: int = 15,
) -> tuple[str, str]:
    """Generate num_trials mutations, pick the one closest to target."""
    return with_micro_search(
        lambda: _apply_one_mutation(parent_dot),
        fallback=parent_dot,
        rasterize=_rasterize_dot,
        orig_img_fast=orig_img_fast,
        num_trials=num_trials,
        default_summary="Mutation: no improvement",
    )


def crossover_with_micro_search(
    dot_a: str,
    dot_b: str,
    orig_img_fast: Image.Image,
    num_trials: int = 15,
) -> tuple[str, str]:
    """Crossover: try swapping attribute blocks between two DOT sources."""
    attrs_b = _ATTR_BLOCK_RE.findall(dot_b)
    if not attrs_b:
        return mutate_with_micro_search(dot_a, orig_img_fast, num_trials)

    def _op() -> tuple[str, str]:
        attr = random.choice(attrs_b)
        return re.sub(r"(\{)", r"\1\n" + attr, dot_a, count=1), "Crossover: attributes"

    return with_micro_search(
        _op,
        fallback=dot_a,
        rasterize=_rasterize_dot,
        orig_img_fast=orig_img_fast,
        num_trials=num_trials,
        default_summary="Crossover: no improvement",
    )
