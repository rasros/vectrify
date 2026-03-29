import io
import logging
import random
import re

from PIL import Image

log = logging.getLogger(__name__)

# ── Attribute pools ────────────────────────────────────────────────────────────

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
_LAYOUTS = ["dot", "neato", "fdp", "circo", "twopi"]
_ARROW_HEADS = ["normal", "vee", "dot", "odot", "none", "box", "open"]


def _parse_node_names(dot: str) -> list[str]:
    """Extract node identifiers from a DOT string."""
    names = []
    for m in re.finditer(
        r'^\s*"?([A-Za-z0-9_]+)"?\s*(?:\[|;|->|--|\n)', dot, re.MULTILINE
    ):
        names.append(m.group(1))
    return list(dict.fromkeys(names))  # unique, order-preserving


def _set_graph_attr(dot: str, key: str, value: str) -> str:
    """Set or replace a top-level graph attribute."""
    pattern = rf"({key}\s*=\s*)[^\s;,\]]+"
    if re.search(pattern, dot):
        return re.sub(pattern + r"([;\s])", rf"\g<1>{value}\2", dot, count=1)
    # Insert after opening brace
    return re.sub(r"(\{)", rf"\1\n    {key}={value};", dot, count=1)


def _random_node_attr_tweak(dot: str) -> str:
    op = random.choice(["shape", "color", "fillcolor", "fontsize", "style"])
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
    return dot


def _random_edge_attr_tweak(dot: str) -> str:
    op = random.choice(["style", "color", "arrowhead"])
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
    return dot


def _random_layout_tweak(dot: str) -> str:
    op = random.choice(["rankdir", "splines", "nodesep"])
    if op == "rankdir":
        val = random.choice(_RANK_DIRS)
        return _set_graph_attr(dot, "rankdir", val)
    if op == "splines":
        val = random.choice(["true", "false", "ortho", "curved", "line"])
        return _set_graph_attr(dot, "splines", val)
    if op == "nodesep":
        val = str(round(random.uniform(0.25, 1.5), 2))
        return _set_graph_attr(dot, "nodesep", val)
    return dot


def _apply_one_mutation(dot: str) -> str:
    ops = [_random_node_attr_tweak, _random_edge_attr_tweak, _random_layout_tweak]
    return random.choice(ops)(dot)


def _rasterize_dot(dot: str) -> bytes | None:
    try:
        import graphviz

        src = graphviz.Source(dot)
        return src.pipe(format="png", quiet=True)
    except Exception:
        return None


def _fast_lab_l1(png_a: bytes, png_b: bytes, size: int = 64) -> float:
    from svgizer.image_utils import resize_long_side
    from svgizer.score.utils import lab_l1

    try:
        img_a = resize_long_side(Image.open(io.BytesIO(png_a)).convert("RGB"), size)
        img_b = resize_long_side(Image.open(io.BytesIO(png_b)).convert("RGB"), size)
        return lab_l1(img_a, img_b)
    except Exception:
        return 1.0


def mutate_with_micro_search(
    parent_dot: str,
    orig_img_fast: Image.Image,
    num_trials: int = 15,
) -> tuple[str, str]:
    """Generate num_trials mutations, pick the one closest to target."""
    orig_buf = io.BytesIO()
    orig_img_fast.save(orig_buf, format="PNG")
    orig_png = orig_buf.getvalue()

    best_dot = parent_dot
    best_score = float("inf")

    for _ in range(num_trials):
        candidate = _apply_one_mutation(parent_dot)
        png = _rasterize_dot(candidate)
        if png is None:
            continue
        score = _fast_lab_l1(orig_png, png)
        if score < best_score:
            best_score = score
            best_dot = candidate

    return best_dot, "local mutation"


def crossover_with_micro_search(
    dot_a: str,
    dot_b: str,
    orig_img_fast: Image.Image,
    num_trials: int = 15,
) -> tuple[str, str]:
    """Crossover: try swapping attribute blocks between two DOT sources."""
    orig_buf = io.BytesIO()
    orig_img_fast.save(orig_buf, format="PNG")
    orig_png = orig_buf.getvalue()

    # Extract attribute lines from both
    attr_pattern = re.compile(
        r"^\s*(?:node|edge|graph)\s*\[[^\]]*\];?\s*$", re.MULTILINE
    )
    attrs_b = attr_pattern.findall(dot_b)

    if not attrs_b:
        return mutate_with_micro_search(dot_a, orig_img_fast, num_trials)

    best_dot = dot_a
    best_score = float("inf")

    for _ in range(num_trials):
        # Take dot_a and inject a random attribute block from dot_b
        attr = random.choice(attrs_b)
        candidate = re.sub(r"(\{)", r"\1\n" + attr, dot_a, count=1)
        png = _rasterize_dot(candidate)
        if png is None:
            continue
        score = _fast_lab_l1(orig_png, png)
        if score < best_score:
            best_score = score
            best_dot = candidate

    return best_dot, "crossover"
