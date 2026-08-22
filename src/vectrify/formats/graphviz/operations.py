"""Syntax-preserving local search operations for DOT diagrams.

DOT is a graph language, so its useful crossover unit is a node plus its
incident edges (or a whole subgraph), rather than renderer-default lines.
These helpers intentionally support conventional, one-statement-per-line DOT
and leave unfamiliar statements untouched.
"""

import random
import re
from collections.abc import Callable

from vectrify.formats.mutations import MutationTable, pick_operator

_NODE_SHAPES = ["box", "ellipse", "circle", "diamond", "hexagon", "octagon"]
_EDGE_STYLES = ["solid", "dashed", "dotted", "bold"]
_COLORS = [
    "black",
    "gray",
    "lightblue",
    "lightgreen",
    "lightyellow",
    "orange",
    "red",
    "blue",
    "green",
    "purple",
]
_FILL_COLORS = [
    "lightblue",
    "lightgreen",
    "lightyellow",
    "lightcoral",
    "white",
    "lightgray",
    "lavender",
]
_FONT_SIZES = ["8", "10", "12", "14", "16", "18"]
_RANK_DIRS = ["TB", "LR", "BT", "RL"]
_ARROW_HEADS = ["normal", "vee", "dot", "odot", "none", "box", "open"]
_ID = r'"(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z0-9_]*'
_NODE_RE = re.compile(
    rf"^(?P<indent>\s*)(?P<id>{_ID})\s*\[(?P<attrs>[^\]]*)\]\s*;?\s*$"
)
_EDGE_RE = re.compile(
    rf"^(?P<indent>\s*)(?P<left>{_ID})(?::[A-Za-z0-9_]+)?\s*(?P<op>->|--)\s*(?P<right>{_ID})(?::[A-Za-z0-9_]+)?(?P<tail>.*)$"
)
_DEFAULT_RE = re.compile(r"^\s*(?:node|edge|graph)\s*\[")
_SUBGRAPH_START_RE = re.compile(
    r"^\s*subgraph\s+(?:\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_]*)?\s*\{"
)


def _unquote(identifier: str) -> str:
    return identifier[1:-1] if identifier.startswith('"') else identifier


def _parse_node_names(dot: str) -> list[str]:
    """Return explicit nodes plus endpoint nodes, in source order."""
    names: list[str] = []
    for line in dot.splitlines():
        node = _NODE_RE.match(line)
        edge = _EDGE_RE.match(line)
        if node and _unquote(node.group("id")).lower() not in {"node", "edge", "graph"}:
            names.append(_unquote(node.group("id")))
        elif edge:
            names.extend([_unquote(edge.group("left")), _unquote(edge.group("right"))])
    return list(dict.fromkeys(names))


def _set_attr(attrs: str, key: str, value: str) -> str:
    pattern = re.compile(
        rf"(?P<prefix>(?:^|,)\s*{re.escape(key)}\s*=\s*)(?:\"(?:[^\"\\]|\\.)*\"|[^,\]]*)"
    )
    if pattern.search(attrs):
        return pattern.sub(lambda m: f"{m.group('prefix')}{value}", attrs, count=1)
    return f"{attrs.rstrip()}, {key}={value}" if attrs.strip() else f"{key}={value}"


def _set_graph_attr(dot: str, key: str, value: str) -> str:
    assignment = re.compile(rf"(?P<prefix>\b{re.escape(key)}\s*=\s*)[^;\n]+;")
    if assignment.search(dot):
        return assignment.sub(lambda m: f"{m.group('prefix')}{value};", dot, count=1)
    brace = dot.find("{")
    return (
        dot
        if brace < 0
        else f"{dot[: brace + 1]}\n    {key}={value};{dot[brace + 1 :]}"
    )


def _replace_line(
    dot: str, predicate: Callable[[str], bool], transform: Callable[[str], str]
) -> str:
    lines = dot.splitlines(keepends=True)
    indexes = [i for i, line in enumerate(lines) if predicate(line.rstrip("\n"))]
    if not indexes:
        return dot
    index = random.choice(indexes)
    lines[index] = transform(lines[index].rstrip("\n")) + (
        "\n" if lines[index].endswith("\n") else ""
    )
    return "".join(lines)


def _random_node_attr_tweak(dot: str) -> str:
    key, value = random.choice(
        [
            ("shape", random.choice(_NODE_SHAPES)),
            ("color", random.choice(_COLORS)),
            ("fillcolor", random.choice(_FILL_COLORS)),
            ("fontsize", random.choice(_FONT_SIZES)),
            ("penwidth", str(round(random.uniform(0.5, 4), 1))),
        ]
    )

    def is_node(line: str) -> bool:
        return _NODE_RE.match(line) is not None and not _DEFAULT_RE.match(line)

    def change(line: str) -> str:
        match = _NODE_RE.match(line)
        assert match
        attrs = _set_attr(match.group("attrs"), key, value)
        return f"{match.group('indent')}{match.group('id')} [{attrs}];"

    changed = _replace_line(dot, is_node, change)
    return (
        changed
        if changed != dot
        else _insert_after_first_brace(dot, f"    node [style=filled, {key}={value}];")
    )


def _random_edge_attr_tweak(dot: str) -> str:
    key, value = random.choice(
        [
            ("style", random.choice(_EDGE_STYLES)),
            ("color", random.choice(_COLORS)),
            ("arrowhead", random.choice(_ARROW_HEADS)),
            ("penwidth", str(round(random.uniform(0.5, 4), 1))),
        ]
    )

    def change(line: str) -> str:
        match = _EDGE_RE.match(line)
        assert match
        tail = match.group("tail")
        attr = re.search(r"\[(?P<attrs>[^\]]*)\]", tail)
        if attr:
            tail = (
                tail[: attr.start()]
                + f"[{_set_attr(attr.group('attrs'), key, value)}]"
                + tail[attr.end() :]
            )
        else:
            tail = tail.rstrip().rstrip(";") + f" [{key}={value}];"
        head = (
            f"{match.group('indent')}{match.group('left')} "
            f"{match.group('op')} {match.group('right')}"
        )
        return f"{head}{tail}"

    changed = _replace_line(dot, lambda line: _EDGE_RE.match(line) is not None, change)
    return (
        changed
        if changed != dot
        else _insert_after_first_brace(dot, f"    edge [{key}={value}];")
    )


_LAYOUT_ATTRS: list[tuple[str, Callable[[], str]]] = [
    ("rankdir", lambda: random.choice(_RANK_DIRS)),
    ("splines", lambda: random.choice(["true", "ortho", "curved", "line"])),
    ("nodesep", lambda: str(round(random.uniform(0.25, 1.5), 2))),
    ("ranksep", lambda: str(round(random.uniform(0.3, 2), 2))),
]


def _random_layout_tweak(dot: str) -> str:
    attr, maker = random.choice(_LAYOUT_ATTRS)
    return _set_graph_attr(dot, attr, maker())


def _remove_node(dot: str) -> str:
    names = _parse_node_names(dot)
    if len(names) <= 1:
        return _random_node_attr_tweak(dot)
    target = random.choice(names)
    result: list[str] = []
    for line in dot.splitlines(keepends=True):
        node = _NODE_RE.match(line.rstrip("\n"))
        edge = _EDGE_RE.match(line.rstrip("\n"))
        if node and _unquote(node.group("id")) == target:
            continue
        if edge and target in {
            _unquote(edge.group("left")),
            _unquote(edge.group("right")),
        }:
            continue
        result.append(line)
    return "".join(result)


MUTATIONS: MutationTable = (
    (_random_node_attr_tweak, "Mutation: node attributes", 3),
    (_random_edge_attr_tweak, "Mutation: edge attributes", 3),
    (_random_layout_tweak, "Mutation: layout tweak", 2),
    (_remove_node, "Mutation: removed node", 1),
)


def apply_mutation(dot: str, operator: str | None = None) -> tuple[str, str]:
    fn, name = pick_operator(MUTATIONS, operator)
    result = fn(dot)
    return (_random_layout_tweak(dot) if result == dot else result), name


def _insert_after_first_brace(dot: str, block: str) -> str:
    brace = dot.find("{")
    if brace == -1:
        return dot
    return f"{dot[: brace + 1]}\n{block}{dot[brace + 1 :]}"


def _fresh_id(name: str, used: set[str]) -> str:
    base = re.sub(r"\W+", "_", name).strip("_") or "node"
    candidate, index = f"donor_{base}", 2
    while candidate in used:
        candidate, index = f"donor_{base}_{index}", index + 1
    return candidate


def _rename_endpoint(line: str, old: str, new: str) -> str:
    return re.sub(rf'(?<![A-Za-z0-9_])"?{re.escape(old)}"?(?![A-Za-z0-9_])', new, line)


def _node_graft(dot_a: str, dot_b: str) -> str | None:
    donor_names = _parse_node_names(dot_b)
    if not donor_names:
        return None
    donor = random.choice(donor_names)
    replacement = (
        _fresh_id(donor, set(_parse_node_names(dot_a)))
        if donor in _parse_node_names(dot_a)
        else donor
    )
    selected: list[str] = []
    declaration_found = False
    for line in dot_b.splitlines():
        node, edge = _NODE_RE.match(line), _EDGE_RE.match(line)
        if node and _unquote(node.group("id")) == donor:
            selected.append(_rename_endpoint(line, donor, replacement))
            declaration_found = True
        elif edge and donor in {
            _unquote(edge.group("left")),
            _unquote(edge.group("right")),
        }:
            selected.append(_rename_endpoint(line, donor, replacement))
    if not declaration_found:
        selected.insert(0, f'    {replacement} [label="{replacement}"];')
    return _insert_after_first_brace(
        dot_a, "\n".join(f"    {line.strip()}" for line in selected)
    )


def _subgraph_blocks(dot: str) -> list[str]:
    blocks: list[str] = []
    lines, index = dot.splitlines(keepends=True), 0
    while index < len(lines):
        if not _SUBGRAPH_START_RE.match(lines[index]):
            index += 1
            continue
        depth, chunk = 0, []
        while index < len(lines):
            line = lines[index]
            chunk.append(line)
            depth += line.count("{") - line.count("}")
            index += 1
            if depth == 0:
                blocks.append("".join(chunk).strip())
                break
    return blocks


def apply_crossover(dot_a: str, dot_b: str) -> tuple[str, str]:
    """Graft a graph unit from B into A, with collision-safe node IDs."""
    subgraphs = _subgraph_blocks(dot_b)
    if subgraphs and random.choice([True, False]):
        block = random.choice(subgraphs)
        # A subgraph can contain several edges and is awkward to rename safely
        # without a full DOT AST.  Only transfer an isolated cluster; otherwise
        # use the collision-safe node-and-edge path below.
        if not set(_parse_node_names(block)) & set(_parse_node_names(dot_a)):
            return _insert_after_first_brace(dot_a, block), "Crossover: subgraph"
    result = _node_graft(dot_a, dot_b)
    if result is not None and result != dot_a:
        return result, "Crossover: node and incident edges"
    return apply_mutation(dot_a)
