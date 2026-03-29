import shutil

import pytest
from PIL import Image

from svgizer.formats.graphviz.operations import (
    _parse_node_names,
    _random_edge_attr_tweak,
    _random_layout_tweak,
    _random_node_attr_tweak,
    _set_graph_attr,
    crossover_with_micro_search,
    mutate_with_micro_search,
)

_DOT_AVAILABLE = shutil.which("dot") is not None

_DOT = """digraph G {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    A -> B;
    B -> C;
}"""


# ── _parse_node_names ─────────────────────────────────────────────────────────


def test_parse_node_names_extracts_names():
    names = _parse_node_names(_DOT)
    assert "A" in names
    assert "B" in names


def test_parse_node_names_empty_string():
    assert _parse_node_names("") == []


# ── _set_graph_attr ───────────────────────────────────────────────────────────


def test_set_graph_attr_replaces_existing():
    dot = "digraph G { rankdir=TB; }"
    result = _set_graph_attr(dot, "rankdir", "LR")
    assert "rankdir=LR" in result
    assert "rankdir=TB" not in result


def test_set_graph_attr_inserts_new():
    dot = "digraph G { A -> B; }"
    result = _set_graph_attr(dot, "splines", "ortho")
    assert "splines=ortho" in result


# ── Attribute tweaks (pure string transformations) ────────────────────────────


def test_random_node_attr_tweak_returns_string():
    result = _random_node_attr_tweak(_DOT)
    assert isinstance(result, str)
    assert "digraph" in result


def test_random_edge_attr_tweak_returns_string():
    result = _random_edge_attr_tweak(_DOT)
    assert isinstance(result, str)
    assert "digraph" in result


def test_random_layout_tweak_returns_string():
    result = _random_layout_tweak(_DOT)
    assert isinstance(result, str)
    assert "digraph" in result


def test_random_layout_tweak_changes_something():
    changed = False
    for _ in range(30):
        if _random_layout_tweak(_DOT) != _DOT:
            changed = True
            break
    assert changed


# ── micro-search wrappers (require graphviz binary) ───────────────────────────


@pytest.mark.skipif(not _DOT_AVAILABLE, reason="graphviz system binary not installed")
def test_mutate_with_micro_search_returns_dot_string():
    target = Image.new("RGB", (32, 32), color="blue")
    result, summary = mutate_with_micro_search(_DOT, target, num_trials=3)
    assert isinstance(result, str)
    assert "digraph" in result or "graph" in result
    assert summary == "local mutation"


@pytest.mark.skipif(not _DOT_AVAILABLE, reason="graphviz system binary not installed")
def test_crossover_with_micro_search_returns_dot_string():
    dot_b = """digraph H {
        node [shape=ellipse, fillcolor=lightgreen];
        X -> Y;
    }"""
    target = Image.new("RGB", (32, 32), color="green")
    result, summary = crossover_with_micro_search(_DOT, dot_b, target, num_trials=3)
    assert isinstance(result, str)
    assert summary in ("crossover", "local mutation")


@pytest.mark.skipif(not _DOT_AVAILABLE, reason="graphviz system binary not installed")
def test_crossover_falls_back_to_mutation_when_no_attrs_in_b():
    dot_b = "digraph H { X -> Y; }"  # no attribute blocks
    target = Image.new("RGB", (32, 32), color="red")
    result, _summary = crossover_with_micro_search(_DOT, dot_b, target, num_trials=3)
    assert isinstance(result, str)
