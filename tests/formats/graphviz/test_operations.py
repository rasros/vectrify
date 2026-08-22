import shutil

import pytest

from vectrify.formats.graphviz.operations import (
    _node_graft,
    _parse_node_names,
    _random_edge_attr_tweak,
    _random_layout_tweak,
    _random_node_attr_tweak,
    _set_graph_attr,
    apply_crossover,
    apply_mutation,
)

_DOT_AVAILABLE = shutil.which("dot") is not None

_DOT = """digraph G {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    A -> B;
    B -> C;
}"""


def test_parse_node_names_extracts_names():
    names = _parse_node_names(_DOT)
    assert "A" in names
    assert "B" in names


def test_parse_node_names_empty_string():
    assert _parse_node_names("") == []


def test_set_graph_attr_replaces_existing():
    dot = "digraph G { rankdir=TB; }"
    result = _set_graph_attr(dot, "rankdir", "LR")
    assert "rankdir=LR" in result
    assert "rankdir=TB" not in result


def test_set_graph_attr_inserts_new():
    dot = "digraph G { A -> B; }"
    result = _set_graph_attr(dot, "splines", "ortho")
    assert "splines=ortho" in result


def test_set_graph_attr_preserves_bracketed_graph_attributes():
    dot = "digraph G {\n    graph [rankdir=LR, ranksep=0.65, nodesep=0.4];\n}"
    result = _set_graph_attr(dot, "ranksep", "1.2")
    assert "graph [rankdir=LR, ranksep=1.2, nodesep=0.4];" in result


@pytest.mark.parametrize(
    "op",
    [_random_node_attr_tweak, _random_edge_attr_tweak, _random_layout_tweak],
)
def test_tweak_keeps_document_structure(op):
    result = op(_DOT)
    assert "digraph" in result
    assert "A -> B" in result


def test_random_layout_tweak_changes_something():
    changed = False
    for _ in range(30):
        if _random_layout_tweak(_DOT) != _DOT:
            changed = True
            break
    assert changed


@pytest.mark.skipif(not _DOT_AVAILABLE, reason="graphviz system binary not installed")
def test_layout_tweaks_keep_bracketed_graph_attributes_valid():
    dot = (
        "digraph G {\n"
        "    graph [rankdir=LR, ranksep=0.65, nodesep=0.4];\n"
        "    A -> B;\n}"
    )
    from vectrify.formats.graphviz.plugin import GraphvizPlugin

    plugin = GraphvizPlugin()
    for _ in range(40):
        valid, error = plugin.validate(_random_layout_tweak(dot))
        assert valid, error


def test_apply_mutation_returns_dot_string():
    result, summary = apply_mutation(_DOT)
    assert "digraph" in result or "graph" in result
    assert summary.startswith("Mutation: ")


def test_apply_crossover_returns_dot_string():
    dot_b = """digraph H {
        node [shape=ellipse, fillcolor=lightgreen];
        X -> Y;
    }"""
    result, summary = apply_crossover(_DOT, dot_b)
    assert "digraph" in result
    assert summary.startswith("Crossover:")


def test_crossover_falls_back_to_mutation_when_no_attrs_in_b():
    dot_b = "digraph H { X -> Y; }"  # no attribute blocks
    _result, summary = apply_crossover(_DOT, dot_b)
    assert summary.startswith("Mutation: ")


def test_crossover_survives_backslash_label_escapes():
    """Regression: the donor attribute block was concatenated into an re.sub
    *replacement template*, so its backslashes were interpreted. \\l, \\r and
    \\N are ordinary Graphviz label escapes, and the crossover died with
    `re.error: bad escape \\l`, silently failing the task.
    """
    from vectrify.formats.graphviz.operations import apply_crossover

    donor = (
        'digraph G {\n    node [shape=box, label="left\\lright\\l"];\n    a -> b;\n}'
    )
    target = "digraph G {\n    a -> b;\n}"

    content, summary = apply_crossover(target, donor)

    assert isinstance(content, str)
    assert isinstance(summary, str)


def test_insert_after_first_brace_keeps_escapes_literal():
    from vectrify.formats.graphviz.operations import _insert_after_first_brace

    block = 'node [label="a\\lb", tooltip="c\\Nd"];'
    out = _insert_after_first_brace("digraph G {\n  a -> b;\n}", block)
    assert block in out  # verbatim, not re-interpreted
    assert out.index(block) > out.index("{")


def test_insert_after_first_brace_without_a_brace_is_a_noop():
    from vectrify.formats.graphviz.operations import _insert_after_first_brace

    assert _insert_after_first_brace("not a graph", "node [];") == "not a graph"


def test_node_mutation_edits_an_explicit_node_not_global_defaults():
    dot = """digraph G {
    node [shape=box];
    A [label=\"first\", shape=ellipse];
    B [label=\"second\"];
    A -> B;
}"""
    result = _random_node_attr_tweak(dot)
    assert "node [shape=box]" in result
    assert result != dot


def test_node_graft_renames_colliding_donor_and_keeps_incident_edge():
    target = "digraph G {\n    A -> B;\n}"
    donor = """digraph H {
    A [label=\"donor\"];
    A -> C [label=\"go\"];
}"""
    # choose the first donor node deterministically for the collision case
    import random

    state = random.getstate()
    random.seed(1)
    try:
        result = _node_graft(target, donor)
    finally:
        random.setstate(state)
    assert result is not None
    assert "donor_A" in result
    assert "donor_A -> C" in result
