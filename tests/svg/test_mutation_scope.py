"""A MutationScope limits which elements and edit kinds a mutation may touch."""

import random
import xml.etree.ElementTree as ET

import pytest

from vectrify.svg.operations import (
    MUTATIONS,
    OPERATOR_KINDS,
    apply_mutation,
    mutation_weights,
    scoped_mutations,
)
from vectrify.svg.selection import MutationScope

SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
    '<g id="left"><rect id="a" x="10" y="10" width="20" height="20" fill="#aa0000"/>'
    '<rect id="b" x="10" y="40" width="20" height="20" fill="#00aa00"/></g>'
    '<rect id="c" x="60" y="10" width="20" height="20" fill="#0000aa"/>'
    '<path id="d" d="M60 60 L80 60 L80 80 Z" fill="#aaaa00"/></svg>'
)


def attributes(svg):
    return {
        el.get("id"): dict(el.attrib)
        for el in ET.fromstring(svg).iter()
        if el.get("id")
    }


def order(svg, parent):
    root = ET.fromstring(svg)
    element = (
        root
        if parent is None
        else next(e for e in root.iter() if e.get("id") == parent)
    )
    return [c.get("id") for c in element]


def changed(svg):
    before, after = attributes(SVG), attributes(svg)
    return {k for k in before if before[k] != after.get(k)}


def test_every_operator_has_declared_kinds():
    assert {name for _fn, name, _w in MUTATIONS} == set(OPERATOR_KINDS)


@pytest.mark.parametrize("seed", range(40))
def test_only_scoped_elements_change(seed):
    random.seed(seed)
    scope = MutationScope(
        frozenset({"left"}), frozenset({"geometry", "paint", "structure"})
    )
    svg, _name = apply_mutation(SVG, scope=scope)
    assert changed(svg) <= {"a", "b"}
    assert order(svg, None) == order(SVG, None)


@pytest.mark.parametrize("seed", range(40))
def test_paint_only_scope_never_moves_geometry(seed):
    random.seed(seed)
    scope = MutationScope(frozenset({"a", "c", "d"}), frozenset({"paint"}))
    svg, name = apply_mutation(SVG, scope=scope)
    assert OPERATOR_KINDS[name] <= {"paint"}
    before, after = attributes(SVG), attributes(svg)
    for key in before:
        for attr in ("x", "y", "width", "height", "d"):
            assert before[key].get(attr) == after[key].get(attr)


def test_forbidden_operator_leaves_the_parent_unchanged():
    scope = MutationScope(frozenset({"a"}), frozenset({"paint"}))
    svg, name = apply_mutation(SVG, "Mutation: path nudge", scope=scope)
    assert svg == SVG
    assert name == "Mutation: path nudge"
    assert {n for _f, n, _w in scoped_mutations(scope)} == {
        "Mutation: color tweak",
        "Mutation: stroke change",
        "Mutation: dropped style property",
    }


def test_reorder_needs_both_siblings_in_scope():
    scope = MutationScope(frozenset({"a"}), frozenset({"structure"}))
    svg, _ = apply_mutation(SVG, "Mutation: reordered elements", scope=scope)
    assert svg == SVG
    scope = MutationScope(frozenset({"a", "b"}), frozenset({"structure"}))
    svg, _ = apply_mutation(SVG, "Mutation: reordered elements", scope=scope)
    assert order(svg, "left") == ["b", "a"]


def test_scope_limits_policy_weights():
    scope = MutationScope(frozenset({"a"}), frozenset({"geometry", "paint"}))
    assert set(mutation_weights(scope)) == {
        name for name, kinds in OPERATOR_KINDS.items() if kinds <= {"geometry", "paint"}
    }
    assert set(mutation_weights()) == set(OPERATOR_KINDS)


def test_unscoped_mutation_is_unchanged_by_the_scope_parameter():
    for seed in range(10):
        random.seed(seed)
        first = apply_mutation(SVG)
        random.seed(seed)
        second = apply_mutation(SVG, scope=None)
        assert first == second
