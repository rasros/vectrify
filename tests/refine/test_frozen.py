"""Tidy may adjust existing handles but cannot grow them at pinned anchors."""

from dataclasses import replace

import pytest

from vectrify.document import Editor, PathNode, import_svg
from vectrify.refine.frozen import without_new_handles
from vectrify.refine.simplify import curved


def pinned_geometry(path, position):
    editor = Editor(import_svg(f'<svg><path id="p" d="{path}"/></svg>'))
    node = editor.snapshot.document.geometry_for("p").subpaths[0].nodes[position]
    editor.pin_node("p", node.id)
    return editor.snapshot.document.geometry_for("p")


@pytest.mark.parametrize("position", [0, 1, 2])
def test_new_handles_are_retracted_on_both_sides_of_a_pin(position):
    original = pinned_geometry("M0 0 L30 0 L30 30", position)
    candidate = curved(original)
    result = without_new_handles(original, candidate)
    nodes = result.subpaths[0].nodes
    point = nodes[position]
    if position > 0:
        assert point.command == "C"
        assert point.values[2:4] == point.endpoint
    if position + 1 < len(nodes):
        assert nodes[position + 1].values[:2] == point.endpoint
    # The handle at the unpinned end of each segment still exists.
    assert all(n.command == "C" for n in nodes[1:])


@pytest.mark.parametrize("position", [0, 3])
def test_pinning_either_explicit_closing_knot_protects_both_sides(position):
    original = pinned_geometry("M0 0 L30 0 L30 30 L0 0 Z", position)
    result = without_new_handles(original, curved(original))
    nodes = result.subpaths[0].nodes
    assert nodes[1].values[:2] == nodes[0].endpoint
    assert nodes[-1].values[2:4] == nodes[0].endpoint
    assert nodes[2] == curved(original).subpaths[0].nodes[2]


@pytest.mark.parametrize("side", [0, 2])
def test_existing_handle_can_move_while_missing_handle_stays_retracted(side):
    path = (
        "M0 0 C10 0 20 0 30 0 C30 0 30 20 30 30"
        if side == 2
        else "M0 0 C10 0 30 0 30 0 C30 10 30 20 30 30"
    )
    original = pinned_geometry(path, 1)
    nodes = original.subpaths[0].nodes
    candidate = original
    for node in nodes[1:]:
        candidate = candidate.replace_node(
            replace(node, values=(5, 5, 25, 25, *node.endpoint))
        )
    result = without_new_handles(original, candidate).subpaths[0].nodes
    if side == 2:
        assert result[1].values[2:4] == (25, 25)
        assert result[2].values[:2] == nodes[1].endpoint
    else:
        assert result[1].values[2:4] == nodes[1].endpoint
        assert result[2].values[:2] == (5, 5)


def test_new_handle_stays_retracted_when_a_preceding_node_is_removed():
    original = pinned_geometry("M0 0 C5 0 10 0 15 0 L30 0 L30 30", 2)
    contour = original.subpaths[0]
    candidate = replace(
        original,
        subpaths=(
            replace(
                contour,
                nodes=(
                    contour.nodes[0],
                    replace(
                        contour.nodes[2], command="C", values=(10, 0, 20, 0, 30, 0)
                    ),
                    contour.nodes[3],
                ),
            ),
        ),
    )
    result = without_new_handles(original, candidate).subpaths[0].nodes
    assert result[1].values == (10, 0, 30, 0, 30, 0)


def test_materializing_an_implicit_closure_cannot_add_a_handle_at_the_pin():
    original = pinned_geometry("M0 0 L30 0 L30 30 Z", 0)
    candidate = curved(original)
    contour = candidate.subpaths[0]
    candidate = replace(
        candidate,
        subpaths=(
            replace(
                contour,
                nodes=(
                    *contour.nodes,
                    PathNode("closing", "C", (20, 20, 10, 10, 0, 0)),
                ),
            ),
        ),
    )
    result = without_new_handles(original, candidate).subpaths[0].nodes
    assert result[1].values[:2] == (0, 0)
    assert result[-1].values[2:4] == (0, 0)
