"""Geometry reduction preserves topology, constraints and exact undo state."""

import math
from dataclasses import replace

import pytest
from shapely import hausdorff_distance
from shapely.geometry import LineString

from vectrify.document import DocumentError, Editor, Selection, import_svg
from vectrify.document.simplify import SimplifyOptions, sampled, simplify_geometry


def circle(radius=30, center=50, count=120, noise=0.1):
    points = []
    for i in range(count):
        angle = i * 2 * math.pi / count
        r = radius + noise * math.sin(i * 1.7)
        points.append((center + r * math.cos(angle), center + r * math.sin(angle)))
    return "M" + " L".join(f"{x:.4f} {y:.4f}" for x, y in points) + " Z"


def document(data):
    return import_svg(
        '<svg width="100" height="100">'
        f'<path id="a" fill-rule="evenodd" d="{data}"/></svg>'
    )


def test_noisy_circle_becomes_fewer_curves_with_bounded_contour_error():
    old = document(circle()).geometry_for("a")
    new = simplify_geometry(old, SimplifyOptions(tolerance=0.75))
    assert len(new.subpaths[0].nodes) < 20
    assert any(n.command == "C" for n in new.subpaths[0].nodes)
    assert len(new.path_data()) < len(old.path_data()) / 2
    a, _ = sampled(old.subpaths[0], 0.005)
    b, _ = sampled(new.subpaths[0], 0.005)
    assert hausdorff_distance(LineString(a), LineString(b), densify=0.1) <= 0.75
    assert new.subpaths[0].nodes[0] == old.subpaths[0].nodes[0]
    assert new.subpaths[0].id == old.subpaths[0].id


def test_holes_and_disconnected_contours_are_kept():
    old = document(circle() + " " + circle(radius=8, noise=0, count=60)).geometry_for(
        "a"
    )
    new = simplify_geometry(old, SimplifyOptions(tolerance=2))
    assert len(new.subpaths) == 2
    assert [s.id for s in new.subpaths] == [s.id for s in old.subpaths]
    assert sum(len(s.nodes) for s in new.subpaths) < 30


def test_pinned_endpoints_survive_reduction_exactly():
    old = document(circle()).geometry_for("a")
    pinned = replace(old.subpaths[0].nodes[23], pinned=True)
    old = old.replace_node(pinned)
    new = simplify_geometry(old, SimplifyOptions(tolerance=2, corners=False))
    assert new.node(pinned.id).endpoint == pinned.endpoint
    assert new.node(pinned.id).pinned


def test_sharp_tips_and_open_endpoints_are_retained():
    old = document(
        "M0 0 L10 0 L20 0 L30 0 L40 0 L30 20 L20 40 L10 20 L0 0 Z"
    ).geometry_for("a")
    new = simplify_geometry(old, SimplifyOptions(tolerance=2))
    assert (20, 40) in [n.endpoint for n in new.subpaths[0].nodes]
    open_geometry = document("M0 0 L10 .1 L20 -.1 L30 .1 L40 0").geometry_for("a")
    new = simplify_geometry(open_geometry, SimplifyOptions(tolerance=0.5), filled=False)
    assert new.subpaths[0].nodes[0] == open_geometry.subpaths[0].nodes[0]
    assert new.subpaths[0].nodes[-1].endpoint == (40, 0)
    assert not new.subpaths[0].closed
    assert len(new.subpaths[0].nodes) == 2


def test_narrow_hole_and_contact_relations_do_not_collapse():
    data = (
        circle(radius=30, count=120, noise=0)
        + " "
        + circle(radius=29.8, count=120, noise=0)
    )
    old = document(data).geometry_for("a")
    new = simplify_geometry(old, SimplifyOptions(tolerance=4, corners=False))
    from shapely.geometry import Polygon

    rings = [Polygon(sampled(s, 0.01)[0]) for s in new.subpaths]
    assert rings[0].contains(rings[1])
    assert not rings[0].boundary.intersects(rings[1].boundary)


def test_simplify_transaction_supports_detached_use_and_undo():
    doc = import_svg(
        f'<svg><defs><path id="source" d="{circle()}"/></defs>'
        '<use id="a" href="#source"/><use id="b" href="#source"/></svg>'
    )
    editor = Editor(doc, selection=Selection(object_ids=frozenset({"a"})))
    with (
        pytest.raises(DocumentError, match="unselected"),
        editor.transaction("Simplify") as tx,
    ):
        tx.simplify_shapes(SimplifyOptions())
    with editor.transaction("Detach") as tx:
        tx.detach_geometry("a")
    before = editor.snapshot.document
    with editor.transaction("Simplify") as tx:
        tx.simplify_shapes(SimplifyOptions())
    assert editor.snapshot.document.geometry_for("a") != before.geometry_for("a")
    assert editor.snapshot.document.geometry_for("b") == before.geometry_for("b")
    editor.undo()
    assert editor.snapshot.document == before


@pytest.mark.parametrize("lock", ["geometry", "structure"])
def test_locks_and_node_selection_are_enforced(lock):
    doc = document(circle())
    editor = Editor(doc, selection=Selection(object_ids=frozenset({"a"})))
    editor.set_locks("a", frozenset({lock}))
    with (
        pytest.raises(DocumentError, match="locked"),
        editor.transaction("Simplify") as tx,
    ):
        tx.simplify_shapes(SimplifyOptions())
    node = doc.geometry_for("a").subpaths[0].nodes[1]
    editor = Editor(
        doc,
        selection=Selection(object_ids=frozenset({"a"}), node_ids=frozenset({node.id})),
    )
    with pytest.raises(DocumentError), editor.transaction("Simplify") as tx:
        tx.simplify_shapes(SimplifyOptions())


def test_empty_selection_cannot_simplify_drawing():
    editor = Editor(document(circle()))
    with pytest.raises(DocumentError), editor.transaction("Simplify") as tx:
        tx.simplify_shapes(SimplifyOptions())


def test_self_crossing_input_and_tiny_triangle_are_left_intact():
    for data in ["M0 0 L20 20 L0 20 L20 0 Z", "M0 0 L1 0 L0 1 Z"]:
        old = document(data).geometry_for("a")
        assert simplify_geometry(old, SimplifyOptions(tolerance=10)) == old


def test_explicit_self_contact_is_preserved_while_adjacent_edges_reduce():
    # Two square loops touch at the shared origin; the second visit is an
    # explicit vertex in a single compound contour, not a crossing to repair.
    path = "M0 0 L5 0 L10 0 L10 5 L10 10 L5 10 L0 10 L0 5 L0 0 "
    path += "L-5 0 L-10 0 L-10 -5 L-10 -10 L-5 -10 L0 -10 L0 -5 Z"
    old = document(path).geometry_for("a")
    new = simplify_geometry(old, SimplifyOptions(tolerance=1))
    assert len(new.subpaths[0].nodes) < len(old.subpaths[0].nodes)
    from vectrify.document.simplify import _junctions

    assert _junctions(LineString(sampled(old.subpaths[0], 0.01)[0])) == _junctions(
        LineString(sampled(new.subpaths[0], 0.01)[0])
    )
