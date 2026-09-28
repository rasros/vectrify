"""Automatic contact matching retains editable curves and durable adjacency."""

import pytest

from vectrify.document import DocumentError, Editor, Selection, import_svg
from vectrify.document.project import load_project, save_project
from vectrify.document.topology import edge


def editor(a="M0 0 L10 0 L10 20 L0 20Z", b="M10 0 L20 0 L20 20 L10 20 L10 10Z"):
    result = Editor(
        import_svg(
            f'<svg width="30" height="30">'
            f'<path id="a" d="{a}"/><path id="b" d="{b}"/></svg>'
        )
    )
    result.select(Selection(object_ids=frozenset({"a", "b"})))
    return result


def test_contact_subdivides_unequal_edges_and_propagates():
    e = editor()
    original = e.snapshot.document
    with e.transaction("Share") as tx:
        assert tx.share_boundaries(0.01) == 2
    doc = e.snapshot.document
    assert load_project(save_project(doc))[0] == doc
    boundary = doc.boundaries[0]
    ref = boundary.members[0]
    owner = next(iter(doc.geometry_users(ref.geometry_id)))
    node = doc.geometry(ref.geometry_id).node(ref.node_id)
    with e.transaction("Move shared node") as tx:
        tx.update_node(owner, node.id, (node.values[0] + 1, node.values[1]))
    changed = e.snapshot.document
    assert (
        edge(changed, boundary.members[0]).points
        == edge(changed, boundary.members[1]).points
    )
    assert changed.geometry_for("a") != doc.geometry_for("a")
    assert changed.geometry_for("b") != doc.geometry_for("b")
    e.undo()
    e.undo()
    assert e.snapshot.document == original


def test_contact_closes_small_gap_but_not_distant_regions():
    e = editor(b="M10.2 0 L20 0 L20 20 L10.2 20Z")
    with (
        pytest.raises(DocumentError, match="No matching"),
        e.transaction("Share") as tx,
    ):
        tx.share_boundaries(0.05)
    with e.transaction("Share") as tx:
        assert tx.share_boundaries(0.3) >= 1
    for boundary in e.snapshot.document.boundaries:
        assert (
            edge(e.snapshot.document, boundary.members[0]).points
            == edge(e.snapshot.document, boundary.members[1]).points
        )


def test_contact_curves_match_in_reverse_without_flattening():
    e = editor(
        a="M0 0 L10 0 C14 5 14 15 10 20 L0 20Z",
        b="M10 20 C14 15 14 5 10 0 L20 0 L20 20Z",
    )
    with e.transaction("Share") as tx:
        assert tx.share_boundaries(0.01) == 1
    doc = e.snapshot.document
    assert len(edge(doc, doc.boundaries[0].members[0]).points) == 4


def test_contact_pins_and_locks_reject_without_partial_changes():
    e = editor(b="M10.2 0 L20 0 L20 20 L10.2 20Z")
    n = e.snapshot.document.geometry_for("a").subpaths[0].nodes[1]
    e.pin_node("a", n.id)
    before = e.snapshot
    with pytest.raises(DocumentError, match="pinned"), e.transaction("Share") as tx:
        tx.share_boundaries(0.3)
    assert e.snapshot == before


def test_contact_subdivides_curves_with_different_node_spacing():
    e = editor(
        a="M0 0 L10 0 C14 5 14 15 10 20 L0 20Z",
        b="M10 20 C12 17.5 13 13.75 13 10 C13 6.25 12 2.5 10 0 L20 0 L20 20Z",
    )
    with e.transaction("Share") as tx:
        assert tx.share_boundaries(0.001) == 2
    assert all(
        len(edge(e.snapshot.document, b.members[0]).points) == 4
        for b in e.snapshot.document.boundaries
    )


def test_corner_contact_does_not_link_crossing_edges():
    e = editor(a="M0 0L10 0L10 10L0 10Z", b="M10 10L20 10L20 20L10 20Z")
    with (
        pytest.raises(DocumentError, match="No matching"),
        e.transaction("Share") as tx,
    ):
        tx.share_boundaries(0.1)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("translate(7 9)", "translate(-4 3)"),
        ("scale(2 3)", "rotate(90)"),
        ("matrix(1 .25 .5 1 8 -3)", "matrix(-2 0 0 3 11 4)"),
    ],
)
def test_transformed_contact_preserves_tree_and_edits_in_common_frame(first, second):
    import numpy as np

    from vectrify.document.hit_test import transform
    from vectrify.document.join import transformed_geometry
    from vectrify.document.topology import close_points, inverse_matrix

    canonical = editor().snapshot.document
    a = transformed_geometry(
        canonical.geometry_for("a"), inverse_matrix(transform(first))
    )
    b = transformed_geometry(
        canonical.geometry_for("b"), inverse_matrix(transform(second))
    )
    e = Editor(
        import_svg(
            f'<svg width="30" height="30"><g transform="{first}" fill="red">'
            f'<path id="a" stroke="blue" stroke-width=".2" d="{a.path_data()}"/>'
            '<path id="unselected" d="M0 0L1 0L1 1Z"/></g>'
            f'<g transform="{second}" fill="green">'
            f'<path id="b" d="{b.path_data()}"/></g></svg>'
        )
    )
    e.select(Selection(object_ids=frozenset({"a", "b"})))
    before = e.snapshot.document
    with e.transaction("Share") as tx:
        assert tx.share_boundaries(0.001) == 2
    doc = e.snapshot.document
    assert doc.root == before.root
    assert doc.geometry_for("unselected") == before.geometry_for("unselected")
    assert load_project(save_project(doc))[0] == doc
    member = doc.boundaries[0].members[0]
    owner = next(iter(doc.geometry_users(member.geometry_id)))
    n = doc.geometry(member.geometry_id).node(member.node_id)
    with e.transaction("Edit") as tx:
        tx.update_node(owner, n.id, (n.values[0] + 0.4, n.values[1] + 0.1))
    for boundary in e.snapshot.document.boundaries:
        assert close_points(
            edge(e.snapshot.document, boundary.members[0]).points,
            edge(e.snapshot.document, boundary.members[1]).points,
        )
    doc = e.snapshot.document
    with e.transaction("Split shared") as tx:
        tx.split_edge(owner, member.node_id, 0.3)
    assert len(e.snapshot.document.boundaries) == len(doc.boundaries) + 1
    e.snapshot.document.validate()
    assert np.isfinite(
        [
            v
            for g in e.snapshot.document.geometries
            for s in g.subpaths
            for n in s.nodes
            for v in n.values
        ]
    ).all()


def test_nonuniform_transform_uses_canvas_contact_distance():
    e = Editor(
        import_svg(
            '<svg><g transform="scale(100 1)">'
            '<path id="a" d="M0 0L.1 0L.1 10L0 10Z"/></g>'
            '<path id="b" d="M10.5 0L20 0L20 10L10.5 10Z"/></svg>'
        )
    )
    e.select(Selection(object_ids=frozenset({"a", "b"})))
    with e.transaction("Share") as tx:
        assert tx.share_boundaries(0.6) == 1


def test_singular_transform_fails_without_modifying_drawing():
    e = Editor(
        import_svg(
            '<svg><path id="a" transform="scale(0 1)" '
            'd="M0 0L10 0L10 10Z"/><path id="b" d="M0 0L10 0L10 10Z"/></svg>'
        )
    )
    e.select(Selection(object_ids=frozenset({"a", "b"})))
    before = e.snapshot
    with pytest.raises(DocumentError, match="collapsed"), e.transaction("Share") as tx:
        tx.share_boundaries(1)
    assert e.snapshot == before
