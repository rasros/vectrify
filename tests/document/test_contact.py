"""Snapping touching edges keeps curves editable and links nothing."""

import math

import pytest

from vectrify.document import DocumentError, Editor, Selection, import_svg
from vectrify.document.contact import edges, regions
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


def close(first, second, tolerance=1e-8):
    return len(first) == len(second) and all(
        math.dist(p, q) <= tolerance for p, q in zip(first, second, strict=True)
    )


def partner(doc, span, object_ids):
    """The other selected path's edge lying on *span*, in root user space."""
    frames = regions(doc, object_ids)
    for gid, matrix in frames.items():
        if gid == span.geometry_id:
            continue
        for e in edges(doc, gid, matrix):
            if close(e.points, edge(doc, span).points) or close(
                e.points[::-1], edge(doc, span).points
            ):
                return e
    return None


def snapped(e, tolerance, ids=None):
    with e.transaction("Snap edges") as tx:
        spans = tx.snap_edges(tolerance)
    ids = ids or e.snapshot.selection.object_ids
    doc = e.snapshot.document
    assert all(partner(doc, span, ids) for span in spans)
    return spans


def test_snapping_subdivides_unequal_edges_and_links_nothing():
    e = editor()
    original = e.snapshot.document
    assert len(snapped(e, 0.01)) == 2
    doc = e.snapshot.document
    assert load_project(save_project(doc))[0] == doc
    # Editing one side afterwards leaves the other where it is.
    node = doc.geometry_for("b").subpaths[0].nodes[1]
    with e.transaction("Move node") as tx:
        tx.update_node("b", node.id, (node.values[0] + 1, node.values[1]))
    assert e.snapshot.document.geometry_for("a") == doc.geometry_for("a")
    e.undo()
    e.undo()
    assert e.snapshot.document == original


def test_two_squares_with_a_small_gap_end_up_with_coincident_edges():
    e = editor(a="M0 0L10 0L10 10L0 10Z", b="M10.3 0L20 0L20 10L10.3 10Z")
    front = e.snapshot.document.geometry_for("b")
    spans = snapped(e, 0.5)
    assert len(spans) == 1
    doc = e.snapshot.document
    # The front path is the reference; the one behind moved onto it.
    assert doc.geometry_for("b") == front
    xs = {n.endpoint[0] for n in doc.geometry_for("a").subpaths[0].nodes}
    assert xs == {0.0, 10.3}
    assert e.undo_labels == ("Snap edges",)
    e.undo()
    assert e.snapshot.document.geometry_for("a").path_data() == (
        import_svg('<svg><path d="M0 0L10 0L10 10L0 10Z"/></svg>')
        .geometries[0]
        .path_data()
    )


def test_snapping_closes_small_gap_but_not_distant_regions():
    e = editor(b="M10.2 0 L20 0 L20 20 L10.2 20Z")
    with (
        pytest.raises(DocumentError, match="No touching"),
        e.transaction("Snap") as tx,
    ):
        tx.snap_edges(0.05)
    assert len(snapped(e, 0.3)) >= 1


def test_snapping_again_changes_nothing():
    e = editor(b="M10.2 0 L20 0 L20 20 L10.2 20Z")
    snapped(e, 0.3)
    doc = e.snapshot.document
    with e.transaction("Snap again") as tx:
        assert tx.snap_edges(0.3)
        assert tx.preview == doc


def test_curves_match_in_reverse_without_flattening():
    e = editor(
        a="M0 0 L10 0 C14 5 14 15 10 20 L0 20Z",
        b="M10 20 C14 15 14 5 10 0 L20 0 L20 20Z",
    )
    spans = snapped(e, 0.01)
    assert len(spans) == 1
    assert len(edge(e.snapshot.document, spans[0]).points) == 4


def test_pins_and_locks_reject_without_partial_changes():
    e = editor(b="M10.2 0 L20 0 L20 20 L10.2 20Z")
    n = e.snapshot.document.geometry_for("a").subpaths[0].nodes[1]
    e.pin_node("a", n.id)
    before = e.snapshot
    with pytest.raises(DocumentError, match="pinned"), e.transaction("Snap") as tx:
        tx.snap_edges(0.3)
    assert e.snapshot == before


def test_subdivides_curves_with_different_node_spacing():
    e = editor(
        a="M0 0 L10 0 C14 5 14 15 10 20 L0 20Z",
        b="M10 20 C12 17.5 13 13.75 13 10 C13 6.25 12 2.5 10 0 L20 0 L20 20Z",
    )
    spans = snapped(e, 0.001)
    assert len(spans) == 2
    assert all(len(edge(e.snapshot.document, s).points) == 4 for s in spans)


def test_corner_contact_does_not_snap_crossing_edges():
    e = editor(a="M0 0L10 0L10 10L0 10Z", b="M10 10L20 10L20 20L10 20Z")
    with (
        pytest.raises(DocumentError, match="No touching"),
        e.transaction("Snap") as tx,
    ):
        tx.snap_edges(0.1)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("translate(7 9)", "translate(-4 3)"),
        ("scale(2 3)", "rotate(90)"),
        ("matrix(1 .25 .5 1 8 -3)", "matrix(-2 0 0 3 11 4)"),
    ],
)
def test_transformed_paths_snap_in_the_common_frame(first, second):
    import numpy as np

    from vectrify.document.hit_test import transform
    from vectrify.document.join import transformed_geometry
    from vectrify.document.topology import inverse_matrix

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
    frames = regions(before, frozenset({"a", "b"}))
    with e.transaction("Snap") as tx:
        spans = tx.snap_edges(0.001)
    assert len(spans) == 2
    doc = e.snapshot.document
    assert doc.root == before.root
    assert doc.geometry_for("unselected") == before.geometry_for("unselected")
    assert load_project(save_project(doc))[0] == doc
    for span in spans:
        points = edge(doc, span).points
        assert any(
            close(o.points[::-1], points, 1e-6) or close(o.points, points, 1e-6)
            for gid, matrix in frames.items()
            if gid != span.geometry_id
            for o in edges(doc, gid, matrix)
        )
    assert np.isfinite(
        [
            v
            for g in doc.geometries
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
    with e.transaction("Snap") as tx:
        assert len(tx.snap_edges(0.6)) == 1


def test_singular_transform_fails_without_modifying_drawing():
    e = Editor(
        import_svg(
            '<svg><path id="a" transform="scale(0 1)" '
            'd="M0 0L10 0L10 10Z"/><path id="b" d="M0 0L10 0L10 10Z"/></svg>'
        )
    )
    e.select(Selection(object_ids=frozenset({"a", "b"})))
    before = e.snapshot
    with pytest.raises(DocumentError, match="collapsed"), e.transaction("Snap") as tx:
        tx.snap_edges(1)
    assert e.snapshot == before


def row(*extra):
    """Three squares in a row, left at the back, plus any *extra* paths."""
    paths = "".join(
        f'<path id="{i}" d="{d}"/>'
        for i, d in (
            ("left", "M0 0L10 0L10 10L0 10Z"),
            ("middle", "M10.2 0L20 0L20 10L10.2 10Z"),
            ("right", "M20.1 0L30 0L30 10L20.1 10Z"),
            *extra,
        )
    )
    return Editor(import_svg(f'<svg width="40" height="40">{paths}</svg>'))


def owner(doc, span):
    return next(iter(doc.geometry_users(span.geometry_id)))


def test_many_regions_snap_every_seam_to_the_front():
    e = row()
    right = e.snapshot.document.geometry_for("right")
    e.select(Selection(object_ids=frozenset({"left", "middle", "right"})))
    spans = snapped(e, 0.5)
    doc = e.snapshot.document
    # Each seam's front edge is the reference the region behind snapped to.
    assert [owner(doc, s) for s in spans] == ["right", "middle"]
    assert doc.geometry_for("right") == right


def test_a_snapped_seam_stays_when_the_next_neighbour_snaps():
    e = row()
    e.select(Selection(object_ids=frozenset({"middle", "right"})))
    snapped(e, 0.5)
    e.select(Selection(object_ids=frozenset({"left", "middle", "right"})))
    spans = snapped(e, 0.5)
    doc = e.snapshot.document
    assert {owner(doc, s) for s in spans} == {"right", "middle"}
    assert {n.endpoint[0] for n in doc.geometry_for("middle").subpaths[0].nodes} == {
        10.2,
        20.1,
    }


def test_many_regions_with_nothing_touching_refuse_without_changes():
    e = row(("far", "M0 30L5 30L5 35L0 35Z"))
    e.select(Selection(object_ids=frozenset({"left", "right", "far"})))
    before = e.snapshot
    with (
        pytest.raises(DocumentError, match="No touching"),
        e.transaction("Snap") as tx,
    ):
        tx.snap_edges(0.5)
    assert e.snapshot == before
