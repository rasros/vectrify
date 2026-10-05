"""Compound paths retain contours instead of connecting or unioning them."""

import numpy as np
import pytest

from tests.document.test_document import render, select
from tests.document.test_join import pair
from vectrify.document import DocumentError, Editor, export_svg, import_svg
from vectrify.document.join import path_style, transformed_geometry


@pytest.mark.parametrize("second", ["M5 0H15V10H5Z", "M10 0H20V10H10Z"])
def test_combine_keeps_overlapping_and_touching_contours_and_pins(second):
    editor = Editor(pair(second=second), selection=select("a", "b"))
    node = editor.snapshot.document.geometry_for("a").subpaths[0].nodes[0]
    editor.pin_node("a", node.id, pinned=True)
    before = editor.snapshot
    contours = tuple(
        s for oid in ("a", "b") for s in before.document.geometry_for(oid).subpaths
    )
    with editor.transaction("Combine") as tx:
        oid = tx.combine_paths(frozenset({"b", "a"}))
    after = editor.snapshot
    assert after.document.geometry_for(oid).subpaths == contours
    assert after.document.geometry_for(oid).node(node.id).pinned
    assert after.document.element(oid).tag == "path"
    assert after.selection.object_ids == {oid}
    assert len(after.document.geometries) == 1
    np.testing.assert_array_equal(
        render(export_svg(before.document)), render(export_svg(after.document))
    )
    undone = editor.undo()
    assert undone.document == before.document
    assert undone.selection == before.selection
    assert editor.redo().document == after.document


def test_combine_preserves_open_curves_without_bridging_nearby_ends():
    doc = pair(
        'fill="none" stroke="red"',
        'fill="none" stroke="red"',
        "M10 0C15 5 20 5 25 0",
    )
    editor = Editor(doc, selection=select("a", "b"))
    with editor.transaction("Combine") as tx:
        oid = tx.combine_paths(frozenset({"a", "b"}))
    assert editor.snapshot.document.geometry_for(oid).subpaths == (
        *doc.geometry_for("a").subpaths,
        *doc.geometry_for("b").subpaths,
    )
    assert not editor.snapshot.document.geometry_for(oid).subpaths[1].closed


@pytest.mark.parametrize("source", [None, "a"])
def test_combine_uses_one_entire_style_at_frontmost_position(source):
    doc = pair(
        'fill="none" stroke="red" stroke-width="2"',
        'fill="blue" stroke="black" stroke-width="4" opacity=".5" fill-rule="evenodd"',
        extra='<rect id="middle" width="5" height="5"/>',
    )
    editor = Editor(doc, selection=select("a", "b"))
    with editor.transaction("Combine") as tx:
        oid = tx.combine_paths(frozenset({"a", "b"}), paint_source=source)
    after = editor.snapshot.document
    assert path_style(after, after.element(oid)) == path_style(
        doc, doc.element(source or "b")
    )
    assert [p.id for p in after.element("g").children] == ["middle", oid]
    assert after.element("middle") == doc.element("middle")
    assert len(after.geometry_for(oid).subpaths) == 2


def test_combine_keeps_nested_evenodd_contours_instead_of_unioning():
    doc = pair('fill-rule="evenodd"', 'fill-rule="evenodd"', "M2 2H8V8H2Z")
    editor = Editor(doc, selection=select("a", "b"))
    with editor.transaction("Combine") as tx:
        oid = tx.combine_paths(frozenset({"a", "b"}))
    after = editor.snapshot.document
    assert after.geometry_for(oid).subpaths == (
        *doc.geometry_for("a").subpaths,
        *doc.geometry_for("b").subpaths,
    )
    assert after.element(oid).get("fill-rule") == "evenodd"
    assert render(export_svg(after))[5, 5, 3] == 0


@pytest.mark.parametrize("grouped", [False, True])
def test_combine_resolves_transforms_without_losing_curves_or_pins(grouped):
    if grouped:
        content = """<g transform="translate(10 10)">
        <path id="a" d="M0 0C0 5 5 5 5 0"/></g>
        <g transform="translate(50 50)">
        <path id="b" d="M0 0C0 5 5 5 5 0"/></g>"""
    else:
        content = """<path id="a" transform="translate(10 10)"
        d="M0 0C0 5 5 5 5 0"/>
        <path id="b" transform="translate(50 50)"
        d="M0 0C0 5 5 5 5 0"/>"""
    editor = Editor(
        import_svg(f'<svg width="100" height="100">{content}</svg>'),
        selection=select("a", "b"),
    )
    node = editor.snapshot.document.geometry_for("a").subpaths[0].nodes[0]
    editor.pin_node("a", node.id, pinned=True)
    before = editor.snapshot.document
    expected = tuple(
        s
        for oid, offset in (("a", 10), ("b", 50))
        for s in transformed_geometry(
            before.geometry_for(oid), (1, 0, 0, 1, offset, offset)
        ).subpaths
    )
    with editor.transaction("Combine") as tx:
        oid = tx.combine_paths(frozenset({"a", "b"}))
    after = editor.snapshot.document
    assert after.geometry_for(oid).subpaths == expected
    assert after.geometry_for(oid).node(node.id).pinned
    np.testing.assert_array_equal(render(export_svg(before)), render(export_svg(after)))


def test_combine_nested_groups_deduplicates_and_removes_empty_selected_groups():
    doc = import_svg("""<svg width="100" height="100"><g id="g" fill="red">
    <g id="nested"><path id="a" d="M0 0H10V10H0Z"/></g>
    <path id="b" d="M5 0H15V10H5Z"/></g>
    <path id="c" fill="red" d="M10 0H20V10H10Z"/></svg>""")
    editor = Editor(doc, selection=select("g", "a", "c"))
    with editor.transaction("Combine") as tx:
        oid = tx.combine_paths(frozenset({"g", "a", "c"}))
    after = editor.snapshot.document
    assert [e.id for e in after.root.children] == [oid]
    assert after.geometry_for(oid).subpaths == tuple(
        s for p in ("a", "b", "c") for s in doc.geometry_for(p).subpaths
    )
    assert editor.undo().document == doc


@pytest.mark.parametrize("lock", ["geometry", "structure", "paint"])
def test_combine_obeys_locks_and_rejects_atomically(lock):
    editor = Editor(pair('fill="red"', 'fill="blue"'), selection=select("a"))
    editor.set_locks("a", frozenset({lock}))
    editor.select(select("a", "b"))
    before = editor.snapshot
    with (
        pytest.raises(DocumentError, match="locked"),
        editor.transaction("Combine") as tx,
    ):
        tx.combine_paths(frozenset({"a", "b"}))
    assert editor.snapshot == before


def test_combine_rejects_clipping_that_would_cut_geometry():
    doc = import_svg("""<svg width="100" height="100">
    <defs><clipPath id="clip"><rect width="10" height="10"/></clipPath></defs>
    <g clip-path="url(#clip)"><path id="a" d="M0 0H20V20H0Z"/></g>
    <path id="b" d="M5 5H25V25H5Z"/></svg>""")
    editor = Editor(doc, selection=select("a", "b"))
    before = editor.snapshot
    with (
        pytest.raises(DocumentError, match="clipping"),
        editor.transaction("Combine") as tx,
    ):
        tx.combine_paths(frozenset({"a", "b"}))
    assert editor.snapshot == before


def test_combine_with_common_clipping_keeps_exact_contours_and_clip():
    doc = import_svg("""<svg width="100" height="100">
    <defs><clipPath id="clip"><rect width="10" height="10"/></clipPath></defs>
    <g id="g" clip-path="url(#clip)"><path id="a" d="M0 0H20V20H0Z"/>
    <path id="b" d="M5 5H25V25H5Z"/></g></svg>""")
    editor = Editor(doc, selection=select("a", "b"))
    with editor.transaction("Combine") as tx:
        oid = tx.combine_paths(frozenset({"a", "b"}))
    after = editor.snapshot.document
    assert after.ancestry(oid)[-2].id == "g"
    assert after.element("g").get("clip-path") == "url(#clip)"
    assert after.geometry_for(oid).subpaths == (
        *doc.geometry_for("a").subpaths,
        *doc.geometry_for("b").subpaths,
    )
