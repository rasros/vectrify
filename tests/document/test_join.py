"""Joining outlines is a geometry operation with rendering and history guards."""

from dataclasses import replace

import numpy as np
import pytest

from tests.document.test_components import drawing
from tests.document.test_document import render, select
from vectrify.document import Editor, EditRejectedError, export_svg, import_svg


def test_split_then_join_restores_exact_compound_geometry_and_selection():
    editor = Editor(drawing(), selection=select("p"))
    original = editor.snapshot.document.geometry_for("p")
    with editor.transaction("Split") as tx:
        ids = tx.split_disconnected("p")
    before = editor.snapshot
    with editor.transaction("Join") as tx:
        joined = tx.join_paths(frozenset(ids))
    after = editor.snapshot
    assert after.selection.object_ids == {joined}
    assert after.document.geometry_for(joined).subpaths == original.subpaths
    assert len(after.document.geometries) == 1
    np.testing.assert_array_equal(
        render(export_svg(before.document)), render(export_svg(after.document))
    )
    assert editor.undo().document == before.document
    assert editor.redo().selection == after.selection


def test_join_group_collapses_to_one_path_and_preserves_pins_and_inherited_paint():
    editor = Editor(drawing(attrs='opacity=".5"'), selection=select("p"))
    geometry = editor.snapshot.document.geometry_for("p")
    node = geometry.subpaths[0].nodes[0]
    editor.pin_node("p", node.id, pinned=True)
    with editor.transaction("Split") as tx:
        ids = tx.split_disconnected("p")
    before = editor.snapshot.document
    group = before.ancestry(ids[0])[-2]
    editor.select(select(group.id))
    with editor.transaction("Join") as tx:
        joined = tx.join_paths(frozenset({group.id}))
    after = editor.snapshot.document
    assert after.geometry_for(joined).node(node.id).pinned
    assert after.ancestry(joined)[-2].id == before.ancestry(group.id)[-2].id
    assert after.element(joined).get("opacity") == ".5"
    np.testing.assert_array_equal(render(export_svg(before)), render(export_svg(after)))


def pair(attrs_a="", attrs_b="", second="M40 0H50V10H40Z", extra=""):
    return import_svg(
        '<svg width="100" height="100"><g id="g" fill="red">'
        f'<path id="a" d="M0 0H10V10H0Z" {attrs_a}/>{extra}'
        f'<path id="b" d="{second}" {attrs_b}/></g></svg>'
    )


def test_nonconsecutive_regions_join_at_frontmost_with_area_weighted_paint():
    doc = pair(
        'fill="red"',
        'fill="blue"',
        "M40 0H70V10H40Z",
        '<rect id="between" width="5" height="5"/>',
    )
    editor = Editor(doc, selection=select("a", "b"))
    with editor.transaction("Join") as tx:
        joined = tx.join_paths(frozenset({"a", "b"}))
    result = editor.snapshot.document
    assert [p.id for p in result.element("g").children] == ["between", joined]
    assert result.element("between") == doc.element("between")
    assert result.element(joined).get("fill") == "#4000bf"
    assert editor.undo().document == doc


def test_nested_regions_are_unioned_instead_of_becoming_holes():
    doc = pair(
        'fill="red" fill-rule="evenodd"',
        'fill="blue" fill-rule="evenodd"',
        "M2 2H8V8H2Z",
    )
    editor = Editor(doc, selection=select("a", "b"))
    with editor.transaction("Join") as tx:
        joined = tx.join_paths(frozenset({"a", "b"}))
    result = editor.snapshot.document
    assert result.element(joined).get("fill-rule") == "nonzero"
    assert render(export_svg(result))[5, 5, 3] == 255


def test_holes_are_excluded_from_paint_weights():
    from vectrify.document.join import painted_weights

    doc = pair('fill="red" fill-rule="evenodd"', 'fill="blue"', "M40 0H46V6H40Z")
    from vectrify.document.svg import parse_path

    geometry = parse_path("M0 0H10V10H0Z M1 1H9V9H1Z")
    old = doc.geometry_for("a")
    doc = doc.replace_geometry(replace(geometry, id=old.id))
    assert painted_weights(doc, [doc.element("a"), doc.element("b")]) == [36, 36]
    editor = Editor(doc, selection=select("a", "b"))
    with editor.transaction("Join") as tx:
        joined = tx.join_paths(frozenset({"a", "b"}))
    assert editor.snapshot.document.element(joined).get("fill") == "#800080"


def test_averaging_respects_paint_locks():
    editor = Editor(pair('fill="blue"'), selection=select("a", "b"))
    editor.set_locks("a", frozenset({"fill"}))
    before = editor.snapshot
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Join") as tx,
    ):
        tx.join_paths(frozenset({"a", "b"}))
    assert editor.snapshot == before


def test_touching_opaque_stroked_outlines_can_join():
    doc = pair('fill="none" stroke="black"', 'fill="none" stroke="black"', "M10 0L20 0")
    editor = Editor(doc, selection=select("a", "b"))
    with editor.transaction("Join") as tx:
        joined = tx.join_paths(frozenset({"a", "b"}))
    assert len(editor.snapshot.document.geometry_for(joined).subpaths) == 2


def test_join_keeps_shared_boundaries_and_locks():
    from vectrify.document import EdgeRef

    doc = pair(second="M10 10H0V20H10Z")
    # Use strokes so two shapes sharing an edge can combine without fill changes.
    for oid in ("a", "b"):
        doc = doc.replace_element(
            replace(
                doc.element(oid), attributes=(("fill", "none"), ("stroke", "black"))
            )
        )
    editor = Editor(doc, selection=select("a", "b"))
    a, b = doc.geometry_for("a"), doc.geometry_for("b")
    with editor.transaction("Link") as tx:
        tx.link_boundary(
            (
                EdgeRef(a.id, a.subpaths[0].nodes[3].id),
                EdgeRef(b.id, b.subpaths[0].nodes[1].id),
            )
        )
    editor.set_locks("a", frozenset({"paint"}))
    with editor.transaction("Join") as tx:
        joined = tx.join_paths(frozenset({"a", "b"}))
    result = editor.snapshot.document
    assert result.element(joined).locks == {"paint"}
    assert {m.geometry_id for m in result.boundaries[0].members} == {
        result.geometry_for(joined).id
    }
    result.validate()


@pytest.mark.parametrize("lock", ["geometry", "structure"])
def test_join_enforces_locks(lock):
    editor = Editor(pair(), selection=select("a", "b"))
    editor.set_locks("a", frozenset({lock}))
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Join") as tx,
    ):
        tx.join_paths(frozenset({"a", "b"}))


def test_union_retains_cubic_curves_and_rejects_pinned_overlap():
    doc = pair('fill="red"', 'fill="blue"', "M5 0C20 0 20 20 5 20Z")
    editor = Editor(doc, selection=select("a", "b"))
    pinned = doc.geometry_for("a").subpaths[0].nodes[0]
    editor.pin_node("a", pinned.id)
    before = editor.snapshot
    with (
        pytest.raises(EditRejectedError, match="Unpin"),
        editor.transaction("Join") as tx,
    ):
        tx.join_paths(frozenset({"a", "b"}))
    assert editor.snapshot == before
    editor.pin_node("a", pinned.id, pinned=False)
    with editor.transaction("Join") as tx:
        joined = tx.join_paths(frozenset({"a", "b"}))
    assert any(
        n.command == "C"
        for s in editor.snapshot.document.geometry_for(joined).subpaths
        for n in s.nodes
    )


def test_transparent_paint_uses_alpha_and_zero_area_falls_back_to_equal_weights():
    from vectrify.document.join import average_paint, path_style

    doc = pair('fill="red" fill-opacity=".5"', 'fill="blue"')
    styles = [path_style(doc, doc.element(i)) for i in ("a", "b")]
    result = average_paint(styles, [100, 300])
    assert result["fill"] == "#2400db"
    assert float(result["fill-opacity"]) == 0.875
    assert average_paint(styles, [0, 0])["fill"] == "#5500aa"


@pytest.mark.parametrize(
    ("source", "fill", "stroke"), [("a", "red", "green"), ("b", "blue", "yellow")]
)
def test_join_can_use_either_sources_colors_and_keep_other_paint_weighted(
    source, fill, stroke
):
    from vectrify.document.join import average_paint, painted_weights, path_style

    doc = pair(
        'fill="red" stroke="green" stroke-width="2" fill-opacity=".4"',
        'fill="blue" stroke="yellow" stroke-width="6" fill-opacity=".8"',
        "M40 0H70V10H40Z",
    )
    paths = [doc.element("a"), doc.element("b")]
    styles = [path_style(doc, p) for p in paths]
    expected = average_paint(styles, painted_weights(doc, paths))
    editor = Editor(doc, selection=select("a", "b"))
    with editor.transaction("Join") as tx:
        oid = tx.join_paths(frozenset({"a", "b"}), color_source=source)
    result = editor.snapshot.document.element(oid)
    assert result.get("fill") == fill
    assert result.get("stroke") == stroke
    assert result.get("fill-opacity") == (".4" if source == "a" else ".8")
    assert result.get("stroke-width") == expected["stroke-width"]
    assert editor.undo().document == doc


def test_group_join_can_choose_an_inherited_child_color():
    doc = pair("", 'fill="blue"')
    editor = Editor(doc, selection=select("g"))
    with editor.transaction("Join") as tx:
        oid = tx.join_paths(frozenset({"g"}), color_source="a")
    assert editor.snapshot.document.element(oid).get("fill") == "red"


def test_source_color_rejects_unselected_source_and_obeys_paint_locks():
    editor = Editor(pair('fill="red"', 'fill="blue"'), selection=select("a", "b"))
    editor.set_locks("b", frozenset({"fill"}))
    before = editor.snapshot
    for source, error in [("outside", "color source"), ("a", "locked")]:
        with (
            pytest.raises(EditRejectedError, match=error),
            editor.transaction("Join") as tx,
        ):
            tx.join_paths(frozenset({"a", "b"}), color_source=source)
        assert editor.snapshot == before
    with editor.transaction("Join") as tx:
        oid = tx.join_paths(frozenset({"a", "b"}), color_source="b")
    assert editor.snapshot.document.element(oid).get("fill") == "blue"


def test_join_nested_group_and_path_deduplicates_selection_and_preserves_history():
    doc = import_svg("""<svg width="100" height="100">
      <g id="g" fill="red" transform="translate(5 5)">
        <path id="a" d="M0 0H10V10H0Z"/>
        <g id="nested" transform="translate(20 0)">
          <path id="b" d="M0 0H10V10H0Z"/>
        </g>
      </g>
      <rect id="between" x="80" width="10" height="10" fill="green"/>
      <path id="c" d="M45 5H55V15H45Z" fill="red"/>
      <rect id="front" x="80" y="20" width="10" height="10"/>
    </svg>""")
    editor = Editor(doc, selection=select("g", "b", "c"))
    before = editor.snapshot
    with editor.transaction("Join") as tx:
        joined = tx.join_paths(frozenset({"g", "b", "c"}), color_source="b")
    after = editor.snapshot
    assert after.selection.object_ids == {joined}
    assert [e.id for e in after.document.root.children] == ["between", joined, "front"]
    assert len(after.document.geometry_for(joined).subpaths) == 3
    assert after.document.element(joined).get("fill") == "red"
    np.testing.assert_array_equal(
        render(export_svg(doc)), render(export_svg(after.document))
    )
    assert editor.undo().document == before.document
    assert editor.snapshot.selection == before.selection
    assert editor.redo().selection == after.selection


def test_join_two_groups_preserves_clipping_and_inherited_color_choice():
    doc = import_svg("""<svg width="100" height="100">
      <defs><clipPath id="clip"><rect width="5" height="10"/></clipPath></defs>
      <g id="g" fill="red" clip-path="url(#clip)">
        <path id="a" d="M0 0H10V10H0Z"/></g>
      <g id="h" fill="blue" transform="translate(20 0)">
        <path id="b" d="M0 0H10V10H0Z"/></g>
    </svg>""")
    editor = Editor(doc, selection=select("g", "h"))
    with editor.transaction("Join") as tx:
        joined = tx.join_paths(frozenset({"g", "h"}), color_source="b")
    from tests.document.test_hit_test import hits
    from vectrify.document import HitIndex

    hit = HitIndex(editor.snapshot.document)
    assert hits(hit, 1, 1, 1, 1) == {joined}
    assert not hits(hit, 7, 1, 1, 1)
    assert hits(hit, 21, 1, 1, 1) == {joined}
    assert editor.snapshot.document.element(joined).get("fill") == "blue"


def test_single_nested_group_join_selects_only_result():
    doc = import_svg("""<svg width="100" height="100"><g id="g">
      <g id="nested"><path id="a" d="M0 0H10V10H0Z"/></g>
      <path id="b" d="M20 0H30V10H20Z"/></g></svg>""")
    editor = Editor(doc, selection=select("g"))
    with editor.transaction("Join") as tx:
        joined = tx.join_paths(frozenset({"g"}))
    assert editor.snapshot.selection.object_ids == {joined}
    assert len(editor.snapshot.document.geometry_for(joined).subpaths) == 2


@pytest.mark.parametrize("lock", ["geometry", "structure"])
def test_mixed_group_join_respects_group_locks(lock):
    editor = Editor(pair(), selection=select("g", "a"))
    editor.set_locks("g", frozenset({lock}))
    before = editor.snapshot
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Join") as tx,
    ):
        tx.join_paths(frozenset({"g", "a"}))
    assert editor.snapshot == before


def test_mixed_group_join_rejects_nonpaths_atomically():
    editor = Editor(
        pair(extra='<rect width="10" height="10"/>'), selection=select("g", "a")
    )
    before = editor.snapshot
    with (
        pytest.raises(EditRejectedError, match="only paths"),
        editor.transaction("Join") as tx,
    ):
        tx.join_paths(frozenset({"g", "a"}))
    assert editor.snapshot == before


def test_mixed_group_join_rejects_referenced_groups_atomically():
    doc = import_svg("""<svg width="100" height="100"><g id="g">
      <path id="a" d="M0 0H10V10H0Z"/></g>
      <path id="b" d="M20 0H30V10H20Z"/><use href="#g" x="50"/></svg>""")
    editor = Editor(doc, selection=select("g", "b"))
    before = editor.snapshot
    with (
        pytest.raises(EditRejectedError, match="Detach references"),
        editor.transaction("Join") as tx,
    ):
        tx.join_paths(frozenset({"g", "b"}))
    assert editor.snapshot == before
