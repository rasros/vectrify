"""Cross-group joins bake selected geometry and keep other artwork in place."""

import numpy as np
import pytest

from tests.document.helpers import render, select
from vectrify.document import DocumentError, Editor, export_svg, import_svg


def joined(svg):
    doc = import_svg(svg)
    editor = Editor(doc, selection=select("a", "b"))
    with editor.transaction("Join") as tx:
        oid = tx.join_paths(frozenset({"a", "b"}))
    return doc, editor, oid


def test_cross_group_transform_and_inherited_paint_preserve_render_and_history():
    before, editor, oid = joined("""<svg width="100" height="100">
    <g transform="translate(10 10)" fill="red"><path id="a" d="M0 0H10V10H0Z"/></g>
    <g transform="translate(50 50) scale(2)" fill="red">
    <path id="b" d="M0 0H10V10H0Z"/></g>
    </svg>""")
    after = editor.snapshot.document
    assert after.ancestry(oid)[-2].tag == "svg"
    assert after.element(oid).get("transform") is None
    np.testing.assert_array_equal(render(export_svg(before)), render(export_svg(after)))
    assert editor.snapshot.selection.object_ids == {oid}
    assert editor.undo().document == before
    assert editor.redo().document == after


def test_clip_definitions_and_use_are_baked_before_joining():
    before, editor, oid = joined("""<svg width="120" height="100">
    <defs><path id="clip-source" d="M0 0H10V20H0Z"/>
    <clipPath id="clip"><use href="#clip-source"/></clipPath>
    <clipPath id="other"><rect width="10" height="10"/></clipPath></defs>
    <g fill="red" transform="translate(10 10)" clip-path="url(#clip)">
    <path id="a" d="M-10 -10H30V30H-10Z"/></g>
    <g fill="red" transform="translate(70 10)" clip-path="url(#other)">
    <path id="b" d="M-10 -10H30V30H-10Z"/></g></svg>""")
    after = editor.snapshot.document
    assert after.element(oid).get("clip-path") is None
    np.testing.assert_array_equal(render(export_svg(before)), render(export_svg(after)))


def test_frontmost_branch_splits_without_reordering_or_restyling_other_objects():
    before, editor, oid = joined("""<svg width="100" height="100">
    <g fill="red"><path id="a" d="M0 0H10V10H0Z"/></g>
    <rect id="between" x="20" y="20" width="20" height="20" fill="blue"/>
    <g id="front" transform="translate(10 0)" fill="red">
      <path id="under" d="M0 20H30V50H0Z"/>
      <g transform="translate(5 0)"><path id="b" d="M0 20H20V40H0Z"/>
        <rect id="above" x="5" y="25" width="10" height="10" fill="yellow"/>
      </g>
      <rect id="last" x="10" y="30" width="10" height="10" fill="green"/>
    </g></svg>""")
    after = editor.snapshot.document
    order = [e.id for e in after.elements()]
    assert (
        order.index("between")
        < order.index("under")
        < order.index(oid)
        < order.index("above")
        < order.index("last")
    )
    for keep in ("between", "under", "above", "last"):
        assert after.element(keep) == before.element(keep)
    np.testing.assert_array_equal(render(export_svg(before)), render(export_svg(after)))


def test_cross_group_average_uses_original_clipped_areas():
    _, editor, oid = joined("""<svg width="100" height="100">
    <defs><clipPath id="clip"><rect width="10" height="10"/></clipPath></defs>
    <g clip-path="url(#clip)" fill="red"><path id="a" d="M0 0H90V90H0Z"/></g>
    <g fill="blue"><path id="b" d="M40 0H70V10H40Z"/></g></svg>""")
    assert editor.snapshot.document.element(oid).get("fill") == "#4000bf"


def test_stroke_only_uniform_scale_resolves_width_and_preserves_cubic():
    _, editor, oid = joined("""<svg width="100" height="100">
    <g transform="scale(2)" fill="none" stroke="red" stroke-width="2">
      <path id="a" d="M0 0C0 5 5 5 5 0"/></g>
    <g transform="translate(40 0)" fill="none" stroke="red" stroke-width="4">
      <path id="b" d="M0 0C0 10 10 10 10 0"/></g></svg>""")
    path = editor.snapshot.document.element(oid)
    assert float(path.get("stroke-width") or 0) == 4
    assert (
        sum(
            n.command == "C"
            for s in editor.snapshot.document.geometry_for(oid).subpaths
            for n in s.nodes
        )
        == 2
    )


@pytest.mark.parametrize("kind", ["geometry", "structure", "paint"])
def test_cross_group_locks_remain_atomic(kind):
    doc = import_svg("""<svg width="100" height="100"><g id="g" fill="red">
    <path id="a" d="M0 0H10V10H0Z"/></g><g fill="blue">
    <path id="b" d="M40 0H60V10H40Z"/></g></svg>""")
    editor = Editor(doc, selection=select("g"))
    editor.set_locks("g", frozenset({kind}))
    editor.select(select("a", "b"))
    before = editor.snapshot
    with pytest.raises(DocumentError, match="locked"), editor.transaction("Join") as tx:
        tx.join_paths(frozenset({"a", "b"}))
    assert editor.snapshot == before


@pytest.mark.parametrize("attrs", ['transform="scale(2 1)"', 'opacity=".5"'])
def test_nonuniform_strokes_and_split_group_opacity_join_and_undo(attrs):
    doc = import_svg(f"""<svg width="100" height="100">
        <g><path id="a" d="M0 0H10V10H0Z" stroke="red"/></g>
        <g {attrs} stroke="red"><path d="M0 30H10V40H0Z"/>
        <path id="b" d="M20 20H30V30H20Z"/><path d="M40 40H50V50H40Z"/></g></svg>""")
    editor = Editor(doc, selection=select("a", "b"))
    with editor.transaction("Join") as tx:
        oid = tx.join_paths(frozenset({"a", "b"}))
    after = editor.snapshot.document
    after.validate()
    assert after.element(oid).tag == "path"
    assert editor.undo().document == doc
    assert editor.redo().document == after


def test_cross_group_join_resolves_object_bounding_box_clips():
    source = """<svg width="100" height="100">
    <defs><clipPath id="clip" clipPathUnits="objectBoundingBox">
      <rect width=".5" height="1"/>
    </clipPath></defs>
    <g fill="red" transform="translate(10 10)" clip-path="url(#clip)">
      <path id="a" d="M0 0H20V20H0Z"/>
    </g>
    <path id="b" fill="red" d="M50 10H60V30H50Z"/>
    </svg>"""
    before, editor, oid = joined(source)
    after = editor.snapshot.document
    assert after.element(oid).get("clip-path") is None
    # Cairo does not support a group's objectBoundingBox clip. Its equivalent
    # user-space rectangle is half the 20 by 20 group bounds.
    expected = source.replace('clipPathUnits="objectBoundingBox"', "").replace(
        'width=".5" height="1"', 'width="10" height="20"'
    )
    # The now-empty original group still references the old definition; Cairo
    # needs user-space units even for that invisible group.
    actual = export_svg(after).replace(
        'clipPathUnits="objectBoundingBox"', 'clipPathUnits="userSpaceOnUse"'
    )
    np.testing.assert_array_equal(render(expected), render(actual))
    assert editor.undo().document == before
    assert editor.redo().document == after


def test_join_clipped_lines_keeps_the_source_clip_instead_of_rejecting():
    source = """<svg width="100" height="100">
    <defs><clipPath id="clip"><rect width="10" height="20"/></clipPath></defs>
    <path id="a" fill="none" stroke="red" d="M0 10H20"/>
    <g transform="translate(40 0)" clip-path="url(#clip)">
      <path id="b" fill="none" stroke="red" d="M0 10H20"/>
    </g></svg>"""
    before, editor, oid = joined(source)
    after = editor.snapshot.document
    assert after.element(oid).get("clip-path")
    assert render(export_svg(after))[10, 45, 3] > 0
    assert render(export_svg(after))[10, 55, 3] == 0
    assert editor.undo().document == before
    assert editor.redo().document == after


def test_clipping_fallback_preserves_polygon_holes(monkeypatch):
    import pathops

    from vectrify.document.join import clip_intersection, curve_path, path_geometry
    from vectrify.document.svg import parse_path

    path = curve_path(parse_path("M0 0H20V20H0Z M5 5V15H15V5Z"))
    clip = curve_path(parse_path("M0 0H10V20H0Z"))

    def fail(*_args, **_kwargs):
        raise pathops.PathOpsError("degenerate contour")

    monkeypatch.setattr(pathops, "op", fail)
    result = path_geometry(clip_intersection(path, clip))
    actual = render(
        f'<svg width="20" height="20"><path d="{result.path_data()}"/></svg>'
    )
    assert actual[2, 2, 3] == 255
    assert actual[10, 7, 3] == 0  # the hole remains open
    assert actual[2, 12, 3] == 0  # the clipped side remains empty


def test_pin_and_group_instance_protection_survive_cross_group_join():
    source = """<svg width="100" height="100"><g id="g" transform="translate(10 0)">
    <path id="a" d="M0 0H10V10H0Z"/></g><g>
    <path id="b" d="M40 0H60V10H40Z"/></g>{extra}</svg>"""
    for extra, pinned, message in [
        ("", True, "Unpin"),
        ('<use href="#g"/>', False, "unselected"),
    ]:
        editor = Editor(
            import_svg(source.format(extra=extra)), selection=select("a", "b")
        )
        if pinned:
            node = editor.snapshot.document.geometry_for("a").subpaths[0].nodes[0]
            editor.pin_node("a", node.id, pinned=True)
        before = editor.snapshot
        with (
            pytest.raises(DocumentError, match=message),
            editor.transaction("Join") as tx,
        ):
            tx.join_paths(frozenset({"a", "b"}))
        assert editor.snapshot == before


@pytest.mark.parametrize(("source", "color"), [("a", "red"), ("b", "blue")])
def test_cross_group_join_uses_chosen_inherited_colors(source, color):
    doc = import_svg("""<svg width="100" height="100">
    <g fill="red" transform="translate(10 10)"><path id="a" d="M0 0H10V10H0Z"/></g>
    <g fill="blue"><path id="b" d="M40 0H70V10H40Z"/></g></svg>""")
    editor = Editor(doc, selection=select("a", "b"))
    with editor.transaction("Join") as tx:
        oid = tx.join_paths(frozenset({"a", "b"}), color_source=source)
    assert editor.snapshot.document.element(oid).get("fill") == color
    assert editor.undo().document == doc
