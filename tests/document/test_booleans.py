"""Union, Subtract, Intersect and Exclude: operands, paint, frames and refusals."""

import pytest

from tests.document.helpers import select
from vectrify.document import Editor, EditRejectedError, import_svg
from vectrify.document.hit_test import mapped
from vectrify.document.holes import filled_region, find_holes
from vectrify.document.join import path_style
from vectrify.document.transforms import root_matrix

BACK = "M0 0H100V100H0Z"
FRONT = "M50 50H150V150H50Z"
INSIDE = "M25 25H75V75H25Z"
FAR = "M200 0H250V50H200Z"
# A circle of radius 50 about (50, 50), drawn with cubic curves.
CIRCLE = (
    "M100 50C100 77.6 77.6 100 50 100C22.4 100 0 77.6 0 50"
    "C0 22.4 22.4 0 50 0C77.6 0 100 22.4 100 50Z"
)


def drawing(*paths, extra=""):
    body = "".join(f'<path id="{oid}" d="{d}" {attrs}/>' for oid, d, attrs in paths)
    return import_svg(f'<svg width="300" height="300">{body}{extra}</svg>')


def painted(document, object_id):
    """The area a path fills, in root user space."""
    style = path_style(document, document.element(object_id))
    region = filled_region(document.geometry_for(object_id), style["fill-rule"])
    return mapped(region, root_matrix(document, object_id))


def run(document, operation, *ids, selection=None):
    editor = Editor(document, selection=select(*(selection or ids)))
    with editor.transaction(operation.capitalize()) as tx:
        result = tx.boolean_paths(frozenset(ids), operation)
    return editor, result


def refuses(document, operation, *ids, match, selection=None):
    editor = Editor(document, selection=select(*(selection or ids)))
    before = editor.snapshot
    with (
        pytest.raises(EditRejectedError, match=match),
        editor.transaction(operation) as tx,
    ):
        tx.boolean_paths(frozenset(ids), operation)
    assert editor.snapshot == before


@pytest.mark.parametrize(
    ("operation", "area"),
    [("union", 17500), ("subtract", 7500), ("intersect", 2500), ("exclude", 15000)],
)
def test_overlapping_paths_keep_the_backmost_paths_identity_paint_and_place(
    operation, area
):
    doc = drawing(
        ("under", FAR, ""),
        ("a", BACK, 'fill="#123456" stroke="#000"'),
        ("middle", FAR, 'fill="green"'),
        ("b", FRONT, 'fill="#ff0000" opacity="0.5"'),
    )
    editor, result = run(doc, operation, "b", "a")
    after = editor.snapshot.document
    assert result == "a"
    assert [e.id for e in after.elements() if e.tag == "path"] == [
        "under",
        "a",
        "middle",
    ]
    assert after.element("a").get("fill") == "#123456"
    assert after.element("a").get("stroke") == "#000"
    assert after.element("a").get("opacity") is None
    assert painted(after, "a").area == pytest.approx(area)
    assert editor.undo_labels == (operation.capitalize(),)
    assert editor.undo().document == doc


def test_subtract_removes_the_front_from_the_back_whichever_is_listed_first():
    doc = drawing(("a", BACK, ""), ("b", FRONT, ""))
    editor, _ = run(doc, "subtract", "a", "b")
    region = painted(editor.snapshot.document, "a")
    assert region.area == pytest.approx(7500)
    assert region.bounds == pytest.approx((0, 0, 100, 100))
    restacked = drawing(("b", FRONT, ""), ("a", BACK, ""))
    editor, result = run(restacked, "subtract", "a", "b")
    assert result == "b"
    region = painted(editor.snapshot.document, "b")
    assert region.area == pytest.approx(7500)
    assert region.bounds == pytest.approx((50, 50, 150, 150))


def test_subtract_takes_away_every_front_path_and_deletes_them():
    doc = drawing(
        ("a", BACK, ""), ("b", FRONT, ""), ("c", "M0 0H20V20H0Z", ""), ("d", FAR, "")
    )
    editor, _ = run(doc, "subtract", "a", "b", "c", "d")
    after = editor.snapshot.document
    assert [e.id for e in after.elements() if e.tag == "path"] == ["a"]
    assert painted(after, "a").area == pytest.approx(10000 - 2500 - 400)


def test_containment_cuts_a_hole_and_the_reverse_would_leave_nothing():
    doc = drawing(("a", BACK, ""), ("b", INSIDE, ""))
    editor, _ = run(doc, "subtract", "a", "b")
    after = editor.snapshot.document
    assert painted(after, "a").area == pytest.approx(10000 - 2500)
    assert len(find_holes(after, "a")) == 1
    for operation, area in (("union", 10000), ("intersect", 2500), ("exclude", 7500)):
        editor, _ = run(doc, operation, "a", "b")
        assert painted(editor.snapshot.document, "a").area == pytest.approx(area)
    inside_behind = drawing(("b", INSIDE, ""), ("a", BACK, ""))
    refuses(inside_behind, "subtract", "a", "b", match="would leave nothing")


def test_holes_in_the_operands_are_kept_or_filled_by_area():
    ring = BACK + "M20 20V80H80V20Z"
    doc = drawing(("a", ring, ""), ("b", "M40 40H60V60H40Z", 'fill="red"'))
    editor, _ = run(doc, "union", "a", "b")
    after = editor.snapshot.document
    # The front square becomes an island inside the ring's hole.
    assert painted(after, "a").area == pytest.approx(10000 - 3600 + 400)
    evenodd = drawing(("a", BACK + INSIDE, 'fill-rule="evenodd"'), ("b", FRONT, ""))
    editor, _ = run(evenodd, "union", "a", "b")
    after = editor.snapshot.document
    assert path_style(after, after.element("a"))["fill-rule"] == "nonzero"
    assert painted(after, "a").area == pytest.approx(17500 - 2500 + 625)


def test_intersect_with_a_hole_where_the_other_path_lies_leaves_nothing():
    ring = BACK + "M20 20V80H80V20Z"
    doc = drawing(("a", ring, ""), ("b", "M40 40H60V60H40Z", ""))
    refuses(doc, "intersect", "a", "b", match="Intersect would leave nothing")


def test_disjoint_paths_union_into_one_path_and_cannot_be_cut():
    doc = drawing(("a", BACK, ""), ("b", FAR, ""))
    for operation in ("union", "exclude"):
        editor, _ = run(doc, operation, "a", "b")
        after = editor.snapshot.document
        assert len(after.geometry_for("a").subpaths) == 2
        assert painted(after, "a").area == pytest.approx(12500)
    refuses(doc, "subtract", "a", "b", match="Nothing to subtract")
    refuses(doc, "intersect", "a", "b", match="Intersect would leave nothing")


def test_identical_paths_cannot_be_excluded():
    doc = drawing(("a", BACK, ""), ("b", BACK, ""))
    refuses(doc, "exclude", "a", "b", match="Exclude would leave nothing")


def test_curves_stay_curves():
    doc = drawing(("a", CIRCLE, ""), ("b", "M50 0H120V100H50Z", ""))
    editor, _ = run(doc, "subtract", "a", "b")
    after = editor.snapshot.document
    commands = {n.command for s in after.geometry_for("a").subpaths for n in s.nodes}
    assert "C" in commands
    half = 3.1416 * 2500 / 2
    assert painted(after, "a").area == pytest.approx(half, rel=1e-3)


def test_the_result_is_written_in_the_targets_frame_across_nested_transforms():
    doc = import_svg(
        '<svg width="300" height="300">'
        '<g id="g" transform="translate(10 0)">'
        f'<path id="a" d="{BACK}" transform="scale(2)" fill="blue"/></g>'
        '<g id="h" transform="scale(2)"><g transform="translate(50 50)">'
        '<path id="b" d="M0 0H25V25H0Z"/></g></g></svg>'
    )
    editor, _ = run(doc, "subtract", "a", "b")
    after = editor.snapshot.document
    assert after.element("a").get("transform") == "scale(2)"
    assert after.ancestry("a")[-2].id == "g"
    region = painted(after, "a")
    # The back path covers (10, 0)-(210, 200); the front one (100, 100)-(150, 150).
    assert region.bounds == pytest.approx((10, 0, 210, 200))
    assert region.area == pytest.approx(40000 - 2500)
    assert "h" in {e.id for e in after.elements()}


def test_clipping_cuts_into_the_operands_and_the_targets_clip_is_removed():
    clip = (
        '<defs><clipPath id="half"><path d="M0 0H300V75H0Z"/></clipPath>'
        '<clipPath id="left"><path d="M0 0H50V300H0Z"/></clipPath></defs>'
    )
    doc = drawing(
        ("a", BACK, 'clip-path="url(#left)"'),
        ("b", FRONT, ""),
        ("c", "M0 0H100V100H0Z", 'clip-path="url(#half)"'),
        extra=clip,
    )
    editor, _ = run(doc, "union", "a", "b")
    after = editor.snapshot.document
    assert after.element("a").get("clip-path") is None
    # The back path's visible half plus the whole front square.
    assert painted(after, "a").area == pytest.approx(5000 + 10000)
    doc = drawing(("a", BACK, ""), ("c", BACK, 'clip-path="url(#half)"'), extra=clip)
    editor, _ = run(doc, "subtract", "a", "c")
    # Only the visible part of the clipped front path is taken away.
    assert painted(editor.snapshot.document, "a").area == pytest.approx(2500)


def test_the_targets_group_clipping_still_applies():
    doc = import_svg(
        '<svg width="300" height="300"><defs><clipPath id="k">'
        '<path d="M0 0H120V300H0Z"/></clipPath></defs>'
        f'<g id="g" clip-path="url(#k)"><path id="a" d="{BACK}"/></g>'
        f'<path id="b" d="{FRONT}"/></svg>'
    )
    editor, _ = run(doc, "union", "a", "b")
    after = editor.snapshot.document
    assert after.element("g").get("clip-path") == "url(#k)"
    assert after.ancestry("a")[-2].id == "g"


@pytest.mark.parametrize(
    ("object_id", "lock"), [("a", "geometry"), ("b", "structure"), ("a", "fill-rule")]
)
def test_locks_on_any_operand_refuse(object_id, lock):
    doc = drawing(("a", BACK, 'fill-rule="evenodd"'), ("b", FRONT, ""))
    editor = Editor(doc, selection=select("a", "b"))
    editor.set_locks(object_id, frozenset({lock}))
    before = editor.snapshot
    with (
        pytest.raises(EditRejectedError, match="locked"),
        editor.transaction("Union") as tx,
    ):
        tx.boolean_paths(frozenset({"a", "b"}), "union")
    assert editor.snapshot == before


@pytest.mark.parametrize("object_id", ["a", "b"])
def test_pinned_points_refuse(object_id):
    doc = drawing(("a", BACK, ""), ("b", FRONT, ""))
    editor = Editor(doc, selection=select("a", "b"))
    node = doc.geometry_for(object_id).subpaths[0].nodes[0].id
    editor.pin_node(object_id, node)
    with (
        pytest.raises(EditRejectedError, match="Unpin"),
        editor.transaction("Union") as tx,
    ):
        tx.boolean_paths(frozenset({"a", "b"}), "union")


def test_references_and_shared_geometry_refuse():
    used = drawing(
        ("a", BACK, ""), ("b", FRONT, ""), extra='<use id="u" href="#b" x="5"/>'
    )
    refuses(used, "union", "a", "b", selection=("a", "b", "u"), match="references")
    shared = import_svg(
        '<svg width="300" height="300">'
        f'<path id="a" d="{BACK}"/><path id="b" d="{FRONT}"/></svg>'
    )
    editor = Editor(shared, selection=select("a", "b"))
    with editor.transaction("Share") as tx:
        tx.share_geometry("b", "a")
    shared = editor.snapshot.document
    refuses(shared, "union", "a", "b", match="shared geometry")


@pytest.mark.parametrize(
    ("extra", "ids", "match"),
    [
        ('<rect id="r" x="0" y="0" width="10" height="10"/>', ("a", "r"), "Convert"),
        (f'<g id="g"><path id="c" d="{FRONT}"/></g>', ("a", "g"), "not groups"),
        (f'<path id="c" d="{FRONT}" fill="none" stroke="red"/>', ("a", "c"), "fill"),
        ("", ("a",), "two or more"),
    ],
)
def test_operands_must_be_two_or_more_filled_drawing_paths(extra, ids, match):
    doc = drawing(("a", BACK, ""), extra=extra)
    refuses(doc, "union", *ids, match=match)


def test_unknown_operations_and_unselected_operands_refuse():
    doc = drawing(("a", BACK, ""), ("b", FRONT, ""))
    refuses(doc, "divide", "a", "b", match="Choose union")
    refuses(doc, "union", "a", "b", selection=("a",), match="unselected")
