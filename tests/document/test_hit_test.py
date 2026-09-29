"""Painted areas follow the drawn geometry, not object bounding boxes."""

import pytest
from shapely.geometry import box

from tests.document.test_document import select
from vectrify.document import (
    DocumentError,
    Editor,
    HitIndex,
    import_svg,
)


def index(body, **kwargs):
    return HitIndex(import_svg(f'<svg width="100" height="100">{body}</svg>'), **kwargs)


def hits(hit_index, x, y, w=1, h=1, mode="intersect", candidates=None):
    """Objects painting in a box: leaf objects by default, or *candidates*.

    A leaf counts when its paint meets the box (with contain, lies inside it);
    a group only when the box covers all of its paint.
    """
    region = box(x, y, x + w, y + h)
    document = hit_index.document
    if candidates is None:
        candidates = {e.id for e in document.elements() if e.tag not in {"svg", "g"}}
    found = set()
    for oid in candidates:
        area = hit_index.area(oid)
        if area is None:
            continue
        group = document.element(oid).tag in {"svg", "g"}
        meets = (
            region.intersects if mode == "intersect" and not group else region.covers
        )
        if meets(area):
            found.add(oid)
    return found


def test_intersection_containment_and_group_areas():
    hit = index(
        """<g id="group">
<rect id="a"
x="10"
y="10"
width="10"
height="10"/>
<circle id="b"
cx="40"
cy="40"
r="10"/>
</g>"""
    )
    assert hits(hit, 15, 15) == {"a"}
    assert hits(hit, 15, 15, mode="contain") == set()
    assert hits(hit, 10, 10, 10, 10, mode="contain") == {"a"}
    assert hit.bounds(frozenset({"group"})) == (10, 10, 50, 50)
    assert hit.bounds(frozenset({"a", "b"})) == (10, 10, 50, 50)
    assert not hits(hit, 90, 90)


def test_triangle_and_circle_empty_bbox_corners_are_not_hits():
    hit = index(
        """<path id="triangle"
d="M0 0 L20 0 L0 20 Z"/>
<circle id="circle"
cx="50"
cy="50"
r="10"/>"""
    )
    assert not hits(hit, 17, 17)
    assert not hits(hit, 40, 40)
    assert hits(hit, 9, 9) == {"triangle"}


@pytest.mark.parametrize(("rule", "hole"), [("evenodd", True), ("nonzero", False)])
def test_fill_rules_and_compound_holes(rule, hole):
    hit = index(
        f'<path id="a" fill-rule="{rule}" d="M0 0 H30 V30 H0 Z M10 10 H20 V20 H10 Z"/>'
    )
    assert bool(hits(hit, 14, 14)) != hole
    assert hits(hit, 2, 2) == {"a"}


def test_opposite_winding_hole_and_self_crossing_path():
    hit = index(
        """<path id="hole"
d="M0 0 H30 V30 H0 Z M10 10 V20 H20 V10 Z"/>
<path id="bow"
d="M50 0 L70 20 L50 20 L70 0 Z"/>"""
    )
    assert not hits(hit, 14, 14)
    assert hits(hit, 59, 2) == {"bow"}
    assert not hits(hit, 50, 8)


def test_transform_order_nested_rotation_and_nonuniform_stroke_scale():
    hit = index(
        """<g transform="translate(50 30)">
<rect id="a"
width="20"
height="10"
transform="rotate(90)"/>
</g>
<line id="stroke"
x1="0"
y1="0"
x2="10"
y2="0"
stroke="black"
stroke-width="2"
transform="translate(10 10) scale(2 4)"/>"""
    )
    assert hits(hit, 42, 32) == {"a"}
    assert not hits(hit, 52, 32)
    assert hits(hit, 20, 6.5) == {"stroke"}
    assert not hits(hit, 20, 4)


def test_curve_overshoot_and_custom_tolerance():
    hit = index(
        '<path id="a" d="M0 10 C50 10 50 10 10 10" fill="none" stroke="black"/>',
        tolerance=0.01,
    )
    assert hits(hit, 35, 9.5) == {"a"}
    assert not hits(hit, 45, 9)


def test_clip_union_holes_and_group_clip_applies_to_child_hits():
    hit = index("""<defs>
<clipPath id="clip">
<path clip-rule="evenodd"
d="M0 0 H40 V40 H0 Z M10 10 H30 V30 H10 Z"/>
<rect x="50"
width="10"
height="10"/>
</clipPath>
</defs>
    <g id="group"
clip-path="url(#clip)">
<rect id="a"
width="80"
height="80"/>
</g>""")
    assert hits(hit, 2, 2) == {"a"}
    assert not hits(hit, 15, 15)
    assert hits(hit, 52, 2) == {"a"}
    assert not hits(hit, 65, 2)
    assert not hits(hit, 15, 15, candidates=frozenset({"group"}))
    assert hits(hit, 0, 0, 60, 40, mode="contain") == {"a"}


def test_object_bounding_box_clip_on_transformed_path_and_group():
    hit = index("""<defs>
<clipPath id="half"
clipPathUnits="objectBoundingBox">
<rect width="0.5"
height="1"/>
</clipPath>
</defs>
    <path id="a"
d="M10 10 H30 V30 H10 Z"
transform="translate(20 0)"
clip-path="url(#half)"/>
    <g id="g"
transform="translate(0 50)"
clip-path="url(#half)">
<rect id="b"
x="10"
width="10"
height="10"/>
<rect id="c"
x="30"
width="10"
height="10"/>
</g>""")
    assert hits(hit, 32, 12) == {"a"}
    assert not hits(hit, 42, 12)
    assert hits(hit, 12, 52) == {"b"}
    assert not hits(hit, 32, 52)


def test_nested_clips_and_use_instances_retain_separate_ids():
    hit = index("""<defs>
<path id="shape"
d="M0 0 H20 V20 H0 Z"/>
    <clipPath id="outer">
<rect width="50"
height="10"/>
</clipPath>
    <clipPath id="inner">
<rect x="5"
width="10"
height="30"/>
</clipPath>
</defs>
    <g clip-path="url(#outer)">
<use id="one"
href="#shape"
clip-path="url(#inner)"/>
    <use id="two"
href="#shape"
x="30"/>
</g>""")
    assert hits(hit, 6, 2) == {"one"}
    assert not hits(hit, 1, 2)
    assert not hits(hit, 6, 12)
    assert hits(hit, 32, 2) == {"two"}
    assert not hits(hit, 32, 12)
    assert not hits(hit, 6, 2, candidates=frozenset({"shape"}))


def test_clip_uses_geometry_even_when_its_paint_is_none_or_transparent():
    hit = index(
        """<defs>
<clipPath id="clip">
<rect width="10"
height="10"
fill="none"
opacity="0"/>
</clipPath>
</defs>
<rect id="a"
width="20"
height="20"
clip-path="url(#clip)"/>"""
    )
    assert hits(hit, 2, 2) == {"a"}
    assert not hits(hit, 12, 2)


def test_stroke_caps_rounded_rects_and_transparent_objects():
    hit = index("""<line id="round"
x1="10"
y1="10"
x2="20"
y2="10"
stroke="black"
stroke-width="4"
stroke-linecap="round"/>
    <line id="butt"
x1="10"
y1="20"
x2="20"
y2="20"
stroke="black"
stroke-width="4"/>
    <rect id="rounded"
x="30"
y="30"
width="20"
height="20"
rx="10"/>
    <rect id="hidden"
width="100"
height="100"
fill="rgba(0,0,0,0)"/>
    <g opacity="0">
<rect id="hidden-child"
width="100"
height="100"/>
</g>""")
    assert hits(hit, 8.5, 9.5, 0.5, 0.5) == {"round"}
    assert not hits(hit, 8.5, 19.5, 0.5, 0.5)
    assert not hits(hit, 30, 30, 1, 1)
    assert hits(hit, 39, 31) == {"rounded"}


def test_index_is_snapshot_scoped():
    doc = import_svg(
        '<svg width="100" height="100"><rect id="a" width="10" height="10"/></svg>'
    )
    old = HitIndex(doc)
    editor = Editor(doc, selection=select("a"))
    with editor.transaction("Move") as tx:
        tx.set_attributes("a", {"transform": "translate(50 0)"})
    assert hits(old, 2, 2) == {"a"}
    new = HitIndex(editor.snapshot.document)
    assert not hits(new, 2, 2)
    assert hits(new, 52, 2) == {"a"}


def test_root_viewbox_coordinates_and_occluded_objects_are_selectable():
    doc = import_svg(
        """<svg width="200"
height="200"
viewBox="0 0 100 100">
<rect id="back"
x="10"
y="10"
width="10"
height="10"/>
<rect id="front"
x="10"
y="10"
width="10"
height="10"
fill="blue"/>
</svg>"""
    )
    hit = HitIndex(doc)
    assert hits(hit, 12, 12) == {"back", "front"}
    assert not hits(hit, 22, 22)


def test_empty_and_singular_shapes_paint_nothing():
    hit = index('<rect id="a" width="10" height="10" transform="scale(0)"/>')
    assert not hits(hit, 0, 0, 100, 100)
    assert hit.bounds(frozenset({"a"})) is None
    with pytest.raises(DocumentError, match="positive"):
        index("", tolerance=0)


def test_referenced_shape_inherits_from_instance_not_definition_ancestors():
    hit = index("""<defs><g fill="none" transform="translate(70 0)">
    <rect id="definition" width="10" height="10"/></g></defs>
    <use id="a" href="#definition" x="20" fill="red"/>
    <use id="b" href="#definition" x="40" fill="none"/>""")
    assert hits(hit, 22, 2) == {"a"}
    assert not hits(hit, 42, 2)
    assert not hits(hit, 92, 2)


def test_matrix_skew_and_rotation_about_an_explicit_center():
    hit = index("""<rect id="a" x="10" y="10" width="10" height="10"
    transform="rotate(90 10 10)"/>
    <rect id="b" width="10" height="10"
    transform="matrix(1 0 0 1 50 0) skewX(45)"/>""")
    assert hits(hit, 2, 12) == {"a"}
    assert not hits(hit, 12, 12)
    assert hits(hit, 60, 5) == {"b"}
    assert not hits(hit, 50, 8)


def test_zero_length_round_and_square_strokes_are_selectable():
    hit = index("""<path id="a" d="M10 10 L10 10" stroke="black"
    stroke-width="4" stroke-linecap="round"/>
    <path id="b" d="M30 10 L30 10" stroke="black"
    stroke-width="4" stroke-linecap="square"/>
    <path id="c" d="M50 10 L50 10" stroke="black" stroke-width="4"/>""")
    assert hits(hit, 9, 9) == {"a"}
    assert hits(hit, 28.1, 8.1, 0.1, 0.1) == {"b"}
    assert not hits(hit, 49, 9)


def test_moveto_only_paths_do_not_paint_a_stroke():
    hit = index("""<path id="open" d="M10 10" stroke="black"
    stroke-width="4" stroke-linecap="round"/>
    <path id="closed" d="M30 10 Z" stroke="black"
    stroke-width="4" stroke-linecap="round"/>""")
    assert not hits(hit, 9, 9)
    assert not hits(hit, 29, 9)


def test_nested_clipped_group_preserves_holes_containment_and_union_area():
    hit = index("""<defs><clipPath id="clip"><path clip-rule="evenodd"
        d="M0 0H30V30H0Z M10 10H20V20H10Z"/></clipPath></defs>
        <g id="outer" transform="translate(40 20)" clip-path="url(#clip)">
          <g id="inner"><rect id="a" x="-5" y="-5" width="45" height="45"/>
          <rect id="b" x="5" y="5" width="20" height="20"/></g>
        </g>""")
    assert not hits(hit, 54, 34)
    assert hits(hit, 46, 26) == {"a", "b"}
    assert hits(hit, 40, 20, 30, 30, mode="contain") == {"a", "b"}
    assert not hits(hit, 46, 26, candidates=frozenset({"outer", "inner"}))
    assert hits(hit, 40, 20, 30, 30, candidates=frozenset({"outer", "inner"})) == {
        "outer",
        "inner",
    }
    assert hit.painted_area("outer") == 800
    assert hit.painted_area("inner") == 800
    assert hit.painted_area("b") == 300


def test_invalid_fast_clip_falls_back_without_changing_selection(monkeypatch):
    from shapely.geometry import Polygon

    import vectrify.document.hit_test as module

    calls = []

    def invalid_crop(*args):
        calls.append(args)
        return Polygon([(0, 0), (20, 20), (0, 20), (20, 0)])

    monkeypatch.setattr(module, "clip_by_rect", invalid_crop)
    hit = index("""<defs><clipPath id="clip"><rect width="10" height="10"/>
        </clipPath></defs><rect id="a" x="5" y="5" width="20" height="20"
        clip-path="url(#clip)"/>""")
    assert calls
    assert hits(hit, 6, 6) == {"a"}
    assert not hits(hit, 11, 6)
    assert hits(hit, 5, 5, 5, 5, mode="contain") == {"a"}
    assert hit.painted_area("a") == 25
