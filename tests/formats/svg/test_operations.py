import random
import re
import statistics
import xml.etree.ElementTree as ET

import pytest

from vectrify.formats.svg.operations import (
    _NAMED_SVG_COLORS,
    apply_crossover,
    apply_mutation,
    crossover,
    mutate_color,
    mutate_drop_style_property,
    mutate_numeric,
    mutate_path,
    mutate_reorder,
    mutate_stroke,
    mutate_translate,
    with_retries,
)

NS = "http://www.w3.org/2000/svg"
SVG_A = (
    f'<svg xmlns="{NS}"><circle cx="10" cy="10" r="5"/>'
    f'<rect width="20" height="20"/></svg>'
)
SVG_B = (
    f'<svg xmlns="{NS}"><ellipse rx="8" ry="4"/>'
    f'<line x1="0" y1="0" x2="10" y2="10"/></svg>'
)
SVG_ONE = (
    f'<svg xmlns="{NS}"><rect width="100" height="50" rx="4"'
    f' font-size="12" opacity="0.8"/></svg>'
)


def test_crossover_returns_valid_svg():
    result = crossover(SVG_A, SVG_B)
    root = ET.fromstring(result)
    assert root.tag.endswith("svg")


def test_crossover_children_only_from_parents():
    result = crossover(SVG_A, SVG_B)
    root = ET.fromstring(result)
    tags = {c.tag.split("}")[-1] for c in root}
    assert tags <= {"circle", "rect", "ellipse", "line"}


def test_crossover_of_unequal_parents_keeps_the_first_parent_intact():
    """This used to assert the child could shrink to a fifth of its parent,
    which was the bug rather than the contract: elements the shorter parent
    lacked at a given index were simply dropped."""
    long_a = f'<svg xmlns="{NS}"><rect/><circle/><ellipse/><line/><path/></svg>'
    result = crossover(long_a, SVG_B)
    assert len(list(ET.fromstring(result))) == 5


def test_crossover_degenerate_single_element():
    single_a = f'<svg xmlns="{NS}"><rect/></svg>'
    single_b = f'<svg xmlns="{NS}"><circle/></svg>'
    result = crossover(single_a, single_b)
    root = ET.fromstring(result)
    assert len(list(root)) == 1


def test_crossover_invalid_svg_returns_a():
    result = crossover("not xml", SVG_B)
    assert result == "not xml"


def test_mutate_numeric_changes_an_attribute():
    changed = False
    for _ in range(20):
        result = mutate_numeric(SVG_ONE)
        if result != SVG_ONE:
            changed = True
            break
    assert changed


def test_mutate_numeric_still_valid_svg():
    result = mutate_numeric(SVG_ONE)
    root = ET.fromstring(result)
    assert root.tag.endswith("svg")


def test_mutate_numeric_opacity_clamped():
    svg = f'<svg xmlns="{NS}"><rect opacity="0.9" width="10" height="10"/></svg>'
    checked = 0
    for _ in range(50):
        result = mutate_numeric(svg)
        root = ET.fromstring(result)
        rect = root.find(".//{http://www.w3.org/2000/svg}rect")
        if rect is not None and "opacity" in rect.attrib:
            val = float(rect.attrib["opacity"])
            assert 0.0 <= val <= 1.0
            checked += 1
    assert checked, "no iteration ever produced an opacity attribute to check"


def test_mutate_numeric_invalid_svg_unchanged():
    result = mutate_numeric("not xml")
    assert result == "not xml"


def test_mutate_numeric_no_numeric_attrs_unchanged():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><g id="foo" fill="red"/></svg>'
    result = mutate_numeric(svg)
    assert ET.fromstring(result).tag.endswith("svg")


@pytest.mark.parametrize(
    "op",
    [
        mutate_numeric,
        mutate_drop_style_property,
        mutate_color,
        mutate_stroke,
        mutate_path,
        mutate_reorder,
    ],
)
def test_mutation_result_stays_parseable(op):
    result = op(SVG_ONE)
    assert ET.fromstring(result).tag.endswith("svg")


def test_with_retries_returns_valid_result_on_first_try():
    result = with_retries(lambda: SVG_A, fallback=SVG_ONE)
    assert result == SVG_A


def test_with_retries_returns_fallback_when_op_always_invalid():
    result = with_retries(lambda: "not xml at all", fallback=SVG_ONE, max_retries=3)
    assert result == SVG_ONE


def test_with_retries_returns_fallback_when_op_always_raises():
    def boom():
        raise RuntimeError("boom")

    result = with_retries(boom, fallback=SVG_ONE, max_retries=3)
    assert result == SVG_ONE


def test_with_retries_succeeds_after_initial_failures():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            return "bad xml"
        return SVG_B

    result = with_retries(flaky, fallback=SVG_ONE, max_retries=5)
    assert result == SVG_B
    assert len(calls) == 3


def test_with_retries_exhausts_all_attempts():
    calls = []

    def always_bad():
        calls.append(1)
        return "bad"

    with_retries(always_bad, fallback=SVG_ONE, max_retries=4)
    assert len(calls) == 4


def test_apply_crossover():
    svg_a = f'<svg xmlns="{NS}"><rect width="10" height="10" fill="red"/></svg>'
    svg_b = f'<svg xmlns="{NS}"><rect width="10" height="10" fill="blue"/></svg>'

    res, summary = apply_crossover(svg_a, svg_b)
    assert "<svg" in res
    assert "crossover" in summary.lower()


def test_apply_mutation():
    svg_a = f'<svg xmlns="{NS}"><rect width="10" height="10" fill="red"/></svg>'

    res, summary = apply_mutation(svg_a)
    assert "<svg" in res
    assert "mutation" in summary.lower()


def test_apply_mutation_falls_back_to_the_parent_when_the_operator_fails():
    unmutable = f'<svg xmlns="{NS}"/>'

    for _ in range(20):
        assert apply_mutation(unmutable)[0] == unmutable


SVG_STYLED = (
    f'<svg xmlns="{NS}"><rect style="fill:red; stroke:blue; opacity:0.5"/></svg>'
)
SVG_SINGLE_PROP = f'<svg xmlns="{NS}"><rect style="fill:red"/></svg>'
SVG_NO_STYLE = f'<svg xmlns="{NS}"><rect width="10"/></svg>'


def test_mutate_drop_style_property_removes_one_property():
    result = mutate_drop_style_property(SVG_STYLED)
    root = ET.fromstring(result)
    rect = root.find(f"{{{NS}}}rect")
    assert rect is not None
    props = [p.strip() for p in rect.get("style", "").split(";") if p.strip()]
    assert len(props) == 2


def test_mutate_drop_style_property_single_prop_unchanged():
    result = mutate_drop_style_property(SVG_SINGLE_PROP)
    assert result == SVG_SINGLE_PROP


def test_mutate_drop_style_property_no_style_unchanged():
    result = mutate_drop_style_property(SVG_NO_STYLE)
    assert result == SVG_NO_STYLE


def test_mutate_drop_style_property_invalid_svg_unchanged():
    result = mutate_drop_style_property("not xml")
    assert result == "not xml"


# ── mutate_color ──────────────────────────────────────────────────────────────

SVG_HEX = f'<svg xmlns="{NS}"><rect width="10" height="10" fill="#3366cc"/></svg>'
SVG_SHORT_HEX = f'<svg xmlns="{NS}"><rect width="10" fill="#f00"/></svg>'
SVG_NAMED = f'<svg xmlns="{NS}"><rect width="10" fill="red"/></svg>'
SVG_COLOR_STYLE = (
    f'<svg xmlns="{NS}"><rect width="4" style="fill:#3366cc; opacity:0.5"/></svg>'
)


def _find(svg: str, tag: str) -> ET.Element:
    el = ET.fromstring(svg).find(f".//{{{NS}}}{tag}")
    assert el is not None, f"no <{tag}> in {svg}"
    return el


def _channels(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def test_mutate_color_nudges_hex_channels_near_the_original():
    for seed in range(30):
        random.seed(seed)
        new = _find(mutate_color(SVG_HEX), "rect").get("fill", "")
        assert re.fullmatch(r"#[0-9a-f]{6}", new)
        for got, want in zip(_channels(new), (0x33, 0x66, 0xCC), strict=True):
            assert abs(got - want) <= 60
            assert 0 <= got <= 255


def test_mutate_color_expands_a_three_digit_hex():
    random.seed(0)
    new = _find(mutate_color(SVG_SHORT_HEX), "rect").get("fill", "")
    r, g, b = _channels(new)
    assert abs(r - 0xFF) <= 60
    assert g <= 60
    assert b <= 60


def test_mutate_color_swaps_a_named_color_for_another_named_color():
    random.seed(0)
    new = _find(mutate_color(SVG_NAMED), "rect").get("fill")
    assert new in _NAMED_SVG_COLORS
    assert new != "red"


def test_mutate_color_leaves_the_rest_of_the_element_alone():
    random.seed(0)
    rect = _find(mutate_color(SVG_HEX), "rect")
    assert rect.get("width") == "10"
    assert rect.get("height") == "10"
    assert set(rect.attrib) == {"width", "height", "fill"}


def test_mutate_color_rewrites_a_style_property_and_keeps_the_others():
    random.seed(0)
    rect = _find(mutate_color(SVG_COLOR_STYLE), "rect")
    props = dict(
        p.split(":", 1) for p in rect.get("style", "").replace(" ", "").split(";") if p
    )
    assert props["opacity"] == "0.5"
    assert props["fill"] != "#3366cc"
    assert rect.get("width") == "4"


def test_mutate_color_changes_exactly_one_element():
    two = (
        f'<svg xmlns="{NS}"><rect fill="#3366cc"/>'
        f'<circle fill="#3366cc"/><ellipse fill="#3366cc"/></svg>'
    )
    random.seed(0)
    fills = [el.get("fill") for el in ET.fromstring(mutate_color(two))]
    assert sorted(f == "#3366cc" for f in fills) == [False, True, True]


def test_mutate_color_ignores_non_colors():
    svg = f'<svg xmlns="{NS}"><rect fill="none" stroke="inherit" width="4"/></svg>'
    assert mutate_color(svg) == svg


def test_mutate_color_no_colors_unchanged():
    svg = f'<svg xmlns="{NS}"><rect width="10" height="10"/></svg>'
    assert mutate_color(svg) == svg


def test_mutate_color_invalid_svg_unchanged():
    assert mutate_color("not xml") == "not xml"


# ── mutate_stroke ─────────────────────────────────────────────────────────────

SVG_STROKED = (
    f'<svg xmlns="{NS}"><rect width="10" height="10" fill="red"'
    f' stroke="blue" stroke-width="7"/></svg>'
)
SVG_UNSTROKED = f'<svg xmlns="{NS}"><rect width="10" height="10" fill="red"/></svg>'


def test_mutate_stroke_adds_a_stroke_and_a_width():
    random.seed(1)
    rect = _find(mutate_stroke(SVG_UNSTROKED), "rect")
    assert rect.get("stroke") in _NAMED_SVG_COLORS
    assert rect.get("stroke-width") in {"1", "2", "3"}


def test_mutate_stroke_removes_an_existing_stroke():
    random.seed(0)
    assert _find(mutate_stroke(SVG_STROKED), "rect").get("stroke") == "none"


def test_mutate_stroke_keeps_an_existing_width_when_changing_color():
    random.seed(1)
    rect = _find(mutate_stroke(SVG_STROKED), "rect")
    assert rect.get("stroke") not in (None, "blue", "none")
    assert rect.get("stroke-width") == "7"


def test_mutate_stroke_touches_nothing_but_the_stroke():
    for seed in range(20):
        random.seed(seed)
        rect = _find(mutate_stroke(SVG_STROKED), "rect")
        assert rect.get("fill") == "red"
        assert rect.get("width") == "10"
        assert rect.get("height") == "10"


def test_mutate_stroke_only_targets_shapes():
    svg = f'<svg xmlns="{NS}"><defs><g id="a"/></defs></svg>'
    result = mutate_stroke(svg)
    assert "stroke" not in result


def test_mutate_stroke_invalid_svg_unchanged():
    assert mutate_stroke("not xml") == "not xml"


# ── mutate_path ───────────────────────────────────────────────────────────────

# Path data is routinely written with leading-dot decimals and no separators,
# which is exactly where a naive number scan goes wrong. Matched here with a
# regex written independently of the one under test.
_NUM_TOKENS = re.compile(r"-?(?:\d+\.\d+|\.\d+|\d+)")
SVG_PATH = f'<svg xmlns="{NS}"><path d="M 1 .5 L 10 20" fill="#123456"/></svg>'
# No viewBox and no width on these fixtures, so the nudge falls back to the
# default canvas span.
_PATH_STEP = max(2.0, 100.0 * 0.03)


def _path_numbers(svg: str) -> list[float]:
    return [float(t) for t in _NUM_TOKENS.findall(_find(svg, "path").get("d", ""))]


def test_mutate_path_nudges_exactly_one_coordinate():
    """A merged replacement changes how many coordinates the path has, reshaping
    it wholesale -- and it still parses, so nothing downstream notices."""
    before = [1.0, 0.5, 10.0, 20.0]
    nudged = 0
    for seed in range(50):
        random.seed(seed)
        after = _path_numbers(mutate_path(SVG_PATH))
        assert len(after) == len(before)
        pairs = enumerate(zip(before, after, strict=True))
        diffs = [i for i, (a, b) in pairs if a != b]
        assert len(diffs) <= 1
        for i in diffs:
            nudged += 1
            # An offset, not a percentage: the step is the same for every
            # number in the path regardless of how large it is.
            assert abs(after[i] - before[i]) <= _PATH_STEP + 0.05
    assert nudged, "no seed ever moved a coordinate"


def test_mutate_path_separates_the_nudged_number_from_its_neighbours():
    svg = f'<svg xmlns="{NS}"><path d="M0 0L3.5.5Z"/></svg>'
    before = [0.0, 0.0, 3.5, 0.5]
    for seed in range(50):
        random.seed(seed)
        after = _path_numbers(mutate_path(svg))
        assert len(after) == len(before)
        for a, b in zip(before, after, strict=True):
            assert abs(b - a) <= _PATH_STEP + 0.05


def test_mutate_path_stays_valid_svg():
    for seed in range(20):
        random.seed(seed)
        assert ET.fromstring(mutate_path(SVG_PATH)).tag.endswith("svg")


def test_mutate_path_leaves_other_attributes_alone():
    random.seed(0)
    assert _find(mutate_path(SVG_PATH), "path").get("fill") == "#123456"


def test_mutate_path_without_a_d_attribute_unchanged():
    svg = f'<svg xmlns="{NS}"><rect width="10" height="10"/></svg>'
    assert mutate_path(svg) == svg


def test_mutate_path_with_no_numbers_unchanged():
    svg = f'<svg xmlns="{NS}"><path d="Z"/></svg>'
    assert mutate_path(svg) == svg


def test_mutate_path_invalid_svg_unchanged():
    assert mutate_path("not xml") == "not xml"


# ── mutate_reorder ────────────────────────────────────────────────────────────

SVG_THREE = f'<svg xmlns="{NS}"><rect id="a"/><circle id="b"/><ellipse id="c"/></svg>'


def _ids(svg: str) -> list[str | None]:
    return [el.get("id") for el in ET.fromstring(svg)]


def test_mutate_reorder_swaps_two_adjacent_siblings():
    orders = set()
    for seed in range(20):
        random.seed(seed)
        ids = _ids(mutate_reorder(SVG_THREE))
        assert sorted(i or "" for i in ids) == ["a", "b", "c"]
        orders.add(tuple(ids))
    assert orders == {("b", "a", "c"), ("a", "c", "b")}


def test_mutate_reorder_preserves_element_content():
    svg = f'<svg xmlns="{NS}"><rect width="10" fill="red"/><circle r="3"/></svg>'
    random.seed(0)
    children = list(ET.fromstring(mutate_reorder(svg)))
    rect = next(c for c in children if c.tag.endswith("rect"))
    assert rect.attrib == {"width": "10", "fill": "red"}
    assert children[0].tag.endswith("circle")


def test_mutate_reorder_works_inside_a_group():
    svg = f'<svg xmlns="{NS}"><g><rect id="a"/><circle id="b"/></g></svg>'
    random.seed(0)
    group = _find(mutate_reorder(svg), "g")
    assert [el.get("id") for el in group] == ["b", "a"]


def test_mutate_reorder_single_child_unchanged():
    svg = f'<svg xmlns="{NS}"><rect id="a"/></svg>'
    assert mutate_reorder(svg) == svg


def test_mutate_reorder_invalid_svg_unchanged():
    assert mutate_reorder("not xml") == "not xml"


def test_crossover_keeps_every_element_of_the_first_parent():
    """Regression: splicing by document order dropped whatever the shorter
    parent lacked at each index -- 23 circles and 16 numerals out of a
    63-element seed. Matching elements to each other makes loss impossible: a
    matched pair contributes one element, an unmatched one is carried."""
    from pathlib import Path

    seeds = sorted(Path("bench/cases/connect-dots/seeds").glob("*.svg"))
    a, b = seeds[0].read_text(), seeds[3].read_text()

    for _ in range(8):
        child = crossover(a, b)
        assert len(re.findall(r"<[a-zA-Z]", child)) == len(re.findall(r"<[a-zA-Z]", a))


def test_crossover_will_not_swap_an_element_for_a_piece_of_itself():
    """One drawing splits a blade into two paths where the other uses one.
    Half a blade covers half of the whole blade, which is enough overlap to
    look like a match, and taking the whole in exchange for the half is how a
    drawing loses content."""
    one_piece = (
        f'<svg xmlns="{NS}" viewBox="0 0 64 64">'
        '<rect x="0" y="0" width="64" height="64" fill="#ffffff"/>'
        '<rect x="8" y="8" width="48" height="48" fill="#336699"/>'
        "</svg>"
    )
    two_pieces = (
        f'<svg xmlns="{NS}" viewBox="0 0 64 64">'
        '<rect x="0" y="0" width="64" height="64" fill="#ffffff"/>'
        '<rect x="8" y="8" width="24" height="48" fill="#336699"/>'
        '<rect x="32" y="8" width="24" height="48" fill="#336699"/>'
        "</svg>"
    )

    for seed in range(20):
        random.seed(seed)
        assert 'width="24"' not in crossover(one_piece, two_pieces)


def test_crossover_swaps_matching_elements_between_parents():
    red = (
        f'<svg xmlns="{NS}" viewBox="0 0 64 64">'
        '<rect x="0" y="0" width="64" height="64" fill="#ffffff"/>'
        '<circle cx="32" cy="32" r="16" fill="#ff0000"/>'
        "</svg>"
    )
    blue = (
        f'<svg xmlns="{NS}" viewBox="0 0 64 64">'
        '<rect x="0" y="0" width="64" height="64" fill="#ffffff"/>'
        '<circle cx="32" cy="32" r="16" fill="#0000ff"/>'
        "</svg>"
    )

    assert any("#0000ff" in crossover(red, blue) for _ in range(20))


def test_a_small_integer_attribute_can_still_grow():
    """A proportional nudge rounded back to an integer cannot move a value of
    1: every factor in the range maps it to 1 again. A rounded corner that
    drifted down to rx="1" was square for the rest of the run, with no mutation
    able to undo it."""
    svg = (
        f'<svg xmlns="{NS}" viewBox="0 0 64 64">'
        '<rect x="8" y="8" width="28" height="18" rx="1" ry="9" fill="#eb93a6"/>'
        "</svg>"
    )

    seen = set()
    for seed in range(60):
        random.seed(seed)
        out = mutate_numeric(svg)
        found = re.search(r'rx="(\d+)"', out)
        if found:
            seen.add(int(found.group(1)))

    assert max(seen) > 1, f"rx never grew past 1, only saw {sorted(seen)}"


def test_translate_moves_both_axes_by_the_same_absolute_step():
    """A coordinate scaled by a factor moves in proportion to its distance from
    the origin, so a far element jumps and a near one barely stirs. A move has
    to be an offset, and it has to carry both axes: an element displaced
    diagonally would otherwise need two separate mutations accepted to arrive."""
    svg = (
        f'<svg xmlns="{NS}" viewBox="0 0 400 400">'
        '<circle cx="20" cy="20" r="5"/>'
        '<circle cx="380" cy="380" r="5"/>'
        "</svg>"
    )

    offsets = []
    for seed in range(40):
        random.seed(seed)
        out = mutate_translate(svg)
        found = re.search(r"translate\((-?[\d.]+) (-?[\d.]+)\)", out)
        assert found, "no translate was applied"
        offsets.append((abs(float(found.group(1))), abs(float(found.group(2)))))

    assert all(dx > 0 and dy > 0 for dx, dy in offsets), "an axis was left behind"
    # Both elements draw offsets from one distribution; nothing scales with
    # where the element happens to sit.
    assert max(max(o) for o in offsets) <= 400 * 0.05 + 0.01


def test_translate_keeps_a_transform_the_element_already_had():
    """Replacing it would silently drop a scale or rotate the drawing depends
    on, and appending would let that transform scale the offset."""
    svg = (
        f'<svg xmlns="{NS}" viewBox="0 0 400 400">'
        '<circle cx="100" cy="100" r="5" transform="scale(2)"/>'
        "</svg>"
    )

    random.seed(1)
    out = mutate_translate(svg)

    assert "scale(2)" in out
    assert re.search(r'transform="translate\([^)]*\) scale\(2\)"', out)


def test_numeric_moves_a_coordinate_by_the_same_step_wherever_it_sits():
    """Scaling a coordinate ties the step to distance from the origin: at a
    factor of 0.7-1.3 a point at cx=20 shifts by at most 6px while one at
    cx=380 leaps 114px. Position is an offset, not a magnitude."""

    def moves(cx: float) -> list[float]:
        svg = (
            f'<svg xmlns="{NS}" viewBox="0 0 400 400">'
            f'<circle cx="{cx:g}" cy="200" r="9"/></svg>'
        )
        out = []
        for seed in range(200):
            random.seed(seed)
            found = re.search(r'\bcx="([-\d.]+)"', mutate_numeric(svg))
            if found and float(found.group(1)) != cx:
                out.append(abs(float(found.group(1)) - cx))
        return out

    near, far = moves(20), moves(380)
    assert near
    assert far
    ratio = statistics.fmean(far) / statistics.fmean(near)
    assert 0.8 < ratio < 1.25, f"step still depends on position (ratio {ratio:.1f})"


def test_numeric_keeps_sizes_proportional():
    """A 3px radius and a 90px radius must not move by the same amount, or the
    small one is destroyed while the large one barely stirs."""

    def moves(r: float) -> list[float]:
        svg = (
            f'<svg xmlns="{NS}" viewBox="0 0 400 400">'
            f'<circle cx="200" cy="200" r="{r:g}"/></svg>'
        )
        out = []
        for seed in range(200):
            random.seed(seed)
            found = re.search(r'\br="([-\d.]+)"', mutate_numeric(svg))
            if found and float(found.group(1)) != r:
                out.append(abs(float(found.group(1)) - r))
        return out

    assert statistics.fmean(moves(90)) > 3 * statistics.fmean(moves(3))


def test_numeric_can_move_an_opacity_off_its_endpoint():
    """opacity="1" is an integer, so a rounded proportional nudge only ever
    returned 1 or 0 -- the element was fully opaque or gone."""
    svg = (
        f'<svg xmlns="{NS}" viewBox="0 0 400 400">'
        '<circle cx="200" cy="200" r="9" opacity="1"/></svg>'
    )

    seen = set()
    for seed in range(200):
        random.seed(seed)
        found = re.search(r'\bopacity="([-\d.]+)"', mutate_numeric(svg))
        if found:
            seen.add(float(found.group(1)))

    assert len(seen) > 5, f"opacity only reached {sorted(seen)}"
    assert all(0.0 <= v <= 1.0 for v in seen)


def test_path_nudge_never_corrupts_an_arc_flag():
    """In `A rx ry rotation large-arc sweep x y` the two flags are booleans.
    Nudged, they become numbers like -1.8, which is not path data: the renderer
    drops the path and the element disappears from the drawing."""
    svg = (
        f'<svg xmlns="{NS}" viewBox="0 0 64 64">'
        '<path d="M16 32 A16 16 0 1 0 48 32 A16 16 0 1 0 16 32 Z"/></svg>'
    )

    changed = 0
    for seed in range(120):
        random.seed(seed)
        out = mutate_path(svg)
        found = re.search(r'd="([^"]*)"', out)
        assert found
        d = found.group(1)
        changed += d not in svg
        for arc in re.finditer(
            r"A\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", d
        ):
            assert arc.group(4) in ("0", "1"), f"large-arc flag corrupted: {d}"
            assert arc.group(5) in ("0", "1"), f"sweep flag corrupted: {d}"

    assert changed > 100, "skipping flags must not stop the operator working"


def test_crossover_pairs_text_by_what_it_says():
    """Overlap can only pair elements already sitting on top of each other,
    which is precisely what two seeds disagreeing about placement do not do. A
    numeral is the same feature in both drawings however far apart they are."""
    left = (
        f'<svg xmlns="{NS}" viewBox="0 0 64 64">'
        '<rect x="0" y="0" width="64" height="64" fill="#ffffff"/>'
        '<text x="4" y="12" font-size="8" fill="#ff0000">7</text>'
        "</svg>"
    )
    right = (
        f'<svg xmlns="{NS}" viewBox="0 0 64 64">'
        '<rect x="0" y="0" width="64" height="64" fill="#ffffff"/>'
        '<text x="52" y="60" font-size="8" fill="#0000ff">7</text>'
        "</svg>"
    )

    swapped = 0
    for seed in range(20):
        random.seed(seed)
        child = crossover(left, right)
        assert len(re.findall(r"<[a-zA-Z]", child)) == len(
            re.findall(r"<[a-zA-Z]", left)
        )
        swapped += "#0000ff" in child

    assert swapped, "the far-away numeral was never paired with its twin"


def test_crossover_does_not_pair_text_that_says_something_else():
    """Matching by label must mean the label, or it degenerates into pairing
    any text with any other and the drawing loses its numbering."""
    left = (
        f'<svg xmlns="{NS}" viewBox="0 0 64 64">'
        '<rect x="0" y="0" width="64" height="64" fill="#ffffff"/>'
        '<text x="4" y="12" font-size="8" fill="#ff0000">7</text>'
        "</svg>"
    )
    right = (
        f'<svg xmlns="{NS}" viewBox="0 0 64 64">'
        '<rect x="0" y="0" width="64" height="64" fill="#ffffff"/>'
        '<text x="52" y="60" font-size="8" fill="#0000ff">3</text>'
        "</svg>"
    )

    for seed in range(20):
        random.seed(seed)
        assert "#0000ff" not in crossover(left, right)


def test_crossover_pairs_each_duplicate_label_only_once():
    """Two elements drawing the same string must not both take the same
    partner, which would leave one unpaired and the count wrong."""
    left = (
        f'<svg xmlns="{NS}" viewBox="0 0 64 64">'
        '<rect x="0" y="0" width="64" height="64" fill="#ffffff"/>'
        '<text x="4" y="12" font-size="8">1</text>'
        '<text x="4" y="30" font-size="8">1</text>'
        "</svg>"
    )
    right = (
        f'<svg xmlns="{NS}" viewBox="0 0 64 64">'
        '<rect x="0" y="0" width="64" height="64" fill="#ffffff"/>'
        '<text x="50" y="12" font-size="8">1</text>'
        "</svg>"
    )

    for seed in range(20):
        random.seed(seed)
        child = crossover(left, right)
        assert len(re.findall(r"<text", child)) == 2
