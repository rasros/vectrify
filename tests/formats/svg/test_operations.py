import random
import re
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from vectrify.formats.svg.operations import (
    _NAMED_SVG_COLORS,
    crossover,
    crossover_with_micro_search,
    mutate_color,
    mutate_drop_style_property,
    mutate_numeric,
    mutate_path,
    mutate_remove_node,
    mutate_reorder,
    mutate_stroke,
    mutate_with_micro_search,
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


def test_crossover_unequal_lengths():
    long_a = f'<svg xmlns="{NS}"><rect/><circle/><ellipse/><line/><path/></svg>'
    result = crossover(long_a, SVG_B, k=2)
    root = ET.fromstring(result)
    assert 1 <= len(list(root)) <= 5


def test_crossover_k_clamped_to_max():
    result = crossover(SVG_A, SVG_B, k=100)
    root = ET.fromstring(result)
    children = list(root)
    tags = {c.tag.split("}")[-1] for c in children}
    assert children
    assert tags <= {"circle", "rect", "ellipse", "line"}


def test_crossover_degenerate_single_element():
    single_a = f'<svg xmlns="{NS}"><rect/></svg>'
    single_b = f'<svg xmlns="{NS}"><circle/></svg>'
    result = crossover(single_a, single_b)
    root = ET.fromstring(result)
    assert len(list(root)) == 1


def test_crossover_invalid_svg_returns_a():
    result = crossover("not xml", SVG_B)
    assert result == "not xml"


def test_mutate_remove_node_reduces_children():
    root_before = ET.fromstring(SVG_A)
    count_before = len(list(root_before))
    result = mutate_remove_node(SVG_A)
    root_after = ET.fromstring(result)
    assert len(list(root_after)) < count_before


def test_mutate_remove_node_still_valid_svg():
    result = mutate_remove_node(SVG_A)
    root = ET.fromstring(result)
    assert root.tag.endswith("svg")


def test_mutate_remove_node_invalid_svg_unchanged():
    result = mutate_remove_node("not xml")
    assert result == "not xml"


def test_mutate_remove_node_no_children_unchanged():
    empty = '<svg xmlns="http://www.w3.org/2000/svg"/>'
    result = mutate_remove_node(empty)
    assert ET.fromstring(result).tag.endswith("svg")


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
        mutate_remove_node,
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


def test_crossover_with_micro_search():
    target_img = Image.new("RGB", (10, 10), color="blue")
    svg_a = f'<svg xmlns="{NS}"><rect width="10" height="10" fill="red"/></svg>'
    svg_b = f'<svg xmlns="{NS}"><rect width="10" height="10" fill="blue"/></svg>'

    res, summary = crossover_with_micro_search(svg_a, svg_b, target_img, num_trials=2)
    assert isinstance(res, str)
    assert "<svg" in res
    assert "crossover" in summary.lower()


def test_mutate_with_micro_search():
    target_img = Image.new("RGB", (10, 10), color="blue")
    svg_a = f'<svg xmlns="{NS}"><rect width="10" height="10" fill="red"/></svg>'

    res, summary = mutate_with_micro_search(svg_a, target_img, num_trials=2)
    assert isinstance(res, str)
    assert "<svg" in res
    assert "mutation" in summary.lower()


def test_micro_search_survives_an_unrenderable_candidate():
    target_img = Image.new("RGB", (8, 8), color="blue")
    broken = f'<svg xmlns="{NS}" viewBox="a b c d"><rect width="8" height="8"/></svg>'

    res, summary = mutate_with_micro_search(broken, target_img, num_trials=2)
    assert res == broken
    assert "no improvement" in summary.lower()


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
            magnitude = max(2.0, abs(before[i]) * 0.15)
            assert abs(after[i] - before[i]) <= magnitude + 0.05
    assert nudged, "no seed ever moved a coordinate"


def test_mutate_path_separates_the_nudged_number_from_its_neighbours():
    """Compact path data has no separators ("L3.5.5"), so a replacement can fuse."""
    svg = f'<svg xmlns="{NS}"><path d="M0 0L3.5.5Z"/></svg>'
    before = [0.0, 0.0, 3.5, 0.5]
    for seed in range(50):
        random.seed(seed)
        after = _path_numbers(mutate_path(svg))
        assert len(after) == len(before)
        for a, b in zip(before, after, strict=True):
            assert abs(b - a) <= max(2.0, abs(a) * 0.15) + 0.05


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
