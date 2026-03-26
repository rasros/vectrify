import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from svgizer.svg.operations import (
    crossover,
    crossover_with_micro_search,
    mutate_drop_style_property,
    mutate_numeric,
    mutate_remove_node,
    mutate_with_micro_search,
    with_micro_search,
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


# ---------------------------------------------------------------------------
# crossover
# ---------------------------------------------------------------------------


def test_crossover_returns_valid_svg():
    result = crossover(SVG_A, SVG_B)
    root = ET.fromstring(result)
    assert root.tag.endswith("svg")


def test_crossover_children_only_from_parents():
    result = crossover(SVG_A, SVG_B)
    root = ET.fromstring(result)
    tags = {c.tag.split("}")[-1] for c in root}
    assert tags <= {"circle", "rect", "ellipse", "line"}


def test_crossover_k1_two_segments():
    # k=1 → one cut-point → child has elements from exactly A then B (or B then A)
    # SVG_A: [circle, rect], SVG_B: [ellipse, line]
    a_tags = {"circle", "rect"}
    b_tags = {"ellipse", "line"}
    result = crossover(SVG_A, SVG_B, k=1)
    root = ET.fromstring(result)
    tags = [c.tag.split("}")[-1] for c in root]
    # All tags come from one of the two parents
    assert all(t in a_tags | b_tags for t in tags)


def test_crossover_unequal_lengths():
    long_a = f'<svg xmlns="{NS}"><rect/><circle/><ellipse/><line/><path/></svg>'
    result = crossover(long_a, SVG_B, k=2)
    root = ET.fromstring(result)
    # Child length is at most max(len_a, len_b)
    assert len(list(root)) <= 5


def test_crossover_k_clamped_to_max():
    # k larger than max_len-1 shouldn't crash
    result = crossover(SVG_A, SVG_B, k=100)
    root = ET.fromstring(result)
    assert root.tag.endswith("svg")


def test_crossover_degenerate_single_element():
    single_a = f'<svg xmlns="{NS}"><rect/></svg>'
    single_b = f'<svg xmlns="{NS}"><circle/></svg>'
    result = crossover(single_a, single_b)
    root = ET.fromstring(result)
    assert len(list(root)) == 1


def test_crossover_invalid_svg_returns_a():
    result = crossover("not xml", SVG_B)
    assert result == "not xml"


# ---------------------------------------------------------------------------
# mutate_remove_node
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# mutate_numeric
# ---------------------------------------------------------------------------


def test_mutate_numeric_changes_an_attribute():
    # Run enough times to ensure at least one attribute changes
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
    for _ in range(50):
        result = mutate_numeric(svg)
        root = ET.fromstring(result)
        rect = root.find(".//{http://www.w3.org/2000/svg}rect")
        if rect is not None and "opacity" in rect.attrib:
            val = float(rect.attrib["opacity"])
            assert 0.0 <= val <= 1.0


def test_mutate_numeric_invalid_svg_unchanged():
    result = mutate_numeric("not xml")
    assert result == "not xml"


def test_mutate_numeric_no_numeric_attrs_unchanged():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><g id="foo" fill="red"/></svg>'
    result = mutate_numeric(svg)
    assert ET.fromstring(result).tag.endswith("svg")


@pytest.mark.parametrize(
    "op", [mutate_remove_node, mutate_numeric, mutate_drop_style_property]
)
def test_mutation_result_is_string(op):
    assert isinstance(op(SVG_ONE), str)


# ---------------------------------------------------------------------------
# with_retries
# ---------------------------------------------------------------------------

FALLBACK = SVG_ONE


def test_with_retries_returns_valid_result_on_first_try():
    result = with_retries(lambda: SVG_A, fallback=FALLBACK)
    assert result == SVG_A


def test_with_retries_returns_fallback_when_op_always_invalid():
    result = with_retries(lambda: "not xml at all", fallback=FALLBACK, max_retries=3)
    assert result == FALLBACK


def test_with_retries_returns_fallback_when_op_always_raises():
    def boom():
        raise RuntimeError("boom")

    result = with_retries(boom, fallback=FALLBACK, max_retries=3)
    assert result == FALLBACK


def test_with_retries_succeeds_after_initial_failures():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            return "bad xml"
        return SVG_B

    result = with_retries(flaky, fallback=FALLBACK, max_retries=5)
    assert result == SVG_B
    assert len(calls) == 3


def test_with_retries_exhausts_all_attempts():
    calls = []

    def always_bad():
        calls.append(1)
        return "bad"

    with_retries(always_bad, fallback=FALLBACK, max_retries=4)
    assert len(calls) == 4


# ---------------------------------------------------------------------------
# with_micro_search
# ---------------------------------------------------------------------------


def test_with_micro_search_finds_improvement():
    # Target image is blue
    target_img = Image.new("RGB", (10, 10), color="blue")
    fallback_svg = f'<svg xmlns="{NS}"><rect width="10" height="10" fill="red"/></svg>'
    better_svg = f'<svg xmlns="{NS}"><rect width="10" height="10" fill="blue"/></svg>'

    yields = [(fallback_svg, "bad"), (better_svg, "good")]

    def op_gen():
        return yields.pop(0)

    best_svg, summary = with_micro_search(
        op_gen, fallback_svg, target_img, num_trials=2, default_summary="none"
    )
    assert best_svg == better_svg
    assert summary == "good"


def test_with_micro_search_no_improvement_returns_fallback():
    # Target image is blue, fallback is already perfect
    target_img = Image.new("RGB", (10, 10), color="blue")
    fallback_svg = f'<svg xmlns="{NS}"><rect width="10" height="10" fill="blue"/></svg>'
    worse_svg = f'<svg xmlns="{NS}"><rect width="10" height="10" fill="red"/></svg>'

    def op_gen():
        return worse_svg, "worse"

    best_svg, summary = with_micro_search(
        op_gen, fallback_svg, target_img, num_trials=2, default_summary="No improvement"
    )
    assert best_svg == fallback_svg
    assert summary == "No improvement"


def test_with_micro_search_ignores_invalid_renders():
    target_img = Image.new("RGB", (10, 10), color="blue")
    fallback_svg = f'<svg xmlns="{NS}"><rect width="10" height="10" fill="red"/></svg>'
    invalid_svg = "not an svg"

    def op_gen():
        return invalid_svg, "invalid"

    best_svg, summary = with_micro_search(
        op_gen, fallback_svg, target_img, num_trials=1, default_summary="none"
    )
    # The invalid SVG should throw inside the rasterizer, be caught, and ignored
    assert best_svg == fallback_svg
    assert summary == "none"


# ---------------------------------------------------------------------------
# High-level Micro Search Wrappers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# mutate_drop_style_property
# ---------------------------------------------------------------------------

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
    assert len(props) == 2  # started with 3, one removed


def test_mutate_drop_style_property_single_prop_unchanged():
    result = mutate_drop_style_property(SVG_SINGLE_PROP)
    assert result == SVG_SINGLE_PROP


def test_mutate_drop_style_property_no_style_unchanged():
    result = mutate_drop_style_property(SVG_NO_STYLE)
    assert result == SVG_NO_STYLE


def test_mutate_drop_style_property_invalid_svg_unchanged():
    result = mutate_drop_style_property("not xml")
    assert result == "not xml"
