import xml.etree.ElementTree as ET

import pytest

from svgizer.svg.operations import (
    crossover,
    mutate_numeric,
    mutate_remove_node,
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


def test_crossover_children_from_parents():
    result = crossover(SVG_A, SVG_B, rate=1.0)
    root = ET.fromstring(result)
    tags = {c.tag.split("}")[-1] for c in root}
    assert tags <= {"circle", "rect", "ellipse", "line"}


def test_crossover_rate_one_takes_all_from_a():
    # rate=1.0 → always pick from A for shared positions
    result = crossover(SVG_A, SVG_B, rate=1.0)
    root = ET.fromstring(result)
    # Both children should come from A (circle, rect)
    tags = [c.tag.split("}")[-1] for c in root]
    assert tags == ["circle", "rect"]


def test_crossover_rate_zero_takes_all_from_b():
    result = crossover(SVG_A, SVG_B, rate=0.0)
    root = ET.fromstring(result)
    tags = [c.tag.split("}")[-1] for c in root]
    assert tags == ["ellipse", "line"]


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


@pytest.mark.parametrize("op", [mutate_remove_node, mutate_numeric])
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
