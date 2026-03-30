import importlib.util

import pytest
from PIL import Image

from svgizer.formats.typst.operations import (
    _mutate_color,
    _random_numeric_tweak,
    _remove_element,
    _reorder_elements,
    crossover_with_micro_search,
    mutate_with_micro_search,
)

_TYPST_AVAILABLE = importlib.util.find_spec("typst") is not None

_TYPST_CODE = (
    "#set page(width: auto, height: auto, margin: 0pt)\n"
    "#rect(width: 100pt, height: 50pt, fill: red)\n"
    "#circle(radius: 20pt, fill: blue)\n"
    "#line(start: (0pt, 0pt), end: (50pt, 50pt), stroke: black)\n"
)


# --- _random_numeric_tweak ---


def test_random_numeric_tweak_changes_value_keeps_unit():
    code = "#rect(width: 100pt)"
    changed = False
    for _ in range(30):
        result = _random_numeric_tweak(code)
        if result != code:
            changed = True
            assert "pt)" in result
            assert "100pt" not in result
            break
    assert changed


def test_random_numeric_tweak_handles_percentages():
    code = "#rect(width: 50%)"
    for _ in range(10):
        result = _random_numeric_tweak(code)
        assert "%" in result


def test_random_numeric_tweak_ignores_no_units():
    code = "#rect(width: 100)"
    result = _random_numeric_tweak(code)
    assert result == code


def test_random_numeric_tweak_result_is_positive():
    # Even very small values should stay ≥ 0.1
    code = "#rect(width: 0.1pt)"
    for _ in range(20):
        result = _random_numeric_tweak(code)
        assert "pt" in result
        # Extract the numeric part and verify it's positive
        import re

        m = re.search(r"(\d+(?:\.\d+)?)pt", result)
        assert m is not None
        assert float(m.group(1)) > 0


# --- _mutate_color ---


def test_mutate_color_changes_fill():
    code = "#rect(fill: red)"
    changed = False
    for _ in range(30):
        result = _mutate_color(code)
        if result != code:
            changed = True
            assert "fill:" in result
            assert "red" not in result
            break
    assert changed


def test_mutate_color_changes_stroke():
    code = "#line(stroke: black)"
    changed = False
    for _ in range(30):
        result = _mutate_color(code)
        if result != code:
            changed = True
            assert "stroke:" in result
            break
    assert changed


def test_mutate_color_no_match_returns_unchanged():
    code = "#rect(width: 10pt)"
    result = _mutate_color(code)
    assert result == code


def test_mutate_color_result_still_has_fill_keyword():
    code = "#circle(radius: 10pt, fill: blue)"
    for _ in range(10):
        result = _mutate_color(code)
        assert "fill:" in result


# --- _remove_element ---


def _count_shapes(code: str) -> int:
    return code.count("#rect") + code.count("#circle") + code.count("#line")


def test_remove_element_removes_one_shape():
    result = _remove_element(_TYPST_CODE)
    assert _count_shapes(result) == _count_shapes(_TYPST_CODE) - 1


def test_remove_element_keeps_page_setup():
    result = _remove_element(_TYPST_CODE)
    assert "#set page" in result


def test_remove_element_does_not_remove_last_element():
    code = "#set page(width: auto, height: auto, margin: 0pt)\n#rect(width: 10pt)\n"
    result = _remove_element(code)
    assert result == code


# --- _reorder_elements ---


def test_reorder_elements_preserves_all_elements():
    result = _reorder_elements(_TYPST_CODE)
    assert "#rect" in result
    assert "#circle" in result
    assert "#line" in result
    assert "#set page" in result


def test_reorder_elements_changes_order():
    changed = False
    for _ in range(30):
        result = _reorder_elements(_TYPST_CODE)
        if result != _TYPST_CODE:
            changed = True
            break
    assert changed


def test_reorder_elements_single_element_unchanged():
    code = "#set page(width: auto, height: auto, margin: 0pt)\n#rect(width: 10pt)\n"
    result = _reorder_elements(code)
    assert result == code


# --- mutate_with_micro_search ---


@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_mutate_with_micro_search_returns_typst_string():
    target = Image.new("RGB", (32, 32), color="red")
    result, summary = mutate_with_micro_search(_TYPST_CODE, target, num_trials=3)
    assert isinstance(result, str)
    assert "#set page" in result
    assert isinstance(summary, str)


@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_mutate_with_micro_search_returns_valid_label():
    target = Image.new("RGB", (64, 64), color=(200, 100, 50))
    _, summary = mutate_with_micro_search(_TYPST_CODE, target, num_trials=5)
    valid_labels = {
        "Mutation: color tweak",
        "Mutation: numeric tweak",
        "Mutation: removed element",
        "Mutation: reordered elements",
        "Mutation: no improvement",
    }
    assert summary in valid_labels


# --- crossover_with_micro_search ---


@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_crossover_with_micro_search_returns_typst_string():
    code_b = (
        "#set page(width: auto, height: auto, margin: 0pt)\n#polygon(fill: green)\n"
    )
    target = Image.new("RGB", (32, 32), color="green")
    result, summary = crossover_with_micro_search(
        _TYPST_CODE, code_b, target, num_trials=3
    )
    assert isinstance(result, str)
    assert summary in ("Crossover: element injection", "Mutation: no improvement")


@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_crossover_falls_back_to_mutation_when_no_elements_in_b():
    code_b = "#set page(width: auto, height: auto, margin: 0pt)\n"
    target = Image.new("RGB", (32, 32), color="red")
    result, _ = crossover_with_micro_search(_TYPST_CODE, code_b, target, num_trials=3)
    assert isinstance(result, str)
    assert "#set page" in result
