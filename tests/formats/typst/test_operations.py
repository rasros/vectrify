import importlib.util

from vectrify.formats.typst.operations import (
    _mutate_color,
    _random_numeric_tweak,
    _remove_element,
    _reorder_elements,
    apply_crossover,
    apply_mutation,
    canonicalize_page_setup,
    scene_units,
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


def test_apply_mutation_returns_typst_string():
    result, _summary = apply_mutation(_TYPST_CODE)
    assert "#set page" in result


def test_apply_mutation_returns_valid_label():
    _, summary = apply_mutation(_TYPST_CODE)
    assert summary in {
        "Mutation: color tweak",
        "Mutation: position tweak",
        "Mutation: size/stroke tweak",
        "Mutation: removed element",
        "Mutation: reordered elements",
        "Mutation: added element",
    }


def test_apply_crossover_returns_typst_string():
    code_b = (
        "#set page(width: auto, height: auto, margin: 0pt)\n#polygon(fill: green)\n"
    )
    result, summary = apply_crossover(_TYPST_CODE, code_b)
    assert "#set page" in result
    assert summary == "Crossover: scene element injection"


def test_scene_units_keep_multiline_place_and_nested_content_together():
    code = """#set page(width: 100pt, height: 100pt, margin: 0pt)
#place(
  dx: 10pt,
  dy: 20pt,
)[
  #rect(width: 20pt, height: 10pt, fill: red)
]
#circle(radius: 4pt)
"""
    units = scene_units(code)
    assert [unit.name for unit in units] == ["place", "circle"]
    assert "#rect" in code[units[0].start : units[0].end]


def test_page_canonicalization_removes_auto_and_duplicate_settings():
    code = "#set page(width: auto)\n#rect(width: 5pt)\n#set page(height: auto)"
    assert canonicalize_page_setup(code, (320, 240)) == (
        "#set page(width: 320pt, height: 240pt, margin: 0pt)\n#rect(width: 5pt)\n"
    )


def test_mutations_never_change_fixed_page_setup():
    code = (
        "#set page(width: 320pt, height: 240pt, margin: 0pt)\n"
        "#rect(width: 10pt, fill: red)\n#circle(radius: 4pt)"
    )
    page = code.splitlines()[0]
    for _ in range(50):
        result, _ = apply_mutation(code)
        assert result.splitlines()[0] == page


def test_crossover_falls_back_to_mutation_when_no_elements_in_b():
    code_b = "#set page(width: auto, height: auto, margin: 0pt)\n"
    result, summary = apply_crossover(_TYPST_CODE, code_b)
    assert "#set page" in result
    assert summary.startswith("Mutation: ")


def test_reorder_keeps_every_element_addressable_without_trailing_newline():
    """Regression: splicing with keepends=True fused two elements onto one line
    when the moved line was last and lacked a newline. _ELEMENT_LINE_RE is
    anchored per line, so the fused element became invisible to every later
    mutation -- permanently losing a mutable site.
    """
    import random

    from vectrify.formats.typst.operations import _ELEMENT_LINE_RE, _reorder_elements

    code = "#set page(width: 100pt)\n#rect(width: 10pt)\n#circle(radius: 5pt)"
    assert len(_ELEMENT_LINE_RE.findall(code)) == 2

    random.seed(0)
    changed = None
    for _ in range(20):
        out = _reorder_elements(code)
        if out != code:
            changed = out
            break

    assert changed is not None, "reorder never produced a change"
    assert len(_ELEMENT_LINE_RE.findall(changed)) == 2
    assert "5pt)#rect" not in changed


def test_crossover_keeps_every_element_addressable():

    from vectrify.formats.typst.operations import (
        _ELEMENT_LINE_RE,
        apply_crossover,
    )

    a = "#set page(width: 100pt)\n#rect(width: 10pt)\n#circle(radius: 5pt)"
    b = "#set page(width: 100pt)\n#polygon()\n#line(length: 9pt)"  # last line bare
    content, _ = apply_crossover(a, b)

    for line in content.splitlines():
        assert (
            line.count("#rect(")
            + line.count("#circle(")
            + line.count("#line(")
            + line.count("#polygon(")
            <= 1
        ), f"elements fused onto one line: {line!r}"
    assert _ELEMENT_LINE_RE.findall(content)


def test_split_lines_normalizes_the_final_newline():
    from vectrify.formats.typst.operations import _split_lines

    assert _split_lines("a\nb") == ["a\n", "b\n"]
    assert _split_lines("a\nb\n") == ["a\n", "b\n"]
    assert _split_lines("") == []
