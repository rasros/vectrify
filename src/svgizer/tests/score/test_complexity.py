from svgizer.score.complexity import svg_complexity


def test_none_returns_zero():
    assert svg_complexity(None) == 0.0


def test_empty_string_returns_zero():
    assert svg_complexity("") == 0.0


def test_byte_length():
    svg = "<svg/>"
    assert svg_complexity(svg) == float(len(svg.encode()))


def test_longer_svg_has_higher_complexity():
    short = "<svg/>"
    long = "<svg>" + "x" * 1000 + "</svg>"
    assert svg_complexity(long) > svg_complexity(short)
