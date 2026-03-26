from svgizer.score.complexity import svg_complexity

NS = "http://www.w3.org/2000/svg"


def test_none_returns_zero():
    assert svg_complexity(None) == 0.0


def test_empty_string_returns_zero():
    assert svg_complexity("") == 0.0


def test_longer_svg_has_higher_complexity():
    short = f'<svg xmlns="{NS}"><rect/></svg>'
    long = f'<svg xmlns="{NS}">' + "<rect/>" * 50 + "</svg>"
    assert svg_complexity(long) > svg_complexity(short)


def test_more_elements_increases_complexity():
    few = f'<svg xmlns="{NS}"><rect/><rect/></svg>'
    many = f'<svg xmlns="{NS}">' + "<rect/>" * 20 + "</svg>"
    assert svg_complexity(many) > svg_complexity(few)


def test_path_vertices_increase_complexity():
    simple = f'<svg xmlns="{NS}"><path d="M0 0 L10 10"/></svg>'
    complex_ = (
        f'<svg xmlns="{NS}"><path d="' + "M0 0 C10 10 20 20 30 30 " * 20 + '"/></svg>'
    )
    assert svg_complexity(complex_) > svg_complexity(simple)


def test_repetitive_svg_compresses_better_than_diverse():
    repetitive = f'<svg xmlns="{NS}">' + '<rect x="1" y="1"/>' * 30 + "</svg>"
    diverse = (
        f'<svg xmlns="{NS}">'
        + "".join(
            f'<rect x="{i * 7}" y="{i * 13}" width="{i * 3 + 1}" height="{i * 5 + 2}" '
            f'fill="rgb({i * 8},{i * 17},{i * 23})" opacity="{0.1 + i * 0.03:.2f}"/>'
            for i in range(30)
        )
        + "</svg>"
    )
    # Both have 30 elements; diverse one carries more unique information
    assert svg_complexity(repetitive) < svg_complexity(diverse)


def test_invalid_svg_falls_back_to_compressed_size():
    malformed = "not xml at all <<<<"
    result = svg_complexity(malformed)
    assert result > 0.0
