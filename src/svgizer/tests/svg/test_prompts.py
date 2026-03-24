from svgizer.svg.prompts import extract_svg_fragment, is_valid_svg


def test_extract_svg_fragment_clean():
    raw = "<svg><rect/></svg>"
    assert extract_svg_fragment(raw) == raw


def test_extract_svg_fragment_markdown_wrapped():
    raw = "Here is your code:\n```xml\n<svg><rect/></svg>\n```\nEnjoy!"
    expected = "<svg><rect/></svg>"
    assert extract_svg_fragment(raw) == expected


def test_extract_svg_fragment_mixed_case():
    raw = "```\n<SVG><circle/></svG>\n```"
    expected = "<SVG><circle/></svG>"
    assert extract_svg_fragment(raw) == expected


def test_extract_svg_fragment_no_tags():
    raw = "I could not generate the SVG."
    assert extract_svg_fragment(raw) == raw


def test_is_valid_svg_happy_path():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
    valid, err = is_valid_svg(svg)
    assert valid is True
    assert err is None


def test_is_valid_svg_malformed_xml():
    svg = "<svg><rect></svg>"  # Unclosed tag
    valid, err = is_valid_svg(svg)
    assert valid is False
    assert err is str
    assert "XML parse error" in err


def test_is_valid_svg_wrong_root_tag():
    xml = "<g><rect/></g>"
    valid, err = is_valid_svg(xml)
    assert valid is False
    assert err is str
    assert "Root tag is not <svg>" in err
