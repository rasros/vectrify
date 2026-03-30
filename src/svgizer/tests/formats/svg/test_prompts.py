from svgizer.formats.svg.prompts import (
    build_svg_gen_prompt,
    extract_svg_fragment,
    is_valid_svg,
)

_IMG_URL = "data:image/png;base64,abc"
_RENDER_URL = "data:image/png;base64,def"
_DIFF_URL = "data:image/png;base64,ghi"
_SVG = "<svg><rect/></svg>"


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
    assert isinstance(err, str)
    assert "XML parse error" in err


def test_is_valid_svg_wrong_root_tag():
    xml = "<g><rect/></g>"
    valid, err = is_valid_svg(xml)
    assert valid is False
    assert isinstance(err, str)
    assert "Root tag is not <svg>" in err


def _text_blocks(blocks: list[dict]) -> list[str]:
    return [b["text"] for b in blocks if b["type"] == "input_text"]


def _image_blocks(blocks: list[dict]) -> list[str]:
    return [b["image_url"] for b in blocks if b["type"] == "input_image"]


def test_gen_prompt_first_attempt_no_svg():
    blocks = build_svg_gen_prompt(_IMG_URL, iter_index=1)
    text = "\n".join(_text_blocks(blocks))
    assert "search/replace" not in text
    assert "CURRENT SVG" not in text


def test_gen_prompt_refinement_includes_svg():
    blocks = build_svg_gen_prompt(_IMG_URL, iter_index=5, svg_prev=_SVG)
    text = "\n".join(_text_blocks(blocks))
    assert "search/replace" in text.lower()
    assert "CURRENT SVG CODE TO MODIFY" in text
    assert _SVG in text


def test_gen_prompt_invalid_msg_shown():
    blocks = build_svg_gen_prompt(
        _IMG_URL, iter_index=2, svg_prev=_SVG, svg_prev_invalid_msg="bad parse"
    )
    text = "\n".join(_text_blocks(blocks))
    assert "bad parse" in text


def test_gen_prompt_goal_included():
    blocks = build_svg_gen_prompt(
        _IMG_URL, iter_index=3, svg_prev=_SVG, goal="fix the circle"
    )
    text = "\n".join(_text_blocks(blocks))
    assert "fix the circle" in text


def test_gen_prompt_iter_index_in_context():
    blocks = build_svg_gen_prompt(_IMG_URL, iter_index=42)
    text = "\n".join(_text_blocks(blocks))
    assert "42" in text


def test_gen_prompt_original_image_always_present():
    blocks = build_svg_gen_prompt(_IMG_URL, iter_index=1)
    assert _IMG_URL in _image_blocks(blocks)


def test_gen_prompt_render_url_added_when_provided():
    blocks = build_svg_gen_prompt(
        _IMG_URL, iter_index=2, svg_prev=_SVG, rasterized_svg_data_url=_RENDER_URL
    )
    images = _image_blocks(blocks)
    assert _RENDER_URL in images


def test_gen_prompt_render_url_absent_when_not_provided():
    blocks = build_svg_gen_prompt(_IMG_URL, iter_index=1)
    assert _RENDER_URL not in _image_blocks(blocks)


def test_gen_prompt_diff_url_added_when_provided():
    blocks = build_svg_gen_prompt(
        _IMG_URL, iter_index=2, svg_prev=_SVG, diff_data_url=_DIFF_URL
    )
    images = _image_blocks(blocks)
    assert _DIFF_URL in images
    text = "\n".join(_text_blocks(blocks))
    assert "Difference Map" in text


def test_gen_prompt_diff_url_absent_when_not_provided():
    blocks = build_svg_gen_prompt(_IMG_URL, iter_index=1)
    assert _DIFF_URL not in _image_blocks(blocks)


def test_gen_prompt_diff_format_instructions_in_edit():
    blocks = build_svg_gen_prompt(_IMG_URL, iter_index=2, svg_prev=_SVG)
    text = "\n".join(_text_blocks(blocks))
    assert "<<<SEARCH>>>" in text
    assert "<<<REPLACE>>>" in text
    assert "<<<END>>>" in text
