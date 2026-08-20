import re

from tests.helpers import image_urls, text_blocks
from vectrify.formats.svg.prompts import (
    build_svg_gen_prompt,
    extract_svg_fragment,
    is_valid_svg,
)

_IMG_URL = "data:image/png;base64,abc"
_RENDER_URL = "data:image/png;base64,def"
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


_text_blocks = text_blocks
_image_blocks = image_urls


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


def test_gen_prompt_diff_format_instructions_in_edit():
    blocks = build_svg_gen_prompt(_IMG_URL, iter_index=2, svg_prev=_SVG)
    text = "\n".join(_text_blocks(blocks))
    assert "<<<SEARCH>>>" in text
    assert "<<<REPLACE>>>" in text
    assert "<<<END>>>" in text


def test_gen_prompt_pins_the_viewbox_to_the_canvas():
    blocks = build_svg_gen_prompt(
        "data:image/png;base64,abc",
        1,
        svg_prev=None,
        canvas=(768, 512),
    )
    text = "\n".join(b["text"] for b in blocks if b["type"] == "input_text")
    assert "viewBox='0 0 768 512'" in text
    assert "0 0 W H" not in text


def test_svg_prompt_states_the_division_of_labour_with_local_search():
    blocks = build_svg_gen_prompt(_IMG_URL, 1, canvas=(512, 512))
    text = "\n".join(_text_blocks(blocks))
    assert "local optimizer" in text
    assert "Rough coordinates and approximate colors are fine" in text


def test_svg_edit_asks_for_structural_change_not_tuning():
    blocks = build_svg_gen_prompt(
        _IMG_URL,
        2,
        svg_prev=_SVG,
        rasterized_svg_data_url=_RENDER_URL,
        canvas=(512, 512),
    )
    text = "\n".join(_text_blocks(blocks))
    assert "the wrong kind of thing, or in the wrong place" in text


def test_svg_edit_never_mentions_the_removed_difference_map():
    """Regression: the edit prompt told the model to 'only modify elements
    visible in the difference map' long after that image stopped being sent,
    scoping every edit to something the model could not see."""
    blocks = build_svg_gen_prompt(
        _IMG_URL,
        2,
        svg_prev=_SVG,
        rasterized_svg_data_url=_RENDER_URL,
        canvas=(512, 512),
    )
    text = "\n".join(_text_blocks(blocks)).lower()
    assert "difference map" not in text
    assert "diff map" not in text


def test_prompt_asks_for_geometry_and_colors_in_mutable_places():
    """<polygon points> renders identically to a path and is mutable by nothing,
    so a candidate built from it looks fine and never improves again."""
    blocks = build_svg_gen_prompt(_IMG_URL, 1, canvas=(384, 384))
    text = "\n".join(_text_blocks(blocks))
    assert "`<path d=" in text
    assert "`stroke-width`" in text
    assert "stop-color" in text


def test_edit_prompt_also_carries_the_mutable_markup_rules():
    blocks = build_svg_gen_prompt(
        _IMG_URL,
        2,
        svg_prev=_SVG,
        rasterized_svg_data_url=_RENDER_URL,
        canvas=(384, 384),
    )
    assert "`<path d=" in "\n".join(_text_blocks(blocks))


def test_prompt_states_only_preferences():
    """Naming what to avoid teaches the model the forbidden markup exists."""
    from vectrify.formats.svg.prompts import MUTABLE_SVG

    lowered = MUTABLE_SVG.lower()
    for banned in ("never", "do not", "don't", "avoid", "instead of", "polygon"):
        assert banned not in lowered


def test_mutable_svg_rules_name_only_attributes_an_operator_reaches():
    """The rules are a promise about the optimizer; if the two drift apart the
    prompt starts steering the model toward markup nothing can move."""
    from vectrify.formats.svg.operations import _COLOR_ATTRS, _NUMERIC_ATTRS
    from vectrify.formats.svg.prompts import MUTABLE_SVG

    quoted = set(re.findall(r"`([a-z-]+)`", MUTABLE_SVG))
    attrs = {a for a in quoted if a not in {"transform", "d"}}
    assert attrs
    assert attrs <= (_NUMERIC_ATTRS | _COLOR_ATTRS)


def test_the_prompt_offers_the_file_name_as_evidence_of_the_subject():
    """Often the only place the subject is stated. One model read a
    connect-the-dots duck as a banana and two moons, named its groups
    accordingly, and drew a crescent where the eye's highlight belonged."""
    blocks = build_svg_gen_prompt(
        _IMG_URL, 1, canvas=(512, 512), source_name="connect-the-dots-little-duck.png"
    )
    text = "\n".join(_text_blocks(blocks))

    assert "connect-the-dots-little-duck.png" in text
    # Evidence, not instruction: plenty of files are called scan_04.png.
    assert "trust the image if they disagree" in text


def test_no_file_name_leaves_the_prompt_unchanged():
    without = "\n".join(
        _text_blocks(build_svg_gen_prompt(_IMG_URL, 1, canvas=(512, 512)))
    )

    assert "The file is named" not in without


def test_the_prompt_asks_what_the_picture_is_before_how_to_draw_it():
    """The group names are the model's record of what it thinks it is drawing,
    and every later edit works from them, so a part named for what it resembles
    gets drawn as that instead."""
    text = "\n".join(_text_blocks(build_svg_gen_prompt(_IMG_URL, 1, canvas=(512, 512))))

    assert "Work out what the picture depicts" in text


def test_the_edit_prompt_names_elements_that_paint_nothing():
    from vectrify.formats.svg.prompts import build_svg_gen_prompt

    blocks = build_svg_gen_prompt(
        "data:image/png;base64,AA",
        1,
        svg_prev="<svg/>",
        invisible=['<ellipse cx="244" cy="217" rx="13"/> inside <g id="eye">'],
    )
    text = "\n".join(b.get("text", "") for b in blocks)
    assert "paint nothing" in text
    assert 'cx="244"' in text


def test_the_prompt_says_nothing_when_everything_paints():
    from vectrify.formats.svg.prompts import build_svg_gen_prompt

    blocks = build_svg_gen_prompt(
        "data:image/png;base64,AA", 1, svg_prev="<svg/>", invisible=[]
    )
    assert "paint nothing" not in "\n".join(b.get("text", "") for b in blocks)


def test_a_first_draft_prompt_never_carries_the_report():
    """There is no parent to inspect, so the section cannot apply."""
    from vectrify.formats.svg.prompts import build_svg_gen_prompt

    blocks = build_svg_gen_prompt(
        "data:image/png;base64,AA", 1, invisible=["<circle/> inside <g id='x'>"]
    )
    assert "paint nothing" not in "\n".join(b.get("text", "") for b in blocks)
