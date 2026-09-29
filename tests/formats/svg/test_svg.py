import io
from xml.etree import ElementTree as ET

import pytest
from PIL import Image

from tests.helpers import TEST_MODEL as _MODEL
from vectrify.formats.svg.operations import apply_crossover, apply_mutation
from vectrify.formats.svg.prompts import (
    build_svg_gen_prompt,
    extract_svg_fragment,
    is_valid_svg,
)
from vectrify.formats.svg.replies import NoUsableOutputError, apply_edit, extract_svg
from vectrify.image_utils import rasterize_svg_to_png_bytes

NS = "http://www.w3.org/2000/svg"
SVG = f'<svg xmlns="{NS}" viewBox="0 0 32 32"><rect width="32" height="32"/></svg>'


def _make_image_data_url(color: str = "blue", size: int = 32) -> str:
    import base64

    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def test_rasterize_renders_at_the_requested_size():
    png = rasterize_svg_to_png_bytes(SVG, out_w=24, out_h=16)
    assert Image.open(io.BytesIO(png)).size == (24, 16)


def test_validate_accepts_svg_and_rejects_other_roots():
    assert is_valid_svg(SVG) == (True, None)
    ok, err = is_valid_svg("<html></html>")
    assert not ok
    assert err


def test_validate_reports_a_parse_error():
    ok, err = is_valid_svg("<svg><rect></svg>")
    assert not ok
    assert err is not None
    assert "parse error" in err


def test_extract_from_llm_strips_prose_and_fences():
    """Compared as markup, not as text: what comes back has been normalised for
    local search, so its serialisation differs from the model's."""
    raw = f"Sure, here you go:\n```xml\n{SVG}\n```\nHope that helps!"

    got = ET.fromstring(extract_svg(raw))
    want = ET.fromstring(SVG)

    assert [el.tag for el in got.iter()] == [el.tag for el in want.iter()]


def test_apply_edit_patches_the_parent_with_a_diff_block():
    raw = '<<<SEARCH>>>\n<rect width="32"\n<<<REPLACE>>>\n<rect width="16"\n<<<END>>>'
    result = apply_edit(SVG, raw)
    assert 'width="16"' in result
    assert result.startswith("<svg")


def test_apply_edit_falls_back_to_a_whole_document():
    whole = f'<svg xmlns="{NS}"><circle r="4"/></svg>'

    got = ET.fromstring(apply_edit(SVG, f"Here it is:\n{whole}"))

    want = ET.fromstring(whole)
    assert [el.tag for el in got.iter()] == [el.tag for el in want.iter()]
    circle = next(el for el in got.iter() if el.tag.endswith("circle"))
    assert circle.get("r") == "4"


def test_generate_prompt_carries_the_target_image_and_canvas():
    url = _make_image_data_url()
    blocks = build_svg_gen_prompt(
        url,
        iter_index=1,
        svg_prev=None,
        rasterized_svg_data_url=None,
        goal="make it blue",
        canvas=(64, 48),
    )
    text = "\n".join(b["text"] for b in blocks if b["type"] == "input_text")
    assert "viewBox='0 0 64 48'" in text
    assert "make it blue" in text
    assert [b["image_url"] for b in blocks if b["type"] == "input_image"] == [url]


def test_the_render_preview_is_only_sent_with_a_parent():
    target, preview = _make_image_data_url("red"), _make_image_data_url("green")
    fresh = build_svg_gen_prompt(
        target,
        iter_index=1,
        svg_prev=None,
        rasterized_svg_data_url=preview,
        goal=None,
        canvas=(32, 32),
    )
    refine = build_svg_gen_prompt(
        target,
        iter_index=2,
        svg_prev=SVG,
        rasterized_svg_data_url=preview,
        goal=None,
        canvas=(32, 32),
    )
    assert [b["image_url"] for b in fresh if b["type"] == "input_image"] == [target]
    assert [b["image_url"] for b in refine if b["type"] == "input_image"] == [
        target,
        preview,
    ]


def test_mutate_returns_valid_svg_and_a_summary():
    content, summary = apply_mutation(SVG)
    assert is_valid_svg(content)[0]
    assert summary.strip()


def test_crossover_returns_valid_svg_and_a_summary():
    other = f'<svg xmlns="{NS}" viewBox="0 0 32 32"><circle r="8"/></svg>'
    content, summary = apply_crossover(SVG, other)
    assert is_valid_svg(content)[0]
    assert summary.strip()


@pytest.mark.llm
def test_llm_svg_generation_produces_valid_svg():
    from vectrify.llm import LLMConfig, get_provider

    client = get_provider("openai")
    prompt = build_svg_gen_prompt(_make_image_data_url("blue"), iter_index=1)
    raw = client.generate(prompt, LLMConfig(model=_MODEL))
    svg = extract_svg_fragment(raw)
    valid, err = is_valid_svg(svg)
    assert valid, f"LLM did not produce valid SVG: {err}\nRaw: {raw[:200]}"


@pytest.mark.llm
def test_llm_svg_refinement_produces_valid_svg():
    from vectrify.llm import LLMConfig, get_provider

    ns = "http://www.w3.org/2000/svg"
    parent_svg = f'<svg xmlns="{ns}"><rect width="32" height="32" fill="blue"/></svg>'
    prompt = build_svg_gen_prompt(
        _make_image_data_url("red"),
        iter_index=2,
        svg_prev=parent_svg,
        goal="Make the fill color match the target image.",
    )
    client = get_provider("openai")
    raw = client.generate(prompt, LLMConfig(model=_MODEL))
    svg = apply_edit(parent_svg, raw)
    valid, err = is_valid_svg(svg)
    assert valid, f"LLM refinement did not produce valid SVG: {err}\nRaw: {raw[:200]}"


_PARENT = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
    '<circle cx="5" cy="5" r="2" /></svg>'
)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("", id="empty"),
        pytest.param("I cannot see the image clearly, so I am skipping.", id="prose"),
        pytest.param("```svg\n```", id="empty-fence"),
    ],
)
def test_a_reply_with_nothing_drawn_in_it_says_so(raw):
    """Not an XML complaint: nothing was drawn, so there is no markup to blame.

    The extractor falls back to returning the whole reply, so prose used to
    reach the parser and come back as "not well-formed (invalid token): line 1,
    column 1", which reads as a broken drawing.
    """
    for call in (
        lambda: apply_edit(_PARENT, raw),
        lambda: extract_svg(raw),
    ):
        with pytest.raises(NoUsableOutputError) as caught:
            call()
        assert "no diff blocks" in str(caught.value)


def test_the_error_quotes_what_came_back_instead():
    with pytest.raises(NoUsableOutputError, match="rate limit"):
        apply_edit(_PARENT, "Sorry, rate limit reached.")
    with pytest.raises(NoUsableOutputError, match="the reply was empty"):
        apply_edit(_PARENT, "   \n  ")


def test_a_usable_reply_is_still_applied():
    edited = apply_edit(_PARENT, '<<<SEARCH>>>\nr="2"\n<<<REPLACE>>>\nr="3"\n<<<END>>>')
    assert 'r="3"' in edited
    assert "<svg" in extract_svg("here it is\n" + _PARENT)


_STROKED = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<g id="curve" fill="none" stroke="#111111" stroke-width="3">'
    '<path d="M 8 32 C 20 20 44 20 56 32" /></g></svg>'
)

_FILLED = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<path fill="#111111" d="M 8 8 L 56 8 L 56 56 L 8 56 Z" /></svg>'
)
