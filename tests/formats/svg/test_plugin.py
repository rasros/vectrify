import io

import pytest
from PIL import Image

from tests.helpers import TEST_MODEL as _MODEL
from vectrify.formats.svg.plugin import SvgPlugin
from vectrify.formats.svg.prompts import (
    build_svg_gen_prompt,
    extract_svg_fragment,
    is_valid_svg,
)

NS = "http://www.w3.org/2000/svg"
SVG = f'<svg xmlns="{NS}" viewBox="0 0 32 32"><rect width="32" height="32"/></svg>'


def _make_image_data_url(color: str = "blue", size: int = 32) -> str:
    import base64

    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def test_plugin_identity_matches_the_files_it_writes():
    plugin = SvgPlugin()
    assert plugin.name == "svg"
    assert plugin.file_extension == ".svg"


def test_rasterize_renders_at_the_requested_size():
    png = SvgPlugin().rasterize(SVG, out_w=24, out_h=16)
    assert Image.open(io.BytesIO(png)).size == (24, 16)


def test_validate_accepts_svg_and_rejects_other_roots():
    plugin = SvgPlugin()
    assert plugin.validate(SVG) == (True, None)
    ok, err = plugin.validate("<html></html>")
    assert not ok
    assert err


def test_validate_reports_a_parse_error():
    ok, err = SvgPlugin().validate("<svg><rect></svg>")
    assert not ok
    assert err is not None
    assert "parse error" in err


def test_extract_from_llm_strips_prose_and_fences():
    raw = f"Sure, here you go:\n```xml\n{SVG}\n```\nHope that helps!"
    assert SvgPlugin().extract_from_llm(raw) == SVG


def test_apply_edit_patches_the_parent_with_a_diff_block():
    raw = '<<<SEARCH>>>\n<rect width="32"\n<<<REPLACE>>>\n<rect width="16"\n<<<END>>>'
    result = SvgPlugin().apply_edit(SVG, raw)
    assert 'width="16"' in result
    assert result.startswith("<svg")


def test_apply_edit_falls_back_to_a_whole_document():
    whole = f'<svg xmlns="{NS}"><circle r="4"/></svg>'
    assert SvgPlugin().apply_edit(SVG, f"Here it is:\n{whole}") == whole


def test_generate_prompt_carries_the_target_image_and_canvas():
    url = _make_image_data_url()
    blocks = SvgPlugin().build_generate_prompt(
        url,
        node_index=1,
        content_prev=None,
        raster_preview_url=None,
        goal="make it blue",
        canvas=(64, 48),
    )
    text = "\n".join(b["text"] for b in blocks if b["type"] == "input_text")
    assert "viewBox='0 0 64 48'" in text
    assert "make it blue" in text
    assert [b["image_url"] for b in blocks if b["type"] == "input_image"] == [url]


def test_the_render_preview_is_only_sent_with_a_parent():
    plugin = SvgPlugin()
    target, preview = _make_image_data_url("red"), _make_image_data_url("green")
    fresh = plugin.build_generate_prompt(
        target,
        node_index=1,
        content_prev=None,
        raster_preview_url=preview,
        goal=None,
        canvas=(32, 32),
    )
    refine = plugin.build_generate_prompt(
        target,
        node_index=2,
        content_prev=SVG,
        raster_preview_url=preview,
        goal=None,
        canvas=(32, 32),
    )
    assert [b["image_url"] for b in fresh if b["type"] == "input_image"] == [target]
    assert [b["image_url"] for b in refine if b["type"] == "input_image"] == [
        target,
        preview,
    ]


def test_mutate_returns_valid_svg_and_a_summary():
    content, summary = SvgPlugin().mutate(SVG)
    assert SvgPlugin().validate(content)[0]
    assert summary.strip()


def test_crossover_returns_valid_svg_and_a_summary():
    other = f'<svg xmlns="{NS}" viewBox="0 0 32 32"><circle r="8"/></svg>'
    content, summary = SvgPlugin().crossover(SVG, other)
    assert SvgPlugin().validate(content)[0]
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
    svg = SvgPlugin().apply_edit(parent_svg, raw)
    valid, err = is_valid_svg(svg)
    assert valid, f"LLM refinement did not produce valid SVG: {err}\nRaw: {raw[:200]}"
