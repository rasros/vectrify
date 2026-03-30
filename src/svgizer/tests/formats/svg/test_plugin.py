import io

import pytest
from PIL import Image

from svgizer.formats.svg.plugin import SvgPlugin
from svgizer.formats.svg.prompts import (
    build_svg_gen_prompt,
    extract_svg_fragment,
    is_valid_svg,
)

_MODEL = "gpt-5.4-nano"


def _make_image_data_url(color: str = "blue", size: int = 32) -> str:
    import base64

    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


@pytest.mark.llm
def test_llm_svg_generation_produces_valid_svg():
    from svgizer.llm import LLMConfig, get_provider

    client = get_provider("openai")
    prompt = build_svg_gen_prompt(_make_image_data_url("blue"), iter_index=1)
    raw = client.generate(prompt, LLMConfig(model=_MODEL))
    svg = extract_svg_fragment(raw)
    valid, err = is_valid_svg(svg)
    assert valid, f"LLM did not produce valid SVG: {err}\nRaw: {raw[:200]}"


@pytest.mark.llm
def test_llm_svg_refinement_produces_valid_svg():
    from svgizer.llm import LLMConfig, get_provider

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
