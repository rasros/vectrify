import io

import pytest
from PIL import Image

from svgizer.formats.svg.plugin import SvgPlugin
from svgizer.formats.svg.prompts import (
    build_svg_gen_prompt,
    extract_svg_fragment,
    is_valid_svg,
)
from svgizer.llm import LLMConfig, get_provider

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
    client = get_provider("openai")
    image_data_url = _make_image_data_url("blue")
    prompt = build_svg_gen_prompt(image_data_url, iter_index=1)
    config = LLMConfig(model=_MODEL)
    raw = client.generate(prompt, config)
    svg = extract_svg_fragment(raw)
    valid, err = is_valid_svg(svg)
    assert valid, f"LLM did not produce valid SVG: {err}\nRaw: {raw[:200]}"


@pytest.mark.llm
def test_llm_svg_refinement_produces_valid_svg():
    client = get_provider("openai")
    image_data_url = _make_image_data_url("red")
    ns = "http://www.w3.org/2000/svg"
    parent_svg = f'<svg xmlns="{ns}"><rect width="32" height="32" fill="blue"/></svg>'
    prompt = build_svg_gen_prompt(
        image_data_url,
        iter_index=2,
        svg_prev=parent_svg,
        goal="Make the fill color match the target image.",
    )
    config = LLMConfig(model=_MODEL)
    raw = client.generate(prompt, config)
    plugin = SvgPlugin()
    svg = plugin.apply_edit(parent_svg, raw)
    valid, err = is_valid_svg(svg)
    assert valid, f"LLM refinement did not produce valid SVG: {err}\nRaw: {raw[:200]}"
