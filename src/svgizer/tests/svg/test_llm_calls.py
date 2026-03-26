import io

import pytest
from PIL import Image

from svgizer.llm import LLMConfig, get_provider
from svgizer.svg.prompts import (
    build_summarize_prompt,
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
    """LLM + gen prompt → extract_svg_fragment → is_valid_svg."""
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
    """LLM + refinement prompt (existing SVG) → extract_svg_fragment → is_valid_svg."""
    client = get_provider("openai")
    image_data_url = _make_image_data_url("red")
    ns = "http://www.w3.org/2000/svg"
    parent_svg = f'<svg xmlns="{ns}"><rect width="32" height="32" fill="blue"/></svg>'
    prompt = build_svg_gen_prompt(
        image_data_url,
        iter_index=2,
        svg_prev=parent_svg,
        change_summary="Make the fill color match the target image.",
    )
    config = LLMConfig(model=_MODEL)
    raw = client.generate(prompt, config)
    svg = extract_svg_fragment(raw)
    valid, err = is_valid_svg(svg)
    assert valid, f"LLM refinement did not produce valid SVG: {err}\nRaw: {raw[:200]}"


@pytest.mark.llm
def test_llm_summarize_prompt_returns_nonempty_text():
    """LLM + summarize prompt → non-empty string."""
    client = get_provider("openai")
    image_data_url = _make_image_data_url("green")
    parent_svg_data_url = _make_image_data_url("blue")
    prompt = build_summarize_prompt(
        image_data_url,
        rasterized_svg_data_url=parent_svg_data_url,
        custom_goal="Make the SVG look more like the target.",
    )
    config = LLMConfig(model=_MODEL)
    result = client.generate(prompt, config)
    assert isinstance(result, str)
    assert len(result.strip()) > 0
