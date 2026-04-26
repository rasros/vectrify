import importlib.util

import pytest

from vectrify.formats.typst.plugin import TypstPlugin

_TYPST_AVAILABLE = importlib.util.find_spec("typst") is not None

_VALID_TYPST = (
    "#set page(width: auto, height: auto, margin: 0pt)\n"
    "#rect(width: 10pt, height: 10pt, fill: red)"
)


def test_extract_fenced_typst_block():
    plugin = TypstPlugin()
    raw = "Here is the code:\n```typst\n#rect()\n```\nEnjoy!"
    result = plugin.extract_from_llm(raw)
    assert result == "#rect()"


def test_extract_fenced_typ_block():
    plugin = TypstPlugin()
    raw = "```typ\n#circle()\n```"
    result = plugin.extract_from_llm(raw)
    assert result == "#circle()"


def test_extract_fenced_case_insensitive():
    plugin = TypstPlugin()
    raw = "```TYPST\n#line()\n```"
    result = plugin.extract_from_llm(raw)
    assert result == "#line()"


def test_extract_fallback_returns_stripped_raw():
    plugin = TypstPlugin()
    raw = "  #rect()  "
    result = plugin.extract_from_llm(raw)
    assert result == "#rect()"


@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_validate_valid_typst():
    plugin = TypstPlugin()
    valid, err = plugin.validate(_VALID_TYPST)
    assert valid is True
    assert err is None


@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_validate_invalid_typst():
    plugin = TypstPlugin()
    valid, err = plugin.validate("#rect(width: missing_var)")
    assert valid is False
    assert err is not None


@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_rasterize_returns_png_bytes():
    plugin = TypstPlugin()
    png = plugin.rasterize(_VALID_TYPST, out_w=64, out_h=64)
    assert isinstance(png, bytes)
    assert png[:4] == b"\x89PNG"


@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_rasterize_output_dimensions():
    import io

    from PIL import Image

    plugin = TypstPlugin()
    png = plugin.rasterize(_VALID_TYPST, out_w=128, out_h=96)
    img = Image.open(io.BytesIO(png))
    assert img.size == (128, 96)


@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_rasterize_fast_returns_png_bytes():
    plugin = TypstPlugin()
    png = plugin.rasterize_fast(_VALID_TYPST, long_side=64)
    assert png is not None
    assert png[:4] == b"\x89PNG"


@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_rasterize_fast_returns_none_on_invalid():
    plugin = TypstPlugin()
    result = plugin.rasterize_fast("#invalid_syntax()", long_side=64)
    assert result is None


@pytest.mark.llm
@pytest.mark.skipif(not _TYPST_AVAILABLE, reason="typst package not installed")
def test_llm_typst_generation_produces_valid_typst():
    import base64
    import io

    from PIL import Image

    from vectrify.formats.typst.prompts import build_typst_gen_prompt
    from vectrify.llm import LLMConfig, get_provider

    img = Image.new("RGB", (32, 32), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    image_data_url = f"data:image/png;base64,{b64}"

    client = get_provider("openai")
    prompt = build_typst_gen_prompt(
        image_data_url,
        node_index=1,
        typst_prev=None,
        rasterized_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    raw = client.generate(prompt, LLMConfig(model="gpt-5.4-nano"))
    code = TypstPlugin().extract_from_llm(raw)
    valid, err = TypstPlugin().validate(code)
    assert valid, f"LLM did not produce valid Typst: {err}\nRaw: {raw[:200]}"
