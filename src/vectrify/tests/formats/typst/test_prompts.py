from vectrify.formats.typst.prompts import build_typst_gen_prompt

_IMG_URL = "data:image/png;base64,abc"
_RENDER_URL = "data:image/png;base64,def"
_DIFF_URL = "data:image/png;base64,ghi"
_TYPST_CODE = "#rect(width: 10pt)"


def _text_blocks(blocks: list[dict]) -> list[str]:
    return [b["text"] for b in blocks if b.get("type") == "input_text"]


def _image_urls(blocks: list[dict]) -> list[str]:
    return [b["image_url"] for b in blocks if b.get("type") == "input_image"]


def test_gen_prompt_first_iteration_no_prev_code():
    blocks = build_typst_gen_prompt(
        _IMG_URL,
        node_index=1,
        typst_prev=None,
        rasterized_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    text = "\n".join(_text_blocks(blocks))
    assert "iteration #1" in text.lower()
    assert _IMG_URL in _image_urls(blocks)
    assert "output only the typst code block" in text.lower()


def test_gen_prompt_first_iteration_asks_for_fenced_code():
    blocks = build_typst_gen_prompt(
        _IMG_URL,
        node_index=1,
        typst_prev=None,
        rasterized_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    text = "\n".join(_text_blocks(blocks))
    assert "```typst" in text


def test_gen_prompt_refinement_includes_previous_code():
    blocks = build_typst_gen_prompt(
        _IMG_URL,
        node_index=3,
        typst_prev=_TYPST_CODE,
        rasterized_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    text = "\n".join(_text_blocks(blocks))
    assert _TYPST_CODE in text
    assert "3" in text


def test_gen_prompt_render_url_included():
    blocks = build_typst_gen_prompt(
        _IMG_URL,
        node_index=2,
        typst_prev=_TYPST_CODE,
        rasterized_data_url=_RENDER_URL,
        goal=None,
        diff_data_url=None,
    )
    assert _RENDER_URL in _image_urls(blocks)


def test_gen_prompt_diff_url_included():
    blocks = build_typst_gen_prompt(
        _IMG_URL,
        node_index=2,
        typst_prev=_TYPST_CODE,
        rasterized_data_url=None,
        goal=None,
        diff_data_url=_DIFF_URL,
    )
    assert _DIFF_URL in _image_urls(blocks)
    text = "\n".join(_text_blocks(blocks))
    assert "Difference map" in text


def test_gen_prompt_goal_included():
    blocks = build_typst_gen_prompt(
        _IMG_URL,
        node_index=2,
        typst_prev=_TYPST_CODE,
        rasterized_data_url=None,
        goal="make it red",
        diff_data_url=None,
    )
    text = "\n".join(_text_blocks(blocks))
    assert "make it red" in text


def test_gen_prompt_system_text_mentions_typst_rules():
    blocks = build_typst_gen_prompt(
        _IMG_URL,
        node_index=1,
        typst_prev=None,
        rasterized_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    text = "\n".join(_text_blocks(blocks))
    assert "#set page(width: auto, height: auto, margin: 0pt)" in text
    assert "NEVER use multiple pages" in text


def test_gen_prompt_diff_format_instructions_in_edit():
    blocks = build_typst_gen_prompt(
        _IMG_URL,
        node_index=2,
        typst_prev=_TYPST_CODE,
        rasterized_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    text = "\n".join(_text_blocks(blocks))
    assert "<<<SEARCH>>>" in text
    assert "<<<REPLACE>>>" in text
    assert "<<<END>>>" in text
    assert "search/replace diff blocks" in text
