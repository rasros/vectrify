from svgizer.formats.graphviz.prompts import build_dot_gen_prompt

_IMG_URL = "data:image/png;base64,abc"
_RENDER_URL = "data:image/png;base64,def"
_DIFF_URL = "data:image/png;base64,ghi"
_DOT = "digraph G { A -> B }"


def _text_blocks(blocks: list[dict]) -> list[str]:
    return [b["text"] for b in blocks if b.get("type") == "text"]


def _image_urls(blocks: list[dict]) -> list[str]:
    return [b["image_url"]["url"] for b in blocks if b.get("type") == "image_url"]


# ── build_dot_gen_prompt ──────────────────────────────────────────────────────


def test_gen_prompt_first_iteration_no_dot():
    blocks = build_dot_gen_prompt(
        _IMG_URL,
        node_index=1,
        dot_prev=None,
        rasterized_dot_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    text = "\n".join(_text_blocks(blocks))
    assert "iteration #1" in text.lower()
    assert _IMG_URL in _image_urls(blocks)


def test_gen_prompt_first_iteration_asks_for_fenced_code():
    blocks = build_dot_gen_prompt(
        _IMG_URL,
        node_index=1,
        dot_prev=None,
        rasterized_dot_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    text = "\n".join(_text_blocks(blocks))
    assert "```dot" in text


def test_gen_prompt_refinement_includes_previous_dot():
    blocks = build_dot_gen_prompt(
        _IMG_URL,
        node_index=3,
        dot_prev=_DOT,
        rasterized_dot_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    text = "\n".join(_text_blocks(blocks))
    assert _DOT in text
    assert "3" in text


def test_gen_prompt_render_url_included():
    blocks = build_dot_gen_prompt(
        _IMG_URL,
        node_index=2,
        dot_prev=_DOT,
        rasterized_dot_data_url=_RENDER_URL,
        goal=None,
        diff_data_url=None,
    )
    assert _RENDER_URL in _image_urls(blocks)


def test_gen_prompt_diff_url_included():
    blocks = build_dot_gen_prompt(
        _IMG_URL,
        node_index=2,
        dot_prev=_DOT,
        rasterized_dot_data_url=None,
        goal=None,
        diff_data_url=_DIFF_URL,
    )
    assert _DIFF_URL in _image_urls(blocks)


def test_gen_prompt_goal_included():
    blocks = build_dot_gen_prompt(
        _IMG_URL,
        node_index=2,
        dot_prev=_DOT,
        rasterized_dot_data_url=None,
        goal="add more nodes",
        diff_data_url=None,
    )
    text = "\n".join(_text_blocks(blocks))
    assert "add more nodes" in text


def test_gen_prompt_system_text_mentions_digraph():
    blocks = build_dot_gen_prompt(
        _IMG_URL,
        node_index=1,
        dot_prev=None,
        rasterized_dot_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    text = "\n".join(_text_blocks(blocks))
    assert "digraph" in text.lower()


def test_gen_prompt_warns_about_arrow_graph_mismatch():
    blocks = build_dot_gen_prompt(
        _IMG_URL,
        node_index=1,
        dot_prev=None,
        rasterized_dot_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    text = "\n".join(_text_blocks(blocks))
    assert "->" in text


def test_gen_prompt_diff_format_instructions_in_edit():
    blocks = build_dot_gen_prompt(
        _IMG_URL,
        node_index=2,
        dot_prev=_DOT,
        rasterized_dot_data_url=None,
        goal=None,
        diff_data_url=None,
    )
    text = "\n".join(_text_blocks(blocks))
    assert "<<<SEARCH>>>" in text
    assert "<<<REPLACE>>>" in text
    assert "<<<END>>>" in text
